from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.data_io import load_movie_feature_store, required_movie_feature_columns
from utils.reranker_scores import RerankerScores
from utils.training_utils import Logger, ensure_dir, print_trainable_parameters, remove_path, save_json, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO for MovieRec movie-token ranking.")
    parser.add_argument("--model-name-or-path", type=str, default="models/sft/Qwen3-4B-SFT-QLoRA/final")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/grpo_movielens_1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/grpo/Qwen3_4B-GRPO"))
    parser.add_argument("--logging-dir", type=Path, default=Path("outputs/grpo/Qwen3-4B-GRPO"))
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mask-history-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--logging-first-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--reranker-score-path", type=Path)
    parser.add_argument("--reranker-reward-weight", type=float, default=0.05)
    parser.add_argument("--exact-match-reward-weight", type=float, default=1.0)
    parser.add_argument("--ndcg-reward-weight", type=float, default=0.5)
    parser.add_argument("--history-token-penalty", type=float, default=-0.3)
    parser.add_argument("--duplicate-token-penalty", type=float, default=-0.2)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="MovieRec")
    return parser.parse_args()


def load_grpo_train_dataset(data_dir: Path):
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return load_dataset("json", data_files=str(train_path), split="train")


def build_movie_maps(raw_dir: Path) -> tuple[list[str], list[str], dict[str, int]]:
    movie_features = load_movie_feature_store(raw_dir, required_movie_feature_columns({"NextMoviePrediction"}))
    movie_tokens = list(movie_features.movie_tokens)
    movie_ids = list(movie_features.movie_ids)
    return movie_tokens, movie_ids, {movie_id: index for index, movie_id in enumerate(movie_ids)}


class ItemGrpoCollator:
    def __init__(self, tokenizer, movie_id_to_index: dict[str, int], max_length: int) -> None:
        self.tokenizer = tokenizer
        self.movie_id_to_index = movie_id_to_index
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [example["prompt"] for example in examples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded["target_movie_index"] = torch.tensor(
            [self.movie_id_to_index[str(example["target_movie_id"])] for example in examples],
            dtype=torch.long,
        )
        encoded["history_movie_indices"] = [
            [self.movie_id_to_index[str(movie_id)] for movie_id in example.get("history_movie_ids", [])]
            for example in examples
        ]
        encoded["user_id"] = [str(example["user_id"]) for example in examples]
        return encoded


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    reversed_mask = attention_mask.flip(dims=[1])
    distance_from_end = reversed_mask.long().argmax(dim=1)
    return attention_mask.shape[1] - 1 - distance_from_end


def final_movie_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, movie_token_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    last_positions = last_non_padding_indices(attention_mask).to(input_ids.device)
    return outputs.logits[rows, last_positions, :].index_select(dim=-1, index=movie_token_ids)


def mask_history_logits(movie_logits: torch.Tensor, history_indices: list[list[int]]) -> torch.Tensor:
    masked = movie_logits.clone()
    for row, history in enumerate(history_indices):
        if history:
            masked[row, torch.tensor(history, device=movie_logits.device, dtype=torch.long)] = -torch.inf
    return masked


def sample_movie_indices(movie_logits: torch.Tensor, num_generations: int, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    sample_logits = movie_logits / max(temperature, 1e-6)
    sample_probs = F.softmax(sample_logits, dim=-1)
    sampled_indices = torch.multinomial(sample_probs, num_samples=num_generations, replacement=True)
    log_probs = F.log_softmax(movie_logits, dim=-1)
    return sampled_indices, log_probs.gather(dim=-1, index=sampled_indices)


def compute_rewards(
    *,
    args: argparse.Namespace,
    sampled_indices: torch.Tensor,
    sampled_log_probs: torch.Tensor,
    target_indices: torch.Tensor,
    history_indices: list[list[int]],
    user_ids: list[str],
    movie_ids: list[str],
    reranker: RerankerScores | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = sampled_indices.device
    batch_size, num_generations = sampled_indices.shape
    rewards = torch.zeros((batch_size, num_generations), device=device, dtype=torch.float32)

    exact_mask = sampled_indices == target_indices[:, None]
    exact_rewards = exact_mask.float() * args.exact_match_reward_weight
    rewards = rewards + exact_rewards

    ranked_indices = sampled_indices.gather(dim=-1, index=torch.argsort(sampled_log_probs.detach(), dim=-1, descending=True))
    ranked_matches = ranked_indices == target_indices[:, None]
    has_target = ranked_matches.any(dim=-1)
    target_rank = ranked_matches.float().argmax(dim=-1) + 1
    rank_values = torch.zeros(batch_size, device=device, dtype=torch.float32)
    if has_target.any():
        rank_values[has_target] = args.ndcg_reward_weight / torch.log2(target_rank[has_target].float() + 1.0)
    ndcg_rewards = torch.zeros_like(rewards)
    ndcg_rewards[exact_mask] = rank_values[:, None].expand_as(rewards)[exact_mask]
    rewards = rewards + ndcg_rewards

    history_mask = torch.zeros_like(rewards, dtype=torch.bool)
    for row, history in enumerate(history_indices):
        if history:
            history_tensor = torch.tensor(history, device=device, dtype=torch.long)
            history_mask[row] = (sampled_indices[row, :, None] == history_tensor[None, :]).any(dim=-1)
    history_rewards = history_mask.float() * args.history_token_penalty
    rewards = rewards + history_rewards

    duplicate_mask = torch.zeros_like(rewards, dtype=torch.bool)
    for position in range(num_generations):
        duplicate_mask[:, position] = (sampled_indices[:, :position] == sampled_indices[:, position : position + 1]).any(dim=-1)
    duplicate_rewards = duplicate_mask.float() * args.duplicate_token_penalty
    rewards = rewards + duplicate_rewards

    reranker_rewards = torch.zeros_like(rewards)
    if reranker is not None and args.reranker_reward_weight != 0.0:
        values = []
        for row in range(batch_size):
            values.append(
                [
                    args.reranker_reward_weight * reranker.percentile(user_ids[row], movie_ids[int(movie_index)])
                    for movie_index in sampled_indices[row].detach().cpu().tolist()
                ]
            )
        reranker_rewards = torch.tensor(values, device=device, dtype=torch.float32)
        rewards = rewards + reranker_rewards

    reward_std = rewards.std(dim=-1)
    return rewards, {
        "exact_rate": float(exact_mask.float().mean().item()),
        "target_in_group_rate": float(has_target.float().mean().item()),
        "history_rate": float(history_mask.float().mean().item()),
        "duplicate_rate": float(duplicate_mask.float().mean().item()),
        "exact_reward": float(exact_rewards.mean().item()),
        "ndcg_reward": float(ndcg_rewards.mean().item()),
        "reranker_reward": float(reranker_rewards.mean().item()),
        "history_reward": float(history_rewards.mean().item()),
        "duplicate_reward": float(duplicate_rewards.mean().item()),
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(reward_std.mean().item()),
    }


def group_normalized_advantages(rewards: torch.Tensor) -> torch.Tensor:
    reward_mean = rewards.mean(dim=-1, keepdim=True)
    reward_std = rewards.std(dim=-1, keepdim=True)
    return torch.where(reward_std > 1e-6, (rewards - reward_mean) / reward_std.clamp_min(1e-6), torch.zeros_like(rewards))


def save_checkpoint(model, tokenizer, output_dir: Path, step: int, logger: Logger) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    remove_path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger(f"Saved checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def save_final(checkpoint_dir: Path, tokenizer, output_dir: Path, data_dir: Path, logger: Logger) -> Path:
    final_dir = output_dir / "final"
    remove_path(final_dir)
    shutil.move(str(checkpoint_dir), str(final_dir))
    tokenizer.save_pretrained(final_dir)
    for filename in ["movie_tokens.json", "user_tokens.json"]:
        source = data_dir / filename
        if source.exists():
            shutil.copyfile(source, final_dir / filename)
    logger(f"Moved final checkpoint {checkpoint_dir} to {final_dir}")
    return final_dir


def train_item_grpo(args: argparse.Namespace, model, tokenizer, train_dataset, movie_token_ids: list[int], movie_id_to_index: dict[str, int], movie_ids: list[str], logger: Logger) -> None:
    device = model.device
    ref_model = load_causal_lm(
        ModelConfig(args.model_name_or_path, load_in_4bit=args.load_in_4bit, attn_implementation=args.attn_implementation),
        tokenizer=tokenizer,
    )
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    reranker = RerankerScores(args.reranker_score_path) if args.reranker_score_path is not None and args.reranker_reward_weight != 0.0 else None
    movie_token_tensor = torch.tensor(movie_token_ids, device=device, dtype=torch.long)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ItemGrpoCollator(tokenizer, movie_id_to_index, args.max_seq_length),
    )
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(dataloader) / max(1, args.accumulation_steps))
    total_steps = int(math.ceil(args.epochs * steps_per_epoch))
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    totals: dict[str, float] = {}
    running_calls = 0
    last_checkpoint: Path | None = None

    while global_step < total_steps:
        for batch in dataloader:
            if global_step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_indices = batch["target_movie_index"].to(device)

            movie_logits = final_movie_logits(model, input_ids, attention_mask, movie_token_tensor)
            if args.mask_history_actions:
                movie_logits = mask_history_logits(movie_logits, batch["history_movie_indices"])
            sampled_indices, sampled_log_probs = sample_movie_indices(movie_logits, args.num_generations, args.temperature)

            with torch.no_grad():
                ref_movie_logits = final_movie_logits(ref_model, input_ids, attention_mask, movie_token_tensor)
                if args.mask_history_actions:
                    ref_movie_logits = mask_history_logits(ref_movie_logits, batch["history_movie_indices"])
                sampled_ref_log_probs = F.log_softmax(ref_movie_logits, dim=-1).gather(dim=-1, index=sampled_indices)

            rewards, reward_stats = compute_rewards(
                args=args,
                sampled_indices=sampled_indices,
                sampled_log_probs=sampled_log_probs,
                target_indices=target_indices,
                history_indices=batch["history_movie_indices"],
                user_ids=batch["user_id"],
                movie_ids=movie_ids,
                reranker=reranker,
            )
            advantages = group_normalized_advantages(rewards)
            policy_loss = -(advantages.detach() * sampled_log_probs).mean()
            log_ratio = sampled_ref_log_probs - sampled_log_probs
            kl_loss = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
            loss = policy_loss + args.beta * kl_loss
            (loss / max(1, args.accumulation_steps)).backward()

            micro_step += 1
            running_calls += 1
            current_stats = {
                **reward_stats,
                "advantage_abs_mean": float(advantages.abs().mean().item()),
                "kl": float(kl_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "loss": float(loss.item()),
            }
            for key, value in current_stats.items():
                totals[key] = totals.get(key, 0.0) + value

            if micro_step % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_interval == 0 or (args.logging_first_step and global_step == 1):
                    summary = {key: value / max(1, running_calls) for key, value in sorted(totals.items())}
                    summary["step"] = global_step
                    summary["learning_rate"] = scheduler.get_last_lr()[0]
                    print("TRAIN_METRICS", json.dumps(summary, ensure_ascii=False), flush=True)
                    totals.clear()
                    running_calls = 0
                if global_step % args.save_interval == 0:
                    last_checkpoint = save_checkpoint(model, tokenizer, args.output_dir, global_step, logger)

    if last_checkpoint is None or last_checkpoint.name != f"checkpoint-{global_step}":
        last_checkpoint = save_checkpoint(model, tokenizer, args.output_dir, global_step, logger)
    save_final(last_checkpoint, tokenizer, args.output_dir, args.data_dir, logger)


def main() -> None:
    args = parse_args()
    logger = Logger()

    logger("1. Setup seed, output directory, and logging")
    setup_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.logging_dir)
    save_json(args.logging_dir / "training_args.json", vars(args))
    if args.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HF_LOGGING_VERBOSITY", "info")

    logger("2. Load GRPO train.jsonl and movie token catalog")
    train_dataset = load_grpo_train_dataset(args.data_dir)
    movie_tokens, movie_ids, movie_id_to_index = build_movie_maps(args.raw_dir)
    if args.reranker_score_path is not None and "user_id" not in train_dataset.column_names:
        raise ValueError("GRPO data must include user_id to use SASRec reranker rewards.")

    logger("3. Load tokenizer and SFT policy model")
    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="left")
    tokenizer.add_tokens(movie_tokens, special_tokens=False)
    user_tokens_path = Path(args.model_name_or_path) / "user_tokens.json"
    if user_tokens_path.exists():
        tokenizer.add_tokens([record["user_token"] for record in json.loads(user_tokens_path.read_text(encoding="utf-8"))], special_tokens=False)
    movie_token_ids = [int(token_id) for token_id in tokenizer.convert_tokens_to_ids(movie_tokens)]
    model = load_causal_lm(
        ModelConfig(
            args.model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
            adapter_is_trainable=True,
        ),
        tokenizer=tokenizer,
    )
    if not isinstance(model, PeftModel):
        raise ValueError("Item-space GRPO expects a PEFT/adapter SFT checkpoint as --model-name-or-path.")

    logger("4. Start item-space GRPO training")
    print_trainable_parameters(model, logger)
    train_item_grpo(args, model, tokenizer, train_dataset, movie_token_ids, movie_id_to_index, movie_ids, logger)


if __name__ == "__main__":
    main()
