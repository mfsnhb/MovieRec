from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import TrainerCallback

from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.data_io import load_movie_feature_store, required_movie_feature_columns
from utils.reranker_scores import RerankerScores
from utils.training_utils import Logger, ensure_dir, print_trainable_parameters, save_json, save_last_checkpoint_as_final, setup_seed


MOVIE_TOKEN_RE = re.compile(r"\bmovie_\d+\b")
NUMBERED_MOVIE_LINE_RE = re.compile(r"^\s*\d+\.\s*(movie_\d+)\s*\|\s*\S+", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA GRPO for MovieRec movie ID token recommendation.")
    parser.add_argument("--model-name-or-path", type=str, default="models/sft/qwen3_4b_QLoRA/final")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/grpo_movielens_1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/grpo/qwen3_4b_QLoRA"))
    parser.add_argument("--logging-dir", type=Path, default=Path("outputs/grpo/qwen3_4b_QLoRA"))
    parser.add_argument("--max-completion-length", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--min-learning-rate", type=float, default=5e-7)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--logging-first-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-log-interval", type=int, default=10)
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-completions-to-print", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--reranker-score-path", type=Path)
    parser.add_argument("--reranker-reward-weight", type=float, default=0.1)
    parser.add_argument("--ndcg-reward-weight", type=float, default=1.0)
    parser.add_argument("--exact-first-reward-weight", type=float, default=0.2)
    parser.add_argument("--format-reward-weight", type=float, default=0.2)
    parser.add_argument("--unique-list-reward-weight", type=float, default=0.2)
    parser.add_argument("--valid-list-reward-weight", type=float, default=0.1)
    parser.add_argument("--invalid-token-penalty", type=float, default=-0.5)
    parser.add_argument("--history-token-penalty", type=float, default=-0.3)
    parser.add_argument("--resume-from-checkpoint", type=str)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="MovieRec")
    return parser.parse_args()


def load_grpo_train_dataset(data_dir: Path):
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return load_dataset("json", data_files=str(train_path), split="train")


def build_grpo_config(args: argparse.Namespace) -> GRPOConfig:
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(args.output_dir),
        logging_dir=str(args.logging_dir),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.min_learning_rate},
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        logging_steps=args.log_interval,
        logging_first_step=args.logging_first_step,
        save_steps=args.save_interval,
        save_strategy="steps",
        save_total_limit=2,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.output_dir.name,
    )


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


def build_lora_config(args: argparse.Namespace, movie_token_ids: list[int]) -> LoraConfig:
    kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": split_csv(args.target_modules),
    }
    fields = set(getattr(LoraConfig, "__dataclass_fields__", {}))
    if movie_token_ids:
        if "trainable_token_indices" not in fields:
            raise RuntimeError(
                "The installed PEFT version does not support trainable_token_indices. "
                "Use the remote PEFT version used for the experiments, or upgrade PEFT."
            )
        kwargs["trainable_token_indices"] = movie_token_ids
    return LoraConfig(**kwargs)


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def extract_valid_movie_tokens(completion: Any, valid_movie_tokens: set[str]) -> list[str]:
    tokens: list[str] = []
    for match in MOVIE_TOKEN_RE.finditer(extract_completion_text(completion)):
        token = match.group(0)
        if token in valid_movie_tokens:
            tokens.append(token)
    return tokens


def unique_valid_movie_tokens(completion: Any, valid_movie_tokens: set[str]) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in extract_valid_movie_tokens(completion, valid_movie_tokens):
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def first_valid_movie_token(completion: Any, valid_movie_tokens: set[str]) -> str | None:
    tokens = unique_valid_movie_tokens(completion, valid_movie_tokens)
    return tokens[0] if tokens else None


class RewardStats:
    def __init__(self, log_interval: int) -> None:
        self.log_interval = max(1, log_interval)
        self.calls = 0
        self.totals: dict[str, float] = {}

    def add(self, **values: float) -> None:
        self.calls += 1
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
        if self.calls % self.log_interval == 0:
            summary = {key: value / self.calls for key, value in sorted(self.totals.items())}
            print("REWARD_STATS", json.dumps(summary, ensure_ascii=False), flush=True)


def build_movie_token_maps(raw_dir: Path) -> tuple[dict[str, str], set[str], list[str]]:
    movie_features = load_movie_feature_store(raw_dir, required_movie_feature_columns({"NextMoviePrediction"}))
    token_to_movie_id = movie_features.movie_token_to_id
    movie_tokens = list(movie_features.movie_tokens)
    return token_to_movie_id, set(movie_tokens), movie_tokens


def build_list_ndcg_reward(args: argparse.Namespace, valid_movie_tokens: set[str], stats: RewardStats):
    def ndcg_reward(completions: list[Any], target_movie_token: list[str], **_: Any) -> list[float]:
        rewards = []
        hits = 0
        ranks = []
        for completion, target in zip(completions, target_movie_token):
            tokens = unique_valid_movie_tokens(completion, valid_movie_tokens)
            rank = tokens.index(target) + 1 if target in tokens else None
            if rank is None:
                rewards.append(0.0)
                continue
            hits += 1
            ranks.append(rank)
            rewards.append(args.ndcg_reward_weight / np.log2(rank + 1))
        stats.add(ndcg_hit_rate=hits / max(1, len(completions)), ndcg_rank_mean=float(np.mean(ranks)) if ranks else 0.0, ndcg_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return ndcg_reward


def build_exact_first_reward(args: argparse.Namespace, valid_movie_tokens: set[str]):
    def exact_first_reward(completions: list[Any], target_movie_token: list[str], **_: Any) -> list[float]:
        rewards = []
        for completion, target in zip(completions, target_movie_token):
            rewards.append(args.exact_first_reward_weight if first_valid_movie_token(completion, valid_movie_tokens) == target else 0.0)
        return rewards

    return exact_first_reward


def build_format_reward(args: argparse.Namespace, valid_movie_tokens: set[str], stats: RewardStats):
    def format_reward(completions: list[Any], top_k: list[int] | None = None, **_: Any) -> list[float]:
        rewards = []
        format_rates = []
        for index, completion in enumerate(completions):
            expected_k = int(top_k[index]) if top_k is not None else 10
            text = extract_completion_text(completion)
            lines = [line for line in text.splitlines() if line.strip()]
            numbered = NUMBERED_MOVIE_LINE_RE.findall(text)
            valid_numbered = [token for token in numbered if token in valid_movie_tokens]
            rate = min(1.0, len(valid_numbered) / max(1, expected_k))
            if lines and len(lines) > expected_k + 2:
                rate *= 0.8
            format_rates.append(rate)
            rewards.append(args.format_reward_weight * rate)
        stats.add(format_rate=float(np.mean(format_rates)) if format_rates else 0.0, format_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return format_reward


def build_unique_list_reward(args: argparse.Namespace, valid_movie_tokens: set[str], stats: RewardStats):
    def unique_list_reward(completions: list[Any], top_k: list[int] | None = None, **_: Any) -> list[float]:
        rewards = []
        unique_rates = []
        for index, completion in enumerate(completions):
            expected_k = int(top_k[index]) if top_k is not None else 10
            tokens = extract_valid_movie_tokens(completion, valid_movie_tokens)
            unique_count = len(dict.fromkeys(tokens[:expected_k]))
            rate = unique_count / max(1, expected_k)
            unique_rates.append(rate)
            rewards.append(args.unique_list_reward_weight * rate)
        stats.add(unique_rate=float(np.mean(unique_rates)) if unique_rates else 0.0, unique_list_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return unique_list_reward


def build_list_validity_reward(args: argparse.Namespace, valid_movie_tokens: set[str], stats: RewardStats):
    def list_validity_reward(completions: list[Any], top_k: list[int] | None = None, **_: Any) -> list[float]:
        rewards = []
        valid_rates = []
        for index, completion in enumerate(completions):
            expected_k = int(top_k[index]) if top_k is not None else 10
            raw_tokens = MOVIE_TOKEN_RE.findall(extract_completion_text(completion))
            valid_count = sum(token in valid_movie_tokens for token in raw_tokens[:expected_k])
            rate = valid_count / max(1, min(expected_k, len(raw_tokens))) if raw_tokens else 0.0
            valid_rates.append(rate)
            rewards.append(args.valid_list_reward_weight * rate if raw_tokens else args.invalid_token_penalty)
        stats.add(list_valid_rate=float(np.mean(valid_rates)) if valid_rates else 0.0, list_validity_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return list_validity_reward


def build_history_exclusion_reward(args: argparse.Namespace, valid_movie_tokens: set[str], stats: RewardStats):
    def history_exclusion_reward(completions: list[Any], history_movie_tokens: list[list[str]], **_: Any) -> list[float]:
        rewards = []
        repeat_rates = []
        for completion, history_tokens in zip(completions, history_movie_tokens):
            tokens = unique_valid_movie_tokens(completion, valid_movie_tokens)
            history = set(history_tokens)
            repeat_count = sum(token in history for token in tokens)
            repeat_rate = repeat_count / max(1, len(tokens)) if tokens else 0.0
            repeat_rates.append(repeat_rate)
            rewards.append(args.history_token_penalty * repeat_count)
        stats.add(history_repeat_rate=float(np.mean(repeat_rates)) if repeat_rates else 0.0, history_exclusion_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return history_exclusion_reward


def build_reranker_reward(
    args: argparse.Namespace,
    token_to_movie_id: dict[str, str],
    valid_movie_tokens: set[str],
    stats: RewardStats,
):
    reranker = RerankerScores(args.reranker_score_path) if args.reranker_score_path is not None and args.reranker_reward_weight != 0.0 else None

    def reranker_reward(completions: list[Any], user_id: list[Any] | None = None, top_k: list[int] | None = None, **_: Any) -> list[float]:
        if reranker is None or user_id is None:
            return [0.0] * len(completions)
        rewards = []
        matched = 0
        list_lengths = []
        for index, (completion, uid) in enumerate(zip(completions, user_id)):
            expected_k = int(top_k[index]) if top_k is not None else 10
            movie_ids = [token_to_movie_id[token] for token in unique_valid_movie_tokens(completion, valid_movie_tokens)[:expected_k] if token in token_to_movie_id]
            list_lengths.append(len(movie_ids))
            if not movie_ids:
                rewards.append(0.0)
                continue
            matched += 1
            rewards.append(args.reranker_reward_weight * float(np.mean([reranker.percentile(uid, movie_id) for movie_id in movie_ids])))
        stats.add(reranker_match_rate=matched / max(1, len(completions)), reranker_list_len_mean=float(np.mean(list_lengths)) if list_lengths else 0.0, reranker_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return reranker_reward


def build_reward_funcs(
    args: argparse.Namespace,
    token_to_movie_id: dict[str, str],
    valid_movie_tokens: set[str],
) -> list:
    stats = RewardStats(args.reward_log_interval)
    return [
        build_list_ndcg_reward(args, valid_movie_tokens, stats),
        build_exact_first_reward(args, valid_movie_tokens),
        build_format_reward(args, valid_movie_tokens, stats),
        build_unique_list_reward(args, valid_movie_tokens, stats),
        build_list_validity_reward(args, valid_movie_tokens, stats),
        build_history_exclusion_reward(args, valid_movie_tokens, stats),
        build_reranker_reward(args, token_to_movie_id, valid_movie_tokens, stats),
    ]


class FlushLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print("TRAIN_METRICS", json.dumps({"step": state.global_step, **logs}, ensure_ascii=False), flush=True)


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
    token_to_movie_id, valid_movie_tokens, movie_tokens = build_movie_token_maps(args.raw_dir)
    if args.reranker_score_path is not None and "user_id" not in train_dataset.column_names:
        raise ValueError("GRPO data must be rebuilt with user_id to use SASRec reranker rewards.")

    logger("3. Load tokenizer and SFT policy model")
    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="left")
    tokenizer.add_tokens(movie_tokens, special_tokens=False)
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
    is_peft_model = isinstance(model, PeftModel)
    if args.load_in_4bit and not is_peft_model:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    logger("4. Build GRPO LoRA and training configuration")
    peft_config = None if is_peft_model else build_lora_config(args, movie_token_ids)
    training_args = build_grpo_config(args)

    logger("5. Start GRPO training")
    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=build_reward_funcs(args, token_to_movie_id, valid_movie_tokens),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(FlushLogCallback())
    print_trainable_parameters(trainer.model, logger)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger("6. Save final GRPO adapter and tokenizer once")
    final_dir = save_last_checkpoint_as_final(trainer, tokenizer, args.output_dir, logger)
    save_json(final_dir / "movie_tokens.json", [{"movie_token": token, "movie_id": token_to_movie_id[token]} for token in movie_tokens])
    logger(f"Final GRPO model is available at {final_dir}")


if __name__ == "__main__":
    main()
