from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import TrainerCallback

from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.data_io import load_movie_feature_store, required_movie_feature_columns
from utils.inference import build_title_lookup, extract_valid_titles, normalize_title_text
from utils.training_utils import Logger, ensure_dir, print_trainable_parameters, save_json, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA GRPO for title-based MovieRec recommendation.")
    parser.add_argument("--model-name-or-path", type=str, default="models/SFT/qwen3_4b_QLoRA/final")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/grpo_movielens_1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/grpo/qwen3_4b_QLoRA"))
    parser.add_argument("--logging-dir", type=Path, default=Path("outputs/grpo/qwen3_4b_QLoRA"))
    parser.add_argument("--max-completion-length", type=int, default=16)
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
    parser.add_argument("--valid-title-reward", type=float, default=0.1)
    parser.add_argument("--invalid-title-penalty", type=float, default=-0.5)
    parser.add_argument("--history-title-penalty", type=float, default=-0.3)
    parser.add_argument("--duplicate-title-penalty", type=float, default=-0.1)
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


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def first_valid_title(completion: Any, valid_title_by_normalized: dict[str, str]) -> str | None:
    titles = extract_valid_titles(extract_completion_text(completion), valid_title_by_normalized)
    return titles[0] if titles else None


def title_lookup(
    completion: Any,
    valid_title_by_normalized: dict[str, str],
    title_to_movie_id: dict[str, str],
) -> tuple[str | None, str | None]:
    title = first_valid_title(completion, valid_title_by_normalized)
    if title is None:
        return None, None
    return title, title_to_movie_id.get(normalize_title_text(title))


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


class RerankerScores:
    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.scores = data["scores"].astype(np.float32, copy=False)
        user_ids = [str(value) for value in data["user_ids"].tolist()]
        movie_ids = [str(value) for value in data["movie_ids"].tolist()]
        self.user_to_row = {user_id: index for index, user_id in enumerate(user_ids)}
        self.movie_to_col = {movie_id: index for index, movie_id in enumerate(movie_ids)}
        order = np.argsort(np.argsort(self.scores, axis=1), axis=1).astype(np.float32)
        self.percentiles = order / max(1, self.scores.shape[1] - 1)

    def percentile(self, user_id: Any, movie_id: Any) -> float:
        row = self.user_to_row.get(str(user_id))
        col = self.movie_to_col.get(str(movie_id))
        if row is None or col is None:
            return 0.0
        return float(self.percentiles[row, col])


def build_title_maps(raw_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    movie_features = load_movie_feature_store(raw_dir, required_movie_feature_columns({"NextMovieTitlePrediction"}))
    valid_titles = [movie_features.title(movie_id) for movie_id in movie_features.movie_ids]
    valid_title_by_normalized = build_title_lookup(valid_titles)
    title_to_movie_id = {normalize_title_text(movie_features.title(movie_id)): movie_id for movie_id in movie_features.movie_ids}
    return title_to_movie_id, valid_title_by_normalized


def build_exact_match_reward(valid_title_by_normalized: dict[str, str]):
    def exact_match_reward(completions: list[Any], target_movie_title: list[str], **_: Any) -> list[float]:
        rewards = []
        for completion, target in zip(completions, target_movie_title):
            title = first_valid_title(completion, valid_title_by_normalized)
            rewards.append(1.0 if title is not None and normalize_title_text(title) == normalize_title_text(target) else 0.0)
        return rewards

    return exact_match_reward


def build_validity_reward(
    args: argparse.Namespace,
    title_to_movie_id: dict[str, str],
    valid_title_by_normalized: dict[str, str],
    stats: RewardStats,
):
    def validity_reward(completions: list[Any], history_movie_titles: list[list[str]], **_: Any) -> list[float]:
        rewards = []
        valid_count = 0
        history_count = 0
        for completion, history_titles in zip(completions, history_movie_titles):
            title, movie_id = title_lookup(completion, valid_title_by_normalized, title_to_movie_id)
            if movie_id is None:
                rewards.append(args.invalid_title_penalty)
                continue
            valid_count += 1
            history = {normalize_title_text(title) for title in history_titles}
            if normalize_title_text(title) in history:
                history_count += 1
                rewards.append(args.history_title_penalty)
            else:
                rewards.append(args.valid_title_reward)
        total = max(1, len(completions))
        stats.add(valid_rate=valid_count / total, history_rate=history_count / total, validity_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return validity_reward


def build_duplicate_title_reward(
    args: argparse.Namespace,
    valid_title_by_normalized: dict[str, str],
    stats: RewardStats,
):
    def duplicate_title_reward(completions: list[Any], prompt: list[Any] | None = None, **_: Any) -> list[float]:
        if prompt is None:
            return [0.0] * len(completions)
        rewards = [0.0] * len(completions)
        duplicate_count = 0
        group_titles: dict[Any, set[str]] = {}
        for index, (completion, prompt_text) in enumerate(zip(completions, prompt)):
            title = first_valid_title(completion, valid_title_by_normalized)
            if title is None:
                continue
            normalized = normalize_title_text(title)
            seen = group_titles.setdefault(prompt_text, set())
            if normalized in seen:
                rewards[index] = args.duplicate_title_penalty
                duplicate_count += 1
            else:
                seen.add(normalized)
        stats.add(duplicate_rate=duplicate_count / max(1, len(completions)), duplicate_title_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return duplicate_title_reward


def build_reranker_reward(
    args: argparse.Namespace,
    title_to_movie_id: dict[str, str],
    valid_title_by_normalized: dict[str, str],
    stats: RewardStats,
):
    reranker = RerankerScores(args.reranker_score_path) if args.reranker_score_path is not None and args.reranker_reward_weight != 0.0 else None

    def reranker_reward(completions: list[Any], user_id: list[Any] | None = None, **_: Any) -> list[float]:
        if reranker is None or user_id is None:
            return [0.0] * len(completions)
        rewards = []
        matched = 0
        for completion, uid in zip(completions, user_id):
            _, movie_id = title_lookup(completion, valid_title_by_normalized, title_to_movie_id)
            if movie_id is None:
                rewards.append(0.0)
                continue
            matched += 1
            rewards.append(args.reranker_reward_weight * reranker.percentile(uid, movie_id))
        stats.add(reranker_match_rate=matched / max(1, len(completions)), reranker_reward_mean=float(np.mean(rewards)) if rewards else 0.0)
        return rewards

    return reranker_reward


def build_reward_funcs(
    args: argparse.Namespace,
    title_to_movie_id: dict[str, str],
    valid_title_by_normalized: dict[str, str],
) -> list:
    stats = RewardStats(args.reward_log_interval)
    return [
        build_exact_match_reward(valid_title_by_normalized),
        build_validity_reward(args, title_to_movie_id, valid_title_by_normalized, stats),
        build_duplicate_title_reward(args, valid_title_by_normalized, stats),
        build_reranker_reward(args, title_to_movie_id, valid_title_by_normalized, stats),
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

    logger("2. Load GRPO train.jsonl")
    train_dataset = load_grpo_train_dataset(args.data_dir)
    title_to_movie_id, valid_title_by_normalized = build_title_maps(args.raw_dir)
    if args.reranker_score_path is not None and "user_id" not in train_dataset.column_names:
        raise ValueError("GRPO data must be rebuilt with user_id to use SASRec reranker rewards.")

    logger("3. Load tokenizer and SFT policy model")
    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="left")
    model = load_causal_lm(
        ModelConfig(
            args.model_name_or_path,
            load_in_4bit=False,
            attn_implementation=args.attn_implementation,
            adapter_is_trainable=False,
        ),
        tokenizer=tokenizer,
    )
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    logger("4. Build GRPO LoRA and training configuration")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[name.strip() for name in args.target_modules.split(",") if name.strip()],
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    print_trainable_parameters(model, logger)

    training_args = build_grpo_config(args)

    logger("5. Start GRPO training")
    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=build_reward_funcs(args, title_to_movie_id, valid_title_by_normalized),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(FlushLogCallback())
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger("6. Save final GRPO adapter and tokenizer")
    final_dir = ensure_dir(args.output_dir / "final")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger(f"Saved final GRPO checkpoint to {final_dir}")


if __name__ == "__main__":
    main()
