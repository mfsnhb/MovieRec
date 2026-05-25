from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

from model.llm import ModelConfig, collect_movie_tokens, load_causal_lm, load_tokenizer
from utils.movie_generation import build_movie_token_id_map, patch_generate_with_movie_constraints
from utils.training_utils import Logger, ensure_dir, print_trainable_parameters, save_json, setup_seed


MOVIE_TOKEN_RE = re.compile(r"<movie_\d+>")
NUM_GENERATIONS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA GRPO for MovieRec generative recommendation.")
    parser.add_argument("--model-name-or-path", type=str, default="outputs/sft/qwen3_4b/final")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/grpo_movielens_1m"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grpo/qwen3_4b"))
    parser.add_argument("--max-completion-length", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--min-learning-rate", type=float, default=5e-7)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--movie-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constrained-generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
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
    return GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.min_learning_rate},
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        save_strategy="steps",
        save_total_limit=2,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        bf16=args.bf16,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.output_dir.name,
    )


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def parse_answer_token(text: str) -> str | None:
    token_match = MOVIE_TOKEN_RE.search(text)
    return token_match.group(0) if token_match else None


def rule_reward(completions: list[Any], target_movie_id: list[str], **_: Any) -> list[float]:
    rewards = []
    for completion, target in zip(completions, target_movie_id):
        predicted = parse_answer_token(extract_completion_text(completion))
        rewards.append(1.0 if predicted == target else 0.0)
    return rewards


def ndcg_rule_reward(completions: list[Any], target_movie_id: list[str], **_: Any) -> list[float]:
    penalties = [-1.0 / math.log2(rank + 2) for rank in range(NUM_GENERATIONS)]
    normalizer = sum(penalties)
    penalties = [-penalty / normalizer for penalty in penalties]
    rewards: list[float] = []
    for start in range(0, len(completions), NUM_GENERATIONS):
        group_completions = completions[start : start + NUM_GENERATIONS]
        group_targets = target_movie_id[start : start + NUM_GENERATIONS]
        group_rewards = []
        has_hit = False
        for rank, (completion, target) in enumerate(zip(group_completions, group_targets)):
            predicted = parse_answer_token(extract_completion_text(completion))
            if predicted == target:
                has_hit = True
                group_rewards.append(0.0)
            else:
                group_rewards.append(penalties[min(rank, len(penalties) - 1)])
        if has_hit:
            rewards.extend(group_rewards)
        else:
            rewards.extend([0.0] * len(group_rewards))
    return rewards


def main() -> None:
    args = parse_args()
    global NUM_GENERATIONS
    NUM_GENERATIONS = args.num_generations
    logger = Logger()

    logger("1. Setup seed, output directory, and logging")
    setup_seed(args.seed)
    ensure_dir(args.output_dir)
    save_json(args.output_dir / "training_args.json", vars(args))
    if args.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    logger("2. Load GRPO train.jsonl")
    train_dataset = load_grpo_train_dataset(args.data_dir)

    logger("3. Load tokenizer and policy model")
    movie_tokens = collect_movie_tokens(args.raw_dir) if args.movie_tokens else []
    tokenizer = load_tokenizer(args.model_name_or_path, movie_tokens=movie_tokens, padding_side="left")
    model = load_causal_lm(ModelConfig(args.model_name_or_path, load_in_4bit=args.load_in_4bit), tokenizer=tokenizer)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.constrained_generation:
        if not movie_tokens:
            raise ValueError("--constrained-generation requires --movie-tokens.")
        movie_token_ids = build_movie_token_id_map(tokenizer, movie_tokens).values()
        patch_generate_with_movie_constraints(model, movie_token_ids, [tokenizer.eos_token_id, tokenizer.pad_token_id])
        logger("Enabled constrained movie-ID generation for GRPO rollouts")
    print_trainable_parameters(model, logger)

    logger("4. Build LoRA and GRPO configuration")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[name.strip() for name in args.target_modules.split(",") if name.strip()],
    )
    training_args = build_grpo_config(args)

    logger("5. Start GRPO training")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[rule_reward, ndcg_rule_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger("6. Save final GRPO adapter and tokenizer")
    final_dir = ensure_dir(args.output_dir / "final")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger(f"Saved final GRPO checkpoint to {final_dir}")


if __name__ == "__main__":
    main()
