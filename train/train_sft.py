from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from model.llm import ModelConfig, collect_movie_tokens, load_causal_lm, load_tokenizer
from utils.training_utils import Logger, ensure_dir, print_trainable_parameters, save_json, setup_seed


RESPONSE_TEMPLATE = "\n\n### Response:\n"

PROMPT_TEMPLATE = """{instruction}

### Input:
{input}{response_template}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for MovieRec Movie-ID recommendation data.")
    parser.add_argument("--model-name-or-path", default="models/Qwen3-4B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft/qwen3_4b"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--movie-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-movie-id", type=int)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--packing-strategy", choices=["bfd", "bfd_split", "wrapped"], default="bfd")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--resume-from-checkpoint", type=str)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="MovieRec")
    return parser.parse_args()


def format_prompt(example: dict) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
        response_template=RESPONSE_TEMPLATE,
    )


def prompt_completion_fields(example: dict) -> dict[str, str]:
    return {
        "prompt": format_prompt(example),
        "completion": example["output"],
    }


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


def load_sft_train_dataset(data_dir: Path):
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return load_dataset("json", data_files=str(train_path), split="train")


def movie_token_ids(tokenizer, movie_tokens: Iterable[str]) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(list(movie_tokens))
    if isinstance(ids, int):
        ids = [ids]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    valid_ids = []
    for token_id in ids:
        if token_id is None:
            continue
        token_id = int(token_id)
        if unk_id is not None and token_id == unk_id:
            continue
        valid_ids.append(token_id)
    return sorted(set(valid_ids))


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    return SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.min_learning_rate},
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        save_strategy="steps",
        save_total_limit=2,
        bf16=args.bf16,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=args.packing,
        packing_strategy=args.packing_strategy,
        max_length=args.max_seq_length,
        completion_only_loss=True,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.output_dir.name,
    )


def build_lora_config(args: argparse.Namespace, target_modules: list[str], trainable_token_ids: list[int]) -> LoraConfig:
    kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules,
    }
    if trainable_token_ids:
        kwargs["trainable_token_indices"] = {
            "embed_tokens": trainable_token_ids,
            "lm_head": trainable_token_ids,
        }
    return LoraConfig(**kwargs)


def main() -> None:
    args = parse_args()
    logger = Logger()

    logger("1. Setup seed, output directory, and logging")
    setup_seed(args.seed)
    ensure_dir(args.output_dir)
    save_json(args.output_dir / "training_args.json", vars(args))
    if args.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    logger("2. Load SFT train.jsonl")
    train_dataset = load_sft_train_dataset(args.data_dir)
    train_dataset = train_dataset.map(prompt_completion_fields, remove_columns=train_dataset.column_names)

    logger("3. Load tokenizer and model")
    movie_tokens = []
    if args.movie_tokens:
        movie_tokens = collect_movie_tokens(args.data_dir, args.raw_dir, args.max_movie_id)
        logger(f"Loaded {len(movie_tokens)} movie tokens")
    tokenizer = load_tokenizer(args.model_name_or_path, movie_tokens=movie_tokens, padding_side="right")
    trainable_movie_token_ids = movie_token_ids(tokenizer, movie_tokens)
    if trainable_movie_token_ids:
        logger(f"Movie-token-only tuning enabled for {len(trainable_movie_token_ids)} token ids")
    model = load_causal_lm(
        ModelConfig(
            args.model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
        ),
        tokenizer=tokenizer,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)


    logger("4. Build LoRA and SFT configuration")
    target_modules = split_csv(args.target_modules)
    peft_config = build_lora_config(args, target_modules, trainable_movie_token_ids)
    training_args = build_sft_config(args)

    logger("5. Start SFT training")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    print_trainable_parameters(trainer.model, logger)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger("6. Save final adapter and tokenizer")
    final_dir = ensure_dir(args.output_dir / "final")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger(f"Saved final SFT checkpoint to {final_dir}")


if __name__ == "__main__":
    main()
