from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.training_utils import Logger, ensure_dir, print_trainable_parameters, save_json, setup_seed


RESPONSE_TEMPLATE = "\n\n### Response:\n"

PROMPT_TEMPLATE = """{instruction}

### Input:
{input}{response_template}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for title-based MovieRec recommendation data.")
    parser.add_argument("--model-name-or-path", default="models/Qwen3-4B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("sft/qwen3_4b_QLoRA"))
    parser.add_argument("--max-train-examples", type=int)
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
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--max-steps", type=int, default=-1)
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


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


def load_sft_train_dataset(data_dir: Path):
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return load_dataset("json", data_files=str(train_path), split="train")


def to_prompt_completion(example: dict) -> dict[str, str]:
    return {
        "prompt": format_prompt(example),
        "completion": str(example["output"]),
    }


def prepare_train_dataset(args: argparse.Namespace, logger: Logger):
    dataset = load_sft_train_dataset(args.data_dir)
    if args.max_train_examples is not None:
        dataset = dataset.select(range(min(args.max_train_examples, len(dataset))))
        logger(f"Using {len(dataset)} training examples because --max-train-examples is set")

    dataset = dataset.map(
        to_prompt_completion,
        remove_columns=dataset.column_names,
        desc="Formatting SFT prompt-completion examples",
    )
    logger(f"Prepared SFT prompt-completion dataset with {len(dataset)} examples")
    return dataset


def sft_config_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr": args.min_learning_rate},
        "num_train_epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.log_interval,
        "save_steps": args.save_interval,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "bf16": args.bf16,
        "optim": args.optim,
        "gradient_checkpointing": args.gradient_checkpointing,
        "packing": args.packing,
        "max_steps": args.max_steps,
        "report_to": "wandb" if args.use_wandb else "none",
        "run_name": args.output_dir.name,
    }
    fields = set(getattr(SFTConfig, "__dataclass_fields__", {}))
    if "max_length" in fields:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in fields:
        kwargs["max_seq_length"] = args.max_seq_length
    if "completion_only_loss" in fields:
        kwargs["completion_only_loss"] = True
    if "remove_unused_columns" in fields:
        kwargs["remove_unused_columns"] = True
    return {key: value for key, value in kwargs.items() if key in fields}


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    return SFTConfig(**sft_config_kwargs(args))


def build_lora_config(args: argparse.Namespace, target_modules: list[str]) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def main() -> None:
    args = parse_args()
    logger = Logger()

    logger("1. Setup seed, output directory, and logging")
    setup_seed(args.seed)
    ensure_dir(args.output_dir)
    save_json(args.output_dir / "training_args.json", vars(args))
    if args.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    logger("2. Load tokenizer and model")
    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="right")
    model = load_causal_lm(
        ModelConfig(
            args.model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
            adapter_is_trainable=True,
        ),
        tokenizer=tokenizer,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    logger("3. Build SFT prompt-completion train dataset")
    train_dataset = prepare_train_dataset(args, logger)

    logger("4. Build LoRA and SFT configuration")
    target_modules = split_csv(args.target_modules)
    peft_config = build_lora_config(args, target_modules)
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
