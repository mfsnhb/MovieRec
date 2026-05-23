from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from model.llm import ModelConfig, collect_movie_tokens, load_causal_lm, load_tokenizer
from utils.training_utils import Logger, ensure_dir, init_wandb, print_trainable_parameters, save_json, setup_seed


RESPONSE_TEMPLATE = "\n\n### Response:\n"

PROMPT_TEMPLATE = """{instruction}

### Input:
{input}{response_template}{output}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for MovieRec Movie-ID recommendation data.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft/qwen2p5_7b"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--movie-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-movie-id", type=int)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--resume-from-checkpoint", type=str)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="MovieRec")
    return parser.parse_args()


def format_example(example: dict) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
        response_template=RESPONSE_TEMPLATE,
        output=example["output"],
    )


def main() -> None:
    args = parse_args()
    logger = Logger()
    if args.packing:
        raise ValueError("Completion-only loss masking is enabled, so --packing must stay disabled.")

    logger("1. Setup seed, output directory, and logging")
    setup_seed(args.seed)
    ensure_dir(args.output_dir)
    save_json(args.output_dir / "training_args.json", vars(args))
    wandb_run = init_wandb(args, run_name=args.output_dir.name)

    logger("2. Load SFT JSONL datasets")
    files = {"train": str(args.data_dir / "train.jsonl")}
    valid_path = args.data_dir / "valid.jsonl"
    if valid_path.exists():
        files["validation"] = str(valid_path)
    dataset = load_dataset("json", data_files=files)

    logger("3. Load tokenizer and model")
    movie_tokens = []
    if args.movie_tokens:
        movie_tokens = collect_movie_tokens(args.data_dir, args.raw_dir, args.max_movie_id)
        logger(f"Loaded {len(movie_tokens)} movie tokens")
    tokenizer = load_tokenizer(args.model_name_or_path, movie_tokens=movie_tokens, padding_side="right")
    model = load_causal_lm(
        ModelConfig(args.model_name_or_path, load_in_4bit=args.load_in_4bit),
        tokenizer=tokenizer,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    print_trainable_parameters(model, logger)

    logger("4. Build LoRA and SFT configuration")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[name.strip() for name in args.target_modules.split(",") if name.strip()],
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.min_learning_rate},
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        eval_steps=args.eval_interval,
        eval_strategy="steps" if "validation" in dataset else "no",
        save_strategy="steps",
        bf16=args.bf16,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=args.packing,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.output_dir.name,
    )

    logger("5. Start SFT training")
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_example,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger("6. Save final adapter and tokenizer")
    final_dir = ensure_dir(args.output_dir / "final")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if wandb_run is not None:
        wandb_run.finish()
    logger(f"Saved final SFT checkpoint to {final_dir}")


if __name__ == "__main__":
    main()
