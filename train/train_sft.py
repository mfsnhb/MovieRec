from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.training_utils import Logger, ensure_dir, print_trainable_parameters, save_json, save_last_checkpoint_as_final, setup_seed


RESPONSE_TEMPLATE = "\n\n### Response:\n"

PROMPT_TEMPLATE = """{instruction}

### Input:
{input}{response_template}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for MovieRec recommendation data.")
    parser.add_argument("--model-name-or-path", default="models/Qwen3-4B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--movie-token-file", type=Path)
    parser.add_argument("--movie-embedding-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/sft/qwen3_4b_QLoRA"))
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["steps", "no"], default="steps")
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--peft-mode", choices=["lora", "trainable_tokens", "full"], default="lora")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", type=str)
    parser.add_argument("--skip-final-save", action="store_true")
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


def load_movie_tokens(data_dir: Path, movie_token_file: Path | None) -> list[str]:
    path = movie_token_file or data_dir / "movie_tokens.json"
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    tokens = []
    for record in records:
        token = record["movie_token"] if isinstance(record, dict) else str(record)
        if token not in tokens:
            tokens.append(token)
    return tokens


def add_movie_tokens(tokenizer, movie_tokens: list[str], logger: Logger) -> list[int]:
    if not movie_tokens:
        logger("No movie token file found; tokenizer vocabulary is left unchanged")
        return []
    added = tokenizer.add_tokens(movie_tokens, special_tokens=False)
    token_ids = tokenizer.convert_tokens_to_ids(movie_tokens)
    logger(f"Loaded {len(movie_tokens)} movie tokens; tokenizer.add_tokens added {added} new tokens")
    return [int(token_id) for token_id in token_ids]


def load_movie_embedding_table(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Movie embedding file not found: {path}")
    data = np.load(path, allow_pickle=False)
    if "embeddings" not in data:
        raise ValueError(f"{path} is missing the 'embeddings' array")
    embeddings = data["embeddings"]
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embeddings array in {path}, got shape {embeddings.shape}")
    if "movie_tokens" in data:
        tokens = [str(token) for token in data["movie_tokens"]]
    elif "movie_ids" in data:
        tokens = [f"movie_{movie_id}" for movie_id in data["movie_ids"]]
    else:
        raise ValueError(f"{path} must include either 'movie_tokens' or 'movie_ids'")
    if len(tokens) != embeddings.shape[0]:
        raise ValueError(f"Embedding row count {embeddings.shape[0]} does not match token count {len(tokens)} in {path}")
    return {token: np.asarray(vector, dtype=np.float32) for token, vector in zip(tokens, embeddings, strict=True)}


def initialize_movie_token_embeddings(
    model,
    tokenizer,
    movie_tokens: list[str],
    movie_embedding_file: Path | None,
    logger: Logger,
) -> dict[str, object] | None:
    if movie_embedding_file is None:
        return None
    if not movie_tokens:
        raise ValueError("--movie-embedding-file requires movie tokens to be loaded first")

    embedding_table = load_movie_embedding_table(movie_embedding_file)
    missing_tokens = [token for token in movie_tokens if token not in embedding_table]
    if missing_tokens:
        preview = ", ".join(missing_tokens[:10])
        raise ValueError(f"{movie_embedding_file} is missing {len(missing_tokens)} movie tokens, e.g. {preview}")

    input_embeddings = model.get_input_embeddings()
    input_weight = input_embeddings.weight
    hidden_size = int(input_weight.shape[1])
    vectors = np.stack([embedding_table[token] for token in movie_tokens], axis=0)
    if vectors.shape[1] != hidden_size:
        raise ValueError(
            f"Movie embedding dim {vectors.shape[1]} does not match model hidden size {hidden_size}. "
            "Use embeddings generated with the same dimension as the LLM."
        )

    movie_token_ids = tokenizer.convert_tokens_to_ids(movie_tokens)
    if any(token_id == tokenizer.unk_token_id for token_id in movie_token_ids):
        raise ValueError("At least one movie token maps to unk_token_id after tokenizer.add_tokens")

    with torch.no_grad():
        vector_tensor = torch.as_tensor(vectors, dtype=input_weight.dtype, device=input_weight.device)
        input_weight[movie_token_ids] = vector_tensor
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.data_ptr() != input_weight.data_ptr():
            output_weight = output_embeddings.weight
            output_weight[movie_token_ids] = vector_tensor.to(dtype=output_weight.dtype, device=output_weight.device)

    vector_norms = np.linalg.norm(vectors, axis=1)
    stats = {
        "movie_embedding_file": str(movie_embedding_file),
        "num_initialized_tokens": len(movie_tokens),
        "embedding_dim": vectors.shape[1],
        "vector_norm_mean": float(vector_norms.mean()),
        "vector_norm_std": float(vector_norms.std()),
        "vector_norm_min": float(vector_norms.min()),
        "vector_norm_max": float(vector_norms.max()),
        "output_embedding_tied": bool(
            model.get_output_embeddings() is not None
            and model.get_output_embeddings().weight.data_ptr() == input_weight.data_ptr()
        ),
    }
    logger(
        "Initialized "
        f"{stats['num_initialized_tokens']} movie token embeddings from {movie_embedding_file} "
        f"(dim={stats['embedding_dim']}, norm_mean={stats['vector_norm_mean']:.4f})"
    )
    return stats


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
        "save_strategy": args.save_strategy,
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


def build_lora_config(args: argparse.Namespace, target_modules: list[str], movie_token_ids: list[int]) -> LoraConfig:
    kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules,
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


def build_trainable_tokens_config(movie_token_ids: list[int]):
    if not movie_token_ids:
        raise ValueError("--peft-mode trainable_tokens requires a non-empty movie token file.")
    try:
        from peft import TrainableTokensConfig
    except ImportError as exc:
        raise RuntimeError(
            "The installed PEFT version does not support TrainableTokensConfig. "
            "Use the remote PEFT version used for the experiments, or upgrade PEFT."
        ) from exc
    return TrainableTokensConfig(task_type="CAUSAL_LM", token_indices=movie_token_ids)


def build_peft_config(args: argparse.Namespace, target_modules: list[str], movie_token_ids: list[int]):
    if args.peft_mode == "full":
        return None
    if args.peft_mode == "trainable_tokens":
        return build_trainable_tokens_config(movie_token_ids)
    return build_lora_config(args, target_modules, movie_token_ids)


def main() -> None:
    args = parse_args()
    logger = Logger()

    logger("1. Setup seed, output directory, and logging")
    if args.peft_mode == "full" and args.load_in_4bit:
        raise ValueError("--peft-mode full requires --no-load-in-4bit; quantized base weights are not full fine-tuned.")
    setup_seed(args.seed)
    ensure_dir(args.output_dir)
    save_json(args.output_dir / "training_args.json", vars(args))
    if args.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    logger("2. Load tokenizer and model")
    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="right")
    movie_tokens = load_movie_tokens(args.data_dir, args.movie_token_file)
    movie_token_ids = add_movie_tokens(tokenizer, movie_tokens, logger)
    model = load_causal_lm(
        ModelConfig(
            args.model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
            adapter_is_trainable=True,
        ),
        tokenizer=tokenizer,
    )
    embedding_init_stats = initialize_movie_token_embeddings(model, tokenizer, movie_tokens, args.movie_embedding_file, logger)
    if embedding_init_stats is not None:
        save_json(args.output_dir / "movie_embedding_init.json", embedding_init_stats)
    is_peft_model = isinstance(model, PeftModel)
    if args.load_in_4bit and not is_peft_model:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    logger("3. Build SFT prompt-completion train dataset")
    train_dataset = prepare_train_dataset(args, logger)

    logger("4. Build PEFT and SFT configuration")
    target_modules = split_csv(args.target_modules)
    peft_config = None if is_peft_model else build_peft_config(args, target_modules, movie_token_ids)
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

    if args.skip_final_save:
        logger("6. Skip final save because --skip-final-save is set")
        return

    logger("6. Save final adapter and tokenizer once")
    token_file = args.movie_token_file or args.data_dir / "movie_tokens.json"
    final_dir = save_last_checkpoint_as_final(
        trainer,
        tokenizer,
        args.output_dir,
        logger,
        extra_files=[(token_file, "movie_tokens.json")],
    )
    logger(f"Final SFT model is available at {final_dir}")


if __name__ == "__main__":
    main()
