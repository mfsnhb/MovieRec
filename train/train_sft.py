from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.data_io import clean_value, load_movie_feature_store, load_ratings, movie_token
from utils.reranker_scores import RerankerScores
from utils.training_utils import Logger, ensure_dir, print_trainable_parameters, save_json, save_last_checkpoint_as_final, setup_seed


RESPONSE_TEMPLATE = "\n\n### Response:\n"
PROMPT_TEMPLATE = """{instruction}

### Input:
{input}{response_template}"""


@dataclass(frozen=True)
class TokenCatalog:
    movie_tokens: list[str]
    movie_token_ids: list[int]
    movie_token_to_index: dict[str, int]
    movie_id_to_index: dict[str, int]
    user_tokens: list[str]
    user_token_ids: list[int]


class NegativeSampler:
    def __init__(
        self,
        token_catalog: TokenCatalog,
        raw_dir: Path,
        seed: int,
        random_count: int,
        popularity_count: int,
        sasrec_count: int,
        popularity_temperature: float,
        reranker_score_path: Path | None,
    ) -> None:
        self.token_catalog = token_catalog
        self.random_count = random_count
        self.popularity_count = popularity_count
        self.sasrec_count = sasrec_count
        self.total_count = random_count + popularity_count + sasrec_count
        self.rng = np.random.default_rng(seed)
        self.movie_indices = np.arange(len(token_catalog.movie_tokens), dtype=np.int64)
        self.movie_ids = [token.removeprefix("movie_") for token in token_catalog.movie_tokens]
        self.movie_id_set = set(self.movie_ids)
        self.movie_id_to_index = token_catalog.movie_id_to_index
        self.reranker = RerankerScores(reranker_score_path) if reranker_score_path is not None and sasrec_count > 0 else None
        self.popularity_probs = self._build_popularity_probs(raw_dir, popularity_temperature)

    def _build_popularity_probs(self, raw_dir: Path, temperature: float) -> np.ndarray:
        movie_features = load_movie_feature_store(raw_dir, {"movie_id", "title"})
        ratings_df = load_ratings(raw_dir, movie_features)
        counts = ratings_df["movie_id"].value_counts()
        popularity = np.array([float(counts.get(movie_id, 0)) for movie_id in self.movie_ids], dtype=np.float64)
        popularity = np.maximum(popularity, 1.0)
        popularity = np.power(popularity, temperature)
        return popularity / popularity.sum()

    def sample(self, user_id: Any, target_movie_id: Any, history_movie_ids: list[Any]) -> tuple[list[int], list[float]]:
        target_id = clean_value(target_movie_id)
        excluded = {target_id, *(clean_value(movie_id) for movie_id in history_movie_ids)}
        sasrec_indices = self._sasrec_indices(user_id, excluded, self.sasrec_count)
        selected: list[int] = []
        self._extend_unique(selected, self._sample_uniform(excluded, self.random_count))
        self._extend_unique(selected, self._sample_popularity(excluded, self.popularity_count))
        self._extend_unique(selected, sasrec_indices)
        if len(selected) < self.total_count:
            self._extend_unique(selected, self._sample_uniform(excluded | {self.movie_ids[index] for index in selected}, self.total_count - len(selected)))
        selected = selected[: self.total_count]
        sasrec_set = set(sasrec_indices)
        log_q = [self._log_q(index, excluded, sasrec_set) for index in selected]
        target_index = self.movie_id_to_index[target_id]
        return [target_index, *selected], [0.0, *log_q]

    def _available_mask(self, excluded: set[str]) -> np.ndarray:
        return np.array([movie_id not in excluded for movie_id in self.movie_ids], dtype=bool)

    def _sample_uniform(self, excluded: set[str], count: int) -> list[int]:
        if count <= 0:
            return []
        available = self.movie_indices[self._available_mask(excluded)]
        if available.size == 0:
            return []
        return self.rng.choice(available, size=min(count, available.size), replace=False).astype(int).tolist()

    def _sample_popularity(self, excluded: set[str], count: int) -> list[int]:
        if count <= 0:
            return []
        mask = self._available_mask(excluded)
        available = self.movie_indices[mask]
        if available.size == 0:
            return []
        probs = self.popularity_probs[mask]
        probs = probs / probs.sum()
        return self.rng.choice(available, size=min(count, available.size), replace=False, p=probs).astype(int).tolist()

    def _sample_sasrec(self, user_id: Any, excluded: set[str], count: int) -> list[int]:
        return self._sasrec_indices(user_id, excluded, count)

    def _sasrec_indices(self, user_id: Any, excluded: set[str], count: int) -> list[int]:
        if count <= 0 or self.reranker is None:
            return []
        movie_ids = self.reranker.top_movie_ids(user_id, excluded, count)
        return [self.movie_id_to_index[movie_id] for movie_id in movie_ids if movie_id in self.movie_id_to_index]

    def _log_q(self, movie_index: int, excluded: set[str], sasrec_indices: set[int]) -> float:
        q = 0.0
        total = max(1, self.total_count)
        available_count = len(self.movie_ids) - len(excluded & self.movie_id_set)
        if self.random_count > 0 and available_count > 0:
            q += (self.random_count / total) * (1.0 / available_count)
        if self.popularity_count > 0:
            mask = self._available_mask(excluded)
            denominator = float(self.popularity_probs[mask].sum())
            if denominator > 0:
                q += (self.popularity_count / total) * float(self.popularity_probs[movie_index] / denominator)
        if self.sasrec_count > 0 and movie_index in sasrec_indices:
            q += (self.sasrec_count / total) * (1.0 / max(1, len(sasrec_indices)))
        return float(np.log(max(q, 1e-12)))

    @staticmethod
    def _extend_unique(selected: list[int], candidates: list[int]) -> None:
        seen = set(selected)
        for index in candidates:
            if index not in seen:
                selected.append(index)
                seen.add(index)


class MovieTokenCollator:
    def __init__(self, tokenizer, token_catalog: TokenCatalog, max_length: int, negative_sampler: NegativeSampler | None = None) -> None:
        self.tokenizer = tokenizer
        self.token_catalog = token_catalog
        self.max_length = max_length
        self.negative_sampler = negative_sampler

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [example["prompt"] for example in examples]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        target_movie_indices = [self.token_catalog.movie_token_to_index[example["target_movie_token"]] for example in examples]
        encoded["target_movie_index"] = torch.tensor(target_movie_indices, dtype=torch.long)
        if self.negative_sampler is not None:
            candidates = []
            log_q = []
            for example in examples:
                candidate_indices, candidate_log_q = self.negative_sampler.sample(
                    example["user_id"],
                    example["target_movie_id"],
                    example.get("history_movie_ids") or [],
                )
                candidates.append(candidate_indices)
                log_q.append(candidate_log_q)
            encoded["candidate_movie_indices"] = torch.tensor(candidates, dtype=torch.long)
            encoded["candidate_log_q"] = torch.tensor(log_q, dtype=torch.float32)
        return encoded


class MovieTokenSFTTrainer(SFTTrainer):
    def __init__(self, *args, movie_token_ids: list[int], loss_mode: str, sampled_logit_correction: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.movie_token_ids = torch.tensor(movie_token_ids, dtype=torch.long)
        self.loss_mode = loss_mode
        self.sampled_logit_correction = sampled_logit_correction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target_movie_index = inputs.pop("target_movie_index")
        candidate_movie_indices = inputs.pop("candidate_movie_indices", None)
        candidate_log_q = inputs.pop("candidate_log_q", None)
        outputs = model(**inputs)
        row_indices = torch.arange(inputs["input_ids"].shape[0], device=outputs.logits.device)
        last_indices = inputs["attention_mask"].sum(dim=1).to(outputs.logits.device) - 1
        movie_token_ids = self.movie_token_ids.to(outputs.logits.device)
        final_logits = outputs.logits[row_indices, last_indices, :]
        if self.loss_mode == "sampled":
            if candidate_movie_indices is None or candidate_log_q is None:
                raise ValueError("Sampled loss requires candidate_movie_indices and candidate_log_q from the data collator.")
            candidate_movie_indices = candidate_movie_indices.to(outputs.logits.device)
            candidate_token_ids = movie_token_ids[candidate_movie_indices]
            sampled_logits = final_logits.gather(dim=-1, index=candidate_token_ids)
            if self.sampled_logit_correction:
                sampled_logits = sampled_logits - candidate_log_q.to(outputs.logits.device)
            labels = torch.zeros(sampled_logits.shape[0], dtype=torch.long, device=outputs.logits.device)
            loss = F.cross_entropy(sampled_logits.float(), labels)
        else:
            movie_logits = final_logits.index_select(dim=-1, index=movie_token_ids)
            loss = F.cross_entropy(movie_logits.float(), target_movie_index.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for MovieRec item-space movie-token prediction.")
    parser.add_argument("--model-name-or-path", default="models/Qwen3-4B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--movie-token-file", type=Path)
    parser.add_argument("--user-token-file", type=Path)
    parser.add_argument("--movie-embedding-file", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
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
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--loss-mode", choices=["full", "sampled"], default="full")
    parser.add_argument("--num-random-negatives", type=int, default=64)
    parser.add_argument("--num-popularity-negatives", type=int, default=64)
    parser.add_argument("--num-sasrec-negatives", type=int, default=0)
    parser.add_argument("--negative-sampling-temperature", type=float, default=0.75)
    parser.add_argument("--reranker-score-path", type=Path)
    parser.add_argument("--sampled-logit-correction", action=argparse.BooleanOptionalAction, default=True)
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


def load_token_records(path: Path, field: str) -> list[str]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    tokens = []
    for record in records:
        token = record[field] if isinstance(record, dict) else str(record)
        if token not in tokens:
            tokens.append(token)
    return tokens


def load_movie_tokens(data_dir: Path, movie_token_file: Path | None) -> list[str]:
    return load_token_records(movie_token_file or data_dir / "movie_tokens.json", "movie_token")


def load_user_tokens(data_dir: Path, user_token_file: Path | None) -> list[str]:
    return load_token_records(user_token_file or data_dir / "user_tokens.json", "user_token")


def add_tokens(tokenizer, tokens: list[str], label: str, logger: Logger) -> list[int]:
    if not tokens:
        logger(f"No {label} token file found; tokenizer vocabulary is left unchanged for {label} tokens")
        return []
    added = tokenizer.add_tokens(tokens, special_tokens=False)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    logger(f"Loaded {len(tokens)} {label} tokens; tokenizer.add_tokens added {added} new tokens")
    return [int(token_id) for token_id in token_ids]


def build_token_catalog(tokenizer, data_dir: Path, movie_token_file: Path | None, user_token_file: Path | None, logger: Logger) -> TokenCatalog:
    movie_tokens = load_movie_tokens(data_dir, movie_token_file)
    user_tokens = load_user_tokens(data_dir, user_token_file)
    movie_token_ids = add_tokens(tokenizer, movie_tokens, "movie", logger)
    user_token_ids = add_tokens(tokenizer, user_tokens, "user", logger)
    return TokenCatalog(
        movie_tokens,
        movie_token_ids,
        {token: index for index, token in enumerate(movie_tokens)},
        {token.removeprefix("movie_"): index for index, token in enumerate(movie_tokens)},
        user_tokens,
        user_token_ids,
    )


def load_movie_embedding_table(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Movie embedding file not found: {path}")
    data = np.load(path, allow_pickle=False)
    embeddings = data["embeddings"]
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embeddings array in {path}, got shape {embeddings.shape}")
    if "movie_tokens" in data:
        tokens = [str(token) for token in data["movie_tokens"]]
    elif "movie_ids" in data:
        tokens = [movie_token(movie_id) for movie_id in data["movie_ids"]]
    else:
        raise ValueError(f"{path} must include either 'movie_tokens' or 'movie_ids'")
    if len(tokens) != embeddings.shape[0]:
        raise ValueError(f"Embedding row count {embeddings.shape[0]} does not match token count {len(tokens)} in {path}")
    return {token: np.asarray(vector, dtype=np.float32) for token, vector in zip(tokens, embeddings, strict=True)}


def set_token_embeddings(model, token_ids: list[int], vectors: np.ndarray) -> dict[str, object]:
    input_weight = model.get_input_embeddings().weight
    if vectors.shape[1] != input_weight.shape[1]:
        raise ValueError(f"Embedding dim {vectors.shape[1]} does not match model hidden size {input_weight.shape[1]}")
    with torch.no_grad():
        vector_tensor = torch.as_tensor(vectors, dtype=input_weight.dtype, device=input_weight.device)
        input_weight[token_ids] = vector_tensor
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.data_ptr() != input_weight.data_ptr():
            output_embeddings.weight[token_ids] = vector_tensor.to(dtype=output_embeddings.weight.dtype, device=output_embeddings.weight.device)
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "num_initialized_tokens": len(token_ids),
        "embedding_dim": vectors.shape[1],
        "vector_norm_mean": float(norms.mean()),
        "vector_norm_std": float(norms.std()),
        "vector_norm_min": float(norms.min()),
        "vector_norm_max": float(norms.max()),
    }


def initialize_movie_token_embeddings(model, token_catalog: TokenCatalog, movie_embedding_file: Path | None, logger: Logger) -> dict[str, object] | None:
    if movie_embedding_file is None:
        return None
    embedding_table = load_movie_embedding_table(movie_embedding_file)
    missing_tokens = [token for token in token_catalog.movie_tokens if token not in embedding_table]
    if missing_tokens:
        raise ValueError(f"{movie_embedding_file} is missing {len(missing_tokens)} movie tokens, e.g. {', '.join(missing_tokens[:10])}")
    vectors = np.stack([embedding_table[token] for token in token_catalog.movie_tokens], axis=0)
    stats = set_token_embeddings(model, token_catalog.movie_token_ids, vectors)
    stats["movie_embedding_file"] = str(movie_embedding_file)
    logger(f"Initialized {stats['num_initialized_tokens']} movie token embeddings from {movie_embedding_file}")
    return stats


def initialize_user_token_embeddings(model, tokenizer, token_catalog: TokenCatalog, raw_dir: Path, logger: Logger) -> dict[str, object] | None:
    if not token_catalog.user_tokens:
        return None
    movie_features = load_movie_feature_store(raw_dir, {"movie_id", "title"})
    ratings_df = load_ratings(raw_dir, movie_features)
    input_weight = model.get_input_embeddings().weight.detach()
    movie_vector_by_token = {
        token: input_weight[token_id].float().cpu().numpy()
        for token, token_id in zip(token_catalog.movie_tokens, token_catalog.movie_token_ids, strict=True)
    }
    vectors = []
    missing_users = 0
    for token in token_catalog.user_tokens:
        user_id = token.removeprefix("user_")
        user_ratings = ratings_df[ratings_df["user_id"] == clean_value(user_id)]
        movie_vectors = [movie_vector_by_token[movie_token(movie_id)] for movie_id in user_ratings["movie_id"] if movie_token(movie_id) in movie_vector_by_token]
        if not movie_vectors:
            missing_users += 1
            vectors.append(np.zeros(input_weight.shape[1], dtype=np.float32))
        else:
            vectors.append(np.mean(movie_vectors, axis=0).astype(np.float32))
    stats = set_token_embeddings(model, token_catalog.user_token_ids, np.stack(vectors, axis=0))
    stats["missing_user_histories"] = missing_users
    logger(f"Initialized {stats['num_initialized_tokens']} user token embeddings from mean pooled interacted movie embeddings")
    return stats


def load_sft_train_dataset(data_dir: Path):
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return load_dataset("json", data_files=str(train_path), split="train")


def to_prompt_target(example: dict) -> dict[str, Any]:
    return {
        "prompt": format_prompt(example),
        "user_id": str(example["user_id"]),
        "target_movie_id": str(example["target_movie_id"]),
        "target_movie_token": str(example["target_movie_token"]),
        "history_movie_ids": [str(movie_id) for movie_id in example.get("history_movie_ids", [])],
    }


def prepare_train_dataset(args: argparse.Namespace, logger: Logger):
    dataset = load_sft_train_dataset(args.data_dir)
    if args.max_train_examples is not None:
        dataset = dataset.select(range(min(args.max_train_examples, len(dataset))))
        logger(f"Using {len(dataset)} training examples because --max-train-examples is set")
    dataset = dataset.map(to_prompt_target, remove_columns=dataset.column_names, desc="Formatting item-space SFT prompts")
    logger(f"Prepared item-space SFT dataset with {len(dataset)} examples")
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
        "max_steps": args.max_steps,
        "report_to": "wandb" if args.use_wandb else "none",
        "run_name": args.output_dir.name,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }
    fields = set(getattr(SFTConfig, "__dataclass_fields__", {}))
    if "max_length" in fields:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in fields:
        kwargs["max_seq_length"] = args.max_seq_length
    return {key: value for key, value in kwargs.items() if key in fields}


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    return SFTConfig(**sft_config_kwargs(args))


def build_lora_config(args: argparse.Namespace, target_modules: list[str], trainable_token_ids: list[int]) -> LoraConfig:
    kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules,
    }
    fields = set(getattr(LoraConfig, "__dataclass_fields__", {}))
    if trainable_token_ids:
        if "trainable_token_indices" not in fields:
            raise RuntimeError("The installed PEFT version does not support trainable_token_indices. Upgrade PEFT.")
        kwargs["trainable_token_indices"] = sorted(set(trainable_token_ids))
    return LoraConfig(**kwargs)


def build_trainable_tokens_config(trainable_token_ids: list[int]):
    if not trainable_token_ids:
        raise ValueError("--peft-mode trainable_tokens requires non-empty movie/user token files.")
    try:
        from peft import TrainableTokensConfig
    except ImportError as exc:
        raise RuntimeError("The installed PEFT version does not support TrainableTokensConfig. Upgrade PEFT.") from exc
    return TrainableTokensConfig(task_type="CAUSAL_LM", token_indices=sorted(set(trainable_token_ids)))


def build_peft_config(args: argparse.Namespace, target_modules: list[str], trainable_token_ids: list[int]):
    if args.peft_mode == "full":
        return None
    if args.peft_mode == "trainable_tokens":
        return build_trainable_tokens_config(trainable_token_ids)
    return build_lora_config(args, target_modules, trainable_token_ids)


def build_negative_sampler(args: argparse.Namespace, token_catalog: TokenCatalog) -> NegativeSampler | None:
    if args.loss_mode != "sampled":
        return None
    total_negatives = args.num_random_negatives + args.num_popularity_negatives + args.num_sasrec_negatives
    if total_negatives <= 0:
        raise ValueError("--loss-mode sampled requires at least one negative sample.")
    if args.num_sasrec_negatives > 0 and args.reranker_score_path is None:
        raise ValueError("--num-sasrec-negatives requires --reranker-score-path.")
    return NegativeSampler(
        token_catalog=token_catalog,
        raw_dir=args.raw_dir,
        seed=args.seed,
        random_count=args.num_random_negatives,
        popularity_count=args.num_popularity_negatives,
        sasrec_count=args.num_sasrec_negatives,
        popularity_temperature=args.negative_sampling_temperature,
        reranker_score_path=args.reranker_score_path,
    )


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

    logger("2. Load tokenizer, expand movie/user tokens, and load model")
    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="right")
    token_catalog = build_token_catalog(tokenizer, args.data_dir, args.movie_token_file, args.user_token_file, logger)
    model = load_causal_lm(
        ModelConfig(
            args.model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
            adapter_is_trainable=True,
        ),
        tokenizer=tokenizer,
    )
    movie_init_stats = initialize_movie_token_embeddings(model, token_catalog, args.movie_embedding_file, logger)
    if movie_init_stats is not None:
        save_json(args.output_dir / "movie_embedding_init.json", movie_init_stats)
    user_init_stats = initialize_user_token_embeddings(model, tokenizer, token_catalog, args.raw_dir, logger)
    if user_init_stats is not None:
        save_json(args.output_dir / "user_embedding_init.json", user_init_stats)

    is_peft_model = isinstance(model, PeftModel)
    if args.load_in_4bit and not is_peft_model:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    logger("3. Build item-space SFT train dataset")
    train_dataset = prepare_train_dataset(args, logger)

    logger("4. Build PEFT, collator, and SFT configuration")
    target_modules = split_csv(args.target_modules)
    trainable_token_ids = token_catalog.movie_token_ids + token_catalog.user_token_ids
    peft_config = None if is_peft_model else build_peft_config(args, target_modules, trainable_token_ids)
    training_args = build_sft_config(args)
    negative_sampler = build_negative_sampler(args, token_catalog)
    data_collator = MovieTokenCollator(tokenizer, token_catalog, args.max_seq_length, negative_sampler)

    logger("5. Start item-space SFT training")
    trainer = MovieTokenSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
        processing_class=tokenizer,
        movie_token_ids=token_catalog.movie_token_ids,
        loss_mode=args.loss_mode,
        sampled_logit_correction=args.sampled_logit_correction,
    )
    print_trainable_parameters(trainer.model, logger)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.skip_final_save:
        logger("6. Skip final save because --skip-final-save is set")
        return

    logger("6. Save final adapter and tokenizer once")
    final_dir = save_last_checkpoint_as_final(
        trainer,
        tokenizer,
        args.output_dir,
        logger,
        extra_files=[
            (args.movie_token_file or args.data_dir / "movie_tokens.json", "movie_tokens.json"),
            (args.user_token_file or args.data_dir / "user_tokens.json", "user_tokens.json"),
        ],
    )
    logger(f"Final SFT model is available at {final_dir}")


if __name__ == "__main__":
    main()
