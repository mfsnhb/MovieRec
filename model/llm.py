from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.data_io import clean_value, movie_id_sort_key, movie_token


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    trust_remote_code: bool = True


def torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


def build_quantization_config(config: ModelConfig) -> BitsAndBytesConfig | None:
    if not config.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch_dtype(config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=True,
    )


def collect_movie_tokens(
    data_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
    max_movie_id: int | None = None,
    token_format: Literal["angle", "plain"] = "angle",
) -> list[str]:
    movie_ids: set[str] = set()
    if raw_dir is not None:
        movies_path = Path(raw_dir) / "movies.pkl"
        if movies_path.exists():
            movies_df = pd.read_pickle(movies_path)
            movie_ids.update(clean_value(movie_id) for movie_id in movies_df["movie_id"].tolist())
    if data_dir is not None:
        stats_path = Path(data_dir) / "stats.json"
        if stats_path.exists() and max_movie_id is None:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            num_movies = stats.get("num_movies")
            if isinstance(num_movies, int) and num_movies > 0:
                max_movie_id = num_movies
    if max_movie_id is not None:
        movie_ids.update(str(movie_id) for movie_id in range(1, max_movie_id + 1))
    return [movie_token(movie_id, token_format) for movie_id in sorted(movie_ids, key=movie_id_sort_key)]


def load_tokenizer(
    model_name_or_path: str,
    movie_tokens: Iterable[str] = (),
    trust_remote_code: bool = True,
    padding_side: Literal["left", "right"] = "right",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    tokens = list(movie_tokens)
    if tokens:
        tokenizer.add_tokens(tokens)
    return tokenizer


def load_causal_lm(config: ModelConfig, tokenizer=None):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=build_quantization_config(config),
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer is not None and len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    return model
