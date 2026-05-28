from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    adapter_is_trainable: bool = False


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


def load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    padding_side: Literal["left", "right"] = "right",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def adapter_base_model_path(model_name_or_path: Path | str) -> str | None:
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    base_model_path = json.loads(adapter_config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if not base_model_path:
        return None
    base_path = Path(base_model_path)
    if base_path.is_absolute():
        return str(base_path)
    cwd_relative = Path.cwd() / base_path
    if cwd_relative.exists():
        return str(cwd_relative.resolve())
    return str((adapter_config_path.parent / base_path).resolve())


def load_causal_lm(config: ModelConfig, tokenizer=None):
    adapter_path = Path(config.model_name_or_path)
    base_model_path = adapter_base_model_path(adapter_path)
    model_path = base_model_path or config.model_name_or_path
    kwargs = {
        "quantization_config": build_quantization_config(config),
        "device_map": "auto",
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if tokenizer is not None and len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if base_model_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=config.adapter_is_trainable)
    model.config.use_cache = False
    return model
