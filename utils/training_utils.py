from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


class Logger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __call__(self, content: Any) -> None:
        if self.enabled:
            print(content)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def print_trainable_parameters(model: torch.nn.Module, logger: Logger | None = None) -> None:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
    ratio = 100 * trainable / total if total else 0.0
    (logger or Logger())(f"Trainable parameters: {trainable:,} / {total:,} ({ratio:.4f}%)")
