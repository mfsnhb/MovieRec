from __future__ import annotations

import json
import math
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
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def init_wandb(args: Any, run_name: str, resume_id: str | None = None):
    if not getattr(args, "use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install wandb or disable --use-wandb.") from exc
    return wandb.init(
        project=getattr(args, "wandb_project", "MovieRec"),
        name=run_name,
        id=resume_id,
        resume="allow" if resume_id else None,
        config=vars(args),
    )


def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        return cosine_lr(step, total_steps, base_lr, min_lr, warmup_steps) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_training_checkpoint(
    output_dir: Path | str,
    model: Any,
    tokenizer: Any,
    trainer_state: dict[str, Any],
    step: int,
    name: str | None = None,
) -> Path:
    checkpoint_dir = ensure_dir(Path(output_dir) / (name or f"checkpoint-{step}"))
    model.save_pretrained(str(checkpoint_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(checkpoint_dir))
    save_json(checkpoint_dir / "trainer_state.json", {"step": step, **trainer_state})
    return checkpoint_dir


def load_training_manifest(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


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
