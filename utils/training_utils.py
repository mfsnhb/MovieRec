from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.trainer_utils import get_last_checkpoint


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


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


def checkpoint_step(path: Path | str) -> int | None:
    match = CHECKPOINT_RE.search(str(path))
    return int(match.group(1)) if match else None


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_existing_files(files: list[tuple[Path, str]], target_dir: Path) -> None:
    for source, target_name in files:
        if source.exists():
            shutil.copyfile(source, target_dir / target_name)


def save_last_checkpoint_as_final(
    trainer: Any,
    tokenizer: Any,
    output_dir: Path | str,
    logger: Logger | None = None,
    extra_files: list[tuple[Path, str]] | None = None,
    alias_name: str = "final",
) -> Path:
    logger = logger or Logger()
    output_dir = ensure_dir(output_dir)
    final_dir = output_dir / alias_name
    global_step = int(getattr(trainer.state, "global_step", 0) or 0)
    last_checkpoint = get_last_checkpoint(str(output_dir))

    if last_checkpoint is not None and checkpoint_step(last_checkpoint) == global_step:
        last_checkpoint_dir = Path(last_checkpoint)
        remove_path(final_dir)
        shutil.move(str(last_checkpoint_dir), str(final_dir))
        logger(f"Moved final checkpoint {last_checkpoint_dir} to {final_dir}")
    else:
        remove_path(final_dir)
        ensure_dir(final_dir)
        trainer.save_model(str(final_dir))
        if last_checkpoint is None:
            logger(f"No training checkpoint was found; saved final model to {final_dir}")
        else:
            logger(
                f"Last checkpoint {last_checkpoint} is not the final training step {global_step}; "
                f"saved final model to {final_dir}"
            )

    tokenizer.save_pretrained(str(final_dir))
    copy_existing_files(extra_files or [], final_dir)
    return final_dir


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
