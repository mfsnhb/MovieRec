from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from model.sasrec import SASRec, SASRecConfig
from utils.training_utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project SASRec item embeddings into LLM movie-token embedding space.")
    parser.add_argument("--sasrec-dir", type=Path, default=Path("models/reranker/sasrec_prefix_teacher"))
    parser.add_argument("--target-model", type=str, default="models/sft/qwen3_4b_stage1_alignment/final")
    parser.add_argument("--output-path", type=Path, default=Path("models/reranker/sasrec_prefix_teacher/qwen_movie_embedding_init.npz"))
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def load_mapping(sasrec_dir: Path) -> tuple[list[str], list[str]]:
    data = json.loads((sasrec_dir / "item_mapping.json").read_text(encoding="utf-8"))
    index_to_movie_id = {int(index): str(movie_id) for index, movie_id in data["index_to_movie_id"].items()}
    movie_ids = [index_to_movie_id[index] for index in range(1, len(index_to_movie_id) + 1)]
    movie_tokens = [f"movie_{movie_id}" for movie_id in movie_ids]
    return movie_ids, movie_tokens


def load_sasrec_embeddings(sasrec_dir: Path, num_items: int) -> np.ndarray:
    checkpoint = torch.load(sasrec_dir / "model.pt", map_location="cpu")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = SASRecConfig(num_items=int(checkpoint["num_items"]))
    else:
        state_dict = checkpoint["state_dict"]
        config = SASRecConfig(**checkpoint["config"])
    model = SASRec(config)
    model.load_state_dict(state_dict)
    embeddings = model.item_embedding.weight.detach().cpu().numpy()[1 : num_items + 1].astype(np.float32)
    return embeddings


def load_target_movie_embeddings(model_name_or_path: str, movie_tokens: list[str], load_in_4bit: bool, attn_implementation: str | None) -> np.ndarray:
    tokenizer = load_tokenizer(model_name_or_path, padding_side="right")
    tokenizer.add_tokens(movie_tokens, special_tokens=False)
    model = load_causal_lm(
        ModelConfig(model_name_or_path, load_in_4bit=load_in_4bit, attn_implementation=attn_implementation),
        tokenizer=tokenizer,
    )
    token_ids = tokenizer.convert_tokens_to_ids(movie_tokens)
    if any(token_id == tokenizer.unk_token_id for token_id in token_ids):
        raise ValueError("At least one movie token maps to unk_token_id")
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()[token_ids].astype(np.float32)
    return embeddings


def fit_ridge_projection(source: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    x = source - source_mean
    y = target - target_mean
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float32)
    projection = np.linalg.solve(xtx + reg, x.T @ y).astype(np.float32)
    projected = (x @ projection + target_mean).astype(np.float32)
    mse = float(np.mean((projected - target) ** 2))
    cosine = np.sum(projected * target, axis=1) / (np.linalg.norm(projected, axis=1) * np.linalg.norm(target, axis=1) + 1e-8)
    stats = {
        "projection": "ridge_sasrec_to_llm_embedding",
        "ridge": float(ridge),
        "projection_mse": mse,
        "projection_cosine_mean": float(cosine.mean()),
        "projection_cosine_std": float(cosine.std()),
        "source_dim": int(source.shape[1]),
        "target_dim": int(target.shape[1]),
        "num_items": int(source.shape[0]),
    }
    return projected, projection, stats


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_path.parent)
    movie_ids, movie_tokens = load_mapping(args.sasrec_dir)
    source = load_sasrec_embeddings(args.sasrec_dir, len(movie_ids))
    target = load_target_movie_embeddings(args.target_model, movie_tokens, args.load_in_4bit, args.attn_implementation)
    projected, projection, stats = fit_ridge_projection(source, target, args.ridge)
    np.savez_compressed(
        args.output_path,
        embeddings=projected.astype(np.float32),
        movie_ids=np.asarray(movie_ids),
        movie_tokens=np.asarray(movie_tokens),
        sasrec_to_llm_projection=projection.astype(np.float32),
    )
    save_json(args.output_path.with_suffix(".manifest.json"), {"sasrec_dir": str(args.sasrec_dir), "target_model": args.target_model, **stats})
    print(json.dumps({"output_path": str(args.output_path), **stats}, indent=2))


if __name__ == "__main__":
    main()
