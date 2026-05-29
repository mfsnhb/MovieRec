from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from dataset.build_sft_dataset import sequence_targets
from model.sasrec import SASRec, SASRecConfig
from utils.data_io import clean_value, load_movie_feature_store, load_ratings
from utils.sasrec_data import SASRecMappings, grouped_user_records, sequence_target_to_input
from utils.training_utils import ensure_dir, save_json, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SASRec top-k labels for each SFT training prefix/window.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--sasrec-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("models/reranker/sasrec_prefix_teacher/prefix_topk_labels.jsonl"))
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--train-windows-per-user", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(sasrec_dir: Path, device: torch.device) -> SASRec:
    checkpoint = torch.load(sasrec_dir / "model.pt", map_location=device)
    if "model_state_dict" in checkpoint:
        num_items = int(checkpoint["num_items"])
        config = SASRecConfig(num_items=num_items)
        state_dict = checkpoint["model_state_dict"]
    else:
        config = SASRecConfig(**checkpoint["config"])
        state_dict = checkpoint["state_dict"]
    model = SASRec(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_mappings(sasrec_dir: Path) -> SASRecMappings:
    data = json.loads((sasrec_dir / "item_mapping.json").read_text(encoding="utf-8"))
    return SASRecMappings(
        movie_id_to_index={str(movie_id): int(index) for movie_id, index in data["movie_id_to_index"].items()},
        index_to_movie_id={int(index): str(movie_id) for index, movie_id in data["index_to_movie_id"].items()},
        user_ids=[str(user_id) for user_id in data["user_ids"]],
    )


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    ensure_dir(args.output_path.parent)
    device = torch.device(args.device)

    movie_features = load_movie_feature_store(args.raw_dir, {"movie_id", "title"})
    ratings_df = load_ratings(args.raw_dir, movie_features)
    records_by_user = grouped_user_records(ratings_df, args.max_users)
    mappings = load_mappings(args.sasrec_dir)
    model = load_checkpoint(args.sasrec_dir, device)

    pending: list[dict[str, Any]] = []
    written = 0

    def flush(handle) -> None:
        nonlocal written
        if not pending:
            return
        input_ids = torch.tensor([item["input_ids"] for item in pending], dtype=torch.long, device=device)
        with torch.no_grad():
            scores = model.score_all(input_ids)
        for row, item in enumerate(pending):
            row_scores = scores[row].clone()
            for item_index in item["history_item_indices"]:
                row_scores[item_index - 1] = -torch.inf
            top_count = min(args.top_k, int(torch.isfinite(row_scores).sum().item()))
            top_scores, top_cols = torch.topk(row_scores, top_count)
            movie_ids = [mappings.index_to_movie_id[int(col) + 1] for col in top_cols.tolist()]
            record = {
                "id": item["id"],
                "user_id": item["user_id"],
                "train_window_index": item["train_window_index"],
                "target_position": item["target_position"],
                "target_movie_id": item["target_movie_id"],
                "label_movie_ids": movie_ids,
                "label_scores": [float(score) for score in top_scores.tolist()],
                "label_source": "sasrec_prefix_teacher",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        pending.clear()

    with args.output_path.open("w", encoding="utf-8") as handle:
        for user_id, records in tqdm(records_by_user.items(), desc="export sasrec prefix labels"):
            for target in sequence_targets(records, args.min_history, args.max_history, args.train_windows_per_user):
                if target.split != "train":
                    continue
                target_movie_id = clean_value(target.target["movie_id"])
                pending.append(
                    {
                        "id": f"NextMoviePrediction:user_{user_id}:train_window_{target.train_window_index}:pos_{target.target_pos}",
                        "user_id": user_id,
                        "train_window_index": target.train_window_index,
                        "target_position": target.target_pos,
                        "target_movie_id": target_movie_id,
                        "input_ids": sequence_target_to_input(target, mappings, args.max_len),
                        "history_item_indices": {
                            mappings.movie_id_to_index[clean_value(event["movie_id"])] for event in target.history
                        },
                    }
                )
                if len(pending) >= args.batch_size:
                    flush(handle)
        flush(handle)

    save_json(
        args.output_path.with_suffix(".manifest.json"),
        {
            "raw_dir": str(args.raw_dir),
            "sasrec_dir": str(args.sasrec_dir),
            "output_path": str(args.output_path),
            "max_len": args.max_len,
            "max_history": args.max_history,
            "min_history": args.min_history,
            "train_windows_per_user": args.train_windows_per_user,
            "top_k": args.top_k,
            "num_labels": written,
            "label_source": "sasrec_prefix_teacher",
        },
    )
    print(f"Wrote {written} SASRec prefix labels to {args.output_path}")


if __name__ == "__main__":
    main()
