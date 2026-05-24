from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from utils.data_io import (
    clean_value,
    load_id_inputs,
    movie_token,
)
from dataset.sft_schema import (
    ID_ONLY_SUFFIX,
    format_id_interaction_history,
    format_user_profile,
)
from dataset.build_sft_dataset import LEAVE_ONE_OUT_SPLITS, loo_targets
from utils.training_utils import ensure_dir, save_json


SOURCE = "funrec-movielens-1m"
GRPO_INSTRUCTION = "You are a helpful movie recommendation assistant. Use the user's profile and interaction history to recommend the next movie."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MovieRec GRPO prompt JSONL data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/grpo_movielens_1m"))
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--movie-token-format", choices=["angle", "plain"], default="angle")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_prompt(
    user: dict[str, Any] | None,
    history: Iterable[dict[str, Any]],
    token_format: str,
) -> str:
    prompt_input = f"""User profile:
{format_user_profile(user)}

Chronological interaction history before the target movie. Each line says that the user gave a MovieLens ID a rating at a specific time:
{format_id_interaction_history(history, token_format)}

Based on this history, predict the next movie the user is most likely to watch.
{ID_ONLY_SUFFIX}"""
    return f"""{GRPO_INSTRUCTION}

### Input:
{prompt_input}

### Response:
"""


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {args.out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(args.out_dir)
    ensure_dir(args.out_dir)

    movie_features, users, ratings_df = load_id_inputs(args.raw_dir)
    user_ids = list(ratings_df["user_id"].drop_duplicates())

    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    grouped = ratings_df.groupby("user_id", sort=False)
    processed_users = 0
    for user_id, user_ratings in grouped:
        if args.max_users is not None and processed_users >= args.max_users:
            break
        processed_users += 1
        records = user_ratings.to_dict("records")
        if len(records) <= args.min_history:
            continue
        for split, target_pos, history, target in loo_targets(records, args.min_history, args.max_history):
            target_movie_id = clean_value(target["movie_id"])
            outputs[split].append(
                {
                    "id": f"grpo:user_{user_id}:loo_{split}:pos_{target_pos}",
                    "split": split,
                    "prompt": build_prompt(users.get(user_id), history, args.movie_token_format),
                    "target_movie_id": movie_token(target_movie_id, args.movie_token_format),
                    "history_movie_ids": [movie_token(event["movie_id"], args.movie_token_format) for event in history],
                    "history_timestamps": [event["timestamp"] for event in history],
                    "candidate_movie_ids": None,
                    "source": SOURCE,
                }
            )

    counts = Counter({split: len(records) for split, records in outputs.items()})
    for split, records in outputs.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", records)
    save_json(
        args.out_dir / "manifest.json",
        {
            "source": SOURCE,
            "raw_dir": str(args.raw_dir),
            "min_history": args.min_history,
            "max_history": args.max_history,
            "movie_token_format": args.movie_token_format,
            "sequence_split_protocol": "leave_one_out",
            "leave_one_out_splits": {split: f"target is the {offset} item from the end" for split, offset in LEAVE_ONE_OUT_SPLITS},
            "splits": dict(counts),
            "num_users": len(user_ids),
            "num_movies": len(movie_features),
        },
    )
    print(f"Wrote GRPO prompts to {args.out_dir}: {dict(counts)}")


if __name__ == "__main__":
    main()
