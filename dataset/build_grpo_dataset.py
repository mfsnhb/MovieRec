from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dataset.build_sft_dataset import LEAVE_ONE_OUT_SPLITS, loo_targets, user_token_records
from dataset.sft_schema import INSTRUCTION, MOVIE_TOKEN_ONLY_SUFFIX, format_id_title_interaction_history
from utils.data_io import clean_value, load_movie_feature_store, load_ratings, required_movie_feature_columns, user_token
from utils.training_utils import ensure_dir, save_json


SOURCE = "funrec-movielens-1m"
GRPO_INSTRUCTION = INSTRUCTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MovieRec GRPO prompt JSONL data with single Movie ID token targets.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/grpo_movielens_1m"))
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--train-windows-per-user", type=int, default=0)
    parser.add_argument("--train-sample-seed", type=int, default=42)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_prompt(user_id: Any, history: Iterable[dict[str, Any]], movie_features) -> str:
    prompt_input = f"""User token: {user_token(user_id)}

Here is the user's recent MovieLens trail in watch order. Each line names a movie by its catalog token and title:
{format_id_title_interaction_history(history, movie_features)}

Use this sequence to predict the next movie this user is likely to watch. {MOVIE_TOKEN_ONLY_SUFFIX}"""
    return f"""{GRPO_INSTRUCTION}

### Input:
{prompt_input}

### Response:
"""


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_record(
    *,
    split: str,
    user_id: str,
    target_pos: int,
    history: list[dict[str, Any]],
    target: dict[str, Any],
    movie_features,
    train_window_index: int | None = None,
) -> dict[str, Any]:
    target_movie_id = clean_value(target["movie_id"])
    prefix = "train" if split == "train" else f"loo_{split}"
    record = {
        "id": f"grpo:user_{user_id}:{prefix}:pos_{target_pos}",
        "split": split,
        "prompt": build_prompt(user_id, history, movie_features),
        "target_movie_token": movie_features.token(target_movie_id),
        "target_movie_title": movie_features.title(target_movie_id),
        "user_id": user_id,
        "user_token": user_token(user_id),
        "target_movie_id": target_movie_id,
        "history_movie_tokens": [movie_features.token(event["movie_id"]) for event in history],
        "history_movie_ids": [clean_value(event["movie_id"]) for event in history],
        "target_position": target_pos,
        "source": SOURCE,
    }
    if train_window_index is not None:
        record["train_window_index"] = train_window_index
    return record


def train_target_positions(num_records: int, min_history: int, windows_per_user: int, seed: int, user_id: str) -> list[int]:
    last_train_pos = num_records - 3
    if last_train_pos < min_history or windows_per_user <= 0:
        return []
    candidates = list(range(min_history, last_train_pos + 1))
    if len(candidates) <= windows_per_user:
        return candidates
    rng = random.Random(f"{seed}:{user_id}")
    sampled = rng.sample(candidates, windows_per_user)
    return sorted(sampled)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {args.out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(args.out_dir)
    ensure_dir(args.out_dir)

    movie_features = load_movie_feature_store(args.raw_dir, required_movie_feature_columns({"NextMoviePrediction"}))
    ratings_df = load_ratings(args.raw_dir, movie_features)
    user_ids = list(ratings_df["user_id"].drop_duplicates())

    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    grouped = ratings_df.groupby("user_id", sort=False)
    processed_users = 0
    for uid, user_ratings in grouped:
        if args.max_users is not None and processed_users >= args.max_users:
            break
        processed_users += 1
        user_id = clean_value(uid)
        records = user_ratings.to_dict("records")
        if len(records) <= args.min_history:
            continue

        for window_index, target_pos in enumerate(
            train_target_positions(len(records), args.min_history, args.train_windows_per_user, args.train_sample_seed, user_id)
        ):
            history = records[:target_pos]
            if args.max_history is not None and args.max_history > 0:
                history = history[-args.max_history :]
            outputs["train"].append(
                make_record(
                    split="train",
                    user_id=user_id,
                    target_pos=target_pos,
                    history=history,
                    target=records[target_pos],
                    movie_features=movie_features,
                    train_window_index=window_index,
                )
            )

        for split, target_pos, history, target in loo_targets(records, args.min_history, args.max_history):
            if split == "train":
                continue
            outputs[split].append(
                make_record(
                    split=split,
                    user_id=user_id,
                    target_pos=target_pos,
                    history=history,
                    target=target,
                    movie_features=movie_features,
                )
            )

    counts = Counter({split: len(records) for split, records in outputs.items()})
    for split, records in outputs.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", records)
    save_json(args.out_dir / "movie_tokens.json", movie_features.token_records())
    save_json(args.out_dir / "user_tokens.json", user_token_records(user_ids))
    save_json(
        args.out_dir / "manifest.json",
        {
            "source": SOURCE,
            "raw_dir": str(args.raw_dir),
            "min_history": args.min_history,
            "max_history": args.max_history,
            "train_windows_per_user": args.train_windows_per_user,
            "train_sample_seed": args.train_sample_seed,
            "target_unit": "movie_id_token",
            "sequence_split_protocol": "leave_one_out",
            "leave_one_out_splits": {split: f"target is the {offset} item from the end" for split, offset in LEAVE_ONE_OUT_SPLITS},
            "grpo_train_sampling_protocol": "sampled_prefix_windows_before_valid_test",
            "splits": dict(counts),
            "num_users": len(user_ids),
            "num_movies": len(movie_features),
        },
    )
    print(f"Wrote GRPO prompts to {args.out_dir}: {dict(counts)}")


if __name__ == "__main__":
    main()
