from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from dataset.sft_schema import (
    ALIGNMENT_TASKS,
    FEATURE_OUTPUT_TASKS,
    ID_OUTPUT_TASKS,
    RECOMMENDATION_TASKS,
    TASKS,
    USER_PROFILE_TASKS,
    RenderedExample,
    build_feature_to_id,
    build_id_to_feature,
    build_next_movie_prediction,
)
from utils.data_io import (
    MovieFeatureStore,
    clean_value,
    load_movie_feature_store,
    load_ratings,
    load_user_profiles,
    required_movie_feature_columns,
    user_token,
)

SOURCE = "funrec-movielens-1m"
LEAVE_ONE_OUT_SPLITS = (
    ("train", 3),
    ("valid", 2),
    ("test", 1),
)


@dataclass(frozen=True)
class SequenceTarget:
    split: str
    target_pos: int
    history: list[dict[str, Any]]
    target: dict[str, Any]
    train_window_index: int | None = None


class JsonlWriters:
    def __init__(self, out_dir: Path, splits: Iterable[str]):
        self.handles = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in splits}

    def write(self, split: str, record: dict[str, Any]) -> None:
        self.handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MovieLens SFT JSONL data with single Movie ID token targets.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--stage", choices=["all", "alignment", "recommendation"], default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--max-train-examples", type=int, default=100_000)
    parser.add_argument("--train-sample-seed", type=int, default=42)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-examples-per-task", type=int)
    parser.add_argument("--sample-per-task", type=int, default=3)
    parser.add_argument("--prompt-template-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_tasks_for_stage(stage: str) -> tuple[str, ...]:
    if stage == "alignment":
        return ALIGNMENT_TASKS
    if stage == "recommendation":
        return RECOMMENDATION_TASKS
    return TASKS


def parse_tasks(tasks: str, stage: str) -> list[str]:
    if tasks == "all":
        return list(default_tasks_for_stage(stage))
    selected = [task.strip() for task in tasks.split(",") if task.strip()]
    unknown = sorted(set(selected) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Available tasks: {list(TASKS)}")
    allowed = set(default_tasks_for_stage(stage))
    if stage != "all":
        disallowed = sorted(set(selected) - allowed)
        if disallowed:
            raise ValueError(f"Tasks {disallowed} do not belong to --stage {stage}.")
    return selected


def record_from_rendered(
    *,
    example_id: str,
    task: str,
    split: str,
    rendered: RenderedExample,
    movie_features: MovieFeatureStore,
    user_id: str | None = None,
    target_movie_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
    target_position: int | None = None,
    train_window_index: int | None = None,
) -> dict[str, Any]:
    record = {
        "id": example_id,
        "task": task,
        "split": split,
        "instruction": rendered.instruction,
        "input": rendered.input,
        "output": rendered.output,
        "source": SOURCE,
    }
    if user_id is not None:
        record["user_id"] = clean_value(user_id)
        record["user_token"] = user_token(user_id)
    if target_movie_id is not None:
        record["target_movie_id"] = clean_value(target_movie_id)
        record["target_movie_token"] = movie_features.token(target_movie_id)
        record["target_movie_title"] = movie_features.title(target_movie_id)
    if target_position is not None:
        record["target_position"] = target_position
    if train_window_index is not None:
        record["train_window_index"] = train_window_index
    if history is not None:
        record["history_movie_ids"] = [clean_value(event["movie_id"]) for event in history]
        record["history_movie_tokens"] = [movie_features.token(event["movie_id"]) for event in history]
        record["history_movie_titles"] = [movie_features.title(event["movie_id"]) for event in history]
    return record


def eval_record_from_target(
    *,
    split: str,
    movie_features: MovieFeatureStore,
    user_id: str,
    target_movie_id: str,
    history: list[dict[str, Any]],
    target_position: int,
) -> dict[str, Any]:
    return {
        "id": f"eval:user_{user_id}:loo_{split}:pos_{target_position}",
        "split": split,
        "user_id": clean_value(user_id),
        "user_token": user_token(user_id),
        "target_position": target_position,
        "target_movie_id": clean_value(target_movie_id),
        "target_movie_token": movie_features.token(target_movie_id),
        "target_movie_title": movie_features.title(target_movie_id),
        "history_movie_ids": [clean_value(event["movie_id"]) for event in history],
        "history_movie_tokens": [movie_features.token(event["movie_id"]) for event in history],
        "history_movie_titles": [movie_features.title(event["movie_id"]) for event in history],
        "source": SOURCE,
    }


def maybe_write(
    record: dict[str, Any],
    writers: JsonlWriters,
    counts: Counter,
    samples: dict[str, list[dict[str, Any]]],
    sample_per_task: int,
    total_written: int,
    max_examples_per_task: int | None,
) -> int:
    if max_examples_per_task is not None and counts[record["task"]] >= max_examples_per_task:
        return total_written
    writers.write(record["split"], record)
    counts[record["task"]] += 1
    if len(samples[record["task"]]) < sample_per_task:
        samples[record["task"]].append(record)
    return total_written + 1


def validate_record(record: dict[str, Any], valid_movie_tokens: set[str]) -> None:
    if not record["instruction"] or not record["input"] or not record["output"]:
        raise ValueError(f"Empty training field in {record['id']}")
    task = record["task"]
    if task in ID_OUTPUT_TASKS and record["output"].strip() not in valid_movie_tokens:
        raise ValueError(f"Invalid movie token output in {record['id']}: {record['output']}")
    if task in FEATURE_OUTPUT_TASKS:
        output = record["output"].lower()
        if "genres" not in output or "story" not in output:
            raise ValueError(f"Natural movie profile missing genre/story context in {record['id']}")
    if task in USER_PROFILE_TASKS and "user_" not in record["input"]:
        raise ValueError(f"User token missing from sequence input in {record['id']}")


def load_inputs(raw_dir: Path, tasks: set[str] | None = None) -> tuple[MovieFeatureStore, dict[str, dict[str, Any]], pd.DataFrame]:
    selected_tasks = tasks or set(TASKS)
    movie_features = load_movie_feature_store(raw_dir, required_movie_feature_columns(selected_tasks))
    ratings_df = load_ratings(raw_dir, movie_features)
    users = load_user_profiles(raw_dir) if selected_tasks & USER_PROFILE_TASKS else {}
    return movie_features, users, ratings_df


def emit_alignment_tasks(
    tasks: set[str],
    movie_features: MovieFeatureStore,
    valid_movie_tokens: set[str],
    writers: JsonlWriters,
    counts: Counter,
    samples: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    total_written: int,
) -> int:
    builders: dict[str, Callable[[str, MovieFeatureStore, random.Random | None], RenderedExample]] = {
        "ID2Feature": build_id_to_feature,
        "Feature2ID": build_feature_to_id,
    }
    rng = random.Random(args.prompt_template_seed)
    for movie_id in movie_features.movie_ids:
        for task, builder in builders.items():
            if task not in tasks:
                continue
            rendered = builder(movie_id, movie_features, rng)
            record = record_from_rendered(
                example_id=f"{task}:movie_{movie_id}",
                task=task,
                split="train",
                rendered=rendered,
                movie_features=movie_features,
                target_movie_id=movie_id,
            )
            validate_record(record, valid_movie_tokens)
            total_written = maybe_write(
                record,
                writers,
                counts,
                samples,
                args.sample_per_task,
                total_written,
                args.max_examples_per_task,
            )
    return total_written


def loo_targets(records: list[dict[str, Any]], min_history: int, max_history: int | None) -> Iterable[tuple[str, int, list[dict[str, Any]], dict[str, Any]]]:
    for split, offset in LEAVE_ONE_OUT_SPLITS:
        target_pos = len(records) - offset
        if target_pos < min_history:
            continue
        history = records[:target_pos]
        if max_history is not None and max_history > 0:
            history = history[-max_history:]
        yield split, target_pos, history, records[target_pos]


def training_targets(
    records_by_user: dict[str, list[dict[str, Any]]],
    min_history: int,
    max_history: int | None,
    max_train_examples: int,
    seed: int,
) -> list[tuple[str, SequenceTarget]]:
    candidates: list[tuple[str, int]] = []
    for user_id, records in records_by_user.items():
        last_train_pos = len(records) - 3
        for target_pos in range(min_history, last_train_pos + 1):
            candidates.append((user_id, target_pos))
    rng = random.Random(seed)
    if len(candidates) > max_train_examples:
        candidates = rng.sample(candidates, max_train_examples)
    candidates.sort(key=lambda item: (item[0], item[1]))

    targets = []
    for index, (user_id, target_pos) in enumerate(candidates):
        records = records_by_user[user_id]
        history = records[:target_pos]
        if max_history is not None and max_history > 0:
            history = history[-max_history:]
        targets.append((user_id, SequenceTarget("train", target_pos, history, records[target_pos], index)))
    return targets


def emit_sequence_tasks(
    tasks: set[str],
    movie_features: MovieFeatureStore,
    valid_movie_tokens: set[str],
    users: dict[str, dict[str, Any]],
    ratings_df: pd.DataFrame,
    writers: JsonlWriters,
    counts: Counter,
    eval_counts: Counter,
    samples: dict[str, list[dict[str, Any]]],
    eval_samples: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    total_written: int,
) -> int:
    grouped = ratings_df.groupby("user_id", sort=False)
    records_by_user: dict[str, list[dict[str, Any]]] = {}
    for user_id, user_ratings in grouped:
        if args.max_users is not None and len(records_by_user) >= args.max_users:
            break
        records = user_ratings.to_dict("records")
        if len(records) > args.min_history:
            records_by_user[clean_value(user_id)] = records

    for user_id, sequence_target in training_targets(
        records_by_user,
        args.min_history,
        args.max_history,
        args.max_train_examples,
        args.train_sample_seed,
    ):
        if "NextMoviePrediction" not in tasks:
            continue
        target_movie_id = sequence_target.target["movie_id"]
        rendered = build_next_movie_prediction(
            user_id,
            sequence_target.history,
            target_movie_id,
            movie_features,
            random.Random(args.prompt_template_seed + (sequence_target.train_window_index or 0)),
        )
        record = record_from_rendered(
            example_id=f"NextMoviePrediction:user_{user_id}:train_{sequence_target.train_window_index}:pos_{sequence_target.target_pos}",
            task="NextMoviePrediction",
            split="train",
            rendered=rendered,
            movie_features=movie_features,
            user_id=user_id,
            target_movie_id=target_movie_id,
            history=sequence_target.history,
            target_position=sequence_target.target_pos,
            train_window_index=sequence_target.train_window_index,
        )
        validate_record(record, valid_movie_tokens)
        total_written = maybe_write(
            record,
            writers,
            counts,
            samples,
            args.sample_per_task,
            total_written,
            args.max_examples_per_task,
        )

    for user_id, records in records_by_user.items():
        for split, target_pos, history, target in loo_targets(records, args.min_history, args.max_history):
            if split == "train":
                continue
            eval_record = eval_record_from_target(
                split=split,
                movie_features=movie_features,
                user_id=user_id,
                target_movie_id=target["movie_id"],
                history=history,
                target_position=target_pos,
            )
            writers.write(split, eval_record)
            eval_counts[split] += 1
            if len(eval_samples[split]) < args.sample_per_task:
                eval_samples[split].append(eval_record)
    return total_written


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def user_token_records(user_ids: Iterable[Any]) -> list[dict[str, str]]:
    return [{"user_id": clean_value(user_id), "user_token": user_token(user_id)} for user_id in user_ids]


def main() -> None:
    args = parse_args()
    selected_tasks = set(parse_tasks(args.tasks, args.stage))
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {args.out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    movie_features, users, ratings_df = load_inputs(args.raw_dir, selected_tasks)
    valid_movie_tokens = set(movie_features.movie_tokens)
    user_ids = list(ratings_df["user_id"].drop_duplicates())

    writers = JsonlWriters(args.out_dir, ["train", "valid", "test"])
    counts: Counter = Counter()
    eval_counts: Counter = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_written = 0
    try:
        if selected_tasks & set(RECOMMENDATION_TASKS):
            total_written = emit_sequence_tasks(
                selected_tasks,
                movie_features,
                valid_movie_tokens,
                users,
                ratings_df,
                writers,
                counts,
                eval_counts,
                samples,
                eval_samples,
                args,
                total_written,
            )
        if selected_tasks & set(ALIGNMENT_TASKS):
            total_written = emit_alignment_tasks(
                selected_tasks, movie_features, valid_movie_tokens, writers, counts, samples, args, total_written
            )
    finally:
        writers.close()

    sample_records = [record for task in TASKS for record in samples.get(task, [])]
    with (args.out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in sample_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    eval_sample_records = [record for split in ["valid", "test"] for record in eval_samples.get(split, [])]
    with (args.out_dir / "eval_samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in eval_sample_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(
        args.out_dir / "stats.json",
        {
            "total_count": total_written + sum(eval_counts.values()),
            "train_count": total_written,
            "valid_count": eval_counts["valid"],
            "test_count": eval_counts["test"],
            "train_count_by_task": dict(counts),
            "num_users": len(user_ids),
            "num_movies": len(movie_features),
            "num_movie_tokens": len(movie_features.movie_tokens),
        },
    )
    write_json(args.out_dir / "movie_tokens.json", movie_features.token_records())
    write_json(args.out_dir / "user_tokens.json", user_token_records(user_ids))
    write_json(
        args.out_dir / "manifest.json",
        {
            "source": SOURCE,
            "raw_dir": str(args.raw_dir),
            "stage": args.stage,
            "tasks": sorted(selected_tasks),
            "min_history": args.min_history,
            "max_history": args.max_history,
            "sequence_split_protocol": "leave_one_out",
            "leave_one_out_splits": {split: f"target is the {offset} item from the end" for split, offset in LEAVE_ONE_OUT_SPLITS},
            "sft_train_sampling_protocol": "overlapping_prefix_sample",
            "max_train_examples": args.max_train_examples,
            "train_sample_seed": args.train_sample_seed,
            "recommendation_output_format": "single_movie_token",
            "max_examples_per_task": args.max_examples_per_task,
            "prompt_template_seed": args.prompt_template_seed,
            "target_unit": "movie_id_token",
            "counts": {
                "train": total_written,
                "valid": eval_counts["valid"],
                "test": eval_counts["test"],
            },
        },
    )
    print(f"Wrote {total_written} examples to {args.out_dir}")
    print(f"Counts by task: {dict(counts)}")


if __name__ == "__main__":
    main()
