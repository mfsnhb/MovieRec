from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from dataset.sft_schema import (
    FEATURE_OUTPUT_TASKS,
    RATING_OUTPUT_TASKS,
    TASKS,
    TITLE_OUTPUT_TASKS,
    USER_PROFILE_TASKS,
    RenderedExample,
    build_next_movie_title_prediction,
    build_seq_rating,
    build_seq_title_to_feature,
    build_single_feature_to_title,
    build_single_title_to_feature,
)
from utils.data_io import (
    MovieFeatureStore,
    clean_value,
    load_movie_feature_store,
    load_ratings,
    load_user_profiles,
    required_movie_feature_columns,
)

SOURCE = "funrec-movielens-1m"
LEAVE_ONE_OUT_SPLITS = (
    ("train", 3),
    ("valid", 2),
    ("test", 1),
)


class JsonlWriters:
    def __init__(self, out_dir: Path, splits: Iterable[str]):
        self.handles = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in splits}

    def write(self, split: str, record: dict[str, Any]) -> None:
        self.handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build title-based MovieLens SFT JSONL data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-examples-per-task", type=int)
    parser.add_argument("--sample-per-task", type=int, default=3)
    parser.add_argument("--prompt-template-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_tasks(tasks: str) -> list[str]:
    if tasks == "all":
        return list(TASKS)
    selected = [task.strip() for task in tasks.split(",") if task.strip()]
    unknown = sorted(set(selected) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Available tasks: {list(TASKS)}")
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
    target_rating: Any | None = None,
    history: list[dict[str, Any]] | None = None,
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
        record["user_id"] = user_id
    if target_movie_id is not None:
        record["target_movie_title"] = movie_features.title(target_movie_id)
    if target_rating is not None:
        record["target_rating"] = clean_value(target_rating)
    if history is not None:
        record["history_movie_titles"] = [movie_features.title(event["movie_id"]) for event in history]
        record["history_ratings"] = [clean_value(event["rating"]) for event in history]
        if history and "timestamp" in history[0]:
            record["history_timestamps"] = [clean_value(event["timestamp"]) for event in history]
    return record


def maybe_write(
    record: dict[str, Any],
    writers: JsonlWriters,
    counts: Counter,
    split_counts: Counter,
    samples: dict[str, list[dict[str, Any]]],
    sample_per_task: int,
    total_written: int,
    max_examples_per_task: int | None,
) -> int:
    if max_examples_per_task is not None and counts[record["task"]] >= max_examples_per_task:
        return total_written
    split = record["split"]
    writers.write(split, record)
    key = (split, record["task"])
    counts[record["task"]] += 1
    split_counts[key] += 1
    if len(samples[record["task"]]) < sample_per_task:
        samples[record["task"]].append(record)
    return total_written + 1


def validate_record(record: dict[str, Any], valid_titles: set[str]) -> None:
    if not record["instruction"] or not record["input"] or not record["output"]:
        raise ValueError(f"Empty training field in {record['id']}")
    task = record["task"]
    if task in TITLE_OUTPUT_TASKS and record["output"].strip() not in valid_titles:
        raise ValueError(f"Invalid movie title output in {record['id']}: {record['output']}")
    if task in FEATURE_OUTPUT_TASKS:
        output = record["output"].lower()
        if "description" not in output or "genre" not in output:
            raise ValueError(f"Feature output missing description or genres in {record['id']}")
    target_title = str(record.get("target_movie_title", "")).strip()
    if task == "Single_Feature2Title" and target_title and target_title in record["input"]:
        raise ValueError(f"Feature-to-title input leaks target title in {record['id']}: {target_title}")
    if task == "Single_Title2Feature" and target_title and target_title in record["output"]:
        raise ValueError(f"Title-to-feature output repeats target title in {record['id']}: {target_title}")
    if task in RATING_OUTPUT_TASKS:
        try:
            float(record["output"])
        except ValueError as exc:
            raise ValueError(f"Invalid rating output in {record['id']}: {record['output']}") from exc
    if task in USER_PROFILE_TASKS:
        if "rated" not in record["input"] or "stars" not in record["input"]:
            raise ValueError(f"Interaction rating missing from sequence input in {record['id']}")
        if "- Gender:" not in record["input"] and "User profile is unavailable." not in record["input"]:
            raise ValueError(f"User profile missing from sequence input in {record['id']}")


def load_inputs(raw_dir: Path, tasks: set[str] | None = None) -> tuple[MovieFeatureStore, dict[str, dict[str, Any]], pd.DataFrame]:
    selected_tasks = tasks or set(TASKS)
    movie_features = load_movie_feature_store(raw_dir, required_movie_feature_columns(selected_tasks))
    ratings_df = load_ratings(raw_dir, movie_features)
    users = load_user_profiles(raw_dir) if selected_tasks & USER_PROFILE_TASKS else {}
    return movie_features, users, ratings_df


def emit_alignment_tasks(
    tasks: set[str],
    movie_features: MovieFeatureStore,
    valid_titles: set[str],
    writers: JsonlWriters,
    counts: Counter,
    split_counts: Counter,
    samples: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    total_written: int,
) -> int:
    builders: dict[str, Callable[[str, MovieFeatureStore, random.Random | None], RenderedExample]] = {
        "Single_Title2Feature": build_single_title_to_feature,
        "Single_Feature2Title": build_single_feature_to_title,
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
            validate_record(record, valid_titles)
            total_written = maybe_write(
                record,
                writers,
                counts,
                split_counts,
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


def emit_sequence_recommendation_tasks(
    tasks: set[str],
    movie_features: MovieFeatureStore,
    valid_titles: set[str],
    users: dict[str, dict[str, Any]],
    ratings_df: pd.DataFrame,
    writers: JsonlWriters,
    counts: Counter,
    split_counts: Counter,
    samples: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    total_written: int,
) -> int:
    grouped = ratings_df.groupby("user_id", sort=False)
    rng = random.Random(args.prompt_template_seed)
    processed_users = 0
    for user_id, user_ratings in grouped:
        if args.max_users is not None and processed_users >= args.max_users:
            break
        processed_users += 1
        records = user_ratings.to_dict("records")
        if len(records) <= args.min_history:
            continue
        for split, target_pos, history, target in loo_targets(records, args.min_history, args.max_history):
            target_movie_id = target["movie_id"]
            task_renderers = []
            if "NextMovieTitlePrediction" in tasks:
                task_renderers.append(
                    (
                        "NextMovieTitlePrediction",
                        build_next_movie_title_prediction(users.get(user_id), history, target_movie_id, movie_features, rng),
                    )
                )
            if "Seq_Title2Feature" in tasks:
                task_renderers.append(
                    (
                        "Seq_Title2Feature",
                        build_seq_title_to_feature(users.get(user_id), history, target_movie_id, movie_features, rng),
                    )
                )
            if "Seq_Rating" in tasks:
                task_renderers.append(
                    (
                        "Seq_Rating",
                        build_seq_rating(
                            users.get(user_id),
                            history,
                            target_movie_id,
                            target["rating"],
                            movie_features,
                            rng,
                        ),
                    )
                )
            for task, rendered in task_renderers:
                record = record_from_rendered(
                    example_id=f"{task}:user_{user_id}:loo_{split}:pos_{target_pos}",
                    task=task,
                    split=split,
                    rendered=rendered,
                    movie_features=movie_features,
                    user_id=user_id,
                    target_movie_id=target_movie_id,
                    target_rating=target["rating"] if task == "Seq_Rating" else None,
                    history=history,
                )
                validate_record(record, valid_titles)
                total_written = maybe_write(
                    record,
                    writers,
                    counts,
                    split_counts,
                    samples,
                    args.sample_per_task,
                    total_written,
                    args.max_examples_per_task,
                )
    return total_written


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_tasks = set(parse_tasks(args.tasks))
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {args.out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    movie_features, users, ratings_df = load_inputs(args.raw_dir, selected_tasks)
    valid_titles = {movie_features.title(movie_id) for movie_id in movie_features.movie_ids}
    user_ids = list(ratings_df["user_id"].drop_duplicates())

    writers = JsonlWriters(args.out_dir, ["train", "valid", "test"])
    counts: Counter = Counter()
    split_counts: Counter = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_written = 0
    try:
        total_written = emit_sequence_recommendation_tasks(
            selected_tasks,
            movie_features,
            valid_titles,
            users,
            ratings_df,
            writers,
            counts,
            split_counts,
            samples,
            args,
            total_written,
        )
        total_written = emit_alignment_tasks(
            selected_tasks, movie_features, valid_titles, writers, counts, split_counts, samples, args, total_written
        )
    finally:
        writers.close()

    sample_records = [record for task in TASKS for record in samples.get(task, [])]
    with (args.out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in sample_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    split_task_counts = {f"{split}:{task}": count for (split, task), count in sorted(split_counts.items())}
    write_json(
        args.out_dir / "stats.json",
        {
            "total_examples": total_written,
            "counts_by_task": dict(counts),
            "counts_by_split_task": split_task_counts,
            "num_users": len(user_ids),
            "num_movies": len(movie_features),
            "num_unique_titles": len(valid_titles),
        },
    )
    write_json(
        args.out_dir / "manifest.json",
        {
            "source": SOURCE,
            "raw_dir": str(args.raw_dir),
            "tasks": sorted(selected_tasks),
            "min_history": args.min_history,
            "max_history": args.max_history,
            "sequence_split_protocol": "leave_one_out",
            "leave_one_out_splits": {split: f"target is the {offset} item from the end" for split, offset in LEAVE_ONE_OUT_SPLITS},
            "max_examples_per_task": args.max_examples_per_task,
            "prompt_template_seed": args.prompt_template_seed,
            "splits": {split: sum(count for (s, _), count in split_counts.items() if s == split) for split in ["train", "valid", "test"]},
        },
    )
    print(f"Wrote {total_written} examples to {args.out_dir}")
    print(f"Counts by task: {dict(counts)}")


if __name__ == "__main__":
    main()
