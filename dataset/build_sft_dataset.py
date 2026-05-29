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
)
from utils.reranker_scores import RerankerScores

SOURCE = "funrec-movielens-1m"
LEAVE_ONE_OUT_SPLITS = (
    ("train", 3),
    ("valid", 2),
    ("test", 1),
)
TRAIN_TARGET_OFFSET = 3


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
    parser = argparse.ArgumentParser(description="Build MovieLens SFT JSONL data with Movie ID tokens.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--stage", choices=["all", "alignment", "recommendation"], default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--train-windows-per-user", type=int, default=5)
    parser.add_argument("--reranker-score-path", type=Path)
    parser.add_argument("--prefix-label-path", type=Path)
    parser.add_argument("--sft-topk-min", type=int, default=5)
    parser.add_argument("--sft-topk-max", type=int, default=10)
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
    candidate_movie_ids: list[str] | None = None,
    label_movie_ids: list[str] | None = None,
    label_scores: list[float] | None = None,
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
    if candidate_movie_ids is not None:
        normalized_candidates = [clean_value(movie_id) for movie_id in candidate_movie_ids]
        record["candidate_movie_ids"] = normalized_candidates
        record["candidate_movie_tokens"] = [movie_features.token(movie_id) for movie_id in normalized_candidates]
        record["candidate_movie_titles"] = [movie_features.title(movie_id) for movie_id in normalized_candidates]
        if target_movie_id is not None:
            record["positive_candidate_index"] = normalized_candidates.index(clean_value(target_movie_id))
    if label_movie_ids is not None:
        normalized_labels = [clean_value(movie_id) for movie_id in label_movie_ids]
        record["label_movie_ids"] = normalized_labels
        record["label_movie_tokens"] = [movie_features.token(movie_id) for movie_id in normalized_labels]
        record["label_movie_titles"] = [movie_features.title(movie_id) for movie_id in normalized_labels]
        record["label_k"] = len(normalized_labels)
        record["label_source"] = "sasrec_prefix_teacher" if label_scores is not None else "sasrec_reranker_scores"
        if label_scores is not None:
            record["label_scores"] = label_scores
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


def output_movie_token(line: str) -> str:
    content = line.strip().split("|", 1)[0].strip()
    if ". " in content:
        content = content.split(". ", 1)[1].strip()
    return content


def validate_record(record: dict[str, Any], valid_movie_tokens: set[str]) -> None:
    if not record["instruction"] or not record["input"] or not record["output"]:
        raise ValueError(f"Empty training field in {record['id']}")
    task = record["task"]
    if task in ID_OUTPUT_TASKS:
        output_tokens = [output_movie_token(line) for line in record["output"].splitlines() if line.strip()]
        if task == "NextMoviePrediction" and "label_k" in record:
            if len(output_tokens) != record["label_k"]:
                raise ValueError(f"Expected {record['label_k']} output tokens in {record['id']}: {record['output']}")
            if len(set(output_tokens)) != len(output_tokens):
                raise ValueError(f"Duplicate movie token output in {record['id']}: {record['output']}")
        elif output_tokens:
            output_tokens = output_tokens[:1]
        for output_token in output_tokens:
            if output_token not in valid_movie_tokens:
                raise ValueError(f"Invalid movie token output in {record['id']}: {record['output']}")
    if task in FEATURE_OUTPUT_TASKS:
        output = record["output"].lower()
        if "genres" not in output or "story" not in output:
            raise ValueError(f"Natural movie profile missing genre/story context in {record['id']}")
    if task in USER_PROFILE_TASKS:
        if "rating:" in record["input"]:
            raise ValueError(f"Interaction rating should not appear in sequence input in {record['id']}")
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


def non_overlapping_train_targets(
    records: list[dict[str, Any]],
    min_history: int,
    max_history: int | None,
    windows_per_user: int,
) -> Iterable[SequenceTarget]:
    if windows_per_user <= 0:
        return
    target_pos = len(records) - TRAIN_TARGET_OFFSET
    window_size = max_history if max_history is not None and max_history > 0 else target_pos
    for window_index in range(windows_per_user):
        if target_pos < min_history:
            break
        history_end = target_pos
        history_start = max(0, history_end - window_size)
        history = records[history_start:history_end]
        if len(history) < min_history:
            break
        yield SequenceTarget("train", target_pos, history, records[target_pos], window_index)
        target_pos = history_start - 1


def sequence_targets(
    records: list[dict[str, Any]],
    min_history: int,
    max_history: int | None,
    train_windows_per_user: int,
) -> Iterable[SequenceTarget]:
    yield from non_overlapping_train_targets(records, min_history, max_history, train_windows_per_user)
    for split, target_pos, history, target in loo_targets(records, min_history, max_history):
        if split == "train":
            continue
        yield SequenceTarget(split, target_pos, history, target)


def load_prefix_labels(path: Path | None) -> dict[str, tuple[list[str], list[float]]]:
    if path is None:
        return {}
    labels: dict[str, tuple[list[str], list[float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            labels[str(record["id"])] = (
                [clean_value(movie_id) for movie_id in record["label_movie_ids"]],
                [float(score) for score in record.get("label_scores", [])],
            )
    return labels


def teacher_augmented_label(target_movie_id: str, teacher_movie_ids: list[str], teacher_scores: list[float], k: int) -> tuple[list[str], list[float]]:
    target_movie_id = clean_value(target_movie_id)
    label_movie_ids = [target_movie_id]
    label_scores = [float("inf")]
    for movie_id, score in zip(teacher_movie_ids, teacher_scores, strict=False):
        movie_id = clean_value(movie_id)
        if movie_id == target_movie_id or movie_id in label_movie_ids:
            continue
        label_movie_ids.append(movie_id)
        label_scores.append(float(score))
        if len(label_movie_ids) >= k:
            break
    return label_movie_ids, label_scores


def topk_label_from_reranker(
    reranker_scores: RerankerScores | None,
    user_id: str,
    history: list[dict[str, Any]],
    k: int,
) -> tuple[list[str], list[float]]:
    excluded = [clean_value(event["movie_id"]) for event in history]
    return reranker_scores.top_movie_ids_with_scores(user_id, excluded, k)


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
    rng = random.Random(args.prompt_template_seed)
    reranker_scores = RerankerScores(args.reranker_score_path) if args.reranker_score_path is not None else None
    prefix_labels = load_prefix_labels(args.prefix_label_path)
    if "NextMoviePrediction" in tasks and reranker_scores is None and not prefix_labels:
        raise ValueError("--prefix-label-path or --reranker-score-path is required for NextMoviePrediction top-k SFT labels.")
    if args.sft_topk_min < 1 or args.sft_topk_max < args.sft_topk_min:
        raise ValueError("Require 1 <= --sft-topk-min <= --sft-topk-max.")

    processed_users = 0
    for user_id, user_ratings in grouped:
        if args.max_users is not None and processed_users >= args.max_users:
            break
        processed_users += 1
        records = user_ratings.to_dict("records")
        if len(records) <= args.min_history:
            continue
        for sequence_target in sequence_targets(records, args.min_history, args.max_history, args.train_windows_per_user):
            target_movie_id = sequence_target.target["movie_id"]
            if sequence_target.split != "train":
                eval_record = eval_record_from_target(
                    split=sequence_target.split,
                    movie_features=movie_features,
                    user_id=user_id,
                    target_movie_id=target_movie_id,
                    history=sequence_target.history,
                    target_position=sequence_target.target_pos,
                )
                writers.write(sequence_target.split, eval_record)
                eval_counts[sequence_target.split] += 1
                if len(eval_samples[sequence_target.split]) < args.sample_per_task:
                    eval_samples[sequence_target.split].append(eval_record)
                continue

            if "NextMoviePrediction" not in tasks:
                continue

            label_k = rng.randint(args.sft_topk_min, args.sft_topk_max)
            example_id = f"NextMoviePrediction:user_{user_id}:train_window_{sequence_target.train_window_index}:pos_{sequence_target.target_pos}"
            if prefix_labels:
                all_label_movie_ids, all_label_scores = prefix_labels[example_id]
            else:
                all_label_movie_ids, all_label_scores = topk_label_from_reranker(
                    reranker_scores,
                    user_id,
                    sequence_target.history,
                    label_k,
                )
            label_movie_ids, label_scores = teacher_augmented_label(
                target_movie_id,
                all_label_movie_ids,
                all_label_scores,
                label_k,
            )
            rendered = build_next_movie_prediction(
                users.get(user_id),
                sequence_target.history,
                label_movie_ids,
                movie_features,
                rng,
            )
            record = record_from_rendered(
                example_id=example_id,
                task="NextMoviePrediction",
                split="train",
                rendered=rendered,
                movie_features=movie_features,
                user_id=user_id,
                target_movie_id=target_movie_id,
                history=sequence_target.history,
                target_position=sequence_target.target_pos,
                train_window_index=sequence_target.train_window_index,
                label_movie_ids=label_movie_ids,
                label_scores=label_scores,
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


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
            "split_protocol_note": (
                "Leave-one-out is applied at the user sequence level. Training JSONL contains SFT task records built from "
                "non-overlapping train windows; valid/test JSONL contain raw sequence-target eval records without task, "
                "instruction, input, or output fields."
            ),
            "train_windows_per_user": args.train_windows_per_user,
            "sft_train_sampling_protocol": "non_overlapping_recent_windows",
            "sft_train_window_rule": (
                "For SFT recommendation training, NextMoviePrediction uses the configured non-overlapping train windows. "
                "Each train label places the true clicked target first, then fills the remaining ranked slots with SASRec "
                "prefix-teacher recommendations after excluding the prompt history and duplicates. "
                "Each earlier train target is the item immediately before the previous max-history window, "
                "so train history windows do not overlap."
            ),
            "recommendation_label_source": (
                "sasrec_prefix_teacher" if args.prefix_label_path is not None else "sasrec_reranker_scores" if args.reranker_score_path is not None else None
            ),
            "prefix_label_path": str(args.prefix_label_path) if args.prefix_label_path is not None else None,
            "reranker_score_path": str(args.reranker_score_path) if args.reranker_score_path is not None else None,
            "recommendation_label_k_range": [args.sft_topk_min, args.sft_topk_max],
            "recommendation_output_format": "one_movie_token_per_line_ranked",
            "max_examples_per_task": args.max_examples_per_task,
            "prompt_template_seed": args.prompt_template_seed,
            "target_unit": "movie_id_token_list",
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
