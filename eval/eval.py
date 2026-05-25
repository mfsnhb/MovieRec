from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tqdm import tqdm

from dataset.build_sft_dataset import loo_targets
from utils.data_io import (
    clean_value,
    load_id_inputs,
    movie_token,
)
from utils.inference import (
    GenerationConfig,
    build_recommendation_prompt,
    generate_ranked_movie_tokens_batch,
    load_inference_components,
    prediction_record,
)
from utils.training_utils import ensure_dir, save_json, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation for MovieRec LLM recommendation.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/leave_one_out"))
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--movie-token-format", choices=["angle", "plain"], default="angle")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def ndcg(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    ensure_dir(args.output_dir)

    movie_features, users, ratings_df = load_id_inputs(args.raw_dir)
    components = load_inference_components(
        args.model_name_or_path,
        args.raw_dir,
        list(movie_features.movie_ids),
        args.movie_token_format,
        args.load_in_4bit,
        args.attn_implementation,
    )
    generation_config = GenerationConfig(top_k=args.top_k)

    predictions_path = args.output_dir / "predictions.jsonl"
    metrics = {
        "total": 0,
        "hr@1": 0,
        "hr@5": 0,
        "hr@10": 0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
    }

    grouped = ratings_df.groupby("user_id", sort=False)
    with predictions_path.open("w", encoding="utf-8") as handle:
        pending: list[dict[str, object]] = []

        def flush_pending() -> None:
            if not pending:
                return
            predictions = generate_ranked_movie_tokens_batch(
                components,
                [str(item["prompt"]) for item in pending],
                generation_config,
                [item["history_tokens"] for item in pending],  # type: ignore[list-item]
                token_format=args.movie_token_format,
            )
            for item, prediction in zip(pending, predictions, strict=True):
                target_token = str(item["target_token"])
                ranked = prediction["predicted_movie_ids"]
                rank = ranked.index(target_token) + 1 if target_token in ranked else None

                metrics["total"] += 1
                metrics["hr@1"] += int(rank == 1)
                metrics["hr@5"] += int(rank is not None and rank <= 5)
                metrics["hr@10"] += int(rank is not None and rank <= 10)
                metrics["ndcg@5"] += ndcg(rank) if rank is not None and rank <= 5 else 0.0
                metrics["ndcg@10"] += ndcg(rank) if rank is not None and rank <= 10 else 0.0

                handle.write(
                    json.dumps(
                        prediction_record(
                            str(item["user_id"]),
                            str(item["split"]),
                            int(item["target_pos"]),
                            target_token,
                            prediction,
                            rank,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            pending.clear()

        processed_users = 0
        for user_id, user_ratings in tqdm(grouped, desc="leave-one-out eval"):
            if args.max_users is not None and processed_users >= args.max_users:
                break
            records = user_ratings.to_dict("records")
            if len(records) <= args.min_history:
                continue
            processed_users += 1
            for split, target_pos, history, target in loo_targets(records, args.min_history, args.max_history):
                if split != "test":
                    continue
                target_token = movie_token(clean_value(target["movie_id"]), args.movie_token_format)
                prompt = build_recommendation_prompt(users.get(user_id), history, args.movie_token_format)
                history_tokens = {movie_token(clean_value(event["movie_id"]), args.movie_token_format) for event in history}
                pending.append(
                    {
                        "user_id": user_id,
                        "split": split,
                        "target_pos": target_pos,
                        "target_token": target_token,
                        "prompt": prompt,
                        "history_tokens": history_tokens,
                    }
                )
                if len(pending) >= args.batch_size:
                    flush_pending()
        flush_pending()

    total = max(1, metrics["total"])
    normalized = {
        "num_users": metrics["total"],
        "HR@1": metrics["hr@1"] / total,
        "HR@5": metrics["hr@5"] / total,
        "HR@10": metrics["hr@10"] / total,
        "NDCG@5": metrics["ndcg@5"] / total,
        "NDCG@10": metrics["ndcg@10"] / total,
    }
    save_json(args.output_dir / "leave_one_out_metrics.json", normalized)
    print(json.dumps(normalized, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
