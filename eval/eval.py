from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from tqdm import tqdm

from dataset.build_sft_dataset import loo_targets
from utils.data_io import (
    clean_value,
    load_movie_feature_store,
    load_ratings,
    load_user_profiles,
)
from utils.inference import (
    RankingConfig,
    build_recommendation_prompt,
    generate_movie_recommendations_batch,
    load_inference_components,
    prediction_record,
    rank_movie_recommendations_batch,
)
from utils.reranker_scores import RerankerScores
from utils.training_utils import ensure_dir, save_json, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation for MovieRec movie ID token recommendation.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/leave_one_out"))
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-negatives", type=int, default=0, help="Use candidate eval with this many random negatives per positive; 0 keeps full-ranking eval.")
    parser.add_argument("--score-mode", choices=["generate", "first_token"], default="generate")
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--reranker-score-path", type=Path, help="Optional SASRec score npz for reporting distillation agreement metrics.")
    parser.add_argument("--seed", type=int, default=42)
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

    movie_features = load_movie_feature_store(args.raw_dir, {"movie_id", "title"})
    users = load_user_profiles(args.raw_dir)
    ratings_df = load_ratings(args.raw_dir, movie_features)
    components = load_inference_components(
        args.model_name_or_path,
        movie_features,
        args.load_in_4bit,
        args.attn_implementation,
    )
    ranking_config = RankingConfig(top_k=args.top_k, generation_max_new_tokens=args.generation_max_new_tokens)

    predictions_path = args.output_dir / "predictions.jsonl"
    metrics = {
        "total": 0,
        "hr@1": 0,
        "hr@5": 0,
        "hr@10": 0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
        "mrr": 0.0,
        "sasrec_overlap@5": 0.0,
        "sasrec_overlap@10": 0.0,
        "target_in_sasrec@10": 0,
    }
    reranker_scores = RerankerScores(args.reranker_score_path) if args.reranker_score_path is not None else None

    grouped = ratings_df.groupby("user_id", sort=False)
    rng = random.Random(args.seed)
    negative_pools: dict[str, list[str]] = {}
    if args.candidate_negatives > 0:
        for pool_user_id, user_ratings in grouped:
            interacted = {clean_value(movie_id) for movie_id in user_ratings["movie_id"]}
            negative_pools[pool_user_id] = [movie_id for movie_id in movie_features.movie_ids if movie_id not in interacted]

    with predictions_path.open("w", encoding="utf-8") as handle:
        pending: list[dict[str, object]] = []

        def flush_pending() -> None:
            if not pending:
                return
            if args.score_mode == "first_token" or args.candidate_negatives > 0:
                predictions = rank_movie_recommendations_batch(
                    components,
                    [str(item["prompt"]) for item in pending],
                    ranking_config,
                    [item["history_movie_ids"] for item in pending],  # type: ignore[list-item]
                    [item["candidate_movie_ids"] for item in pending],  # type: ignore[list-item]
                )
            else:
                predictions = generate_movie_recommendations_batch(
                    components,
                    [str(item["prompt"]) for item in pending],
                    ranking_config,
                    [item["history_movie_ids"] for item in pending],  # type: ignore[list-item]
                )
            for item, prediction in zip(pending, predictions, strict=True):
                target_movie_id = str(item["target_movie_id"])
                ranked = prediction["predicted_movie_ids"]
                rank = ranked.index(target_movie_id) + 1 if target_movie_id in ranked else None

                metrics["total"] += 1
                metrics["hr@1"] += int(rank == 1)
                metrics["hr@5"] += int(rank is not None and rank <= 5)
                metrics["hr@10"] += int(rank is not None and rank <= 10)
                metrics["ndcg@5"] += ndcg(rank) if rank is not None and rank <= 5 else 0.0
                metrics["ndcg@10"] += ndcg(rank) if rank is not None and rank <= 10 else 0.0
                metrics["mrr"] += 1.0 / rank if rank is not None else 0.0

                record = prediction_record(
                    str(item["user_id"]),
                    str(item["split"]),
                    int(item["target_pos"]),
                    target_movie_id,
                    str(item["target_movie_token"]),
                    str(item["target_movie_title"]),
                    prediction,
                    rank,
                )
                if reranker_scores is not None:
                    sasrec_top10 = reranker_scores.top_movie_ids(str(item["user_id"]), item["history_movie_ids"], 10)  # type: ignore[arg-type]
                    predicted_at_5 = set(ranked[:5])
                    predicted_at_10 = set(ranked[:10])
                    sasrec_at_5 = set(sasrec_top10[:5])
                    sasrec_at_10 = set(sasrec_top10[:10])
                    metrics["sasrec_overlap@5"] += len(predicted_at_5 & sasrec_at_5) / 5
                    metrics["sasrec_overlap@10"] += len(predicted_at_10 & sasrec_at_10) / 10
                    metrics["target_in_sasrec@10"] += int(target_movie_id in sasrec_at_10)
                    record["sasrec_top10_movie_ids"] = sasrec_top10
                    record["sasrec_overlap@10"] = len(predicted_at_10 & sasrec_at_10) / 10
                if item["candidate_movie_ids"] is not None:
                    record["candidate_movie_ids"] = item["candidate_movie_ids"]
                    record["candidate_movie_tokens"] = [movie_features.token(movie_id) for movie_id in item["candidate_movie_ids"]]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            pending.clear()

        processed_users = 0
        for user_id, user_ratings in tqdm(grouped, desc="leave-one-out movie token eval"):
            if args.max_users is not None and processed_users >= args.max_users:
                break
            records = user_ratings.to_dict("records")
            if len(records) <= args.min_history:
                continue
            processed_users += 1
            for split, target_pos, history, target in loo_targets(records, args.min_history, args.max_history):
                if split != "test":
                    continue
                target_movie_id = clean_value(target["movie_id"])
                candidate_movie_ids = None
                if args.candidate_negatives > 0:
                    pool = negative_pools[user_id]
                    if len(pool) < args.candidate_negatives:
                        raise ValueError(f"User {user_id} has only {len(pool)} available negatives; need {args.candidate_negatives}.")
                    candidate_movie_ids = [target_movie_id] + rng.sample(pool, args.candidate_negatives)
                    rng.shuffle(candidate_movie_ids)
                prompt = build_recommendation_prompt(users.get(user_id), history, movie_features, candidate_movie_ids)
                pending.append(
                    {
                        "user_id": user_id,
                        "split": split,
                        "target_pos": target_pos,
                        "target_movie_id": target_movie_id,
                        "target_movie_token": movie_features.token(target_movie_id),
                        "target_movie_title": movie_features.title(target_movie_id),
                        "prompt": prompt,
                        "history_movie_ids": {clean_value(event["movie_id"]) for event in history},
                        "candidate_movie_ids": candidate_movie_ids,
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
        "MRR": metrics["mrr"] / total,
        "SASRecOverlap@5": metrics["sasrec_overlap@5"] / total if reranker_scores is not None else None,
        "SASRecOverlap@10": metrics["sasrec_overlap@10"] / total if reranker_scores is not None else None,
        "TargetInSASRec@10": metrics["target_in_sasrec@10"] / total if reranker_scores is not None else None,
        "top_k": args.top_k,
        "max_history": args.max_history,
        "prediction_unit": "movie_id_token",
        "candidate_negatives": args.candidate_negatives,
        "reranker_score_path": str(args.reranker_score_path) if args.reranker_score_path is not None else None,
        "eval_protocol": "candidate" if args.candidate_negatives > 0 else "full_ranking",
        "score_mode": args.score_mode,
        "generation_max_new_tokens": args.generation_max_new_tokens,
    }
    save_json(args.output_dir / "leave_one_out_metrics.json", normalized)
    print(json.dumps(normalized, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
