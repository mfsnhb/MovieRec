from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from model.llm import ModelConfig, collect_movie_tokens, load_causal_lm, load_tokenizer
from dataset.sft_schema import ID_ONLY_SUFFIX, format_id_interaction_history, format_user_profile
from utils.data_io import (
    clean_value,
    load_id_inputs,
    movie_token,
)
from utils.movie_generation import append_movie_logits_processor, build_movie_token_id_map
from utils.training_utils import ensure_dir, save_json, setup_seed


MOVIE_TOKEN_RE = re.compile(r"<movie_\d+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation for MovieRec LLM recommendation.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/leave_one_out"))
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--num-return-sequences", type=int, default=10)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--movie-token-format", choices=["angle", "plain"], default="angle")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_eval_prompt(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    token_format: str,
) -> str:
    prompt_input = f"""User profile:
{format_user_profile(user)}

Chronological interaction history before the target movie. Each line says that the user gave a MovieLens ID a rating at a specific time:
{format_id_interaction_history(history, token_format)}

Based on all interactions before the target, predict the next movie the user is most likely to watch.
{ID_ONLY_SUFFIX}"""
    return f"""You are a helpful movie recommendation assistant. Use the user's profile and interaction history to recommend the next movie.

### Input:
{prompt_input}

### Response:
"""


def parse_movie_tokens(text: str) -> list[str]:
    seen = set()
    tokens = []
    for token in MOVIE_TOKEN_RE.findall(text):
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


def ndcg(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    ensure_dir(args.output_dir)

    movie_features, users, ratings_df = load_id_inputs(args.raw_dir)
    valid_movie_tokens = sorted(movie_token(movie_id, args.movie_token_format) for movie_id in movie_features.movie_ids)
    tokenizer = load_tokenizer(
        args.model_name_or_path,
        movie_tokens=collect_movie_tokens(raw_dir=args.raw_dir, token_format=args.movie_token_format),
        padding_side="left",
    )
    movie_token_id_map = build_movie_token_id_map(tokenizer, valid_movie_tokens)
    movie_token_ids = list(movie_token_id_map.values())
    model = load_causal_lm(ModelConfig(args.model_name_or_path, load_in_4bit=args.load_in_4bit), tokenizer=tokenizer)
    model.eval()

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
        processed_users = 0
        for user_id, user_ratings in tqdm(grouped, desc="leave-one-out eval"):
            if args.max_users is not None and processed_users >= args.max_users:
                break
            records = user_ratings.to_dict("records")
            if len(records) <= args.min_history:
                continue
            processed_users += 1
            history = records[:-1]
            if args.max_history is not None and args.max_history > 0:
                history = history[-args.max_history :]
            target = records[-1]
            target_token = movie_token(clean_value(target["movie_id"]), args.movie_token_format)
            prompt = build_eval_prompt(users.get(user_id), history, args.movie_token_format)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            logits_processor = append_movie_logits_processor(
                None,
                movie_token_ids,
                inputs["input_ids"].shape[1],
                [tokenizer.eos_token_id, tokenizer.pad_token_id],
            )
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "num_beams": max(args.num_beams, args.num_return_sequences) if not args.do_sample else args.num_beams,
                "num_return_sequences": args.num_return_sequences,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "logits_processor": logits_processor,
            }
            if args.do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            decoded = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            ranked = []
            for text in decoded:
                for token in parse_movie_tokens(text):
                    if token not in ranked:
                        ranked.append(token)
            rank = ranked.index(target_token) + 1 if target_token in ranked else None

            metrics["total"] += 1
            metrics["hr@1"] += int(rank == 1)
            metrics["hr@5"] += int(rank is not None and rank <= 5)
            metrics["hr@10"] += int(rank is not None and rank <= 10)
            metrics["ndcg@5"] += ndcg(rank) if rank is not None and rank <= 5 else 0.0
            metrics["ndcg@10"] += ndcg(rank) if rank is not None and rank <= 10 else 0.0

            handle.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "target_movie_id": target_token,
                        "predicted_movie_ids": ranked,
                        "rank": rank,
                        "generations": decoded,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

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
