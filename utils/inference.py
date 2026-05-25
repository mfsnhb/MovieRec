from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from dataset.sft_schema import ID_ONLY_SUFFIX, format_id_interaction_history, format_user_profile
from dataset.build_sft_dataset import loo_targets
from model.llm import ModelConfig, collect_movie_tokens, load_causal_lm, load_tokenizer
from utils.data_io import clean_value, load_id_inputs, movie_token
from utils.movie_generation import build_movie_token_id_map
from utils.training_utils import ensure_dir, setup_seed


RECOMMENDATION_INSTRUCTION = "You are a helpful movie recommendation assistant. Use the user's profile and interaction history to recommend the next movie."


@dataclass(frozen=True)
class GenerationConfig:
    top_k: int = 10


@dataclass(frozen=True)
class InferenceComponents:
    tokenizer: Any
    model: Any
    movie_token_id_map: dict[str, int]
    movie_tokens: list[str]
    movie_token_ids: torch.Tensor


def build_recommendation_prompt(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    token_format: Literal["angle", "plain"] = "angle",
) -> str:
    prompt_input = f"""User profile:
{format_user_profile(user)}

Chronological interaction history before the target movie. Each line says that the user gave a MovieLens ID a rating at a specific time:
{format_id_interaction_history(history, token_format)}

Based on this profile and all interactions before the target, predict the next movie the user is most likely to watch.
{ID_ONLY_SUFFIX}"""
    return f"""{RECOMMENDATION_INSTRUCTION}

### Input:
{prompt_input}

### Response:
"""


def movie_token_regex(token_format: Literal["angle", "plain"] = "angle") -> re.Pattern[str]:
    pattern = r"<movie_\d+>" if token_format == "angle" else r"(?<![\w<])movie_\d+(?![\w>])"
    return re.compile(pattern)


def parse_movie_tokens(text: str, token_format: Literal["angle", "plain"] = "angle") -> list[str]:
    seen = set()
    tokens = []
    for token in movie_token_regex(token_format).findall(text):
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


def load_inference_components(
    model_name_or_path: str,
    raw_dir,
    valid_movie_ids: list[str],
    token_format: Literal["angle", "plain"] = "angle",
    load_in_4bit: bool = True,
    attn_implementation: str | None = None,
) -> InferenceComponents:
    valid_movie_tokens = [movie_token(movie_id, token_format) for movie_id in valid_movie_ids]
    tokenizer = load_tokenizer(
        model_name_or_path,
        movie_tokens=collect_movie_tokens(raw_dir=raw_dir, token_format=token_format),
        padding_side="left",
    )
    movie_token_id_map = build_movie_token_id_map(tokenizer, valid_movie_tokens)
    model = load_causal_lm(
        ModelConfig(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            attn_implementation=attn_implementation,
        ),
        tokenizer=tokenizer,
    )
    model.eval()
    movie_tokens = list(movie_token_id_map)
    movie_token_ids = torch.tensor(
        [movie_token_id_map[token] for token in movie_tokens],
        device=model.device,
        dtype=torch.long,
    )
    return InferenceComponents(
        tokenizer=tokenizer,
        model=model,
        movie_token_id_map=movie_token_id_map,
        movie_tokens=movie_tokens,
        movie_token_ids=movie_token_ids,
    )


def _rank_movie_logit_row(
    movie_tokens: list[str],
    logits: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    k = min(top_k, int(torch.isfinite(logits).sum().item()))
    if k == 0:
        return {"generations": [], "predicted_movie_ids": [], "scores": {}}
    scores, indices = torch.topk(F.log_softmax(logits, dim=-1), k=k)
    ranked = [movie_tokens[index] for index in indices.tolist()]
    return {
        "generations": ranked,
        "predicted_movie_ids": ranked,
        "scores": {token: float(score) for token, score in zip(ranked, scores.tolist(), strict=True)},
    }


def rank_movie_tokens_from_logits_batch(
    components: InferenceComponents,
    prompts: list[str],
    top_k: int,
    excluded_movie_tokens: list[set[str] | None] | None = None,
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    tokenizer = components.tokenizer
    model = components.model
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, components.movie_token_ids]

    if excluded_movie_tokens:
        token_to_column = {token: index for index, token in enumerate(components.movie_tokens)}
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for row, excluded in enumerate(excluded_movie_tokens):
            if not excluded:
                continue
            columns = [token_to_column[token] for token in excluded if token in token_to_column]
            if columns:
                mask[row, columns] = True
        logits = logits.masked_fill(mask, -torch.inf)

    return [_rank_movie_logit_row(components.movie_tokens, row_logits, top_k) for row_logits in logits]


def rank_movie_tokens_from_logits(
    components: InferenceComponents,
    prompt: str,
    top_k: int,
    excluded_movie_tokens: set[str] | None = None,
) -> dict[str, Any]:
    return rank_movie_tokens_from_logits_batch(components, [prompt], top_k, [excluded_movie_tokens])[0]


def generate_ranked_movie_tokens(
    components: InferenceComponents,
    prompt: str,
    generation_config: GenerationConfig,
    excluded_movie_tokens: set[str] | None = None,
    token_format: Literal["angle", "plain"] = "angle",
) -> dict[str, Any]:
    return rank_movie_tokens_from_logits(components, prompt, generation_config.top_k, excluded_movie_tokens)


def generate_ranked_movie_tokens_batch(
    components: InferenceComponents,
    prompts: list[str],
    generation_config: GenerationConfig,
    excluded_movie_tokens: list[set[str] | None] | None = None,
    token_format: Literal["angle", "plain"] = "angle",
) -> list[dict[str, Any]]:
    return rank_movie_tokens_from_logits_batch(components, prompts, generation_config.top_k, excluded_movie_tokens)


def prediction_record(
    user_id: str,
    split: str,
    target_pos: int,
    target_token: str,
    prediction: dict[str, Any],
    rank: int | None = None,
) -> dict[str, Any]:
    record = {
        "user_id": user_id,
        "split": split,
        "target_position": target_pos,
        "target_movie_id": target_token,
        "predicted_movie_ids": prediction["predicted_movie_ids"],
        "generations": prediction["generations"],
    }
    if rank is not None:
        record["rank"] = rank
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MovieRec constrained movie-ID inference.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/inference/predictions.jsonl"))
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--movie-token-format", choices=["angle", "plain"], default="angle")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    ensure_dir(args.output_path.parent)

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

    grouped = ratings_df.groupby("user_id", sort=False)
    with args.output_path.open("w", encoding="utf-8") as handle:
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
                handle.write(
                    json.dumps(
                        prediction_record(
                            str(item["user_id"]),
                            str(item["split"]),
                            int(item["target_pos"]),
                            str(item["target_token"]),
                            prediction,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            pending.clear()

        processed_users = 0
        for user_id, user_ratings in grouped:
            if args.max_users is not None and processed_users >= args.max_users:
                break
            records = user_ratings.to_dict("records")
            if len(records) <= args.min_history:
                continue
            processed_users += 1
            for split, target_pos, history, target in loo_targets(records, args.min_history, args.max_history):
                if split != args.split:
                    continue
                prompt = build_recommendation_prompt(users.get(user_id), history, args.movie_token_format)
                history_tokens = {movie_token(clean_value(event["movie_id"]), args.movie_token_format) for event in history}
                target_token = movie_token(clean_value(target["movie_id"]), args.movie_token_format)
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


if __name__ == "__main__":
    main()
