from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import re

import torch

from dataset.build_sft_dataset import loo_targets
from dataset.sft_schema import (
    CANDIDATE_MOVIE_TOKEN_ONLY_SUFFIX,
    INSTRUCTION,
    MOVIE_TOKEN_LIST_SUFFIX,
    format_candidate_movies,
    format_id_interaction_history,
    format_id_title_interaction_history,
    format_user_profile,
)
from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.data_io import (
    MovieFeatureStore,
    clean_value,
    load_movie_feature_store,
    load_ratings,
    load_user_profiles,
)
from utils.training_utils import ensure_dir, setup_seed


RECOMMENDATION_INSTRUCTION = INSTRUCTION


@dataclass(frozen=True)
class RankingConfig:
    top_k: int = 10
    generation_max_new_tokens: int = 64


@dataclass(frozen=True)
class InferenceComponents:
    tokenizer: Any
    model: Any
    movie_ids: list[str]
    movie_tokens: list[str]
    movie_token_ids: list[int]
    movie_id_to_index: dict[str, int]
    movie_id_to_title: dict[str, str]


def build_recommendation_prompt(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    movie_features: MovieFeatureStore,
    candidate_movie_ids: list[str] | None = None,
    output_k: int = 10,
) -> str:
    candidate_movie_ids = [clean_value(movie_id) for movie_id in candidate_movie_ids] if candidate_movie_ids is not None else None
    if candidate_movie_ids is None:
        history_text = format_id_title_interaction_history(history, movie_features)
        candidate_text = ""
        suffix = MOVIE_TOKEN_LIST_SUFFIX.format(k=output_k)
        task_text = "Use the profile and the pattern in this movie sequence to recommend the next movies this user is likely to watch."
    else:
        history_text = format_id_title_interaction_history(history, movie_features)
        candidate_text = f"\n\nCandidate movies:\n{format_candidate_movies(candidate_movie_ids, movie_features)}"
        suffix = CANDIDATE_MOVIE_TOKEN_ONLY_SUFFIX
        task_text = "Use the profile and the movie sequence to choose the movie this user is most likely to watch next from the candidate movies."

    prompt_input = f"""User profile:
{format_user_profile(user)}

Here is the user's recent MovieLens trail. Each line names a movie by its catalog token and title:
{history_text}{candidate_text}

{task_text} {suffix}"""
    return f"""{RECOMMENDATION_INSTRUCTION}

### Input:
{prompt_input}

### Response:
"""


def load_inference_components(
    model_name_or_path: str,
    movie_features: MovieFeatureStore,
    load_in_4bit: bool = True,
    attn_implementation: str | None = None,
) -> InferenceComponents:
    tokenizer = load_tokenizer(model_name_or_path, padding_side="left")
    movie_tokens = list(movie_features.movie_tokens)
    tokenizer.add_tokens(movie_tokens, special_tokens=False)
    model = load_causal_lm(
        ModelConfig(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            attn_implementation=attn_implementation,
        ),
        tokenizer=tokenizer,
    )
    model.eval()
    movie_token_ids = [int(token_id) for token_id in tokenizer.convert_tokens_to_ids(movie_tokens)]
    movie_ids = list(movie_features.movie_ids)
    return InferenceComponents(
        tokenizer=tokenizer,
        model=model,
        movie_ids=movie_ids,
        movie_tokens=movie_tokens,
        movie_token_ids=movie_token_ids,
        movie_id_to_index={movie_id: index for index, movie_id in enumerate(movie_ids)},
        movie_id_to_title={movie_id: movie_features.title(movie_id) for movie_id in movie_ids},
    )


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    reversed_mask = attention_mask.flip(dims=[1])
    distance_from_end = reversed_mask.long().argmax(dim=1)
    return attention_mask.shape[1] - 1 - distance_from_end


def restrict_scores_to_candidates(
    components: InferenceComponents,
    movie_scores: torch.Tensor,
    candidate_movie_ids: list[list[str] | None],
) -> tuple[torch.Tensor, list[list[str]], list[list[str]]]:
    token_by_movie_id = dict(zip(components.movie_ids, components.movie_tokens, strict=True))
    candidate_ids_by_row: list[list[str]] = []
    candidate_tokens_by_row: list[list[str]] = []
    candidate_indices: list[list[int]] = []
    for row_candidates in candidate_movie_ids:
        if row_candidates is None:
            ids = components.movie_ids
        else:
            ids = [clean_value(movie_id) for movie_id in row_candidates]
        candidate_ids_by_row.append(ids)
        candidate_tokens_by_row.append([token_by_movie_id[movie_id] for movie_id in ids])
        candidate_indices.append([components.movie_id_to_index[movie_id] for movie_id in ids])

    max_candidates = max(len(indices) for indices in candidate_indices)
    restricted_scores = torch.full((len(candidate_indices), max_candidates), -torch.inf, device=movie_scores.device, dtype=movie_scores.dtype)
    for row, indices in enumerate(candidate_indices):
        index_tensor = torch.tensor(indices, device=movie_scores.device, dtype=torch.long)
        restricted_scores[row, : len(indices)] = movie_scores[row].index_select(dim=0, index=index_tensor)
    return restricted_scores, candidate_ids_by_row, candidate_tokens_by_row


MOVIE_TOKEN_PATTERN = re.compile(r"\bmovie_(\d+)\b")
NUMBERED_MOVIE_LINE_PATTERN = re.compile(r"^\s*\d+\.\s*movie_(\d+)\s*\|\s*(.+?)\s*$")


def canonical_title(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def parse_generated_movie_ids(
    text: str,
    valid_movie_ids: set[str],
    excluded_movie_ids: set[str] | None,
    top_k: int,
    movie_id_to_title: dict[str, str] | None = None,
) -> tuple[list[str], list[bool]]:
    excluded = {clean_value(movie_id) for movie_id in excluded_movie_ids or set()}
    movie_ids: list[str] = []
    title_matches: list[bool] = []
    for line in text.splitlines():
        numbered_match = NUMBERED_MOVIE_LINE_PATTERN.match(line)
        if numbered_match:
            movie_id = clean_value(numbered_match.group(1))
            generated_title = numbered_match.group(2).strip()
            title_matches.append(
                movie_id_to_title is not None
                and movie_id in movie_id_to_title
                and canonical_title(generated_title) == canonical_title(movie_id_to_title[movie_id])
            )
        else:
            token_match = MOVIE_TOKEN_PATTERN.search(line)
            if not token_match:
                continue
            movie_id = clean_value(token_match.group(1))
            title_matches.append(False)
        if movie_id not in valid_movie_ids or movie_id in excluded or movie_id in movie_ids:
            continue
        movie_ids.append(movie_id)
        if len(movie_ids) >= top_k:
            break
    return movie_ids, title_matches


def constrained_generated_text(movie_ids: list[str], movie_id_to_title: dict[str, str]) -> str:
    return "\n".join(f"{index}. movie_{movie_id} | {movie_id_to_title[movie_id]}" for index, movie_id in enumerate(movie_ids, 1))


def fill_with_first_token_ranking(
    components: InferenceComponents,
    prompt: str,
    ranking_config: RankingConfig,
    excluded_movie_ids: set[str] | None,
    existing_movie_ids: list[str],
) -> list[str]:
    filled = list(existing_movie_ids)
    blocked = set(existing_movie_ids) | {clean_value(movie_id) for movie_id in excluded_movie_ids or set()}
    prediction = rank_movie_recommendations_batch(
        components,
        [prompt],
        RankingConfig(top_k=len(components.movie_ids), generation_max_new_tokens=ranking_config.generation_max_new_tokens),
        [blocked],
        None,
    )[0]
    for movie_id in prediction["predicted_movie_ids"]:
        if movie_id not in filled:
            filled.append(movie_id)
        if len(filled) >= ranking_config.top_k:
            break
    return filled


def generate_movie_recommendations_batch(
    components: InferenceComponents,
    prompts: list[str],
    ranking_config: RankingConfig,
    excluded_movie_ids: list[set[str] | None] | None = None,
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    if excluded_movie_ids is not None and len(excluded_movie_ids) != len(prompts):
        raise ValueError("excluded_movie_ids length must match prompts length")

    tokenizer = components.tokenizer
    model = components.model
    top_k = max(1, ranking_config.top_k)
    valid_movie_ids = set(components.movie_ids)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=ranking_config.generation_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    results: list[dict[str, Any]] = []
    for row in range(len(prompts)):
        generated_ids = output_ids[row, prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_movie_ids, title_matches = parse_generated_movie_ids(
            generated_text,
            valid_movie_ids,
            excluded_movie_ids[row] if excluded_movie_ids is not None else None,
            top_k,
            components.movie_id_to_title,
        )
        results.append(
            {
                "predicted_movie_ids": predicted_movie_ids,
                "predicted_movie_tokens": [f"movie_{movie_id}" for movie_id in predicted_movie_ids],
                "scores": [],
                "generated_text": generated_text,
                "raw_title_match_rate": sum(title_matches) / max(1, len(title_matches)),
            }
        )
    return results


def rank_movie_recommendations_batch(
    components: InferenceComponents,
    prompts: list[str],
    ranking_config: RankingConfig,
    excluded_movie_ids: list[set[str] | None] | None = None,
    candidate_movie_ids: list[list[str] | None] | None = None,
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    if excluded_movie_ids is not None and len(excluded_movie_ids) != len(prompts):
        raise ValueError("excluded_movie_ids length must match prompts length")
    if candidate_movie_ids is not None and len(candidate_movie_ids) != len(prompts):
        raise ValueError("candidate_movie_ids length must match prompts length")

    tokenizer = components.tokenizer
    model = components.model
    top_k = max(1, ranking_config.top_k)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    movie_token_ids = torch.tensor(components.movie_token_ids, device=model.device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(**inputs)
        row_indices = torch.arange(inputs["input_ids"].shape[0], device=model.device)
        last_indices = last_non_padding_indices(inputs["attention_mask"])
        next_token_logits = outputs.logits[row_indices, last_indices, :]
        movie_scores = next_token_logits.index_select(dim=-1, index=movie_token_ids)

    movie_ids_by_row = [components.movie_ids] * len(prompts)
    movie_tokens_by_row = [components.movie_tokens] * len(prompts)
    if candidate_movie_ids is not None:
        movie_scores, movie_ids_by_row, movie_tokens_by_row = restrict_scores_to_candidates(components, movie_scores, candidate_movie_ids)

    if excluded_movie_ids is not None:
        movie_scores = movie_scores.clone()
        for row, excluded in enumerate(excluded_movie_ids):
            if not excluded:
                continue
            row_index = {movie_id: index for index, movie_id in enumerate(movie_ids_by_row[row])}
            for movie_id in excluded:
                index = row_index.get(clean_value(movie_id))
                if index is not None:
                    movie_scores[row, index] = -torch.inf

    top_count = min(top_k, movie_scores.shape[1])
    top_scores, top_indices = torch.topk(movie_scores, top_count, dim=1)
    results: list[dict[str, Any]] = []
    for row in range(len(prompts)):
        predicted_movie_ids: list[str] = []
        predicted_movie_tokens: list[str] = []
        scores: list[float] = []
        for score, index in zip(top_scores[row].tolist(), top_indices[row].tolist(), strict=True):
            if score == float("-inf"):
                continue
            predicted_movie_ids.append(movie_ids_by_row[row][index])
            predicted_movie_tokens.append(movie_tokens_by_row[row][index])
            scores.append(float(score))
        results.append(
            {
                "predicted_movie_ids": predicted_movie_ids,
                "predicted_movie_tokens": predicted_movie_tokens,
                "scores": scores,
            }
        )
    return results


def rank_movie_recommendations(
    components: InferenceComponents,
    prompt: str,
    ranking_config: RankingConfig,
    excluded_movie_ids: set[str] | None = None,
    candidate_movie_ids: list[str] | None = None,
) -> dict[str, Any]:
    return rank_movie_recommendations_batch(
        components,
        [prompt],
        ranking_config,
        [excluded_movie_ids],
        [candidate_movie_ids] if candidate_movie_ids is not None else None,
    )[0]


def prediction_record(
    user_id: str,
    split: str,
    target_pos: int,
    target_movie_id: str,
    target_movie_token: str,
    target_movie_title: str,
    prediction: dict[str, Any],
    rank: int | None = None,
) -> dict[str, Any]:
    record = {
        "user_id": user_id,
        "split": split,
        "target_position": target_pos,
        "target_movie_id": target_movie_id,
        "target_movie_token": target_movie_token,
        "target_movie_title": target_movie_title,
        "predicted_movie_ids": prediction["predicted_movie_ids"],
        "predicted_movie_tokens": prediction["predicted_movie_tokens"],
        "scores": prediction["scores"],
    }
    if "generated_text" in prediction:
        record["generated_text"] = prediction["generated_text"]
    if "raw_generated_text" in prediction:
        record["raw_generated_text"] = prediction["raw_generated_text"]
    if "raw_title_match_rate" in prediction:
        record["raw_title_match_rate"] = prediction["raw_title_match_rate"]
    if rank is not None:
        record["rank"] = rank
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MovieRec movie ID token inference.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/inference/predictions.jsonl"))
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    ensure_dir(args.output_path.parent)

    movie_features = load_movie_feature_store(args.raw_dir, {"movie_id", "title"})
    users = load_user_profiles(args.raw_dir)
    ratings_df = load_ratings(args.raw_dir, movie_features)
    components = load_inference_components(
        args.model_name_or_path,
        movie_features,
        args.load_in_4bit,
        args.attn_implementation,
    )
    ranking_config = RankingConfig(top_k=args.top_k)

    grouped = ratings_df.groupby("user_id", sort=False)
    with args.output_path.open("w", encoding="utf-8") as handle:
        pending: list[dict[str, object]] = []

        def flush_pending() -> None:
            if not pending:
                return
            predictions = rank_movie_recommendations_batch(
                components,
                [str(item["prompt"]) for item in pending],
                ranking_config,
                [item["history_movie_ids"] for item in pending],  # type: ignore[list-item]
            )
            for item, prediction in zip(pending, predictions, strict=True):
                handle.write(
                    json.dumps(
                        prediction_record(
                            str(item["user_id"]),
                            str(item["split"]),
                            int(item["target_pos"]),
                            str(item["target_movie_id"]),
                            str(item["target_movie_token"]),
                            str(item["target_movie_title"]),
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
                target_movie_id = clean_value(target["movie_id"])
                prompt = build_recommendation_prompt(users.get(user_id), history, movie_features)
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
                    }
                )
                if len(pending) >= args.batch_size:
                    flush_pending()
        flush_pending()


if __name__ == "__main__":
    main()
