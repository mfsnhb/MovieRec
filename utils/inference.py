from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dataset.build_sft_dataset import loo_targets
from dataset.sft_schema import (
    INSTRUCTION,
    TITLE_ONLY_SUFFIX,
    format_title_interaction_history,
    format_user_profile,
)
from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.data_io import (
    clean_value,
    load_movie_feature_store,
    load_ratings,
    load_user_profiles,
    required_movie_feature_columns,
)
from utils.training_utils import ensure_dir, setup_seed


RECOMMENDATION_INSTRUCTION = INSTRUCTION
_TITLE_STRIP_CHARS = string.whitespace + "\"'`.,;:!?，。；：！？"


@dataclass(frozen=True)
class GenerationConfig:
    top_k: int = 10
    max_new_tokens: int = 48


@dataclass(frozen=True)
class InferenceComponents:
    tokenizer: Any
    model: Any
    valid_title_by_normalized: dict[str, str]


def normalize_title_text(text: str) -> str:
    text = clean_value(text).lower()
    text = text.replace("&", "and")
    text = re.sub(r"\s+", " ", text)
    return text.strip(_TITLE_STRIP_CHARS)


def build_title_lookup(valid_titles: list[str]) -> dict[str, str]:
    return {normalize_title_text(title): title for title in valid_titles}


def build_recommendation_prompt(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    movie_features,
) -> str:
    prompt_input = f"""User profile:
{format_user_profile(user)}

The user's MovieLens history is listed below as movie title and rating:
{format_title_interaction_history(history, movie_features)}

Based on this profile and rating history, recommend the movie the user is most likely to watch next. {TITLE_ONLY_SUFFIX}"""
    return f"""{RECOMMENDATION_INSTRUCTION}

### Input:
{prompt_input}

### Response:
"""


def load_inference_components(
    model_name_or_path: str,
    valid_titles: list[str],
    load_in_4bit: bool = True,
    attn_implementation: str | None = None,
) -> InferenceComponents:
    tokenizer = load_tokenizer(model_name_or_path, padding_side="left")
    model = load_causal_lm(
        ModelConfig(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            attn_implementation=attn_implementation,
        ),
        tokenizer=tokenizer,
    )
    model.eval()
    valid_title_by_normalized = build_title_lookup(valid_titles)
    return InferenceComponents(
        tokenizer=tokenizer,
        model=model,
        valid_title_by_normalized=valid_title_by_normalized,
    )


def _split_candidate_fragments(text: str) -> list[str]:
    fragments = [text]
    fragments.extend(text.splitlines())
    fragments.extend(re.split(r"\s*(?:\d+[\).\s]+|[-*•]\s+|,|;)\s*", text))
    return [fragment.strip(_TITLE_STRIP_CHARS) for fragment in fragments if fragment.strip(_TITLE_STRIP_CHARS)]


def extract_valid_titles(
    text: str,
    valid_title_by_normalized: dict[str, str],
    excluded_titles: set[str] | None = None,
) -> list[str]:
    excluded = {normalize_title_text(title) for title in excluded_titles or set()}
    seen: set[str] = set()
    matches: list[str] = []

    def add(normalized: str) -> None:
        title = valid_title_by_normalized.get(normalized)
        if title is None or normalized in excluded or title in seen:
            return
        seen.add(title)
        matches.append(title)

    for fragment in _split_candidate_fragments(text):
        add(normalize_title_text(fragment))
    return matches


def generate_title_recommendations_batch(
    components: InferenceComponents,
    prompts: list[str],
    generation_config: GenerationConfig,
    excluded_titles: list[set[str] | None] | None = None,
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    tokenizer = components.tokenizer
    model = components.model
    top_k = max(1, generation_config.top_k)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=False,
            num_beams=top_k,
            num_return_sequences=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    results: list[dict[str, Any]] = []
    for row in range(len(prompts)):
        raw_generations = decoded[row * top_k : (row + 1) * top_k]
        row_excluded = excluded_titles[row] if excluded_titles else None
        predicted_titles: list[str] = []
        for text in raw_generations:
            for title in extract_valid_titles(
                text,
                components.valid_title_by_normalized,
                row_excluded,
            ):
                if title not in predicted_titles:
                    predicted_titles.append(title)
            if len(predicted_titles) >= top_k:
                break
        results.append(
            {
                "generations": raw_generations,
                "predicted_movie_titles": predicted_titles[:top_k],
            }
        )
    return results


def generate_title_recommendations(
    components: InferenceComponents,
    prompt: str,
    generation_config: GenerationConfig,
    excluded_titles: set[str] | None = None,
) -> dict[str, Any]:
    return generate_title_recommendations_batch(components, [prompt], generation_config, [excluded_titles])[0]


def prediction_record(
    user_id: str,
    split: str,
    target_pos: int,
    target_movie_title: str,
    prediction: dict[str, Any],
    rank: int | None = None,
) -> dict[str, Any]:
    record = {
        "user_id": user_id,
        "split": split,
        "target_position": target_pos,
        "target_movie_title": target_movie_title,
        "predicted_movie_titles": prediction["predicted_movie_titles"],
        "generations": prediction["generations"],
    }
    if rank is not None:
        record["rank"] = rank
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MovieRec title-based inference.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/inference/predictions.jsonl"))
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    ensure_dir(args.output_path.parent)

    movie_features = load_movie_feature_store(args.raw_dir, required_movie_feature_columns({"NextMovieTitlePrediction"}))
    users = load_user_profiles(args.raw_dir)
    ratings_df = load_ratings(args.raw_dir, movie_features)
    valid_titles = [movie_features.title(movie_id) for movie_id in movie_features.movie_ids]
    components = load_inference_components(
        args.model_name_or_path,
        valid_titles,
        args.load_in_4bit,
        args.attn_implementation,
    )
    generation_config = GenerationConfig(top_k=args.top_k, max_new_tokens=args.max_new_tokens)

    grouped = ratings_df.groupby("user_id", sort=False)
    with args.output_path.open("w", encoding="utf-8") as handle:
        pending: list[dict[str, object]] = []

        def flush_pending() -> None:
            if not pending:
                return
            predictions = generate_title_recommendations_batch(
                components,
                [str(item["prompt"]) for item in pending],
                generation_config,
                [item["history_titles"] for item in pending],  # type: ignore[list-item]
            )
            for item, prediction in zip(pending, predictions, strict=True):
                handle.write(
                    json.dumps(
                        prediction_record(
                            str(item["user_id"]),
                            str(item["split"]),
                            int(item["target_pos"]),
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
                history_titles = {movie_features.title(event["movie_id"]) for event in history}
                pending.append(
                    {
                        "user_id": user_id,
                        "split": split,
                        "target_pos": target_pos,
                        "target_movie_title": movie_features.title(target_movie_id),
                        "prompt": prompt,
                        "history_titles": history_titles,
                    }
                )
                if len(pending) >= args.batch_size:
                    flush_pending()
        flush_pending()


if __name__ == "__main__":
    main()
