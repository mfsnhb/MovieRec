from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from dataset.sft_schema import ID_ONLY_SUFFIX, format_id_interaction_history, format_user_profile
from dataset.build_sft_dataset import loo_targets
from model.llm import ModelConfig, collect_movie_tokens, load_causal_lm, load_tokenizer
from utils.data_io import clean_value, load_id_inputs, movie_token
from utils.movie_generation import append_movie_logits_processor, build_movie_token_id_map
from utils.training_utils import ensure_dir, setup_seed


RECOMMENDATION_INSTRUCTION = "You are a helpful movie recommendation assistant. Use the user's profile and interaction history to recommend the next movie."


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 2
    num_return_sequences: int = 10
    num_beams: int = 10
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(frozen=True)
class InferenceComponents:
    tokenizer: Any
    model: Any
    movie_token_ids: list[int]


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
) -> InferenceComponents:
    valid_movie_tokens = [movie_token(movie_id, token_format) for movie_id in valid_movie_ids]
    tokenizer = load_tokenizer(
        model_name_or_path,
        movie_tokens=collect_movie_tokens(raw_dir=raw_dir, token_format=token_format),
        padding_side="left",
    )
    movie_token_id_map = build_movie_token_id_map(tokenizer, valid_movie_tokens)
    model = load_causal_lm(ModelConfig(model_name_or_path, load_in_4bit=load_in_4bit), tokenizer=tokenizer)
    model.eval()
    return InferenceComponents(tokenizer=tokenizer, model=model, movie_token_ids=list(movie_token_id_map.values()))


def generate_ranked_movie_tokens(
    components: InferenceComponents,
    prompt: str,
    generation_config: GenerationConfig,
    token_format: Literal["angle", "plain"] = "angle",
) -> dict[str, Any]:
    tokenizer = components.tokenizer
    model = components.model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits_processor = append_movie_logits_processor(
        None,
        components.movie_token_ids,
        inputs["input_ids"].shape[1],
        [tokenizer.eos_token_id, tokenizer.pad_token_id],
    )
    generation_kwargs = {
        "max_new_tokens": generation_config.max_new_tokens,
        "do_sample": generation_config.do_sample,
        "num_beams": max(generation_config.num_beams, generation_config.num_return_sequences)
        if not generation_config.do_sample
        else generation_config.num_beams,
        "num_return_sequences": generation_config.num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "logits_processor": logits_processor,
    }
    if generation_config.do_sample:
        generation_kwargs["temperature"] = generation_config.temperature
        generation_kwargs["top_p"] = generation_config.top_p
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    generations = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    ranked = []
    for text in generations:
        for token in parse_movie_tokens(text, token_format):
            if token not in ranked:
                ranked.append(token)
    return {"generations": generations, "predicted_movie_ids": ranked}


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
    parser.add_argument("--max-history", type=int, default=None)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
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
    )
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    grouped = ratings_df.groupby("user_id", sort=False)
    with args.output_path.open("w", encoding="utf-8") as handle:
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
                prediction = generate_ranked_movie_tokens(components, prompt, generation_config, args.movie_token_format)
                target_token = movie_token(clean_value(target["movie_id"]), args.movie_token_format)
                handle.write(
                    json.dumps(
                        prediction_record(
                            user_id,
                            split,
                            target_pos,
                            target_token,
                            prediction,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
