from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import string
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset.sft_schema import build_feature_to_id, build_id_to_feature
from model.llm import ModelConfig, load_causal_lm, load_tokenizer
from utils.data_io import MovieFeatureStore, clean_value, load_movie_feature_store, required_movie_feature_columns
from utils.training_utils import ensure_dir, save_json, setup_seed


PROMPT_TEMPLATE = """{instruction}

### Input:
{input}

### Response:
"""

MOVIE_TOKEN_RE = re.compile(r"\bmovie_\d+\b")
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "because",
    "being",
    "between",
    "from",
    "have",
    "into",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "when",
    "where",
    "while",
    "with",
    "will",
    "would",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporary alignment evaluation for MovieRec movie ID token SFT.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_alignment/qwen3_4b_QLoRA_alignment"))
    parser.add_argument("--max-movies", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def format_prompt(rendered) -> str:
    return PROMPT_TEMPLATE.format(instruction=rendered.instruction, input=rendered.input)


def normalize_text(text: str) -> str:
    text = clean_value(text).lower().replace("&", "and")
    text = re.sub(r"\s+", " ", text)
    return text.strip(string.whitespace + "\"'`.,;:!?，。；：！？")


def description_keywords(description: str) -> set[str]:
    words = WORD_RE.findall(normalize_text(description))
    return {word for word in words if len(word) >= 4 and word not in STOPWORDS}


def title_in_output(title: str, output: str) -> bool:
    return normalize_text(title) in normalize_text(output)


def genre_sets(movie_features: MovieFeatureStore, movie_id: str) -> tuple[set[str], set[str]]:
    true_genres = {normalize_text(genre) for genre in movie_features.get(movie_id).get("genres", "").split("|") if clean_value(genre) != "Unknown"}
    all_genres = set()
    for candidate_id in movie_features.movie_ids:
        all_genres.update(
            normalize_text(genre)
            for genre in movie_features.get(candidate_id).get("genres", "").split("|")
            if clean_value(genre) != "Unknown"
        )
    return true_genres, all_genres


def output_genres(output: str, all_genres: set[str]) -> set[str]:
    normalized = normalize_text(output)
    return {genre for genre in all_genres if genre and genre in normalized}


def f1_score(predicted: set[str], truth: set[str]) -> float:
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    overlap = len(predicted & truth)
    precision = overlap / len(predicted)
    recall = overlap / len(truth)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_movie_token(text: str, valid_tokens: set[str]) -> str | None:
    for match in MOVIE_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token in valid_tokens:
            return token
    return None


def batch_generate(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    reversed_mask = attention_mask.flip(dims=[1])
    distance_from_end = reversed_mask.long().argmax(dim=1)
    return attention_mask.shape[1] - 1 - distance_from_end


def feature_to_id_rank_metrics(model, tokenizer, prompts: list[str], target_indices: list[int], movie_token_ids: list[int]) -> dict[str, float]:
    if not prompts:
        return {"acc@1": 0.0, "hr@10": 0.0, "mrr": 0.0, "mean_rank": 0.0, "median_rank": 0.0}
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    movie_token_tensor = torch.tensor(movie_token_ids, device=model.device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(**inputs)
        row_indices = torch.arange(inputs["input_ids"].shape[0], device=model.device)
        last_indices = last_non_padding_indices(inputs["attention_mask"])
        logits = outputs.logits[row_indices, last_indices, :].index_select(dim=-1, index=movie_token_tensor)
    ranks = []
    for row, target_index in enumerate(target_indices):
        order = torch.argsort(logits[row], descending=True)
        rank = int((order == target_index).nonzero(as_tuple=False).item()) + 1
        ranks.append(rank)
    return {
        "acc@1": sum(rank == 1 for rank in ranks) / len(ranks),
        "hr@10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": float(statistics.median(ranks)),
    }


def evaluate_feature_to_id(model, tokenizer, movie_features: MovieFeatureStore, movie_ids: list[str], output_dir: Path, batch_size: int) -> dict[str, Any]:
    rng = random.Random(1234)
    movie_tokens = list(movie_features.movie_tokens)
    valid_tokens = set(movie_tokens)
    token_to_index = {token: index for index, token in enumerate(movie_tokens)}
    movie_token_ids = [int(token_id) for token_id in tokenizer.convert_tokens_to_ids(movie_tokens)]
    records = []
    rank_totals = {"acc@1": 0.0, "hr@10": 0.0, "mrr": 0.0, "mean_rank": 0.0, "median_rank": 0.0}
    generated_exact = 0
    valid_generation = 0
    total = 0
    path = output_dir / "feature_to_id_predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for start in tqdm(range(0, len(movie_ids), batch_size), desc="Feature2ID eval"):
            batch_ids = movie_ids[start : start + batch_size]
            rendered = [build_feature_to_id(movie_id, movie_features, rng) for movie_id in batch_ids]
            prompts = [format_prompt(example) for example in rendered]
            targets = [movie_features.token(movie_id) for movie_id in batch_ids]
            target_indices = [token_to_index[target] for target in targets]
            batch_rank = feature_to_id_rank_metrics(model, tokenizer, prompts, target_indices, movie_token_ids)
            for key, value in batch_rank.items():
                rank_totals[key] += value * len(batch_ids)

            outputs = batch_generate(model, tokenizer, prompts, max_new_tokens=8)
            for movie_id, prompt, output, target in zip(batch_ids, prompts, outputs, targets, strict=True):
                predicted = extract_movie_token(output, valid_tokens)
                valid_generation += int(predicted is not None)
                generated_exact += int(predicted == target)
                total += 1
                record = {
                    "movie_id": movie_id,
                    "target_movie_token": target,
                    "predicted_movie_token": predicted,
                    "raw_output": output,
                    "prompt": prompt,
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "num_examples": total,
        "generated_exact_acc": generated_exact / max(1, total),
        "generated_valid_rate": valid_generation / max(1, total),
        **{key: value / max(1, total) for key, value in rank_totals.items()},
    }


def evaluate_id_to_feature(
    model,
    tokenizer,
    movie_features: MovieFeatureStore,
    movie_ids: list[str],
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    rng = random.Random(5678)
    all_genres = set()
    for movie_id in movie_features.movie_ids:
        all_genres.update(
            normalize_text(genre)
            for genre in movie_features.get(movie_id).get("genres", "").split("|")
            if clean_value(genre) != "Unknown"
        )

    total = 0
    title_hits = 0
    genre_f1_total = 0.0
    description_recall_total = 0.0
    path = output_dir / "id_to_feature_predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for start in tqdm(range(0, len(movie_ids), batch_size), desc="ID2Feature eval"):
            batch_ids = movie_ids[start : start + batch_size]
            rendered = [build_id_to_feature(movie_id, movie_features, rng) for movie_id in batch_ids]
            prompts = [format_prompt(example) for example in rendered]
            outputs = batch_generate(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            for movie_id, prompt, output in zip(batch_ids, prompts, outputs, strict=True):
                movie = movie_features.get(movie_id)
                title = movie_features.title(movie_id)
                true_genres = {
                    normalize_text(genre)
                    for genre in clean_value(movie.get("genres")).split("|")
                    if clean_value(genre) != "Unknown"
                }
                predicted_genres = output_genres(output, all_genres)
                keywords = description_keywords(clean_value(movie.get("description")))
                output_words = set(WORD_RE.findall(normalize_text(output)))
                description_recall = len(keywords & output_words) / max(1, len(keywords))

                title_hit = title_in_output(title, output)
                genre_f1 = f1_score(predicted_genres, true_genres)

                total += 1
                title_hits += int(title_hit)
                genre_f1_total += genre_f1
                description_recall_total += description_recall

                handle.write(
                    json.dumps(
                        {
                            "movie_id": movie_id,
                            "movie_token": movie_features.token(movie_id),
                            "title": title,
                            "title_contains": title_hit,
                            "genre_f1": genre_f1,
                            "description_keyword_recall": description_recall,
                            "raw_output": output,
                            "prompt": prompt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return {
        "num_examples": total,
        "title_contains_acc": title_hits / max(1, total),
        "genre_f1": genre_f1_total / max(1, total),
        "description_keyword_recall": description_recall_total / max(1, total),
    }


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    movie_features = load_movie_feature_store(args.raw_dir, required_movie_feature_columns({"ID2Feature", "Feature2ID"}))
    movie_ids = list(movie_features.movie_ids)
    if args.max_movies is not None:
        movie_ids = movie_ids[: args.max_movies]

    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="left")
    tokenizer.add_tokens(list(movie_features.movie_tokens), special_tokens=False)
    model = load_causal_lm(
        ModelConfig(
            args.model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
        ),
        tokenizer=tokenizer,
    )
    model.eval()

    id_to_feature = evaluate_id_to_feature(
        model,
        tokenizer,
        movie_features,
        movie_ids,
        output_dir,
        args.batch_size,
        args.max_new_tokens,
    )
    feature_to_id = evaluate_feature_to_id(model, tokenizer, movie_features, movie_ids, output_dir, args.batch_size)
    metrics = {
        "model_name_or_path": args.model_name_or_path,
        "num_movies_evaluated": len(movie_ids),
        "id_to_feature": id_to_feature,
        "feature_to_id": feature_to_id,
    }
    save_json(output_dir / "alignment_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
