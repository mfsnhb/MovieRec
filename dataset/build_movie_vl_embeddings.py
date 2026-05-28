from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from utils.data_io import clean_value, load_movie_feature_store, movie_id_sort_key
from utils.training_utils import ensure_dir, save_json


DEFAULT_API_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_INSTRUCTION = "Represent this movie for recommendation, item identity alignment, and semantic retrieval."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2560-d multimodal movie embeddings from poster + text metadata.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/movie_vl_embeddings"))
    parser.add_argument("--api-base", default=os.getenv("EMBEDDING_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.getenv("SILICONFLOW_API_KEY") or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--dimensions", type=int, default=2560)
    parser.add_argument("--input-schema", choices=["openai_content", "qwen_dict", "text_only"], default="openai_content")
    parser.add_argument("--encoding-format", choices=["float", "base64"], default="float")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-movies", type=int)
    parser.add_argument("--movie-id")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def text_feature(movie: dict[str, Any]) -> str:
    title = clean_value(movie.get("title"))
    genres = clean_value(movie.get("genres")).replace("|", ", ")
    description = clean_value(movie.get("description"))
    return "\n".join(
        [
            f"Movie title: {title}",
            f"Genres: {genres}",
            f"Description: {description}",
        ]
    )


def image_data_uri(path: Path, max_side: int, jpeg_quality: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_side > 0:
            image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_input(args: argparse.Namespace, text: str, poster_uri: str | None) -> Any:
    text_with_instruction = f"Instruction: {args.instruction}\n\n{text}"
    if args.input_schema == "text_only" or poster_uri is None:
        return text_with_instruction
    if args.input_schema == "qwen_dict":
        payload = {"text": text, "instruction": args.instruction}
        if poster_uri is not None:
            payload["image"] = poster_uri
        return payload
    return [
        {"type": "text", "text": text_with_instruction},
        {"type": "image", "image": poster_uri},
    ]


def decode_embedding(value: Any, encoding_format: str) -> list[float]:
    if encoding_format == "float":
        return value
    raw = base64.b64decode(value)
    return np.frombuffer(raw, dtype=np.float32).astype(np.float32).tolist()


def request_embedding(args: argparse.Namespace, content: Any) -> tuple[list[float], dict[str, Any]]:
    if not args.api_key:
        raise ValueError("Missing API key. Set SILICONFLOW_API_KEY, EMBEDDING_API_KEY, OPENAI_API_KEY, or pass --api-key.")

    url = args.api_base.rstrip("/") + "/embeddings"
    body = {
        "model": args.model,
        "input": content,
        "dimensions": args.dimensions,
        "encoding_format": args.encoding_format,
    }
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=args.timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)
                continue
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
            embedding = decode_embedding(data["data"][0]["embedding"], args.encoding_format)
            return embedding, data.get("usage", {})
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(args.retry_sleep * attempt)
    raise RuntimeError(f"Embedding request failed after {args.max_retries} attempts") from last_error


def completed_indices(done_path: Path, movie_id_to_index: dict[str, int]) -> set[int]:
    if not done_path.exists():
        return set()
    done: set[int] = set()
    with done_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            movie_id = clean_value(record.get("movie_id"))
            if movie_id in movie_id_to_index:
                done.add(movie_id_to_index[movie_id])
    return done


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    movie_features = load_movie_feature_store(args.raw_dir, {"movie_id", "title", "genres", "description"})
    movie_ids = list(sorted(movie_features.movie_ids, key=movie_id_sort_key))
    if args.movie_id is not None:
        movie_ids = [clean_value(args.movie_id)]
    if args.max_movies is not None:
        movie_ids = movie_ids[: args.max_movies]

    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
    embeddings_path = out_dir / f"movie_vl_embeddings_{args.dimensions}.npy"
    done_path = out_dir / "done.jsonl"
    errors_path = out_dir / "errors.jsonl"
    records_path = out_dir / "records.jsonl"
    final_npz_path = out_dir / f"movie_vl_embeddings_{args.dimensions}.npz"

    if args.dry_run:
        movie_id = movie_ids[0]
        movie = movie_features.get(movie_id)
        poster_path = args.raw_dir / "image" / f"{movie_id}.png"
        poster_uri = image_data_uri(poster_path, args.max_image_side, args.jpeg_quality) if poster_path.exists() else None
        print(json.dumps({"movie_id": movie_id, "input": build_input(args, text_feature(movie), poster_uri)}, ensure_ascii=False, indent=2)[:4000])
        return

    embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+" if embeddings_path.exists() and args.resume else "w+", dtype=np.float32, shape=(len(movie_ids), args.dimensions))
    done = completed_indices(done_path, movie_id_to_index) if args.resume else set()

    save_json(
        out_dir / "manifest.json",
        {
            "raw_dir": str(args.raw_dir),
            "model": args.model,
            "api_base": args.api_base,
            "dimensions": args.dimensions,
            "input_schema": args.input_schema,
            "instruction": args.instruction,
            "num_movies": len(movie_ids),
            "embeddings_path": str(embeddings_path),
            "final_npz_path": str(final_npz_path),
        },
    )

    with done_path.open("a", encoding="utf-8") as done_handle, errors_path.open("a", encoding="utf-8") as error_handle, records_path.open("a", encoding="utf-8") as record_handle:
        for index, movie_id in enumerate(tqdm(movie_ids, desc="movie VL embeddings")):
            if index in done:
                continue
            movie = movie_features.get(movie_id)
            poster_path = args.raw_dir / "image" / f"{movie_id}.png"
            poster_uri = image_data_uri(poster_path, args.max_image_side, args.jpeg_quality) if poster_path.exists() else None
            content = build_input(args, text_feature(movie), poster_uri)
            try:
                embedding, usage = request_embedding(args, content)
                if len(embedding) != args.dimensions:
                    raise ValueError(f"Expected embedding dim {args.dimensions}, got {len(embedding)}")
                vector = np.asarray(embedding, dtype=np.float32)
                if args.l2_normalize:
                    norm = float(np.linalg.norm(vector))
                    if norm > 0:
                        vector = vector / norm
                embeddings[index] = vector
                embeddings.flush()
                record = {
                    "movie_id": movie_id,
                    "movie_token": f"movie_{movie_id}",
                    "title": movie_features.title(movie_id),
                    "poster_path": str(poster_path) if poster_path.exists() else None,
                    "usage": usage,
                }
                done_handle.write(json.dumps({"movie_id": movie_id, "index": index}, ensure_ascii=False) + "\n")
                done_handle.flush()
                record_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_handle.flush()
                if args.request_sleep > 0:
                    time.sleep(args.request_sleep)
            except Exception as exc:
                error_handle.write(json.dumps({"movie_id": movie_id, "index": index, "error": str(exc)}, ensure_ascii=False) + "\n")
                error_handle.flush()
                raise

    np.savez_compressed(
        final_npz_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        movie_ids=np.asarray(movie_ids),
        movie_tokens=np.asarray([f"movie_{movie_id}" for movie_id in movie_ids]),
        titles=np.asarray([movie_features.title(movie_id) for movie_id in movie_ids]),
    )
    print(f"Saved {len(movie_ids)} embeddings to {final_npz_path}")


if __name__ == "__main__":
    main()
