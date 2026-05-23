from __future__ import annotations

from types import MethodType
from typing import Iterable

import torch
from transformers import LogitsProcessor, LogitsProcessorList


def build_movie_token_id_map(tokenizer, movie_tokens: Iterable[str]) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    skipped: list[str] = []
    for token in movie_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            token_ids[token] = ids[0]
        else:
            skipped.append(token)
    if not token_ids:
        raise ValueError("No movie tokens are single-token vocabulary entries. Call load_tokenizer(..., movie_tokens=...) first.")
    if skipped:
        print(f"Skipped {len(skipped)} movie tokens that are not single-token entries.")
    return token_ids


class MovieTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, movie_token_ids: Iterable[int], prompt_length: int, stop_token_ids: Iterable[int | None]):
        self.movie_token_ids = sorted(set(int(token_id) for token_id in movie_token_ids))
        self.prompt_length = prompt_length
        self.stop_token_ids = sorted({int(token_id) for token_id in stop_token_ids if token_id is not None})
        if not self.movie_token_ids:
            raise ValueError("movie_token_ids must not be empty.")
        if not self.stop_token_ids:
            raise ValueError("At least one stop token id is required for constrained movie generation.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_length = input_ids.shape[1] - self.prompt_length
        allowed_ids = self.movie_token_ids if generated_length == 0 else self.stop_token_ids
        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed_ids] = scores[:, allowed_ids]
        return mask


def append_movie_logits_processor(
    logits_processor,
    movie_token_ids: Iterable[int],
    prompt_length: int,
    stop_token_ids: Iterable[int | None],
) -> LogitsProcessorList:
    processors = LogitsProcessorList(logits_processor or [])
    processors.append(MovieTokenLogitsProcessor(movie_token_ids, prompt_length, stop_token_ids))
    return processors


def patch_generate_with_movie_constraints(model, movie_token_ids: Iterable[int], stop_token_ids: Iterable[int | None]) -> None:
    movie_token_ids = list(movie_token_ids)
    stop_token_ids = list(stop_token_ids)
    original_generate = model.generate

    def constrained_generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            kwargs["logits_processor"] = append_movie_logits_processor(
                kwargs.get("logits_processor"),
                movie_token_ids,
                input_ids.shape[1],
                stop_token_ids,
            )
        return original_generate(*args, **kwargs)

    model.generate = MethodType(constrained_generate, model)
