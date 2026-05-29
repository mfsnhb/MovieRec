from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.build_sft_dataset import SequenceTarget
from utils.data_io import MovieFeatureStore, clean_value


@dataclass(frozen=True)
class SASRecMappings:
    movie_id_to_index: dict[str, int]
    index_to_movie_id: dict[int, str]
    user_ids: list[str]

    @classmethod
    def from_movie_features(cls, movie_features: MovieFeatureStore, user_ids: Iterable[Any]) -> "SASRecMappings":
        movie_ids = [clean_value(movie_id) for movie_id in movie_features.movie_ids]
        return cls(
            movie_id_to_index={movie_id: index + 1 for index, movie_id in enumerate(movie_ids)},
            index_to_movie_id={index + 1: movie_id for index, movie_id in enumerate(movie_ids)},
            user_ids=[clean_value(user_id) for user_id in user_ids],
        )

    @property
    def num_items(self) -> int:
        return len(self.movie_id_to_index)

    @property
    def movie_ids(self) -> list[str]:
        return [self.index_to_movie_id[index] for index in range(1, self.num_items + 1)]


def grouped_user_records(ratings_df: pd.DataFrame, max_users: int | None = None) -> dict[str, list[dict[str, Any]]]:
    records_by_user: dict[str, list[dict[str, Any]]] = {}
    for user_id, user_ratings in ratings_df.groupby("user_id", sort=False):
        if max_users is not None and len(records_by_user) >= max_users:
            break
        records_by_user[clean_value(user_id)] = user_ratings.to_dict("records")
    return records_by_user


def encode_history(history: Iterable[dict[str, Any]], mappings: SASRecMappings, max_len: int) -> list[int]:
    indices = [mappings.movie_id_to_index[clean_value(event["movie_id"])] for event in history]
    return indices[-max_len:]


def left_padded_sequence(indices: list[int], max_len: int) -> list[int]:
    if not indices:
        raise ValueError("SASRec history must contain at least one item")
    truncated = indices[-max_len:]
    return [0] * (max_len - len(truncated)) + truncated


def sequence_target_to_input(sequence_target: SequenceTarget, mappings: SASRecMappings, max_len: int) -> list[int]:
    return left_padded_sequence(encode_history(sequence_target.history, mappings, max_len), max_len)


def user_interaction_sets(records_by_user: dict[str, list[dict[str, Any]]], mappings: SASRecMappings) -> dict[str, set[int]]:
    return {
        user_id: {mappings.movie_id_to_index[clean_value(record["movie_id"])] for record in records}
        for user_id, records in records_by_user.items()
    }


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        records_by_user: dict[str, list[dict[str, Any]]],
        mappings: SASRecMappings,
        max_len: int,
        seed: int,
        train_offset: int = 3,
    ) -> None:
        self.mappings = mappings
        self.max_len = max_len
        self.rng = random.Random(seed)
        self.all_items = list(range(1, mappings.num_items + 1))
        self.interacted = user_interaction_sets(records_by_user, mappings)
        self.examples: list[tuple[str, list[int], int]] = []
        for user_id, records in records_by_user.items():
            train_records = records[: max(0, len(records) - train_offset + 1)]
            indices = [mappings.movie_id_to_index[clean_value(record["movie_id"])] for record in train_records]
            for target_index in range(1, len(indices)):
                self.examples.append((user_id, indices, target_index))

    def __len__(self) -> int:
        return len(self.examples)

    def sample_negative(self, user_id: str) -> int:
        interacted = self.interacted[user_id]
        while True:
            item = self.rng.choice(self.all_items)
            if item not in interacted:
                return item

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_id, sequence, target_index = self.examples[index]
        input_ids = left_padded_sequence(sequence[:target_index], self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "positive_ids": torch.tensor(sequence[target_index], dtype=torch.long),
            "negative_ids": torch.tensor(self.sample_negative(user_id), dtype=torch.long),
        }


def sasrec_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]}


def sasrec_bce_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, positive_ids: torch.Tensor) -> torch.Tensor:
    valid = positive_ids.ne(0)
    return -(
        torch.nn.functional.logsigmoid(positive_scores[valid]).mean()
        + torch.nn.functional.logsigmoid(-negative_scores[valid]).mean()
    )


@dataclass(frozen=True)
class SASRecEvalExample:
    user_id: str
    split: str
    input_ids: list[int]
    target_item_index: int
    history_item_indices: set[int]


def build_eval_examples(
    records_by_user: dict[str, list[dict[str, Any]]],
    mappings: SASRecMappings,
    max_len: int,
    min_history: int,
) -> list[SASRecEvalExample]:
    examples: list[SASRecEvalExample] = []
    split_offsets = {"valid": 2, "test": 1}
    for user_id, records in records_by_user.items():
        for split, offset in split_offsets.items():
            target_pos = len(records) - offset
            if target_pos < min_history:
                continue
            history = records[:target_pos]
            target = records[target_pos]
            examples.append(
                SASRecEvalExample(
                    user_id=user_id,
                    split=split,
                    input_ids=left_padded_sequence(encode_history(history, mappings, max_len), max_len),
                    target_item_index=mappings.movie_id_to_index[clean_value(target["movie_id"])],
                    history_item_indices={mappings.movie_id_to_index[clean_value(event["movie_id"])] for event in history},
                )
            )
    return examples


def ndcg(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / np.log2(rank + 1)
