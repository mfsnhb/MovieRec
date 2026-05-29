from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.data_io import clean_value


class RerankerScores:
    def __init__(self, path: Path | str) -> None:
        data = np.load(path, allow_pickle=False)
        self.scores = data["scores"].astype(np.float32, copy=False)
        self.user_ids = [str(value) for value in data["user_ids"].tolist()]
        self.movie_ids = [str(value) for value in data["movie_ids"].tolist()]
        self.user_to_row = {user_id: index for index, user_id in enumerate(self.user_ids)}
        self.movie_to_col = {movie_id: index for index, movie_id in enumerate(self.movie_ids)}
        order = np.argsort(np.argsort(self.scores, axis=1), axis=1).astype(np.float32)
        self.percentiles = order / max(1, self.scores.shape[1] - 1)

    def percentile(self, user_id: Any, movie_id: Any) -> float:
        row = self.user_to_row.get(str(user_id))
        col = self.movie_to_col.get(str(movie_id))
        if row is None or col is None:
            return 0.0
        return float(self.percentiles[row, col])

    def top_movie_ids(self, user_id: Any, excluded_movie_ids: Iterable[Any], top_k: int) -> list[str]:
        row = self.user_to_row.get(str(user_id))
        if row is None:
            raise KeyError(f"User {user_id} is not available in reranker scores.")
        scores = self.scores[row].copy()
        for movie_id in excluded_movie_ids:
            col = self.movie_to_col.get(clean_value(movie_id))
            if col is not None:
                scores[col] = -np.inf
        finite_cols = np.flatnonzero(np.isfinite(scores))
        if finite_cols.size == 0:
            return []
        top_count = min(max(1, top_k), finite_cols.size)
        candidate_positions = np.argpartition(-scores[finite_cols], top_count - 1)[:top_count]
        candidate_cols = finite_cols[candidate_positions]
        ranked_cols = candidate_cols[np.argsort(-scores[candidate_cols])]
        return [self.movie_ids[int(col)] for col in ranked_cols]

    def top_movie_ids_with_scores(self, user_id: Any, excluded_movie_ids: Iterable[Any], top_k: int) -> tuple[list[str], list[float]]:
        movie_ids = self.top_movie_ids(user_id, excluded_movie_ids, top_k)
        row = self.user_to_row[str(user_id)]
        scores = [float(self.scores[row, self.movie_to_col[movie_id]]) for movie_id in movie_ids]
        return movie_ids, scores
