from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


_MOVIE_ID_PREFIX = "movie_"
_USER_ID_PREFIX = "user_"
MOVIE_ID_FEATURE_COLUMNS = frozenset({"movie_id", "title"})
_TEXT_FEATURE_COLUMNS = frozenset({"movie_id", "title", "genres", "description"})

_MOVIE_FEATURE_COLUMNS_BY_TASK = {
    "NextMoviePrediction": MOVIE_ID_FEATURE_COLUMNS,
    "ID2Feature": _TEXT_FEATURE_COLUMNS,
    "Feature2ID": _TEXT_FEATURE_COLUMNS,
}


_DEFAULT_REQUIRED_COLUMNS = {
    "movies.pkl": set(MOVIE_ID_FEATURE_COLUMNS),
    "ratings.pkl": {"user_id", "movie_id", "rating", "timestamp"},
    "users.pkl": {"user_id", "gender", "age", "occupation"},
}


def clean_value(value: Any) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text or text == "\\N" or text.lower() == "nan":
        return "Unknown"
    return text.replace("&apos;", "'").replace("‘", "'").replace("’", "'")


def movie_id_sort_key(movie_id: Any) -> tuple[int, int | str]:
    text = clean_value(movie_id)
    return (0, int(text)) if text.isdigit() else (1, text)


def movie_token(movie_id: Any) -> str:
    return f"{_MOVIE_ID_PREFIX}{clean_value(movie_id)}"


def user_token(user_id: Any) -> str:
    return f"{_USER_ID_PREFIX}{clean_value(user_id)}"


def required_movie_feature_columns(tasks: Iterable[str]) -> set[str]:
    columns: set[str] = set()
    for task in tasks:
        columns.update(_MOVIE_FEATURE_COLUMNS_BY_TASK[task])
    return columns


@dataclass(frozen=True)
class MovieFeatureStore:
    features_by_id: dict[str, dict[str, Any]]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "MovieFeatureStore":
        features_by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            movie_id = clean_value(record.get("movie_id"))
            if movie_id == "Unknown":
                raise ValueError("Movie feature record is missing movie_id")
            features_by_id[movie_id] = {**dict(record), "movie_id": movie_id}
        return cls(features_by_id)

    def __len__(self) -> int:
        return len(self.features_by_id)

    def __contains__(self, movie_id: object) -> bool:
        return clean_value(movie_id) in self.features_by_id

    @property
    def movie_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.features_by_id, key=movie_id_sort_key))

    @property
    def movie_id_set(self) -> set[str]:
        return set(self.features_by_id)

    @property
    def movie_tokens(self) -> tuple[str, ...]:
        return tuple(movie_token(movie_id) for movie_id in self.movie_ids)

    @property
    def movie_token_to_id(self) -> dict[str, str]:
        return {movie_token(movie_id): movie_id for movie_id in self.movie_ids}

    def token(self, movie_id: Any) -> str:
        normalized_id = clean_value(movie_id)
        if normalized_id not in self.features_by_id:
            raise KeyError(f"Unknown MovieLens movie_id: {normalized_id}")
        return movie_token(normalized_id)

    def get(self, movie_id: Any) -> dict[str, Any]:
        normalized_id = clean_value(movie_id)
        try:
            return self.features_by_id[normalized_id]
        except KeyError as exc:
            raise KeyError(f"Unknown MovieLens movie_id: {normalized_id}") from exc

    def genres(self, movie_id: Any) -> str:
        return clean_value(self.get(movie_id).get("genres")).replace("|", ", ")

    def title(self, movie_id: Any) -> str:
        return clean_value(self.get(movie_id).get("title"))

    def full_feature(self, movie_id: Any) -> str:
        movie = self.get(movie_id)
        title = clean_value(movie.get("title"))
        genres = self.genres(movie_id)
        description = clean_value(movie.get("description")).rstrip(".")
        if description == "Unknown":
            story_sentence = "No story summary is available."
        else:
            story_sentence = f"Its story follows this premise: {description}."

        return " ".join(
            [
                f"{title} is listed under the {genres} genres.",
                story_sentence,
            ]
        )

    def token_records(self) -> list[dict[str, str]]:
        return [
            {
                "movie_id": movie_id,
                "movie_token": movie_token(movie_id),
                "title": self.title(movie_id),
            }
            for movie_id in self.movie_ids
        ]


def load_dataframe(raw_dir: Path, name: str, required_columns: set[str] | None = None) -> pd.DataFrame:
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")
    df = pd.read_pickle(path)
    columns = required_columns if required_columns is not None else _DEFAULT_REQUIRED_COLUMNS[name]
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    return df.copy()


def load_movie_feature_store(raw_dir: Path, required_columns: set[str] | None = None) -> MovieFeatureStore:
    movies_df = normalize_ids(load_dataframe(raw_dir, "movies.pkl", required_columns), ["movie_id"])
    return MovieFeatureStore.from_records(movies_df.to_dict("records"))


def load_ratings(raw_dir: Path, movie_features: MovieFeatureStore | None = None) -> pd.DataFrame:
    ratings_df = normalize_ids(load_dataframe(raw_dir, "ratings.pkl"), ["user_id", "movie_id"])
    if movie_features is not None:
        ratings_df = ratings_df[ratings_df["movie_id"].isin(movie_features.movie_id_set)]
    ratings_df = ratings_df[["user_id", "movie_id", "rating", "timestamp"]].copy()
    return ratings_df.sort_values(["user_id", "timestamp"], kind="mergesort")


def load_user_profiles(raw_dir: Path) -> dict[str, dict[str, Any]]:
    users_df = normalize_ids(load_dataframe(raw_dir, "users.pkl"), ["user_id"])
    return {row["user_id"]: row for row in users_df.to_dict("records")}


def normalize_ids(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].map(clean_value)
    return df
