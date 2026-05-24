from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import pandas as pd


_MOVIE_ID_PREFIX = "movie_"
MOVIE_ID_FEATURE_COLUMNS = frozenset({"movie_id"})
_BASIC_TEXT_FEATURE_COLUMNS = frozenset({"movie_id", "title", "genres", "description"})
_INTERACTION_FEATURE_COLUMNS = frozenset({"movie_id", "title", "genres"})
_FULL_FEATURE_COLUMNS = frozenset(
    {
        "movie_id",
        "title",
        "genres",
        "description",
        "isAdult",
        "runtimeMinutes",
        "averageRating",
        "numVotes",
    }
)

_MOVIE_FEATURE_COLUMNS_BY_TASK = {
    "NextMoviePrediction": MOVIE_ID_FEATURE_COLUMNS,
    "Seq_ID2Title": _BASIC_TEXT_FEATURE_COLUMNS,
    "Seq_Title2ID": _INTERACTION_FEATURE_COLUMNS,
    "Single_ID2Title": _BASIC_TEXT_FEATURE_COLUMNS,
    "Single_Title2ID": _BASIC_TEXT_FEATURE_COLUMNS,
    "ID2Feature": _FULL_FEATURE_COLUMNS,
    "Feature2ID": _FULL_FEATURE_COLUMNS,
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


def movie_token(movie_id: Any, token_format: Literal["angle", "plain"] = "angle") -> str:
    token = f"{_MOVIE_ID_PREFIX}{clean_value(movie_id)}"
    return f"<{token}>" if token_format == "angle" else token


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

    def get(self, movie_id: Any) -> dict[str, Any]:
        normalized_id = clean_value(movie_id)
        try:
            return self.features_by_id[normalized_id]
        except KeyError as exc:
            raise KeyError(f"Unknown MovieLens movie_id: {normalized_id}") from exc

    def genres(self, movie_id: Any) -> str:
        return clean_value(self.get(movie_id).get("genres")).replace("|", ", ")

    def basic_text(self, movie_id: Any) -> str:
        movie = self.get(movie_id)
        return (
            f"The movie is {clean_value(movie.get('title'))}. "
            f"It belongs to the {self.genres(movie_id)} genre(s). "
            f"Its description is: {clean_value(movie.get('description'))}"
        )

    def interaction_text(self, movie_id: Any) -> str:
        movie = self.get(movie_id)
        return f"title: {clean_value(movie.get('title'))}; genres: {self.genres(movie_id)}"

    def full_feature(self, movie_id: Any) -> str:
        movie = self.get(movie_id)
        adult = clean_value(movie.get("isAdult"))
        if adult in {"1", "1.0", "True", "true"}:
            adult_text = "adult title"
        elif adult in {"0", "0.0", "False", "false"}:
            adult_text = "non-adult title"
        else:
            adult_text = f"adult flag is {adult}"
        return (
            f"{clean_value(movie.get('title'))} is a {adult_text} with genre(s) {self.genres(movie_id)}. "
            f"The runtime is {clean_value(movie.get('runtimeMinutes'))} minutes, and its external average rating is "
            f"{clean_value(movie.get('averageRating'))} based on {clean_value(movie.get('numVotes'))} votes. "
            f"Description: {clean_value(movie.get('description'))}"
        )


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
    return ratings_df.sort_values(["user_id", "timestamp", "movie_id"])


def load_user_profiles(raw_dir: Path) -> dict[str, dict[str, Any]]:
    users_df = normalize_ids(load_dataframe(raw_dir, "users.pkl"), ["user_id"])
    return {row["user_id"]: row for row in users_df.to_dict("records")}


def load_id_inputs(raw_dir: Path) -> tuple[MovieFeatureStore, dict[str, dict[str, Any]], pd.DataFrame]:
    movie_features = load_movie_feature_store(raw_dir, set(MOVIE_ID_FEATURE_COLUMNS))
    ratings_df = load_ratings(raw_dir, movie_features)
    users = load_user_profiles(raw_dir)
    return movie_features, users, ratings_df


def normalize_ids(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].map(clean_value)
    return df

