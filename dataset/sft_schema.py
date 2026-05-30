from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from utils.data_io import MovieFeatureStore, clean_value, user_token

TASKS = (
    "NextMoviePrediction",
    "ID2Feature",
    "Feature2ID",
)
ALIGNMENT_TASKS = (
    "ID2Feature",
    "Feature2ID",
)
RECOMMENDATION_TASKS = ("NextMoviePrediction",)
ID_OUTPUT_TASKS = {
    "NextMoviePrediction",
    "Feature2ID",
}
FEATURE_OUTPUT_TASKS = {"ID2Feature"}
USER_PROFILE_TASKS = {"NextMoviePrediction"}

INSTRUCTION = (
    "You are a helpful movie recommendation assistant. Use the user token, "
    "MovieLens movie ID tokens, and movie information to make clear predictions."
)
MOVIE_TOKEN_ONLY_SUFFIX = "Answer with exactly one MovieLens movie ID token, such as movie_1, and no other words."

PROMPT_TEMPLATES = {
    "NextMoviePrediction": [
        """User token: {user_token}

Here is the user's recent MovieLens trail in watch order. Each line names a movie by its catalog token and title:
{history}

Use this sequence to predict the next movie this user is likely to watch. {movie_token_only_suffix}""",
        """The recommendation is for {user_token}.

Recent watched movies, in chronological order:
{history}

Infer the next MovieLens movie token that best fits this user's taste. {movie_token_only_suffix}""",
        """Profile token:
{user_token}

The user's movie trail is:
{history}

Based on this trail, choose the next MovieLens movie token for the user. {movie_token_only_suffix}""",
    ],
    "ID2Feature": [
        """The MovieLens movie ID token is {movie_token}.

Write a short, natural MovieLens catalog note for this token. Mention the movie name, genres, and story.""",
        """For {movie_token}, write the kind of movie profile a recommender could use. Keep it to the title, genres, and plot summary.""",
        """Turn {movie_token} into a natural description of the movie: what it is called, what kind of movie it is, and what it is about.""",
        """Write a concise movie profile for {movie_token}, using only the movie name, genres, and story summary.""",
    ],
    "Feature2ID": [
        """Read this MovieLens catalog note:
{movie_feature}

Which MovieLens movie ID token refers to this movie? {movie_token_only_suffix}""",
        """Given this natural movie profile:
{movie_feature}

Identify the matching MovieLens movie ID token. {movie_token_only_suffix}""",
        """This catalog description belongs to one MovieLens movie:
{movie_feature}

Return the MovieLens ID token for that movie. {movie_token_only_suffix}""",
        """Match the following movie profile to its MovieLens ID token:
{movie_feature}

{movie_token_only_suffix}""",
    ],
}


@dataclass(frozen=True)
class RenderedExample:
    instruction: str
    input: str
    output: str


def render_prompt_template(task: str, rng: random.Random | None = None, **kwargs: Any) -> str:
    sampler = rng if rng is not None else random
    return sampler.choice(PROMPT_TEMPLATES[task]).format(**kwargs)


def format_id_interaction_history(history: Iterable[dict[str, Any]], movie_features: MovieFeatureStore) -> str:
    lines = []
    for event in history:
        movie_id = clean_value(event.get("movie_id"))
        lines.append(f"- {movie_features.token(movie_id)}")
    return "\n".join(lines)


def format_id_title_interaction_history(history: Iterable[dict[str, Any]], movie_features: MovieFeatureStore) -> str:
    lines = []
    for event in history:
        movie_id = clean_value(event.get("movie_id"))
        lines.append(f"- {movie_features.token(movie_id)} | {movie_features.title(movie_id)}")
    return "\n".join(lines)


def movie_token_answer(movie_id: Any, movie_features: MovieFeatureStore) -> str:
    return movie_features.token(movie_id)


def build_next_movie_prediction(
    user_id: Any,
    history: list[dict[str, Any]],
    target_movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "NextMoviePrediction",
        rng,
        user_token=user_token(user_id),
        history=format_id_title_interaction_history(history, movie_features),
        movie_token_only_suffix=MOVIE_TOKEN_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, movie_token_answer(target_movie_id, movie_features))


def build_id_to_feature(
    movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "ID2Feature",
        rng,
        movie_token=movie_features.token(movie_id),
    )
    return RenderedExample(INSTRUCTION, input_text, movie_features.full_feature(movie_id))


def build_feature_to_id(
    movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Feature2ID",
        rng,
        movie_feature=movie_features.full_feature(movie_id),
        movie_token_only_suffix=MOVIE_TOKEN_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, movie_token_answer(movie_id, movie_features))
