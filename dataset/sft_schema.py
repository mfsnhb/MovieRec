from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from utils.data_io import MovieFeatureStore, clean_value

TASKS = (
    "NextMoviePrediction",
    "Seq_ID2Feature",
    "Seq_Feature2ID",
    "ID2Feature",
    "Feature2ID",
)
ALIGNMENT_TASKS = (
    "ID2Feature",
    "Feature2ID",
)
RECOMMENDATION_TASKS = (
    "NextMoviePrediction",
    "Seq_ID2Feature",
    "Seq_Feature2ID",
)
ID_OUTPUT_TASKS = {
    "NextMoviePrediction",
    "Seq_Feature2ID",
    "Feature2ID",
}
FEATURE_OUTPUT_TASKS = {"Seq_ID2Feature", "ID2Feature"}
USER_PROFILE_TASKS = {"NextMoviePrediction", "Seq_ID2Feature", "Seq_Feature2ID"}

INSTRUCTION = (
    "You are a helpful movie recommendation assistant. Use the user's profile, "
    "MovieLens movie ID tokens and movie information to make clear predictions."
)
MOVIE_TOKEN_ONLY_SUFFIX = "Answer with exactly one MovieLens movie ID token, such as movie_1, and no other words."
MOVIE_TOKEN_LIST_SUFFIX = "Answer with exactly {k} ranked movies, one per line, in the format: 1. movie_1 | Movie Title. Do not output any other words."
CANDIDATE_MOVIE_TOKEN_ONLY_SUFFIX = "Answer with exactly one candidate movie in the format: movie_1 | Movie Title. Do not output any other words."

PROMPT_TEMPLATES = {
    "NextMoviePrediction": [
        """User profile:
{user_profile}

Here is the user's recent MovieLens trail. Each line names a movie by its catalog token and title:
{history}

Use the profile and the pattern in this movie sequence to recommend the next movies this user is likely to watch. {movie_token_list_suffix}""",
        """The user can be described as follows:
{user_profile}

Below is the user's viewing sequence in watch order. Each movie is represented by its MovieLens ID token and title:
{history}

Infer a ranked list of future movies that best fits this user's taste. {movie_token_list_suffix}""",
        """Profile:
{user_profile}

The user's movie trail is:
{history}

Based on the profile and this trail, recommend the next MovieLens movie tokens for the user. {movie_token_list_suffix}""",
        """Consider this user's profile:
{user_profile}

Their recent movies are shown below in watch order:
{history}

Recommend a ranked list of movies that matches this user's observed preferences. {movie_token_list_suffix}""",
    ],
    "Seq_ID2Feature": [
        """User profile:
{user_profile}

Recent movies this user watched, in order:
{history}

Based on this user's taste, describe the next movie they would probably like. Include the movie title, genres, and a brief story summary.""",
        """The user can be described as follows:
{user_profile}

Their recent viewing history is:
{history}

Write a short catalog-style description of the movie that should come next for this user, including title, genres, and plot.""",
    ],
    "Seq_Feature2ID": [
        """User profile:
{user_profile}

Recent movies this user watched, in order:
{history}

The next movie is described as:
{movie_feature}

Return the MovieLens ID token for this next movie. {movie_token_only_suffix}""",
        """The user can be described as follows:
{user_profile}

Their recent viewing history is:
{history}

Here is the catalog profile of the next movie:
{movie_feature}

Which MovieLens ID token matches this movie? {movie_token_only_suffix}""",
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


def format_user_profile(user: dict[str, Any] | None) -> str:
    if not user:
        return "User profile is unavailable."
    return "\n".join(
        [
            f"- Gender: {clean_value(user.get('gender'))}",
            f"- Age group: {clean_value(user.get('age'))}",
            f"- Occupation: {clean_value(user.get('occupation'))}",
        ]
    )


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


def format_candidate_movies(candidate_movie_ids: Iterable[str], movie_features: MovieFeatureStore) -> str:
    lines = []
    for index, movie_id in enumerate(candidate_movie_ids, 1):
        normalized_id = clean_value(movie_id)
        lines.append(f"{index}. {movie_features.token(normalized_id)} | {movie_features.title(normalized_id)}")
    return "\n".join(lines)


def movie_token_answer(movie_id: Any, movie_features: MovieFeatureStore) -> str:
    return movie_features.token(movie_id)


def movie_token_title_answer(movie_id: Any, movie_features: MovieFeatureStore) -> str:
    normalized_id = clean_value(movie_id)
    return f"{movie_features.token(normalized_id)} | {movie_features.title(normalized_id)}"


def movie_token_list_answer(movie_ids: Iterable[Any], movie_features: MovieFeatureStore) -> str:
    lines = []
    for index, movie_id in enumerate(movie_ids, 1):
        normalized_id = clean_value(movie_id)
        lines.append(f"{index}. {movie_features.token(normalized_id)} | {movie_features.title(normalized_id)}")
    return "\n".join(lines)


def build_next_movie_prediction(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    target_movie_ids: str | list[str],
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    if isinstance(target_movie_ids, str):
        target_movie_ids = [target_movie_ids]
    output_k = len(target_movie_ids)
    input_text = render_prompt_template(
        "NextMoviePrediction",
        rng,
        user_profile=format_user_profile(user),
        history=format_id_title_interaction_history(history, movie_features),
        movie_token_list_suffix=MOVIE_TOKEN_LIST_SUFFIX.format(k=output_k),
    )
    return RenderedExample(INSTRUCTION, input_text, movie_token_list_answer(target_movie_ids, movie_features))


def build_seq_id_to_feature(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    target_movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Seq_ID2Feature",
        rng,
        user_profile=format_user_profile(user),
        history=format_id_interaction_history(history, movie_features),
    )
    return RenderedExample(INSTRUCTION, input_text, movie_features.full_feature(target_movie_id))


def build_seq_feature_to_id(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    target_movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Seq_Feature2ID",
        rng,
        user_profile=format_user_profile(user),
        history=format_id_interaction_history(history, movie_features),
        movie_feature=movie_features.full_feature(target_movie_id),
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
