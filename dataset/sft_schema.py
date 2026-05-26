from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from utils.data_io import MovieFeatureStore, clean_value

TASKS = (
    "NextMovieTitlePrediction",
    "Seq_Title2Feature",
    "Seq_Rating",
    "Single_Title2Feature",
    "Single_Feature2Title",
)
TITLE_OUTPUT_TASKS = {
    "NextMovieTitlePrediction",
    "Single_Feature2Title",
}
FEATURE_OUTPUT_TASKS = {
    "Seq_Title2Feature",
    "Single_Title2Feature",
}
RATING_OUTPUT_TASKS = {
    "Seq_Rating",
}
USER_PROFILE_TASKS = {
    "NextMovieTitlePrediction",
    "Seq_Title2Feature",
    "Seq_Rating",
}

INSTRUCTION = (
    "You are a helpful movie recommendation assistant. Use the user's viewing history "
    "and movie information to make clear, concise recommendations."
)
TITLE_ONLY_SUFFIX = "Answer with exactly one movie title from MovieLens, including the year in parentheses, and no other words."
RATING_ONLY_SUFFIX = "Answer with exactly one rating number on the same scale as the user's ratings, and no other words."

PROMPT_TEMPLATES = {
    "NextMovieTitlePrediction": [
        """User profile:
{user_profile}

Here is the user's movie-watching history. Each line gives the movie title and the rating the user gave it:
{history}

Based on this profile and rating history, recommend the movie the user is most likely to watch next. {title_only_suffix}""",
        """The user can be described as follows:
{user_profile}

The user's interactions are listed below as movie titles with the ratings they gave:
{history}

Use this profile and sequence to predict the next movie title. {title_only_suffix}""",
        """Profile:
{user_profile}

The following records summarize the user's rated movies:
{history}

What movie title should come next for this user? {title_only_suffix}""",
        """Consider this user's profile:
{user_profile}

Their movie sequence is below. Each line says which title the user rated and how many stars they gave it:
{history}

Please infer the next likely MovieLens title from the profile and ratings. {title_only_suffix}""",
    ],
    "Seq_Title2Feature": [
        """User profile:
{user_profile}

The user has this MovieLens interaction history, represented by movie titles and ratings:
{history}

Describe the kind of movie this user is likely to watch next. Include the title, genres, and description naturally in your answer.""",
        """The user can be described as follows:
{user_profile}

Here is the user's sequence of rated movie titles:
{history}

Write a natural description of the next movie, including its title, genres, and description.""",
        """Profile:
{user_profile}

The user's MovieLens title history is shown below:
{history}

What movie does this profile and sequence of ratings point to next? Answer in prose and include the title, genres, and description.""",
        """A user with this profile:
{user_profile}

has the following ordered list of movie titles and ratings:
{history}

Summarize the likely next movie in natural language, making sure the title, genres, and description are present.""",
    ],
    "Seq_Rating": [
        """User profile:
{user_profile}

The following training interactions summarize this user's known movie ratings:
{history}

Candidate movie:
{candidate_title}

Predict the rating this user would give the candidate movie. {rating_only_suffix}""",
        """The user can be described as follows:
{user_profile}

Known training-set ratings:
{history}

Given candidate title: {candidate_title}

What rating would this user give this movie? {rating_only_suffix}""",
        """Profile:
{user_profile}

Training history, as movie titles with user ratings:
{history}

Now estimate the user's rating for {candidate_title}. {rating_only_suffix}""",
        """Use the user's profile and all available training-set ratings to estimate a rating.

Profile:
{user_profile}

Training ratings:
{history}

Movie to rate: {candidate_title}

Return the rating. {rating_only_suffix}""",
    ],
    "Single_Title2Feature": [
        """The movie title is {movie_title}.

Describe this movie in natural language using its genres and description. Do not repeat the title.""",
        """Please explain the movie {movie_title}. Include its genres and description in a natural sentence, without restating the title.""",
        """What kind of movie is {movie_title}? Describe it with genres and description, but do not copy the title into the answer.""",
        """Turn the movie title {movie_title} into concise natural-language movie features with genres and description. Do not repeat the title.""",
    ],
    "Single_Feature2Title": [
        """Here is a natural-language description of a movie:
{movie_text}

Which MovieLens title does this description refer to? {title_only_suffix}""",
        """Identify the MovieLens title from this description:
{movie_text}

{title_only_suffix}""",
        """A movie is described below:
{movie_text}

Return the matching MovieLens title. {title_only_suffix}""",
        """Match this movie description to its MovieLens title:
{movie_text}

{title_only_suffix}""",
    ],
}


def render_prompt_template(task: str, rng: random.Random | None = None, **kwargs: Any) -> str:
    sampler = rng if rng is not None else random
    return sampler.choice(PROMPT_TEMPLATES[task]).format(**kwargs)


@dataclass(frozen=True)
class RenderedExample:
    instruction: str
    input: str
    output: str


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


def format_title_interaction_history(history: Iterable[dict[str, Any]], movie_features: MovieFeatureStore) -> str:
    lines = []
    for event in history:
        movie_id = clean_value(event.get("movie_id"))
        title = movie_features.title(movie_id)
        rating = clean_value(event.get("rating"))
        lines.append(f"- {title} | rating: {rating}")
    return "\n".join(lines)


def movie_title_answer(movie_id: Any, movie_features: MovieFeatureStore) -> str:
    return movie_features.title(movie_id)


def rating_answer(rating: Any) -> str:
    return clean_value(rating)


def build_next_movie_title_prediction(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    target_movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "NextMovieTitlePrediction",
        rng,
        user_profile=format_user_profile(user),
        history=format_title_interaction_history(history, movie_features),
        title_only_suffix=TITLE_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, movie_title_answer(target_movie_id, movie_features))


def build_seq_title_to_feature(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    target_movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Seq_Title2Feature",
        rng,
        user_profile=format_user_profile(user),
        history=format_title_interaction_history(history, movie_features),
    )
    return RenderedExample(INSTRUCTION, input_text, movie_features.basic_text(target_movie_id))


def build_seq_rating(
    user: dict[str, Any] | None,
    train_history: list[dict[str, Any]],
    candidate_movie_id: str,
    candidate_rating: Any,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Seq_Rating",
        rng,
        user_profile=format_user_profile(user),
        history=format_title_interaction_history(train_history, movie_features),
        candidate_title=movie_features.title(candidate_movie_id),
        rating_only_suffix=RATING_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, rating_answer(candidate_rating))


def build_single_title_to_feature(
    movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Single_Title2Feature",
        rng,
        movie_title=movie_features.title(movie_id),
    )
    return RenderedExample(INSTRUCTION, input_text, movie_features.feature_text(movie_id))


def build_single_feature_to_title(
    movie_id: str,
    movie_features: MovieFeatureStore,
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Single_Feature2Title",
        rng,
        movie_text=movie_features.full_feature(movie_id),
        title_only_suffix=TITLE_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, movie_title_answer(movie_id, movie_features))
