from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from utils.data_io import MovieFeatureStore, clean_value, movie_token

TASKS = (
    "NextMoviePrediction",
    "Seq_ID2Title",
    "Seq_Title2ID",
    "Single_ID2Title",
    "Single_Title2ID",
    "ID2Feature",
    "Feature2ID",
)
ID_OUTPUT_TASKS = {
    "NextMoviePrediction",
    "Seq_Title2ID",
    "Single_Title2ID",
    "Feature2ID",
}
INSTRUCTION = "You are a helpful movie recommendation assistant. Use the user's viewing history and movie information to make clear, concise recommendations."
ID_ONLY_SUFFIX = "Answer with exactly one MovieLens ID and no other words."


PROMPT_TEMPLATES = {
    "NextMoviePrediction": [
        """Here is a user's recent movie-watching history. Each line says that the user gave a MovieLens ID a rating at a specific time:
{history}

Based on this chronological rating history, recommend the movie ID the user is most likely to watch next. {id_only_suffix}""",
        """A user has the following chronological interaction log. Each line means the user gave the MovieLens ID a rating at that timestamp:
{history}

Use this sequence to predict the next movie. {id_only_suffix}""",
        """The following records summarize that the user rated each MovieLens ID at the shown timestamp:
{history}

What MovieLens ID should come next for this user? {id_only_suffix}""",
        """Consider this user's recent viewing sequence. Each line says when the user rated a MovieLens ID and how many stars they gave it:
{history}

Please infer the next likely MovieLens item from when the user rated each Movie ID and how much they liked it. {id_only_suffix}""",
    ],
    "Seq_ID2Title": [
        """A user has this chronological MovieLens interaction history. Each line says that the user gave a MovieLens ID a rating at a specific time:
{history}

Describe the kind of movie this user is likely to watch next. Include the title, genres, and description naturally in your answer.""",
        """Here is a user's recent sequence of ratings. Each line says when the user rated a MovieLens ID and how many stars they gave it:
{history}

Instead of returning another ID, write a natural description of the next movie, including its title, genres, and description.""",
        """The user's recent MovieLens history is shown below. Each line means the user gave that MovieLens ID a rating at that timestamp:
{history}

What movie does this sequence of dated ratings point to next? Answer in prose and include the title, genres, and description.""",
        """Given this chronological list of MovieLens interactions, where each line records the timestamp and rating for a MovieLens ID:
{history}

Summarize the likely next movie in natural language, making sure the title, genres, and description are present.""",
    ],
    "Seq_Title2ID": [
        """User profile:
{user_profile}

Here is the user's chronological viewing history. Each line says when the user rated a movie, how many stars they gave it, and what the movie is:
{history}

Given this profile and history, recommend the next MovieLens movie ID. {id_only_suffix}""",
        """The user can be described as follows:
{user_profile}

Their recent interactions are listed below. Each line says the timestamp, the rating the user gave, and the title and genres of the rated movie:
{history}

Based on these preferences, which MovieLens ID is the most plausible next item? {id_only_suffix}""",
        """Use the profile and interaction details below to make a next-movie prediction.

Profile:
{user_profile}

Interactions, where each line says when the user rated a movie, how many stars they gave it, and what the movie is:
{history}

Return the next MovieLens ID. {id_only_suffix}""",
        """A user with this profile:
{user_profile}

has watched and rated movies described as follows. Each line records a timestamp, the user's rating, and the movie's title and genres:
{history}

Predict the next item in MovieLens ID form. {id_only_suffix}""",
    ],
    "Single_ID2Title": [
        """I know this MovieLens item only by its ID: {movie_id}.

Tell me what movie it refers to in natural language, including its title, genres, and description.""",
        """Please explain which movie is represented by the MovieLens ID {movie_id}. Include the title, genres, and description in a natural sentence.""",
        """What movie does {movie_id} correspond to? Describe it with its title, genres, and description.""",
        """Turn the MovieLens ID {movie_id} into a concise natural-language movie description with title, genres, and description.""",
    ],
    "Single_Title2ID": [
        """Here is a natural-language description of a movie:
{movie_text}

Which MovieLens ID does this description refer to? {id_only_suffix}""",
        """Identify the MovieLens item from this description:
{movie_text}

{id_only_suffix}""",
        """A movie is described below:
{movie_text}

Return the matching MovieLens ID. {id_only_suffix}""",
        """Match this movie description to its MovieLens ID:
{movie_text}

{id_only_suffix}""",
    ],
    "ID2Feature": [
        """Please describe the MovieLens item {movie_id} in a compact but informative way.

Include the title, genres, description, runtime, adult flag, average rating, and vote count.""",
        """Write a natural description for MovieLens ID {movie_id}. Mention its title, genres, description, runtime, whether it is adult content, average rating, and vote count.""",
        """For the item {movie_id}, summarize the important movie features: title, genres, description, runtime, adult flag, average rating, and vote count.""",
        """Convert {movie_id} into a readable movie profile containing title, genres, description, runtime minutes, adult status, average rating, and vote count.""",
    ],
    "Feature2ID": [
        """A movie can be described this way:
{movie_text}

Which MovieLens ID does this movie correspond to? {id_only_suffix}""",
        """Find the MovieLens ID for the following movie profile:
{movie_text}

{id_only_suffix}""",
        """This movie feature profile describes one MovieLens item:
{movie_text}

Return its MovieLens ID. {id_only_suffix}""",
        """Match the following movie profile back to the correct MovieLens ID:
{movie_text}

{id_only_suffix}""",
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


def readable_timestamp(value: Any) -> str:
    text = clean_value(value)
    if text == "Unknown":
        return text
    try:
        return datetime.fromtimestamp(int(float(text)), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OSError):
        return text


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


def format_id_interaction_history(
    history: Iterable[dict[str, Any]],
    token_format: Literal["angle", "plain"] = "angle",
) -> str:
    lines = []
    for event in history:
        movie_id = clean_value(event.get("movie_id"))
        lines.append(
            f"At {readable_timestamp(event.get('timestamp'))}, the user gave {movie_token(movie_id, token_format)} "
            f"a rating of {clean_value(event.get('rating'))} stars."
        )
    return "\n".join(lines)


def format_feature_interaction_history(history: Iterable[dict[str, Any]], movie_features: MovieFeatureStore) -> str:
    lines = []
    for event in history:
        movie_id = clean_value(event.get("movie_id"))
        lines.append(
            f"- At {readable_timestamp(event.get('timestamp'))}, the user gave this movie a rating of {clean_value(event.get('rating'))} stars; "
            f"{movie_features.interaction_text(movie_id)}."
        )
    return "\n".join(lines)


def movie_id_answer(movie_id: Any, token_format: Literal["angle", "plain"] = "angle") -> str:
    return movie_token(movie_id, token_format)


def identify_movie_answer(movie_id: Any, token_format: Literal["angle", "plain"] = "angle") -> str:
    return movie_token(movie_id, token_format)


def build_next_movie_prediction(
    history: list[dict[str, Any]],
    target_movie_id: str,
    token_format: Literal["angle", "plain"],
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "NextMoviePrediction",
        rng,
        history=format_id_interaction_history(history, token_format),
        id_only_suffix=ID_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, movie_id_answer(target_movie_id, token_format))


def build_seq_id_to_title(
    history: list[dict[str, Any]],
    target_movie_id: str,
    movie_features: MovieFeatureStore,
    token_format: Literal["angle", "plain"],
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Seq_ID2Title",
        rng,
        history=format_id_interaction_history(history, token_format),
    )
    return RenderedExample(INSTRUCTION, input_text, movie_features.basic_text(target_movie_id))


def build_seq_title_to_id(
    user: dict[str, Any] | None,
    history: list[dict[str, Any]],
    movie_features: MovieFeatureStore,
    target_movie_id: str,
    token_format: Literal["angle", "plain"],
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Seq_Title2ID",
        rng,
        user_profile=format_user_profile(user),
        history=format_feature_interaction_history(history, movie_features),
        id_only_suffix=ID_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, movie_id_answer(target_movie_id, token_format))


def build_single_id_to_title(
    movie_id: str,
    movie_features: MovieFeatureStore,
    token_format: Literal["angle", "plain"],
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template("Single_ID2Title", rng, movie_id=movie_token(movie_id, token_format))
    return RenderedExample(INSTRUCTION, input_text, movie_features.basic_text(movie_id))


def build_single_title_to_id(
    movie_id: str,
    movie_features: MovieFeatureStore,
    token_format: Literal["angle", "plain"],
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Single_Title2ID",
        rng,
        movie_text=movie_features.basic_text(movie_id),
        id_only_suffix=ID_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, identify_movie_answer(movie_id, token_format))


def build_id_to_feature(
    movie_id: str,
    movie_features: MovieFeatureStore,
    token_format: Literal["angle", "plain"],
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template("ID2Feature", rng, movie_id=movie_token(movie_id, token_format))
    return RenderedExample(INSTRUCTION, input_text, movie_features.full_feature(movie_id))


def build_feature_to_id(
    movie_id: str,
    movie_features: MovieFeatureStore,
    token_format: Literal["angle", "plain"],
    rng: random.Random | None = None,
) -> RenderedExample:
    input_text = render_prompt_template(
        "Feature2ID",
        rng,
        movie_text=movie_features.full_feature(movie_id),
        id_only_suffix=ID_ONLY_SUFFIX,
    )
    return RenderedExample(INSTRUCTION, input_text, identify_movie_answer(movie_id, token_format))
