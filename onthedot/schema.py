"""The small public query contract, independent of any database or model."""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .dataset import active_catalog, manifest

PLAYERS = active_catalog()["players"]
TEAMS = active_catalog()["teams"]
SEASONS = [active_catalog()["season"]]
COMPETITION = active_catalog()["competition"]
DATA_NOTICE = manifest()["notice"]
PLAYER_METRICS = ("goals", "assists", "shots", "minutes", "goals_per_90", "assists_per_90")
TEAM_METRICS = ("goals", "points", "wins", "win_rate")
Metric = Literal["goals", "assists", "shots", "minutes", "goals_per_90", "assists_per_90", "points", "wins", "win_rate"]


def catalog() -> dict:
    return {
        "players": list(PLAYERS), "teams": list(TEAMS),
        "aliases": {name: list(aliases) for name, aliases in (PLAYERS | TEAMS).items()},
        "seasons": list(SEASONS),
        "metrics": {"player": list(PLAYER_METRICS), "team": list(TEAM_METRICS)},
        "competition": COMPETITION, "notice": DATA_NOTICE,
    }


def _canonical(value: str, entities: dict) -> str:
    if not isinstance(value, str):
        raise ValueError("Entity names must be strings.")
    for name, aliases in entities.items():
        if value.strip().casefold() in {alias.casefold() for alias in aliases}:
            return name
    raise ValueError(f"Unknown demo entity: {value}")


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    operation: Literal["player_metric", "player_comparison", "player_trend", "team_metric", "clarification", "unsupported"]
    metric: Metric | None = None
    players: list[str] = Field(default_factory=list, max_length=4)
    team: str | None = None
    season: str | None = None
    venue: Literal["all", "home", "away"] = "all"
    message: str | None = Field(default=None, min_length=1, max_length=300)

    @field_validator("season")
    @classmethod
    def supported_season(cls, value):
        if value is not None and value not in SEASONS:
            raise ValueError("Choose a season present in the dataset.")
        return value

    @field_validator("players", mode="before")
    @classmethod
    def canonical_players(cls, value):
        if not isinstance(value, list):
            raise ValueError("Players must be a list.")
        return [_canonical(name, PLAYERS) for name in value]

    @field_validator("team", mode="before")
    @classmethod
    def canonical_team(cls, value):
        return None if value is None else _canonical(value, TEAMS)

    @model_validator(mode="after")
    def check_operation(self):
        if len(set(self.players)) != len(self.players):
            raise ValueError("Players must be distinct.")
        if self.operation in {"clarification", "unsupported"}:
            if not self.message or self.metric or self.players or self.team or self.season or self.venue != "all":
                raise ValueError("Control queries require a message and no analytical fields.")
        elif self.message is not None:
            raise ValueError("Analytical queries cannot contain a message.")
        elif self.operation == "team_metric":
            if not self.team or self.players or self.metric not in TEAM_METRICS:
                raise ValueError("Team queries require a team and a supported team metric.")
        else:
            valid_count = 2 <= len(self.players) <= 4 if self.operation == "player_comparison" else len(self.players) == 1
            if not valid_count or self.metric not in PLAYER_METRICS:
                raise ValueError("Player queries require the right number of players and a supported player metric.")
        return self


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_query(raw_text: str) -> Query:
    # accept one json object only, never executable code or model-generated sql.
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    if not isinstance(raw_text, str) or len(raw_text) > 10_000:
        raise ValueError("Model output must be a JSON object under 10,000 characters.")
    raw = json.loads(raw_text, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    if not isinstance(raw, dict):
        raise ValueError("Model output must be one JSON object.")
    return Query.model_validate(raw)


def _mentions(question: str, entities: dict) -> set[str]:
    matches = [
        (name, match.start(), match.end())
        for name, aliases in entities.items() for alias in aliases
        for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", question, re.I)
    ]
    # a full name takes precedence over another player's shorter nested alias
    return {name for name, start, end in matches if not any(
        other != name and left <= start and right >= end and right - left > end - start
        for other, left, right in matches
    )}


def validate_grounding(query: Query, question: str) -> None:
    # lexical checks protect entity and filter slots, not the model's choice of intent.
    if query.operation in {"clarification", "unsupported"}:
        return
    if set(query.players) != _mentions(question, PLAYERS):
        raise ValueError("Player selections must match names present in the question.")
    if ({query.team} if query.team else set()) != _mentions(question, TEAMS):
        raise ValueError("Team filters must match names present in the question.")
    # typography must not let an explicit season disappear from the scope.
    season_text = question.translate(str.maketrans({dash: "-" for dash in "‐‑‒–—−﹣－"}))
    seasons = {
        f"{start}-{end[-2:]}" for start, end in re.findall(r"(?<!\d)(20\d{2})\s*[-/]\s*(20\d{2}|\d{2})(?!\d)", season_text)
    }
    if ({query.season} if query.season else set()) != seasons:
        raise ValueError("Season filters must match the explicit season in the question.")
    venues = {venue for venue in ("home", "away") if re.search(r"\b" + venue + r"\b", question, re.I)}
    if re.search(r"\bon the road\b", question, re.I):
        venues.add("away")
    expected_venue = next(iter(venues)) if len(venues) == 1 else "all"
    if query.venue != expected_venue:
        raise ValueError("Venue filters must match the explicit venue in the question.")
