import json

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_catalog")


def test_a_full_name_does_not_select_another_players_nested_alias(monkeypatch):
    from onthedot import schema
    monkeypatch.setattr(schema, "PLAYERS", {
        "Mohamed Salah": ["Mohamed Salah", "Salah"],
        "Salah Oulad M'Hand": ["Salah Oulad M'Hand"],
    })
    query = schema.Query(operation="player_metric", metric="minutes", players=["Salah Oulad M'Hand"])
    schema.validate_grounding(query, "Salah Oulad M'Hand minutes")
    comparison = schema.Query(operation="player_comparison", metric="minutes", players=["Mohamed Salah", "Salah Oulad M'Hand"])
    schema.validate_grounding(comparison, "Compare Mohamed Salah with Salah Oulad M'Hand by minutes")
from pydantic import ValidationError

from onthedot.schema import Query, catalog, parse_query, validate_grounding


def player_query(**changes):
    return Query.model_validate({"operation": "player_metric", "metric": "goals", "players": ["Mercer"], **changes})


def test_aliases_are_canonical_and_optional_filters_stay_unset():
    query = player_query(players=[" mErCeR "], team="NORTHBRIDGE")
    assert query.players == ["Alex Mercer"]
    assert query.team == "Northbridge FC"
    assert query.season is None and query.venue == "all"
    assert catalog()["notice"] == "Synthetic demo data"


@pytest.mark.parametrize("changes", [
    {"operation": "arbitrary_sql"}, {"sql": "DROP TABLE fixtures"},
    {"players": ["Lionel Messi"]}, {"team": "Arsenal"},
    {"players": ["Mercer", "Alex Mercer"]},
    {"players": ["Mercer", "Blake"]},
    {"operation": "player_comparison"},
    {"operation": "player_trend", "players": ["Mercer", "Blake"]},
    {"operation": "team_metric", "players": []},
    {"operation": "team_metric", "team": "Northbridge"},
    {"operation": "team_metric", "players": [], "team": "Northbridge", "metric": "goals_per_90"},
    {"metric": "points"}, {"metric": "expected_goals"},
    {"message": "An analytical query cannot supply prose."},
    {"season": "2023-24"}, {"venue": "neutral"},
    {"players": "Mercer"}, {"players": [1]}, {"metric": True},
])
def test_invalid_or_irrelevant_fields_are_rejected(changes):
    with pytest.raises(ValidationError):
        player_query(**changes)


@pytest.mark.parametrize("operation", ["clarification", "unsupported"])
def test_control_queries_need_only_a_message(operation):
    query = Query(operation=operation, message="Please specify a metric.")
    assert query.players == []
    with pytest.raises(ValidationError):
        Query(operation=operation)
    with pytest.raises(ValidationError):
        Query(operation=operation, message=" ")
    with pytest.raises(ValidationError):
        Query(operation=operation, message="Please specify a metric.", season="2024-25")


@pytest.mark.parametrize("text", [
    '```json\n{"operation":"unsupported","message":"No."}\n```',
    '{"operation":"unsupported","message":"No."} trailing',
    '{"operation":"unsupported","message":"No."} {}',
    '[]', 'null', '"DROP TABLE fixtures"', 'SELECT * FROM fixtures;',
    '{"operation":"unsupported","message":"No.","value":NaN}',
    '{"operation":"unsupported","message":"No.","value":Infinity}',
    '{"operation":"unsupported","message":"No.","value":-Infinity}',
    '{"operation":"unsupported","message":"No.","value":1e999}',
    '{"operation":"unsupported","operation":"clarification","message":"No."}',
])
def test_only_one_strict_json_object_is_accepted(text):
    with pytest.raises(ValueError):
        parse_query(text)


def test_json_can_have_whitespace_and_round_trips_defaults():
    original = player_query()
    assert parse_query(" \n" + original.model_dump_json() + "\t") == original
    assert parse_query(json.dumps({"operation": "player_comparison", "metric": "shots", "players": ["Blake", "Silva"]})).players == ["Theo Blake", "Noah Silva"]


@pytest.mark.parametrize("question,changes", [
    ("Mercer goals in 2024-25", {"season": "2024-25"}),
    ("MERCER goals in 2024/25 away for Northbridge", {"season": "2024-25", "venue": "away", "team": "Northbridge"}),
    ("Alex Mercer goals in 2024-2025 at home", {"season": "2024-25", "venue": "home"}),
    ("Mercer goals on the road", {"venue": "away"}),
    ("Mercer goals home and away", {}),
])
def test_grounding_preserves_explicit_entities_and_filters(question, changes):
    validate_grounding(player_query(**changes), question)


@pytest.mark.parametrize("season", ["2024–25", "2024—25", "2024‑25", "2024 – 25", "2024—2025"])
def test_typographic_seasons_are_preserved_and_cannot_be_dropped(season):
    question = f"Mercer goals in {season}"
    validate_grounding(player_query(season="2024-25"), question)
    with pytest.raises(ValueError, match="Season filters"):
        validate_grounding(player_query(), question)


@pytest.mark.parametrize("question,changes", [
    ("Blake goals", {}),
    ("Mercer and Blake goals", {}),
    ("Mercer goals", {"team": "Northbridge"}),
    ("Mercer goals for Northbridge", {}),
    ("Mercer goals in 2024-25", {}),
    ("Mercer goals", {"season": "2025-26"}),
    ("Mercer goals in 2024-25", {"season": "2025-26"}),
    ("Mercer goals in 2023-24", {}),
    ("Mercer goals away", {}),
    ("Mercer goals", {"venue": "home"}),
])
def test_invented_or_dropped_entities_and_filters_are_rejected(question, changes):
    with pytest.raises(ValueError):
        validate_grounding(player_query(**changes), question)


def test_declining_a_question_does_not_require_fabricating_selectors():
    validate_grounding(Query(operation="unsupported", message="This analysis is unavailable."), "Compare Northbridge and Riverside in 2023-24.")
