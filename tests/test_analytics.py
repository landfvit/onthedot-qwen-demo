import json
import sqlite3

import pytest

from conftest import SYNTHETIC_PATH

pytestmark = pytest.mark.usefixtures("synthetic_dataset")

from onthedot import analytics
from onthedot.schema import Query


def player_result(**changes):
    return analytics.execute_query(Query.model_validate({"operation": "player_metric", "metric": "goals", "players": ["Mercer"], "season": "2024-25", **changes}))


def test_rate_weights_minutes_instead_of_averaging_match_rates():
    result = player_result(metric="goals_per_90")
    assert result["rows"][0] == {"label": "Alex Mercer", "value": 1.2, "matches": 6, "known_matches": 6, "minutes": 450, "data_status": "available"}
    assert result["unit"] == "goals per 90"
    assert result["notice"] == "Synthetic demo data"
    assert result["scope"] == {"competition": "Demo League", "team": None, "season": "2024-25", "venue": "all"}


def test_unknown_inputs_keep_partial_totals_but_invalidate_rates():
    total = player_result(season="2025-26")["rows"][0]
    assert total["value"] == 5 and total["matches"] == 6 and total["known_matches"] == 5
    assert total["data_status"] == "partial"
    rate = player_result(season="2025-26", metric="goals_per_90")["rows"][0]
    assert rate["value"] is None and rate["data_status"] == "insufficient_data"
    combined = player_result(season=None)["rows"][0]
    assert combined["value"] == 11 and combined["known_matches"] == 11 and combined["matches"] == 12


def test_observed_zero_unknown_and_empty_samples_are_distinct():
    unknown = player_result(players=["Grant"], metric="shots")["rows"][0]
    assert unknown["value"] is None and unknown["matches"] == 6 and unknown["known_matches"] == 0
    zero = player_result(players=["Grant"], season="2025-26", venue="home")["rows"][0]
    assert zero["value"] == 0 and zero["matches"] == zero["known_matches"] == 2
    assert zero["data_status"] == "available"
    empty = player_result(team="Riverside")["rows"][0]
    assert empty["value"] is None and empty["matches"] == empty["known_matches"] == 0
    assert empty["minutes"] == 0


def test_season_and_home_away_filters_scope_the_same_appearances():
    home = player_result(venue="home")["rows"][0]
    away = player_result(venue="away")["rows"][0]
    assert (home["value"], home["minutes"], home["matches"]) == (4, 240, 3)
    assert (away["value"], away["minutes"], away["matches"]) == (2, 210, 3)
    assert player_result(metric="goals_per_90", venue="home")["rows"][0]["value"] == 1.5
    assert player_result(metric="goals_per_90", venue="away")["rows"][0]["value"] == pytest.approx(2 * 90 / 210, abs=1e-6)


def test_comparison_preserves_subject_order_and_one_common_scope():
    result = player_result(operation="player_comparison", players=["Blake", "Mercer"], venue="away")
    assert [row["label"] for row in result["rows"]] == ["Theo Blake", "Alex Mercer"]
    assert [row["value"] for row in result["rows"]] == [1, 2]
    assert all(row["matches"] == 3 for row in result["rows"])


def test_monthly_trends_keep_chronology_and_use_weighted_period_totals():
    result = player_result(operation="player_trend", metric="goals_per_90")
    assert [row["label"] for row in result["rows"]] == ["2024-08", "2024-09", "2024-10"]
    assert [row["value"] for row in result["rows"]] == [1.5, 0.5, 1.8]
    assert player_result(operation="player_trend", team="Riverside")["rows"] == []
    all_months = player_result(operation="player_trend", season=None)["rows"]
    assert [row["label"] for row in all_months] == ["2024-08", "2024-09", "2024-10", "2025-08", "2025-09", "2025-10"]
    assert all_months[4]["value"] == 2 and all_months[4]["data_status"] == "partial"


@pytest.mark.parametrize("team,season,metric,value", [
    ("Northbridge", "2024-25", "goals", 9),
    ("Northbridge", "2024-25", "points", 10),
    ("Northbridge", "2024-25", "wins", 3),
    ("Northbridge", "2024-25", "win_rate", 50.0),
    ("Northbridge", "2025-26", "points", 11),
    ("Riverside", "2024-25", "points", 7),
    ("Riverside", "2025-26", "wins", 1),
])
def test_team_statistics_respect_home_away_score_orientation(team, season, metric, value):
    result = analytics.execute_query(Query(operation="team_metric", metric=metric, team=team, season=season))
    assert result["rows"][0]["value"] == value
    assert result["rows"][0]["matches"] == result["rows"][0]["known_matches"] == 6


def test_team_home_away_rates_have_the_correct_denominator():
    home = analytics.execute_query(Query(operation="team_metric", metric="win_rate", team="Northbridge", season="2024-25", venue="home"))
    away = analytics.execute_query(Query(operation="team_metric", metric="win_rate", team="Northbridge", season="2024-25", venue="away"))
    assert home["rows"][0]["value"] == pytest.approx(200 / 3, abs=1e-6)
    assert away["rows"][0]["value"] == pytest.approx(100 / 3, abs=1e-6)
    assert home["unit"] == "%"


@pytest.mark.parametrize("operation", ["clarification", "unsupported"])
def test_control_queries_do_not_open_a_database(monkeypatch, operation):
    def unexpected_database():
        raise AssertionError("Control queries must not execute analytics.")
    monkeypatch.setattr(analytics, "_database", unexpected_database)
    result = analytics.execute_query(Query(operation=operation, message="Please specify a supported metric."))
    assert result["status"] == operation and result["rows"] == []
    assert result["message"] == "Please specify a supported metric."
    assert result["sql"] == []


@pytest.fixture
def recorded_sql(monkeypatch):
    statements = []
    databases = []

    class RecordedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            statements.append({"statement": sql, "parameters": list(parameters)})
            return super().execute(sql, parameters)

    connect = sqlite3.connect

    def recorded_connect(path):
        databases.append(path)
        return connect(path, factory=RecordedConnection)

    monkeypatch.setattr(analytics.sqlite3, "connect", recorded_connect)
    return statements, databases


def test_user_selections_are_sql_parameters_and_database_is_disposable(recorded_sql):
    statements, databases = recorded_sql
    result = player_result(team="Northbridge", venue="home")
    sql, parameters = statements[-1]["statement"], statements[-1]["parameters"]
    assert parameters == ["Alex Mercer", "Northbridge FC", "2024-25"]
    assert all(value not in sql for value in parameters)
    assert "a.team = ?" in sql and "f.season = ?" in sql and "f.home = a.team" in sql
    assert result["sql"] == statements
    assert len(result["sql"]) == 1
    assert databases == [":memory:"]


@pytest.mark.parametrize("query,parameters,clauses", [
    ({"operation": "player_comparison", "metric": "shots", "players": ["Blake", "Mercer"], "season": "2025-26", "venue": "away"},
        [["Theo Blake", "2025-26"], ["Alex Mercer", "2025-26"]], ["COUNT(a.shots)", "f.season = ?", "f.away = a.team"]),
    ({"operation": "player_trend", "metric": "goals", "players": ["Mercer"], "season": "2024-25"},
        [["Alex Mercer", "2024-25"]], ["SUM(a.goals)", "GROUP BY month ORDER BY month", "f.season = ?"]),
    ({"operation": "team_metric", "metric": "points", "team": "Northbridge", "season": "2024-25", "venue": "away"},
        [["Northbridge FC", "Northbridge FC", "Northbridge FC", "Northbridge FC", "2024-25", "Northbridge FC"]],
        ["WITH selected", "SUM(CASE WHEN gf > ga THEN 3", "season = ?", "away = ?"]),
])
def test_sql_display_is_exactly_the_executed_selects_in_execution_order(recorded_sql, query, parameters, clauses):
    statements, _ = recorded_sql
    result = analytics.execute_query(Query.model_validate(query))
    assert result["sql"] == statements
    assert [entry["parameters"] for entry in result["sql"]] == parameters
    for entry in result["sql"]:
        assert all(clause in entry["statement"] for clause in clauses)
        assert entry["statement"].count("?") == len(entry["parameters"])
        assert entry["statement"].lstrip().startswith(("SELECT", "WITH"))
        assert "CREATE TABLE" not in entry["statement"] and "INSERT INTO" not in entry["statement"]


def test_fictional_dataset_has_consistent_scores_and_explicit_unknown_values():
    fixtures = json.loads(SYNTHETIC_PATH.read_text())["fixtures"]
    assert len(fixtures) == 12
    assert len({f["id"] for f in fixtures}) == 12
    for fixture in fixtures:
        assert len(fixture["appearances"]) == 4
        for side in ("home", "away"):
            goals = [a["goals"] for a in fixture["appearances"] if a["team"] == fixture[side]]
            if all(goal is not None for goal in goals):
                assert sum(goals) == fixture[f"{side}_goals"]
