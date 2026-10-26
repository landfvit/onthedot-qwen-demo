from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools import export_season as exporter


@pytest.fixture
def complete_season():
    season = {"id": 8, "name": "2024", "competition": "Premier League", "country": "England"}
    fixtures, appearances = [], []
    teams = [f"Team {i:02}" for i in range(1, 21)]
    for home_id, home in enumerate(teams, 1):
        for away_id, away in enumerate(teams, 1):
            if home_id == away_id:
                continue
            match_id = len(fixtures) + 1
            gameweek = (match_id - 1) // 10 + 1
            fixtures.append({"id": match_id, "kickoff": datetime(2024, 8, 16, tzinfo=timezone.utc) + timedelta(weeks=gameweek - 1),
                "status": "finished", "gameweek": gameweek, "home": home, "home_id": home_id,
                "away": away, "away_id": away_id, "home_score": 0, "away_score": 0})
            for player_id, team in ((home_id, home), (away_id, away)):
                appearances.append({"match_id": match_id, "player_id": player_id,
                    "player_name": "A. Onana" if player_id in {1, 2} else f"Player {player_id:02}",
                    "team": team, "team_id": player_id, "minutes": 90, "goals": 0,
                    "assists": 0, "shots": None, "source_payload_id": match_id})
    profiles = [{"player_id": 1, "name": "A. Onana", "firstname": "André", "lastname": "Onana"},
        {"player_id": 2, "name": "A. Onana", "firstname": "Amadou", "lastname": "Onana"}]
    return season, fixtures, appearances, profiles


def test_complete_season_keeps_ids_nulls_zero_minutes_and_all_rows(complete_season):
    season, fixtures, appearances, profiles = complete_season
    appearances[0]["minutes"] = 0
    appearances[1]["minutes"] = None
    appearances[2]["goals"] = None
    result = exporter.build_dataset(season, fixtures, appearances, profiles)
    assert len(result["fixtures"]) == 380
    exported = [row for match in result["fixtures"] for row in match["appearances"]]
    assert len(exported) == len(appearances) == 760
    by_key = {(row["player_id"], match["id"]): row for match in result["fixtures"] for row in match["appearances"]}
    for row in appearances:
        converted = by_key[(row["player_id"], row["match_id"])]
        assert all(converted[field] == row[field] for field in ("player_id", "team_id", "minutes", "goals", "assists", "shots", "source_payload_id"))
    assert result["provenance"]["counts"]["eligible_appearances"] == 758
    assert result["provenance"]["eligible_missing_fields"]["goals"] == 1
    assert result["provenance"]["eligible_missing_fields"]["shots"] == 758
    assert result["provenance"]["missing_minutes"] == 1
    assert result["catalog"]["season"] == "2024-25"


def test_ambiguous_short_names_and_surnames_do_not_merge_players(complete_season):
    result = exporter.build_dataset(*complete_season)
    players = result["catalog"]["players"]
    assert "André Onana" in players and "Amadou Onana" in players
    assert "Andre Onana" in players["André Onana"]
    assert all("A. Onana" not in aliases and "Onana" not in aliases for aliases in players.values())
    assert "a. onana" in result["provenance"]["excluded_ambiguous_aliases"]
    ids = {row["player_id"] for match in result["fixtures"] for row in match["appearances"] if row["player"] in {"André Onana", "Amadou Onana"}}
    assert ids == {1, 2}


def test_equal_full_names_get_distinct_club_and_id_labels(complete_season):
    season, fixtures, appearances, profiles = complete_season
    for profile in profiles:
        profile.update(firstname="Alex", lastname="Smith")
    result = exporter.build_dataset(season, fixtures, appearances, profiles)
    labels = [label for label in result["catalog"]["players"] if label.startswith("Alex Smith")]
    assert len(labels) == 2 and labels[0] != labels[1]
    assert any("#1" in label for label in labels) and any("#2" in label for label in labels)
    assert all("Alex Smith" not in result["catalog"]["players"][label] for label in labels)


def test_common_full_name_aliases_do_not_make_shared_surnames_unique(complete_season):
    season, fixtures, appearances, profiles = complete_season
    profiles[0].update(firstname="Cole", lastname="Jermaine Palmer", name="C. Palmer")
    profiles[1].update(firstname="Alex", lastname="Palmer", name="A. Palmer")
    for row in appearances:
        if row["player_id"] in {1, 2}:
            row["player_name"] = "C. Palmer" if row["player_id"] == 1 else "A. Palmer"
    result = exporter.build_dataset(season, fixtures, appearances, profiles)
    aliases = result["catalog"]["players"]
    assert "Cole Palmer" in aliases["Cole Jermaine Palmer"]
    assert "Alex Palmer" in aliases["Alex Palmer"]
    assert all("Palmer" not in values for values in aliases.values())


@pytest.mark.parametrize("defect", ["missing_fixture", "duplicate_fixture", "duplicate_pair", "unscored", "unfinished", "missing_gameweek", "missing_statistics", "duplicate_player"])
def test_incomplete_or_ambiguous_season_is_not_exported(complete_season, defect):
    season, fixtures, appearances, profiles = deepcopy(complete_season)
    if defect == "missing_fixture":
        fixtures.pop()
    elif defect == "duplicate_fixture":
        fixtures[-1]["id"] = fixtures[0]["id"]
    elif defect == "duplicate_pair":
        fixtures[-1].update(home=fixtures[0]["home"], away=fixtures[0]["away"])
    elif defect == "unscored":
        fixtures[0]["home_score"] = None
    elif defect == "unfinished":
        fixtures[0]["status"] = "scheduled"
    elif defect == "missing_gameweek":
        fixtures[0]["gameweek"] = None
    elif defect == "missing_statistics":
        appearances[:] = [row for row in appearances if row["match_id"] != fixtures[0]["id"]]
    else:
        appearances.append(deepcopy(appearances[0]))
    with pytest.raises(exporter.ExportError):
        exporter.build_dataset(season, fixtures, appearances, profiles)


def test_public_manifest_has_no_match_rows_and_hashes_the_exact_private_bytes(complete_season, monkeypatch, tmp_path):
    dataset = exporter.build_dataset(*complete_season)
    monkeypatch.setattr(exporter, "read_season", lambda *_: dataset)
    data_path, metadata_path = tmp_path / "private.json", tmp_path / "catalog.json"
    manifest = exporter.export_season("not logged", 2024, data_path, metadata_path)
    assert manifest["dataset_id"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert json.loads(data_path.read_text())["catalog"] == manifest["catalog"]
    assert json.loads(metadata_path.read_text()) == manifest
    assert set(manifest) == {"dataset_id", "catalog", "notice", "source", "counts", "date_range"}
    assert "fixtures" not in manifest and "goals" not in json.dumps(manifest)


def test_failed_export_leaves_existing_files_and_connection_details_private(monkeypatch, tmp_path, capsys):
    data_path, metadata_path = tmp_path / "private.json", tmp_path / "catalog.json"
    data_path.write_text("previous data")
    metadata_path.write_text("previous catalog")

    def fail(*_):
        raise RuntimeError("password=do-not-log-this")

    monkeypatch.setattr(exporter, "read_season", fail)
    assert exporter.main(["--database", "password=do-not-log-this", "--output", str(data_path), "--catalog", str(metadata_path)]) == 1
    assert "do-not-log-this" not in capsys.readouterr().err
    assert data_path.read_text() == "previous data" and metadata_path.read_text() == "previous catalog"


def test_source_session_is_read_only_and_queries_are_bound(complete_season, monkeypatch):
    season, fixtures, appearances, profiles = complete_season
    calls, connection_options = [], {}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement, parameters=()):
            calls.append((statement, parameters))
            if statement.startswith("SHOW"):
                return SimpleNamespace(fetchone=lambda: {"transaction_read_only": "on"})
            index = len(calls) - 2
            return SimpleNamespace(fetchall=lambda: [[season], fixtures, appearances, profiles][index])

    def connect(database, **options):
        assert database == "private source DSN"
        connection_options.update(options)
        return Connection()

    module = ModuleType("psycopg")
    rows = ModuleType("psycopg.rows")
    module.connect, rows.dict_row = connect, object()
    monkeypatch.setitem(sys.modules, "psycopg", module)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows)
    result = exporter.read_season("private source DSN", 2024)
    assert result["provenance"]["read_only"]
    assert "default_transaction_read_only=on" in connection_options["options"]
    assert all(statement.strip().startswith(("SELECT", "SHOW")) for statement, _ in calls)
    assert calls[1][1] == ("2024", "Premier League", "England")
    assert calls[2][1] == calls[3][1] == (8,)
    assert calls[4][1] == ("API-Football", 2024, 8)
