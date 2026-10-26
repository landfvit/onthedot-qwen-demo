"""Deterministic, parameter-bound analytics over a local season export."""

from contextlib import closing
import sqlite3
from textwrap import dedent

from .dataset import data_available, load_dataset
from .schema import Query, catalog

PLAYER_COLUMNS = {"goals": "goals", "assists": "assists", "shots": "shots", "minutes": "minutes", "goals_per_90": "goals", "assists_per_90": "assists"}
TEAM_EXPRESSIONS = {"goals": "SUM(gf)", "wins": "SUM(gf > ga)", "points": "SUM(CASE WHEN gf > ga THEN 3 WHEN gf = ga THEN 1 ELSE 0 END)", "win_rate": "SUM(gf > ga)"}
UNITS = {"goals": "goals", "assists": "assists", "shots": "shots", "minutes": "minutes", "goals_per_90": "goals per 90", "assists_per_90": "assists per 90", "points": "points", "wins": "wins", "win_rate": "%"}


def _database() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE fixtures (id INTEGER PRIMARY KEY, date TEXT, season TEXT,
            home TEXT, away TEXT, home_goals INTEGER, away_goals INTEGER);
        CREATE TABLE appearances (fixture_id INTEGER, player TEXT, team TEXT,
            minutes INTEGER, goals INTEGER, assists INTEGER, shots INTEGER);
    """)
    fixtures = load_dataset()["fixtures"]
    db.executemany("INSERT INTO fixtures VALUES (?, ?, ?, ?, ?, ?, ?)", [
        (f["id"], f["date"], f["season"], f["home"], f["away"], f["home_goals"], f["away_goals"]) for f in fixtures
    ])
    db.executemany("INSERT INTO appearances VALUES (?, ?, ?, ?, ?, ?, ?)", [
        (f["id"], a["player"], a["team"], a["minutes"], a["goals"], a["assists"], a["shots"])
        for f in fixtures for a in f["appearances"]
    ])
    return db


def _select(db: sqlite3.Connection, statement: str, parameters: list, sql: list):
    # retain the exact statement and ordered bindings used by sqlite.
    statement = dedent(statement).strip()
    sql.append({"statement": statement, "parameters": list(parameters)})
    return db.execute(statement, parameters)


def _player_rows(db: sqlite3.Connection, query: Query, player: str, sql: list) -> list[dict]:
    conditions, params = ["a.player = ?", "a.minutes > 0"], [player]
    if query.team:
        conditions.append("a.team = ?")
        params.append(query.team)
    if query.season:
        conditions.append("f.season = ?")
        params.append(query.season)
    if query.venue != "all":
        conditions.append(f"f.{query.venue} = a.team")
    column = PLAYER_COLUMNS[query.metric]
    trend = query.operation == "player_trend"
    grouping = "strftime('%Y-%m', f.date) AS month, " if trend else ""
    # only whitelisted identifiers enter sql, while all user selections are bound.
    rows = _select(db, f"""
        SELECT {grouping}COUNT(*) AS matches,
            SUM(a.minutes) AS minutes, COUNT(a.{column}) AS known_matches,
            SUM(a.{column}) AS total
        FROM appearances a JOIN fixtures f ON f.id = a.fixture_id
        WHERE {' AND '.join(conditions)}
        {'GROUP BY month ORDER BY month' if trend else ''}
    """, params, sql).fetchall()
    result = []
    for row in rows:
        rate = query.metric.endswith("_per_90")
        value = row["total"]
        status = "available"
        if row["known_matches"] == 0 or (rate and row["known_matches"] < row["matches"]):
            value, status = None, "insufficient_data"
        elif row["known_matches"] < row["matches"]:
            status = "partial"
        if rate and value is not None:
            value = round(value * 90 / row["minutes"], 6)
        result.append({"label": row["month"] if trend else player, "value": value,
            "matches": row["matches"], "known_matches": row["known_matches"],
            "minutes": row["minutes"] or 0, "data_status": status})
    return result


def _team_rows(db: sqlite3.Connection, query: Query, sql: list) -> list[dict]:
    conditions, params = ["(home = ? OR away = ?)"], [query.team, query.team]
    if query.season:
        conditions.append("season = ?")
        params.append(query.season)
    if query.venue != "all":
        conditions.append(f"{query.venue} = ?")
        params.append(query.team)
    row = _select(db, f"""
        WITH selected AS (
            SELECT CASE WHEN home = ? THEN home_goals ELSE away_goals END AS gf,
                CASE WHEN home = ? THEN away_goals ELSE home_goals END AS ga
            FROM fixtures WHERE {' AND '.join(conditions)}
        )
        SELECT COUNT(*) AS matches, {TEAM_EXPRESSIONS[query.metric]} AS total FROM selected
    """, [query.team, query.team, *params], sql).fetchone()
    value = row["total"]
    if query.metric == "win_rate" and row["matches"]:
        value = round(value * 100 / row["matches"], 6)
    return [{"label": query.team, "value": value, "matches": row["matches"],
        "known_matches": row["matches"], "data_status": "available" if row["matches"] else "insufficient_data"}]


def execute_query(query: Query) -> dict:
    scope = catalog()
    result = {"status": "ok", "metric": query.metric, "unit": UNITS.get(query.metric),
        "scope": {"competition": scope["competition"], "team": query.team, "season": query.season, "venue": query.venue},
        "rows": [], "notice": scope["notice"], "sql": []}
    if query.operation in {"clarification", "unsupported"}:
        return result | {"status": query.operation, "message": query.message}
    if not data_available():
        return result | {"status": "data_required", "message": "Import the authorized season dataset to calculate statistics. See the setup guide."}
    with closing(_database()) as db:
        if query.operation == "team_metric":
            result["rows"] = _team_rows(db, query, result["sql"])
        else:
            result["rows"] = [row for player in query.players for row in _player_rows(db, query, player, result["sql"])]
    return result
