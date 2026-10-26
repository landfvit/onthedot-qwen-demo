"""Export one existing Premier League season without changing its source database."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unicodedata


class ExportError(ValueError):
    pass


TEAM_ALIASES = {
    "Manchester City": ["Man City"], "Manchester United": ["Man United", "Man Utd"],
    "Tottenham": ["Tottenham Hotspur", "Spurs"], "Newcastle": ["Newcastle United"],
    "West Ham": ["West Ham United"], "Wolves": ["Wolverhampton Wanderers"],
    "Ipswich": ["Ipswich Town"], "Leicester": ["Leicester City"],
    "Bournemouth": ["AFC Bournemouth"], "Brighton": ["Brighton and Hove Albion"],
}


def _plain(value):
    return " ".join(unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().split())


def _player_catalog(appearances, profiles):
    people, clubs, names = {}, defaultdict(set), defaultdict(set)
    for row in appearances:
        people[row["player_id"]] = row["player_name"]
        clubs[row["player_id"]].add(row["team"])
        names[row["player_id"]].add(row["player_name"])
    preferred = {}
    for row in profiles:
        player_id = row["player_id"]
        if player_id not in people:
            continue
        first, last = row["firstname"], row["lastname"]
        full = f"{first.strip()} {last.strip()}" if first and first.strip() and last and last.strip() else ""
        preferred.setdefault(player_id, full or people[player_id])
        names[player_id].update(filter(None, (full, row["name"], row["lastname"])))
        if full:
            names[player_id].add(f"{first.split()[0]} {last.strip()}")
            if "." in people[player_id].split()[0] and len(people[player_id].split()) > 1:
                names[player_id].add(f"{first.split()[0]} {people[player_id].split()[-1]}")
    base = {player_id: preferred.get(player_id, name) for player_id, name in people.items()}
    collisions = Counter(name.casefold() for name in base.values())
    labels = {
        player_id: (f"{name} ({' / '.join(sorted(clubs[player_id]))}, #{player_id})" if collisions[name.casefold()] > 1 else name)
        for player_id, name in base.items()
    }
    for player_id, name in people.items():
        names[player_id].update((labels[player_id], name.split()[-1]))
        if "Mohamed Salah" in names[player_id]:
            names[player_id].add("Mo Salah")
        names[player_id].update(_plain(alias) for alias in list(names[player_id]))
    owners = defaultdict(set)
    for player_id, aliases in names.items():
        for alias in aliases:
            if alias.strip():
                owners[alias.strip().casefold()].add(player_id)
    catalog = {}
    for player_id, label in sorted(labels.items(), key=lambda item: item[1]):
        aliases = {alias.strip() for alias in names[player_id] if alias.strip() and len(owners[alias.strip().casefold()]) == 1}
        catalog[label] = [label, *sorted(aliases - {label})]
    ambiguous = sorted(alias for alias, player_ids in owners.items() if len(player_ids) > 1)
    return labels, catalog, ambiguous


def build_dataset(season, fixtures, appearances, profiles):
    # validate a complete double round robin before exporting any rows.
    teams = {row[side] for row in fixtures for side in ("home", "away")}
    pairs = Counter((row["home"], row["away"]) for row in fixtures)
    expected_pairs = {(home, away) for home in teams for away in teams if home != away}
    ids = {row["id"] for row in fixtures}
    if len(fixtures) != 380 or len(ids) != 380 or len(teams) != 20 or set(pairs) != expected_pairs or any(count != 1 for count in pairs.values()):
        raise ExportError("The selected season must contain all 380 fixtures and 20 distinct teams.")
    if any(row["status"] != "finished" or row["home_score"] is None or row["away_score"] is None for row in fixtures):
        raise ExportError("Every exported fixture must be finished with both scores recorded.")
    if {row["gameweek"] for row in fixtures} != set(range(1, 39)):
        raise ExportError("The selected season must contain gameweeks 1 through 38.")
    if {row["match_id"] for row in appearances} != ids:
        raise ExportError("Every selected fixture must have existing player statistics.")
    if len({(row["match_id"], row["player_id"]) for row in appearances}) != len(appearances):
        raise ExportError("Duplicate player identities within a fixture cannot be exported.")
    labels, player_aliases, ambiguous = _player_catalog(appearances, profiles)
    season_label = f"{season['name']}-{str(int(season['name']) + 1)[-2:]}"
    catalog = {"players": player_aliases, "teams": {team: [team, *TEAM_ALIASES.get(team, [])] for team in sorted(teams)},
        "season": season_label, "competition": season["competition"]}
    grouped = defaultdict(list)
    for row in appearances:
        grouped[row["match_id"]].append({"player": labels[row["player_id"]], "player_id": row["player_id"],
            "team": row["team"], "team_id": row["team_id"], "minutes": row["minutes"],
            "goals": row["goals"], "assists": row["assists"], "shots": row["shots"],
            "source_payload_id": row["source_payload_id"]})
    exported = [{"id": row["id"], "date": row["kickoff"].astimezone(timezone.utc).date().isoformat(),
        "kickoff": row["kickoff"].astimezone(timezone.utc).isoformat(), "season": season_label, "gameweek": row["gameweek"],
        "home": row["home"], "home_id": row["home_id"], "away": row["away"], "away_id": row["away_id"],
        "home_goals": row["home_score"], "away_goals": row["away_score"], "appearances": grouped[row["id"]]}
        for row in sorted(fixtures, key=lambda item: (item["kickoff"], item["id"]))]
    eligible = [row for row in appearances if row["minutes"] is not None and row["minutes"] > 0]
    counts = {"fixtures": len(exported), "teams": len(teams), "players": len(labels), "appearances": len(appearances),
        "eligible_appearances": len(eligible), "eligible_players": len({row["player_id"] for row in eligible})}
    notice = f"Premier League {season_label}: 380 matches, 20 teams"
    return {"competition": season["competition"], "notice": notice, "catalog": catalog, "fixtures": exported,
        "provenance": {"source": "API-Football", "source_season": season["name"], "source_season_id": season["id"],
            "country": season["country"], "exported_at": datetime.now(timezone.utc).isoformat(), "read_only": True,
            "counts": counts, "date_range": {"start": exported[0]["date"], "end": exported[-1]["date"]},
            "eligible_missing_fields": {field: sum(row[field] is None for row in eligible) for field in ("goals", "assists", "shots")},
            "missing_minutes": sum(row["minutes"] is None for row in appearances),
            "excluded_ambiguous_aliases": ambiguous,
            "null_policy": "Source nulls remain null. No assumed zeros. All source rows are retained; analytics require positive recorded minutes.",
            "redistribution": "Public redistribution rights have not been established. Keep the full data local."}}


def read_season(database, year):
    # database access is optional and its session cannot modify the source.
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(database, row_factory=dict_row, connect_timeout=5,
            options="-c default_transaction_read_only=on -c statement_timeout=30000") as connection:
        if connection.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"] != "on":
            raise ExportError("The source connection must be read-only.")
        seasons = connection.execute("""SELECT s.id,s.name,c.name AS competition,c.country
            FROM seasons s JOIN competitions c ON c.id=s.competition_id
            WHERE s.name=%s AND c.name=%s AND c.country=%s""", (str(year), "Premier League", "England")).fetchall()
        if len(seasons) != 1:
            raise ExportError("Select exactly one existing Premier League season in England.")
        season = seasons[0]
        fixtures = connection.execute("""SELECT m.id,m.kickoff,m.status,m.gameweek,
            h.id AS home_id,h.name AS home,a.id AS away_id,a.name AS away,m.home_score,m.away_score
            FROM matches m JOIN teams h ON h.id=m.home_team_id JOIN teams a ON a.id=m.away_team_id
            WHERE m.season_id=%s ORDER BY m.kickoff,m.id""", (season["id"],)).fetchall()
        appearances = connection.execute("""SELECT ps.match_id,ps.player_id,p.name AS player_name,
            ps.team_id,t.name AS team,ps.minutes_played AS minutes,ps.goals,ps.assists,ps.shots_total AS shots,ps.source_payload_id
            FROM player_match_stats ps JOIN players p ON p.id=ps.player_id JOIN teams t ON t.id=ps.team_id
            JOIN matches m ON m.id=ps.match_id WHERE m.season_id=%s ORDER BY ps.match_id,ps.team_id,ps.player_id""", (season["id"],)).fetchall()
        profiles = connection.execute("""SELECT pe.player_id,r->'player'->>'name' AS name,
            r->'player'->>'firstname' AS firstname,r->'player'->>'lastname' AS lastname
            FROM api_football_payloads a CROSS JOIN LATERAL jsonb_array_elements(
                CASE WHEN jsonb_typeof(a.payload->'response')='array' THEN a.payload->'response' ELSE '[]'::jsonb END) r
            JOIN player_external_ids pe ON pe.external_id=r->'player'->>'id'
            JOIN data_providers dp ON dp.id=pe.provider_id AND dp.name=%s
            WHERE a.endpoint='players' AND a.season=%s AND a.successful
                AND EXISTS (SELECT 1 FROM player_match_stats ps JOIN matches m ON m.id=ps.match_id
                    WHERE ps.player_id=pe.player_id AND m.season_id=%s)
            ORDER BY a.retrieved_at DESC,a.id DESC""", ("API-Football", year, season["id"])).fetchall()
        return build_dataset(season, fixtures, appearances, profiles)


def export_season(database, year, destination, manifest_path):
    dataset = read_season(database, year)
    payload = (json.dumps(dataset, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    provenance = dataset["provenance"]
    manifest = {"dataset_id": digest, "catalog": dataset["catalog"], "notice": dataset["notice"],
        "source": {"name": "API-Football", "redistribution": "Public redistribution rights have not been established."},
        "counts": provenance["counts"], "date_range": provenance["date_range"]}
    for path, content in ((Path(destination), payload),
            (Path(manifest_path), (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="Explicit PostgreSQL DSN. It is never logged.")
    parser.add_argument("--season", type=int, default=2024, help="Provider starting year, for example 2024 means 2024-25.")
    parser.add_argument("--output", type=Path, default=Path(".local/real-season.json"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.json"))
    args = parser.parse_args(argv)
    try:
        manifest = export_season(args.database, args.season, args.output, args.catalog)
    except ImportError:
        print("Install the export extra with psycopg to read the local source database.", file=sys.stderr)
        return 1
    except ExportError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception:
        print("Read-only export failed. Check source availability and the selected season. Connection details are not logged.", file=sys.stderr)
        return 1
    print(f"Exported {manifest['counts']['fixtures']} fixtures and {manifest['counts']['appearances']} player rows.")
    print(f"Dataset SHA256: {manifest['dataset_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
