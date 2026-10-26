from collections import defaultdict

from onthedot.dataset import active_catalog, manifest
from onthedot.schema import Query, validate_grounding


def test_catalog_describes_one_complete_pl_season():
    scope = manifest()
    catalog = active_catalog()
    assert catalog["season"] == "2024-25" and catalog["competition"] == "Premier League"
    assert scope["counts"]["fixtures"] == 380
    assert scope["counts"]["teams"] == len(catalog["teams"]) == 20
    assert scope["counts"]["players"] == len(catalog["players"]) == 685
    assert "fixtures" not in scope


def test_a_short_alias_cannot_identify_two_people():
    owners = defaultdict(set)
    for name, aliases in active_catalog()["players"].items():
        assert name in aliases
        for alias in aliases:
            owners[alias.casefold()].add(name)
    assert all(len(names) == 1 for names in owners.values())
    assert "palmer" not in owners and "a. onana" not in owners


def test_common_real_player_names_keep_the_requested_subject_and_scope():
    query = Query(operation="player_comparison", metric="assists", players=["Erling Haaland", "Mohamed Salah"], season="2024-25")
    validate_grounding(query, "Compare Erling Haaland and Mohamed Salah assists in 2024–25")
    assert query.players == ["Erling Braut Haaland", "Mohamed Salah Hamed Mahrous Ghaly"]
