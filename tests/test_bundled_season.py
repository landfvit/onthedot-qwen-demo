from collections import Counter
from hashlib import sha256
import json

from fastapi.testclient import TestClient

import app as server
from onthedot import dataset, parser


def test_default_export_is_bundled_and_matches_the_recorded_snapshot(monkeypatch):
    monkeypatch.delenv("ONTHEDOT_DATA_PATH", raising=False)
    path = dataset.dataset_path()
    assert path == dataset.ROOT / "data" / "season.json"
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == dataset.dataset_id()
    season = json.loads(raw)
    fixtures = season["fixtures"]
    assert len(fixtures) == len({fixture["id"] for fixture in fixtures}) == 380
    teams = Counter(team for fixture in fixtures for team in [fixture["home"], fixture["away"]])
    assert len(teams) == 20 and set(teams.values()) == {38}
    assert sum(len(fixture["appearances"]) for fixture in fixtures) == 15188
    assert {fixture["season"] for fixture in fixtures} == {"2024-25"}


def test_bundled_preview_returns_real_statistics_and_sql_without_a_model(monkeypatch):
    monkeypatch.delenv("ONTHEDOT_DATA_PATH", raising=False)
    dataset.load_dataset.cache_clear()

    def unexpected_model_load():
        raise AssertionError("Recorded preview must not load a language model")

    monkeypatch.setattr(parser, "_load_runtime", unexpected_model_load)
    try:
        with TestClient(server.create_app()) as client:
            assert client.get("/api/meta").json()["data_available"]
            result = client.post("/api/ask", json={
                "question": "Compare Haaland and Salah by assists in 2024-25.",
                "mode": "recorded",
            }).json()
        assert result["validation"]["ok"]
        assert [(row["label"], row["value"], row["matches"]) for row in result["result"]["rows"]] == [
            ("Erling Braut Haaland", 3, 31),
            ("Mohamed Salah Hamed Mahrous Ghaly", 18, 38),
        ]
        assert len(result["sql"]) == 2
        assert result["sql"][0]["parameters"] == ["Erling Braut Haaland", "2024-25"]
        assert result["sql"][1]["parameters"] == ["Mohamed Salah Hamed Mahrous Ghaly", "2024-25"]
    finally:
        dataset.load_dataset.cache_clear()
