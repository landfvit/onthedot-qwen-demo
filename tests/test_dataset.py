from hashlib import sha256
import json

import pytest

from onthedot import dataset


@pytest.fixture(autouse=True)
def reset_dataset_cache():
    dataset.load_dataset.cache_clear()
    yield
    dataset.load_dataset.cache_clear()


def test_missing_season_is_distinct_from_an_empty_season(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTHEDOT_DATA_PATH", str(tmp_path / "missing.json"))
    assert not dataset.data_available()
    with pytest.raises(RuntimeError, match="authorized Premier League"):
        dataset.load_dataset()


def test_catalog_hash_prevents_using_a_different_season_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "season.json"
    path.write_text('{"fixtures":[]}')
    monkeypatch.setenv("ONTHEDOT_DATA_PATH", str(path))
    monkeypatch.setattr(dataset, "dataset_id", lambda: "different")
    with pytest.raises(RuntimeError, match="does not match"):
        dataset.load_dataset()


def test_matching_export_is_read_without_a_database_or_provider(tmp_path, monkeypatch):
    raw = json.dumps({"fixtures": [{"id": 123}]}).encode()
    path = tmp_path / "season.json"
    path.write_bytes(raw)
    monkeypatch.setenv("ONTHEDOT_DATA_PATH", str(path))
    monkeypatch.setattr(dataset, "dataset_id", lambda: sha256(raw).hexdigest())
    assert dataset.data_available()
    assert dataset.load_dataset() == {"fixtures": [{"id": 123}]}
