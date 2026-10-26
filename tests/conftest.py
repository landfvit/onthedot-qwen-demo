import json
from pathlib import Path

import pytest

from onthedot import analytics, schema


SYNTHETIC_PATH = Path(__file__).parent / "data" / "fixtures.json"


@pytest.fixture
def synthetic_catalog(monkeypatch):
    # small controlled inputs isolate arithmetic and schema edge cases from source data
    monkeypatch.setattr(schema, "PLAYERS", {
        "Alex Mercer": ["Alex Mercer", "Mercer"], "Theo Blake": ["Theo Blake", "Blake"],
        "Noah Silva": ["Noah Silva", "Silva"], "Liam Grant": ["Liam Grant", "Grant"],
    })
    monkeypatch.setattr(schema, "TEAMS", {"Northbridge FC": ["Northbridge FC", "Northbridge"], "Riverside FC": ["Riverside FC", "Riverside"]})
    monkeypatch.setattr(schema, "SEASONS", ["2024-25", "2025-26"])
    monkeypatch.setattr(schema, "COMPETITION", "Demo League")
    monkeypatch.setattr(schema, "DATA_NOTICE", "Synthetic demo data")


@pytest.fixture
def synthetic_dataset(monkeypatch, synthetic_catalog):
    monkeypatch.setattr(analytics, "load_dataset", lambda: json.loads(SYNTHETIC_PATH.read_text()))
    monkeypatch.setattr(analytics, "data_available", lambda: True)
