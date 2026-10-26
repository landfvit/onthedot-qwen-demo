"""The bundled season snapshot and its matching query catalog."""

from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "catalog.json"


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def active_catalog() -> dict:
    return manifest()["catalog"]


def dataset_id() -> str:
    return manifest()["dataset_id"]


def dataset_path() -> Path:
    return Path(os.environ.get("ONTHEDOT_DATA_PATH", ROOT / "data" / "season.json")).expanduser()


def data_available() -> bool:
    return dataset_path().is_file()


@lru_cache(maxsize=1)
def load_dataset() -> dict:
    try:
        raw = dataset_path().read_bytes()
    except OSError as error:
        raise RuntimeError("Load the authorized Premier League season export before running analytics. See the setup guide.") from error
    # the model catalog and analytics must describe the same season snapshot
    if sha256(raw).hexdigest() != dataset_id():
        raise RuntimeError("The local season file does not match data/catalog.json. Export both together and restart.")
    return json.loads(raw)
