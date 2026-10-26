"""Live Qwen inference and explicit replay of real, recorded model outputs."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
from threading import Lock
from time import perf_counter

from onthedot import ollama_backend
from onthedot.prompt import DATASET_ID, PROMPT_SHA256, SYSTEM_PROMPT


MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
MAX_TOKENS = 400
DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
TRACE_PATH = DATA_DIRECTORY / "traces.json"

_inference_lock = Lock()
_runtime = None


def _backend() -> str:
    return os.environ.get("ONTHEDOT_BACKEND", "mlx").strip().lower()


def setup_hint() -> str:
    if _backend() == "ollama":
        return ollama_backend.setup_hint()
    if _backend() != "mlx":
        return "Set ONTHEDOT_BACKEND to mlx or ollama, then restart the demo."
    return (
        "Live MLX inference needs Apple Silicon and the mlx extra. "
        "Run uv sync --extra mlx, use local Ollama, or choose a recorded example."
    )


def live_model_info() -> dict:
    """Describe the selected live backend without importing GPU libraries."""
    if _backend() == "ollama":
        return ollama_backend.describe_model() or {
            "model_id": ollama_backend.MODEL_ID,
            "model_revision": "local-unverified",
            "runtime": "Ollama",
        }
    revision = "local-unverified" if os.environ.get("ONTHEDOT_MODEL_PATH") else MODEL_REVISION
    if _runtime is not None:
        revision = _runtime[4]
    return {"model_id": MODEL_ID, "model_revision": revision, "runtime": "MLX"}


def live_available() -> bool:
    """Check the selected optional runtime without loading model weights."""
    if _backend() == "ollama":
        return ollama_backend.describe_model() is not None
    if _backend() != "mlx":
        return False
    return (
        platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"}
        and find_spec("mlx_lm") is not None
    )


def _recordings() -> dict:
    try:
        bundle = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Recorded Qwen examples are unavailable or unreadable.") from exc
    if (
        bundle.get("model_id") != MODEL_ID
        or bundle.get("model_revision") != MODEL_REVISION
        or bundle.get("prompt_sha256") != PROMPT_SHA256
        or bundle.get("source") != "live_qwen_inference"
        or bundle.get("dataset_id") != DATASET_ID
        or not isinstance(bundle.get("traces"), list)
    ):
        raise RuntimeError("Recorded examples do not match the current model, prompt, and dataset.")
    return bundle


def examples() -> list[dict]:
    return [
        {"id": trace["id"], "question": trace["question"]}
        for trace in _recordings()["traces"]
    ]


def _load_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    if not live_available():
        raise RuntimeError(setup_hint())

    # import and load only when live inference is explicitly requested
    try:
        from mlx_lm import generate as mlx_generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError(
            "The live Qwen runtime could not access its libraries or Metal GPU. "
            "Run locally on Apple Silicon with the mlx extra, or use recorded mode."
        ) from exc

    local_path = os.environ.get("ONTHEDOT_MODEL_PATH")
    if local_path:
        model_path = Path(local_path).expanduser()
        if not model_path.is_dir():
            raise RuntimeError("ONTHEDOT_MODEL_PATH must name a local model directory.")
        try:
            model, tokenizer = load(str(model_path))
        except (OSError, ValueError) as exc:
            raise RuntimeError("The local Qwen model could not load. Check ONTHEDOT_MODEL_PATH.") from exc
        # a local directory name cannot establish the identity of its weights
        revision = "local-unverified"
    else:
        try:
            model, tokenizer = load(MODEL_ID, revision=MODEL_REVISION)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "The pinned Qwen model could not load. Download the model or set ONTHEDOT_MODEL_PATH."
            ) from exc
        revision = MODEL_REVISION
    _runtime = model, tokenizer, mlx_generate, make_sampler, revision
    return _runtime


def generate(question: str, mode: str = "recorded") -> dict:
    """Return raw output for independent schema validation and analytics."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a football question.")
    if len(question) > 1000:
        raise ValueError("Questions must contain at most 1000 characters.")
    if mode not in {"live", "recorded"}:
        raise ValueError("Choose live or recorded mode.")

    if mode == "recorded":
        # replay is deliberately exact and never substitutes a rules-based parser
        for trace in _recordings()["traces"]:
            if trace["question"] == question:
                return {
                    "mode": "recorded",
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "raw_text": trace["raw_text"],
                    "inference_ms": trace["inference_ms"],
                    "trace_id": trace["id"],
                }
        raise ValueError(
            "Recorded mode only replays the exact saved example questions. "
            "Choose an example or switch to live Qwen for your own question."
        )

    # one lock covers both lazy loading and generation to avoid shared GPU races
    with _inference_lock:
        if _backend() == "ollama":
            return ollama_backend.generate(question, SYSTEM_PROMPT, MAX_TOKENS)
        if _backend() != "mlx":
            raise RuntimeError(setup_hint())
        model, tokenizer, mlx_generate, make_sampler, revision = _load_runtime()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            started = perf_counter()
            output = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                sampler=make_sampler(temp=0.0),
                verbose=False,
            )
        except (ImportError, OSError, ValueError) as exc:
            raise RuntimeError(
                "Live Qwen generation failed. Check the local model and Metal runtime, "
                "or use a recorded example."
            ) from exc
        inference_ms = round((perf_counter() - started) * 1000, 2)
    return {
        "mode": "live",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "runtime": "MLX",
        "raw_text": output,
        "inference_ms": inference_ms,
    }
