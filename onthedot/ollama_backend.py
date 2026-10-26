"""Optional Qwen inference through an explicitly selected local Ollama server."""

from http.client import HTTPException
import json
import re
from time import perf_counter
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


BASE_URL = "http://127.0.0.1:11434"
MODEL_ID = "qwen3:4b-instruct-2507-q4_K_M"
TAGS_TIMEOUT = 1.0
CHAT_TIMEOUT = 120.0
MAX_RESPONSE_BYTES = 128 * 1024
CONTEXT_TOKENS = 32768


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        # a local service must never forward a question to another endpoint
        return None


def setup_hint() -> str:
    return (
        "Start Ollama locally and run ollama pull " + MODEL_ID + ". "
        "Use ONTHEDOT_BACKEND=ollama, or select a recorded example."
    )


def _request_json(endpoint: str, *, payload: dict | None = None, timeout: float) -> dict:
    if endpoint not in {"/api/tags", "/api/chat"}:
        raise RuntimeError("Only the fixed local Ollama endpoints are supported.")
    request = Request(
        BASE_URL + endpoint,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    # disable environment proxies and redirects so questions remain on loopback
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("response too large")
        result = json.loads(raw)
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("invalid response")
    except (HTTPException, URLError, OSError, ValueError) as exc:
        raise RuntimeError("Local Ollama is unavailable or returned an invalid response. " + setup_hint()) from exc
    return result


def describe_model() -> dict | None:
    """Return only the exact configured tag, with its locally reported digest."""
    try:
        payload = _request_json("/api/tags", timeout=TAGS_TIMEOUT)
    except RuntimeError:
        return None
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    for model in models:
        if isinstance(model, dict) and model.get("name") == MODEL_ID:
            digest = model.get("digest")
            revision = (
                digest if isinstance(digest, str) and re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", digest)
                else "local-unverified"
            )
            return {"model_id": MODEL_ID, "model_revision": revision, "runtime": "Ollama"}
    return None


def generate(question: str, system_prompt: str, max_tokens: int) -> dict:
    info = describe_model() or {
        "model_id": MODEL_ID, "model_revision": "local-unverified", "runtime": "Ollama",
    }
    started = perf_counter()
    result = _request_json(
        "/api/chat",
        payload={
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens, "num_ctx": CONTEXT_TOKENS},
        },
        timeout=CHAT_TIMEOUT,
    )
    message = result.get("message")
    if (
        result.get("model") != MODEL_ID
        or result.get("done") is not True
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
    ):
        raise RuntimeError("Local Ollama returned an incomplete response or a different model. " + setup_hint())
    return {
        "mode": "live",
        **info,
        "raw_text": message["content"],
        "inference_ms": round((perf_counter() - started) * 1000, 2),
    }
