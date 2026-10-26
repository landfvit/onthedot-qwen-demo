from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from onthedot.analytics import execute_query
from onthedot.dataset import data_available
from onthedot.parser import examples, generate, live_available, live_model_info, setup_hint
from onthedot.schema import catalog, parse_query, validate_grounding


ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["recorded", "live"] = "recorded"

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a question.")
        return value


def create_app():
    application = FastAPI(title="ONTHEDOT Qwen demo", docs_url=None, redoc_url=None, openapi_url=None)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @application.middleware("http")
    async def local_requests(request: Request, call_next):
        #questions are submitted from the local viewer, without opening cross-origin access
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"error": {"message": "Use the demo from its own localhost page."}}, status_code=403)
        return await call_next(request)

    @application.exception_handler(RequestValidationError)
    async def bad_request(_request, _error):
        return JSONResponse({"error": {"message": "Enter a question up to 1000 characters and choose recorded or live mode."}}, status_code=422)

    @application.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})

    @application.get("/api/meta")
    def meta():
        try:
            saved_examples = examples()
        except RuntimeError:
            saved_examples = []
        return {
            **live_model_info(),
            "live_available": live_available(),
            "data_available": data_available(),
            "examples": saved_examples,
            "catalog": catalog(),
        }

    @application.post("/api/ask")
    async def ask(body: AskRequest):
        started = perf_counter()
        try:
            generation = await asyncio.to_thread(generate, body.question, body.mode)
        except ValueError as error:
            return JSONResponse({"error": {"message": str(error)}}, status_code=422)
        except RuntimeError:
            logger.exception("Local Qwen inference is unavailable")
            message = setup_hint() if body.mode == "live" else "Recorded examples are unavailable. Check the data files in this checkout."
            return JSONResponse({"error": {"message": message}}, status_code=503)

        response = {
            "question": body.question,
            "generation": generation,
            "query": None,
            "validation": {"ok": False, "error": None},
            "result": None,
            "sql": [],
            "timing": {"validation_ms": 0.0, "analytics_ms": 0.0, "request_ms": 0.0},
        }
        validated_at = perf_counter()
        try:
            query = parse_query(generation["raw_text"])
            validate_grounding(query, body.question)
        except ValueError as error:
            response["validation"]["error"] = str(error)
        else:
            response["query"] = query.model_dump(mode="json")
            response["validation"] = {"ok": True, "error": None}
        response["timing"]["validation_ms"] = round((perf_counter() - validated_at) * 1000, 3)

        #only validated intent reaches the fixed analytics implementation
        if response["validation"]["ok"]:
            computed_at = perf_counter()
            try:
                response["result"] = execute_query(query)
            except RuntimeError:
                logger.exception("Local season data is unavailable")
                return JSONResponse({"error": {"message": "The season dataset is missing or does not match its catalog. Check the export and restart."}}, status_code=503)
            response["sql"] = response["result"].pop("sql", [])
            response["timing"]["analytics_ms"] = round((perf_counter() - computed_at) * 1000, 3)
        response["timing"]["request_ms"] = round((perf_counter() - started) * 1000, 3)
        return response

    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return application


app = create_app()
