"""Illustrative parser smoke cases, not a held-out accuracy benchmark."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from onthedot.parser import MAX_TOKENS, MODEL_ID, MODEL_REVISION, generate, live_model_info
from onthedot.prompt import DATASET_ID, PROMPT_SHA256
from onthedot.schema import parse_query, validate_grounding


def semantic_query(query) -> dict:
    # follow-up wording can vary without changing the intended query
    value = query.model_dump(mode="json")
    value.pop("message", None)
    return value


def evaluate(mode: str, cases_path: Path) -> tuple[dict, list[dict]]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results, traces = [], []
    identity = live_model_info() if mode == "live" else {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    identities = set()
    for case in cases:
        result = {
            "id": case["id"], "question": case["question"],
            "passed": False, "validation_ok": False,
        }
        try:
            expected = parse_query(json.dumps(case["expected"]))
            result["expected"] = semantic_query(expected)
            generation = generate(case["question"], mode=mode)
            identity = {"model_id": generation["model_id"], "model_revision": generation["model_revision"]}
            identities.add((identity["model_id"], identity["model_revision"]))
            traces.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "raw_text": generation["raw_text"],
                    "inference_ms": generation["inference_ms"],
                    "model_id": generation["model_id"],
                    "model_revision": generation["model_revision"],
                }
            )
            actual = parse_query(generation["raw_text"])
            validate_grounding(actual, case["question"])
            result["validation_ok"] = True
            result["actual"] = semantic_query(actual)
            result["passed"] = result["actual"] == result["expected"]
            result["inference_ms"] = generation["inference_ms"]
        except (ValueError, RuntimeError) as exc:
            result["error"] = str(exc)
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {case['id']}", flush=True)

    report = {
        "description": "Illustrative season-query smoke cases, not a held-out benchmark.",
        "mode": mode,
        "model_id": identity["model_id"] if len(identities) <= 1 else "multiple",
        "model_revision": identity["model_revision"] if len(identities) <= 1 else "multiple",
        "prompt_sha256": PROMPT_SHA256,
        "dataset_id": DATASET_ID,
        "passed": sum(row["passed"] for row in results),
        "validation_passed": sum(row["validation_ok"] for row in results),
        "validation_blocked": sum(not row["validation_ok"] for row in results),
        "total": len(results),
        "results": results,
    }
    return report, traces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("recorded", "live"), default="recorded")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "cases.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.record and args.mode != "live":
        parser.error("--record requires --mode live, so replay can never masquerade as inference")
    if args.record:
        identity = live_model_info()
        if identity["model_id"] != MODEL_ID or identity["model_revision"] != MODEL_REVISION:
            parser.error("recording requires the pinned MLX model, not Ollama or an unverified local model")
    report, traces = evaluate(args.mode, args.cases)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.record:
        if len(traces) != report["total"]:
            parser.error("some live calls failed, so the existing recordings were left unchanged")
        if any(trace["model_id"] != MODEL_ID or trace["model_revision"] != MODEL_REVISION for trace in traces):
            parser.error("recording requires the pinned model snapshot, not an unverified local model")
        for trace in traces:
            trace.pop("model_id")
            trace.pop("model_revision")
        bundle = {
            "source": "live_qwen_inference",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt_sha256": PROMPT_SHA256,
            "dataset_id": DATASET_ID,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "runtime": "mlx-lm 0.31.3 on Apple Silicon",
            "generation": {"temperature": 0.0, "max_tokens": MAX_TOKENS},
            "timing": "milliseconds for prompt processing and generation, excluding initial model load",
            "traces": traces,
        }
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"{report['passed']}/{report['total']} illustrative cases passed")
    print(f"{report['validation_passed']} outputs validated, {report['validation_blocked']} blocked before analytics")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
