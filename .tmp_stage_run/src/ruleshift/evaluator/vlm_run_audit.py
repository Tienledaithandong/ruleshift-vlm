"""Resolve Kaggle run directories and audit raw model-call accounting.

The historical kernel copied the runner output one directory deeper than the
artifact collector searched.  This module accepts both layouts, prefers the
nested layout, and checks that summary counters agree with the raw JSONL logs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_run_dir(output_root: Path) -> Path:
    """Find real run output in either historical or current Kaggle layout."""

    root = Path(output_root)
    candidates = (root / "real" / "real", root / "real", root)
    for candidate in candidates:
        if candidate.is_dir() and any(
            (candidate / name).is_file()
            for name in ("request_accounting.json", "summary.json", "responses.jsonl")
        ):
            return candidate
    raise FileNotFoundError(f"no run artifacts found under {root}")


def summarise_run(output_root: Path) -> dict[str, Any]:
    """Return the run summary plus counters derived from raw request logs."""

    run_dir = resolve_run_dir(Path(output_root))
    response_rows = read_jsonl(run_dir / "responses.jsonl") if (run_dir / "responses.jsonl").is_file() else []
    parsed_rows = read_jsonl(run_dir / "parsed.jsonl") if (run_dir / "parsed.jsonl").is_file() else []
    request_rows = read_jsonl(run_dir / "requests.jsonl") if (run_dir / "requests.jsonl").is_file() else []
    accounting = read_json(run_dir / "request_accounting.json") if (run_dir / "request_accounting.json").is_file() else {}
    summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else {}
    raw = {
        "requests": len(request_rows),
        "responses": len(response_rows),
        "ok_responses": sum(row.get("status") == "ok" for row in response_rows),
        "error_responses": sum(row.get("status") == "error" for row in response_rows),
        "parsed": len(parsed_rows),
    }
    observed = {
        "planned": raw["requests"],
        "attempted": raw["responses"],
        "completed": raw["ok_responses"],
        "model_calls": raw["responses"],
        "parsed": raw["parsed"],
    }
    checks = {
        "planned_matches_requests": int(accounting.get("planned", -1)) == raw["requests"],
        "attempted_matches_responses": int(accounting.get("attempted", -1)) == raw["responses"],
        "completed_matches_ok_responses": int(accounting.get("completed", -1)) == raw["ok_responses"],
        "model_calls_matches_responses": int(accounting.get("model_calls", -1)) == raw["responses"],
        "summary_model_calls_matches": int(summary.get("model_calls", -1)) == raw["responses"] if summary else False,
        "parsed_matches_ok_responses": raw["parsed"] == raw["ok_responses"],
    }
    return {
        "run_dir": str(run_dir),
        "resolution": "nested_real_real" if run_dir == Path(output_root) / "real" / "real" else "real" if run_dir == Path(output_root) / "real" else "root",
        "summary": summary,
        "request_accounting": accounting,
        "raw_counts": raw,
        "observed_accounting": observed,
        "consistency": {"checks": checks, "ok": all(checks.values())},
    }
