"""Execute or stub-dry-run the frozen 48-request VLM decision smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_smoke_backend import FrozenQwenBackend, StubSmokeBackend
from ruleshift.evaluator.vlm_smoke import (
    MAX_NEW_TOKENS,
    PARSER_VERSION,
    baseline_summary,
    build_plan,
    control_summary,
    make_model_request,
    manifest_record,
    score_records,
    sha256_text,
    verify_freeze,
    write_json,
)
from ruleshift.evaluator.vlm_decision_visual import visual_digest
from ruleshift.evaluator.vlm_smoke_parser import parse_action
from ruleshift.evaluator.vlm_smoke_checkpoint import verify_checkpoint_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = REPO_ROOT / "results" / "vlm_decision_smoke" / "frozen"
DEFAULT_OUT = REPO_ROOT / "results" / "vlm_decision_smoke"
DEFAULT_SNAPSHOT = Path(
    r"C:\Users\Lenovo\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct"
    r"\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3"
)
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "VLM_DECISION_SMOKE_PROTOCOL.md"


def source_paths() -> list[Path]:
    return [
        REPO_ROOT / "src" / "ruleshift" / "evaluator" / "vlm_decision.py",
        REPO_ROOT / "src" / "ruleshift" / "evaluator" / "vlm_decision_visual.py",
        REPO_ROOT / "src" / "ruleshift" / "evaluator" / "vlm_smoke.py",
        REPO_ROOT / "src" / "ruleshift" / "evaluator" / "vlm_smoke_parser.py",
        REPO_ROOT / "src" / "ruleshift" / "evaluator" / "vlm_smoke_checkpoint.py",
        REPO_ROOT / "src" / "ruleshift" / "agents" / "vlm_backend.py",
        REPO_ROOT / "src" / "ruleshift" / "agents" / "vlm_smoke_backend.py",
        REPO_ROOT / "scripts" / "prepare_vlm_decision_smoke.py",
        REPO_ROOT / "scripts" / "preflight_vlm_decision_smoke.py",
        REPO_ROOT / "scripts" / "run_vlm_decision_smoke.py",
        DEFAULT_PROTOCOL,
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def environment_metadata() -> dict[str, Any]:
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_total_bytes": (
                    int(torch.cuda.get_device_properties(0).total_memory)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - environment report
        metadata["torch_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def build_request_rows(plan) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    request_objects: dict[str, Any] = {}
    for item in plan.requests:
        request = make_model_request(item)
        record = manifest_record(item)
        row = {
            "request_id": item.request_id,
            "arm": item.arm,
            "kind": item.kind,
            "context_id": item.context_id,
            "base_context_id": item.base_context_id,
            "diagnostic_role": item.diagnostic_role,
            "input_mode": request.input_mode,
            "prompt_sha256": record["prompt_sha256"],
            "public_payload_sha256": record["public_payload_sha256"],
            "image_pixel_digest": record["image_pixel_digest"],
            "image_png_sha256": record["image_png_sha256"],
            "feasible_actions": list(request.available_actions),
            "candidate_rules": list(request.candidate_rules),
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "decoding_seed": request.decoding_seed,
            "request_fingerprint": request.fingerprint(),
            "prompt": request.prompt,
        }
        rows.append(row)
        request_objects[item.request_id] = request
    return rows, request_objects


def _parsed_rows_for(parsed_path: Path, request_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(request_ids)
    rows: list[dict[str, Any]] = []
    for line in parsed_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["request_id"] in wanted:
            rows.append(row)
    if len(rows) != len(wanted):
        raise ValueError(f"expected {len(wanted)} parsed rows, found {len(rows)}")
    return rows


def run_plan(
    *,
    frozen_dir: Path,
    out_dir: Path,
    snapshot: Path,
    mode: str,
) -> int:
    actual = mode == "execute"
    out_dir = Path(out_dir)
    target = out_dir / ("real" if actual else "dry_run")
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {target}")
    target.mkdir(parents=True, exist_ok=True)

    freeze = verify_freeze(frozen_dir, source_paths=[path.resolve() for path in source_paths()])
    manifest = json.loads((frozen_dir / "request_manifest.json").read_text(encoding="utf-8"))
    plan = build_plan()
    expected_manifest = [manifest_record(item) for item in plan.requests]
    if manifest != expected_manifest:
        raise ValueError("frozen request manifest does not match current prompts/images")
    requests_rows, request_objects = build_request_rows(plan)
    for row in requests_rows:
        append_jsonl(target / "requests.jsonl", row)

    checkpoint = json.loads(
        (frozen_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_started = time.perf_counter()
    verify_checkpoint_manifest(checkpoint, hash_files=actual)
    checkpoint_verify_seconds = time.perf_counter() - checkpoint_started

    state: dict[str, Any] = {
        "mode": mode,
        "started_utc": utc_now(),
        "model_calls": 0,
        "stub_calls": 0,
        "attempted_request_ids": [],
        "completed_request_ids": [],
        "in_flight": None,
        "failed": None,
        "stopped_reason": None,
    }
    write_json(target / "execution_state.json", state)
    write_json(
        target / "run_start.json",
        {
            "mode": mode,
            "started_utc": state["started_utc"],
            "frozen_dir": str(frozen_dir),
            "freeze_sha256": (frozen_dir / "freeze.sha256").read_text(encoding="utf-8").strip(),
            "request_manifest_sha256": freeze["request_manifest_sha256"],
            "planned_requests": len(requests_rows),
            "checkpoint_verify_seconds": checkpoint_verify_seconds,
            "checkpoint_aggregate_sha256": checkpoint["aggregate_sha256"],
            "environment": environment_metadata(),
        },
    )

    summaries: dict[str, Any] = {}
    response_rows: list[dict[str, Any]] = []
    parsed_by_id: dict[str, dict[str, Any]] = {}
    stop_reason: str | None = None
    failure: dict[str, Any] | None = None

    if actual:
        backend = FrozenQwenBackend(snapshot, device="cuda", dtype="bfloat16")
        print("loading frozen Qwen2.5-VL-3B-Instruct on CUDA...", flush=True)
        load_started = time.perf_counter()
        try:
            backend.load()
        except Exception as exc:
            load_seconds = time.perf_counter() - load_started
            failure = {
                "stage": "model_load",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            stop_reason = "model_load_exception"
            state["failed"] = failure
            state["stopped_reason"] = stop_reason
            write_json(target / "execution_state.json", state)
            write_json(target / "failure.json", failure)
            write_json(
                target / "model_load.json",
                {
                    "loaded": False,
                    "wall_seconds": load_seconds,
                    "error": failure["error"],
                    "traceback": failure["traceback"],
                    "model_path": str(snapshot),
                    "model_id": backend.model_id,
                    "revision": backend.revision,
                },
            )
        else:
            load_seconds = time.perf_counter() - load_started
            write_json(
                target / "model_load.json",
                {**backend.load_metadata, "wall_seconds": load_seconds},
            )
    else:
        backend = StubSmokeBackend()
        load_seconds = 0.0
        write_json(target / "model_load.json", {"backend": "STUB_SYNTHETIC", "loaded": False})

    cache = ResponseCache(target / "cache.jsonl")
    if (target / "cache.jsonl").exists() and (target / "cache.jsonl").stat().st_size > 0:
        raise RuntimeError("fresh cache file unexpectedly non-empty")

    group_specs = [
        ("A1", [item for item in plan.base_requests if item.arm == "A1"], "base"),
        ("A2", [item for item in plan.base_requests if item.arm == "A2"], "base"),
        ("A3", [item for item in plan.base_requests if item.arm == "A3"], "base"),
        ("RGB_NUISANCE", list(plan.nuisance_requests), "nuisance"),
        ("RGB_DIAGNOSTIC", list(plan.diagnostic_requests), "diagnostic"),
    ]

    try:
        for group_name, specs, kind in group_specs:
            if stop_reason is not None:
                break
            for item in specs:
                if actual and state["model_calls"] >= 48:
                    raise RuntimeError("request cap reached before the planned request")
                request = request_objects[item.request_id]
                expected = next(row for row in requests_rows if row["request_id"] == item.request_id)
                if sha256_text(request.prompt) != expected["prompt_sha256"]:
                    raise ValueError(f"prompt changed in flight: {item.request_id}")
                observed_image_digest = (
                    visual_digest(request.image) if request.image is not None else None
                )
                if observed_image_digest != expected["image_pixel_digest"]:
                    raise ValueError(f"image digest changed in flight: {item.request_id}")
                if cache.get(request) is not None:
                    raise RuntimeError(f"unexpected cache hit in fresh run: {item.request_id}")

                state["attempted_request_ids"].append(item.request_id)
                state["in_flight"] = {
                    "request_id": item.request_id,
                    "attempt": 1,
                    "started_utc": utc_now(),
                }
                if actual:
                    state["model_calls"] += 1
                else:
                    state["stub_calls"] += 1
                write_json(target / "execution_state.json", state)
                call_started = time.perf_counter()
                try:
                    response = backend.generate(request)
                except Exception as exc:
                    error_text = traceback.format_exc()
                    append_jsonl(
                        target / "responses.jsonl",
                        {
                            "request_id": item.request_id,
                            "arm": item.arm,
                            "context_id": item.context_id,
                            "attempt": 1,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": error_text,
                            "input_mode": request.input_mode,
                            "image_pixel_digest": expected["image_pixel_digest"],
                            "vision_tensor_sha256": None,
                            "image_seen": False,
                        },
                    )
                    state["failed"] = {
                        "request_id": item.request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": error_text,
                    }
                    state["in_flight"] = None
                    state["stopped_reason"] = "model_request_exception"
                    write_json(target / "execution_state.json", state)
                    failure = state["failed"]
                    raise
                call_seconds = time.perf_counter() - call_started
                metadata = dict(getattr(backend, "last_metadata", {}))
                expected_rgb = request.input_mode == "rgb"
                image_seen = bool(metadata.get("image_seen", False))
                vision_sha = metadata.get("vision_tensor_sha256")
                if expected_rgb and not (image_seen and vision_sha):
                    raise RuntimeError(f"RGB request did not forward an image tensor: {item.request_id}")
                if not expected_rgb and (image_seen or vision_sha is not None):
                    raise RuntimeError(f"text request unexpectedly forwarded an image tensor: {item.request_id}")
                if int(metadata.get("input_tokens", response.input_tokens)) != int(response.input_tokens):
                    raise RuntimeError(f"input token accounting mismatch: {item.request_id}")

                parsed = parse_action(response.text, request.available_actions)
                parsed_row = {
                    "request_id": item.request_id,
                    "arm": item.arm,
                    "context_id": item.context_id,
                    "selected_action": parsed.selected_action,
                    "valid": parsed.valid,
                    "fallback": parsed.fallback,
                    "errors": list(parsed.errors),
                }
                parsed_by_id[item.request_id] = parsed_row
                append_jsonl(target / "parsed.jsonl", parsed_row)
                response_row = {
                    "request_id": item.request_id,
                    "arm": item.arm,
                    "context_id": item.context_id,
                    "attempt": 1,
                    "status": "ok",
                    "raw_text": response.text,
                    "input_tokens": int(response.input_tokens),
                    "output_tokens": int(response.output_tokens),
                    "latency_ms": float(response.latency_ms),
                    "wall_seconds": call_seconds,
                    "backend_id": response.backend_id,
                    "model_id": response.model_id,
                    "revision": checkpoint["revision"],
                    "prompt_sha256": expected["prompt_sha256"],
                    "request_fingerprint": expected["request_fingerprint"],
                    "image_pixel_digest": expected["image_pixel_digest"],
                    "vision_tensor_sha256": vision_sha,
                    "vision_tensor_shape": metadata.get("vision_tensor_shape"),
                    "image_grid_thw": metadata.get("image_grid_thw"),
                    "image_seen": image_seen,
                    "input_ids_sha256": metadata.get("input_ids_sha256"),
                    "gpu_time_s": float(metadata.get("gpu_time_s", 0.0)),
                    "cuda_peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
                    "cuda_peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
                    "parsed": parsed.as_dict(),
                }
                append_jsonl(target / "responses.jsonl", response_row)
                response_rows.append(response_row)
                cache.put(request, response)
                state["completed_request_ids"].append(item.request_id)
                state["in_flight"] = None
                write_json(target / "execution_state.json", state)
                print(
                    f"{item.request_id}: {parsed.selected_action} valid={parsed.valid} "
                    f"image={image_seen} gpu={response_row['gpu_time_s']:.3f}s",
                    flush=True,
                )

            parsed_rows = _parsed_rows_for(
                target / "parsed.jsonl", [item.request_id for item in specs]
            )
            summary = score_records(
                parsed_rows, plan, kind=kind, arm=group_name if kind == "base" else None
            )
            summaries[group_name] = summary
            write_json(target / "arm_summaries.json", summaries)
            print(
                f"{group_name}: valid={summary['valid_count']}/{summary['n']} "
                f"gate={summary['gate_pass']}",
                flush=True,
            )
            if actual and group_name in ("A1", "A2", "A3") and not summary["gate_pass"]:
                stop_reason = f"{group_name}_feasibility_gate_failed"
                state["stopped_reason"] = stop_reason
                write_json(target / "execution_state.json", state)
                break
            if actual and group_name == "A3":
                a3_ids = [item.request_id for item in specs]
                a3_responses = [row for row in response_rows if row["request_id"] in set(a3_ids)]
                technical = {
                    "all_12_dispatched": len(a3_responses) == 12,
                    "all_images_forwarded": len(a3_responses) == 12
                    and all(row["image_seen"] and row["vision_tensor_sha256"] for row in a3_responses),
                    "at_least_one_valid": any(parsed_by_id[rid]["valid"] for rid in a3_ids),
                }
                summaries["A3_technical"] = technical
                write_json(target / "arm_summaries.json", summaries)
                if not all(technical.values()):
                    stop_reason = "A3_technical_precondition_failed"
                    state["stopped_reason"] = stop_reason
                    write_json(target / "execution_state.json", state)
                    break
    except Exception as exc:
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        state["failed"] = failure
        state["in_flight"] = None
        if stop_reason is None:
            stop_reason = "exception_during_run"
        state["stopped_reason"] = stop_reason
        write_json(target / "execution_state.json", state)
        write_json(target / "failure.json", failure)

    if summary_controls_ready := all(
        name in summaries for name in ("A3", "RGB_NUISANCE", "RGB_DIAGNOSTIC")
    ):
        summaries["controls"] = control_summary(
            plan,
            summaries["A3"],
            summaries["RGB_NUISANCE"],
            summaries["RGB_DIAGNOSTIC"],
        )
    summaries["baselines"] = baseline_summary(plan)
    write_json(target / "arm_summaries.json", summaries)

    gpu_time = sum(float(row.get("gpu_time_s", 0.0)) for row in response_rows)
    final = {
        "run_status": (
            "dry_run_complete"
            if not actual and stop_reason is None and len(response_rows) == 48
            else "complete"
            if actual and stop_reason is None and len(response_rows) == 48
            else "stopped"
        ),
        "mode": mode,
        "stopped_reason": stop_reason,
        "failure": failure,
        "planned_requests": 48,
        "attempted_requests": len(state["attempted_request_ids"]),
        "completed_requests": len(state["completed_request_ids"]),
        "model_calls": state["model_calls"],
        "stub_calls": state["stub_calls"],
        "request_cap": 48,
        "retries": 0,
        "gpu_time_s": gpu_time,
        "model_load_seconds": load_seconds,
        "checkpoint_verify_seconds": checkpoint_verify_seconds,
        "backend_id": getattr(backend, "backend_id", None),
        "model_id": getattr(backend, "model_id", None),
        "revision": checkpoint["revision"],
        "arm_summaries": summaries,
        "environment": environment_metadata(),
        "finished_utc": utc_now(),
    }
    write_json(target / "summary.json", final)
    write_json(
        target / "request_accounting.json",
        {
            "planned": 48,
            "attempted": len(state["attempted_request_ids"]),
            "completed": len(state["completed_request_ids"]),
            "model_calls": state["model_calls"],
            "stub_calls": state["stub_calls"],
            "cap": 48,
            "retries": 0,
            "stopped_reason": stop_reason,
        },
    )
    checksums = {}
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums[str(path.relative_to(target))] = file_sha256(path)
    write_json(target / "checksums.json", checksums)
    print(f"run finished: {final['run_status']} at {target}", flush=True)
    return 0 if stop_reason is None and failure is None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "execute"), required=True)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args(argv)
    try:
        return run_plan(
            frozen_dir=args.frozen_dir.resolve(),
            out_dir=args.out.resolve(),
            snapshot=args.snapshot.resolve(),
            mode=args.mode,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
