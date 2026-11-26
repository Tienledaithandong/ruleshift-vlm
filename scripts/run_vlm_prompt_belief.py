"""Execute the frozen prompt/belief diagnostic; no automatic retries."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ruleshift.agents.vlm_backend import ModelRequest, ModelResponse
from ruleshift.agents.vlm_smoke_backend import FrozenQwenBackend, StubSmokeBackend
from ruleshift.evaluator.vlm_prompt_belief import (
    ARMS,
    REQUEST_CAP,
    baseline_diagnostics,
    build_plan,
    make_model_request,
    manifest_record,
    paired_comparisons,
    score_arm,
    sha256_text,
    source_paths,
    verify_freeze,
)
from ruleshift.evaluator.vlm_smoke import assert_prompt_public
from ruleshift.evaluator.vlm_smoke_checkpoint import verify_checkpoint_manifest
from ruleshift.evaluator.vlm_smoke_parser import parse_action

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = REPO_ROOT / "results" / "vlm_prompt_belief_diagnostic" / "frozen"
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "VLM_PROMPT_BELIEF_DIAGNOSTIC_PROTOCOL.md"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
        handle.flush()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _snapshot_override(manifest: dict) -> dict:
    override = os.environ.get("RULESHIFT_SNAPSHOT_PATH", "").strip()
    if not override:
        return manifest
    updated = dict(manifest)
    updated["snapshot_path"] = override
    updated["snapshot_path_source"] = "RULESHIFT_SNAPSHOT_PATH"
    return updated


def _run_metadata(mode: str, backend: Any) -> dict[str, Any]:
    return {
        "experiment_id": "vlm-prompt-belief-diagnostic-v1",
        "mode": mode,
        "backend": getattr(backend, "backend_id", type(backend).__name__),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "cwd": str(Path.cwd()),
        "timestamp_utc": _now(),
        "request_cap": REQUEST_CAP,
        "automatic_retries": 0,
        "image_inputs": False,
    }


def _runtime_audit(backend: FrozenQwenBackend) -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0)
    parameter_dtypes = Counter(str(parameter.dtype) for parameter in backend._model.parameters())
    parameter_count = int(sum(parameter.numel() for parameter in backend._model.parameters()))
    try:
        autocast_dtype = str(torch.get_autocast_dtype("cuda"))
    except Exception:
        autocast_dtype = None
    try:
        allow_bf16_reduction = bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    except Exception:
        allow_bf16_reduction = None
    capability = (int(props.major), int(props.minor))
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": props.name,
        "capability": list(capability),
        "total_memory_bytes": int(props.total_memory),
        "parameter_count": parameter_count,
        "parameter_dtypes": dict(parameter_dtypes),
        "model_parameter_dtype": str(next(backend._model.parameters()).dtype),
        "model_config_torch_dtype": str(getattr(backend._model.config, "torch_dtype", None)),
        "attention_implementation": getattr(backend._model.config, "_attn_implementation", None),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "autocast_dtype": autocast_dtype,
        "allow_bf16_reduced_precision_reduction": allow_bf16_reduction,
        "bf16_supported_flag": bool(torch.cuda.is_bf16_supported()),
        "bf16_hardware_native": bool(capability[0] >= 8),
        "hardware_native_rule": "capability.major >= 8; T4 is not native BF16",
        "is_t4": "T4" in str(props.name),
        "compute_condition": "no explicit autocast; bfloat16 parameters; SDPA",
        "precision_changed": False,
        "load_metadata": dict(backend.load_metadata),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _request_record(item: Any, request: ModelRequest) -> dict[str, Any]:
    payload = dataclasses.asdict(request)
    payload.pop("image", None)
    return {
        "request_manifest": manifest_record(item),
        "model_request": payload,
        "request_fingerprint": request.fingerprint(),
        "prompt": request.prompt,
    }


def _load_frozen_manifest(frozen: Path) -> list[dict[str, Any]]:
    return json.loads((frozen / "request_manifest.json").read_text(encoding="utf-8"))


def _make_backend(args: argparse.Namespace) -> Any:
    if args.mode == "dry_run":
        if args.backend != "stub":
            raise ValueError("dry-run mode requires --backend stub")
        return StubSmokeBackend()
    if args.backend != "qwen":
        raise ValueError("execute mode requires --backend qwen")
    return FrozenQwenBackend(args.snapshot, device="cuda", dtype="bfloat16")


def run(frozen: Path, out: Path, *, mode: str, backend_name: str, snapshot: Path | None, protocol: Path) -> dict[str, Any]:
    frozen = Path(frozen)
    out = Path(out)
    protocol = Path(protocol)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty run output: {out}")
    freeze = verify_freeze(frozen, source_paths=source_paths(protocol))
    plan = build_plan()
    frozen_manifest = _load_frozen_manifest(frozen)
    if [manifest_record(item) for item in plan.requests] != frozen_manifest:
        raise ValueError("rebuilt manifest differs from the frozen manifest")
    checkpoint_manifest = _snapshot_override(json.loads((frozen / "checkpoint_manifest.json").read_text(encoding="utf-8")))
    verify_checkpoint_manifest(checkpoint_manifest, hash_files=True)
    backend = _make_backend(argparse.Namespace(mode=mode, backend=backend_name, snapshot=snapshot))
    if mode == "execute":
        backend.load()
    runtime = {
        "started_utc": _now(),
        "freeze_sha256": (frozen / "freeze.sha256").read_text(encoding="utf-8").strip(),
        "protocol_sha256": freeze["protocol_sha256"],
        "checkpoint_aggregate_sha256": checkpoint_manifest.get("aggregate_sha256"),
        "checkpoint_snapshot_path": checkpoint_manifest.get("snapshot_path"),
    }
    if mode == "execute":
        runtime["audit"] = _runtime_audit(backend)
    else:
        runtime["audit"] = {"synthetic_stub": True, "device": "cpu"}
    _write_json(out / "run_metadata.json", _run_metadata(mode, backend))
    _write_json(out / "runtime_audit.json", runtime)

    responses: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    stopped: dict[str, Any] | None = None
    started = time.perf_counter()
    for item in plan.requests:
        if len(responses) >= REQUEST_CAP:
            stopped = {"reason": "request_budget_exceeded", "request_id": item.request_id}
            break
        if item.input_mode != "text":
            stopped = {"reason": "non_text_input", "request_id": item.request_id}
            break
        assert_prompt_public(item.prompt)
        frozen_row = frozen_manifest[item.request_index - 1]
        if frozen_row["request_id"] != item.request_id or frozen_row["prompt_sha256"] != sha256_text(item.prompt):
            stopped = {"reason": "frozen_prompt_mismatch", "request_id": item.request_id}
            break
        request = dataclasses.replace(make_model_request(item), backend_id=backend.backend_id)
        _append_jsonl(out / "requests.jsonl", _request_record(item, request))
        request_started = time.perf_counter()
        try:
            response = backend.generate(request)
            metadata = dict(getattr(backend, "last_metadata", {}))
            if metadata.get("image_seen"):
                raise RuntimeError("unexpected image tensor in a text-only request")
            if int(response.input_tokens) <= 0:
                raise RuntimeError("invalid input-token accounting")
            row = {
                "request_id": item.request_id,
                "arm": item.arm,
                "status": "ok",
                "text": response.text,
                "input_tokens": int(response.input_tokens),
                "output_tokens": int(response.output_tokens),
                "latency_ms": float(response.latency_ms),
                "backend_id": response.backend_id,
                "metadata": metadata,
                "error": None,
            }
            parsed = parse_action(response.text, item.context.feasible_actions())
            parsed_rows.append({
                "request_id": item.request_id,
                "arm": item.arm,
                "context_id": item.context_id,
                "selected_action": parsed.selected_action,
                "valid": bool(parsed.valid),
                "fallback": bool(parsed.fallback),
                "errors": list(parsed.errors),
                "raw_text": response.text,
            })
        except Exception as exc:
            row = {
                "request_id": item.request_id,
                "arm": item.arm,
                "status": "error",
                "text": None,
                "input_tokens": None,
                "output_tokens": None,
                "latency_ms": (time.perf_counter() - request_started) * 1000,
                "backend_id": backend.backend_id,
                "metadata": dict(getattr(backend, "last_metadata", {})),
                "error": f"{type(exc).__name__}: {exc}",
            }
            stopped = {"reason": "request_error", "request_id": item.request_id, "error": row["error"]}
        responses.append(row)
        _append_jsonl(out / "responses.jsonl", row)
        if stopped is not None:
            break
    elapsed = time.perf_counter() - started

    ok_responses = [row for row in responses if row["status"] == "ok"]
    accounting = {
        "planned": REQUEST_CAP,
        "attempted": len(responses),
        "completed": len(ok_responses),
        "failed": len(responses) - len(ok_responses),
        "model_calls": len(responses),
        "parsed": len(parsed_rows),
        "automatic_retries": 0,
        "elapsed_seconds": elapsed,
        "stopped": stopped,
    }
    _write_json(out / "request_accounting.json", accounting)
    _write_jsonl(out / "parsed.jsonl", parsed_rows)
    _write_json(out / "parsed_actions.json", {"rows": parsed_rows})
    baselines = baseline_diagnostics(plan)
    arms: dict[str, Any] = {}
    all_arms_complete = True
    for arm in ARMS:
        arm_ids = [item.request_id for item in plan.arm_requests(arm)]
        arm_records = [row for row in parsed_rows if row["request_id"] in arm_ids]
        if len(arm_records) == 12 and len(ok_responses) == REQUEST_CAP:
            arms[arm] = score_arm(arm_records, plan, arm, baselines=baselines)
        else:
            all_arms_complete = False
            arms[arm] = {
                "arm": arm,
                "n": len(arm_records),
                "complete": False,
                "eligible_for_visual_smoke_consideration": False,
                "details": arm_records,
            }
    paired = paired_comparisons(plan, arms) if all_arms_complete else {}
    if mode == "dry_run":
        for arm in ARMS:
            arms[arm]["eligible_for_visual_smoke_consideration"] = False
            arms[arm]["eligibility_note"] = "synthetic stub dry run; not scientific"
    summary = {
        "experiment_id": "vlm-prompt-belief-diagnostic-v1",
        "mode": mode,
        "backend": backend.backend_id,
        "request_accounting": accounting,
        "model_calls": len(responses),
        "attempted": len(responses),
        "completed": len(ok_responses),
        "all_arms_complete": all_arms_complete,
        "synthetic_stub": mode == "dry_run",
        "technical_stop": stopped,
        "arms": arms,
        "baselines": baselines,
        "paired_comparisons": paired,
        "runtime": runtime,
        "frozen_hashes": {key: freeze[key] for key in sorted(freeze) if key.endswith("_sha256")},
    }
    _write_json(out / "summary.json", summary)
    _write_json(
        out / "aggregate.json",
        {
            "experiment_id": summary["experiment_id"],
            "mode": mode,
            "request_accounting": accounting,
            "arm_eligibility": {arm: arms[arm].get("eligible_for_visual_smoke_consideration", False) for arm in ARMS},
            "eligible_arms": [] if mode == "dry_run" else [arm for arm in ARMS if arms[arm].get("eligible_for_visual_smoke_consideration", False)],
            "technical_stop": stopped,
            "synthetic_stub": mode == "dry_run",
        },
    )
    print(json.dumps({
        "mode": mode,
        "attempted": len(responses),
        "completed": len(ok_responses),
        "stop": stopped,
        "eligible_arms": [arm for arm in ARMS if arms[arm].get("eligible_for_visual_smoke_consideration", False)],
    }, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry_run", "execute"), required=True)
    parser.add_argument("--backend", choices=("stub", "qwen"), default="qwen")
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    try:
        if args.mode == "execute" and args.snapshot is None:
            raise ValueError("execute mode requires --snapshot")
        summary = run(
            args.frozen.resolve(),
            args.out.resolve(),
            mode=args.mode,
            backend_name=args.backend,
            snapshot=args.snapshot.resolve() if args.snapshot else None,
            protocol=args.protocol.resolve(),
        )
        return 0 if summary["technical_stop"] is None and summary["all_arms_complete"] else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())