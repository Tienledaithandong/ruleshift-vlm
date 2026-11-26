"""CPU-only preflight for the prompt/belief diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from ruleshift.evaluator.vlm_prompt_belief import (
    REQUEST_CAP,
    build_plan,
    manifest_record,
    source_paths,
    sha256_text,
    verify_freeze,
)
from ruleshift.evaluator.vlm_smoke import assert_prompt_public, sha256_json
from ruleshift.evaluator.vlm_smoke_checkpoint import verify_checkpoint_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = REPO_ROOT / "results" / "vlm_prompt_belief_diagnostic" / "frozen"
DEFAULT_OUT = REPO_ROOT / "results" / "vlm_prompt_belief_diagnostic" / "preflight"
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "VLM_PROMPT_BELIEF_DIAGNOSTIC_PROTOCOL.md"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _snapshot_override(manifest: dict) -> dict:
    override = os.environ.get("RULESHIFT_SNAPSHOT_PATH", "").strip()
    if not override:
        return manifest
    updated = dict(manifest)
    updated["snapshot_path"] = override
    updated["snapshot_path_source"] = "RULESHIFT_SNAPSHOT_PATH"
    return updated


def preflight(frozen: Path, out: Path, protocol: Path) -> dict:
    frozen = Path(frozen)
    out = Path(out)
    protocol = Path(protocol)
    freeze = verify_freeze(frozen, source_paths=source_paths(protocol))
    plan = build_plan()
    manifest = json.loads((frozen / "request_manifest.json").read_text(encoding="utf-8"))
    expected_manifest = [manifest_record(item) for item in plan.requests]
    if manifest != expected_manifest:
        raise ValueError("rebuilt request manifest differs from the frozen manifest")
    if len(manifest) != REQUEST_CAP:
        raise ValueError(f"request count is not {REQUEST_CAP}")
    if freeze["request_count"] != REQUEST_CAP:
        raise ValueError("freeze request count is not 48")
    trigger_checks = []
    for item, row in zip(plan.requests, manifest):
        if sha256_text(item.prompt) != row["prompt_sha256"]:
            raise ValueError(f"{item.request_id}: prompt hash mismatch")
        if sha256_json(dict(item.public_payload)) != row["public_payload_sha256"]:
            raise ValueError(f"{item.request_id}: public payload hash mismatch")
        assert_prompt_public(item.prompt)
        trigger_checks.append({
            "request_id": item.request_id,
            "arm": item.arm,
            "posterior_in_payload": "posterior" in item.public_payload,
            "concrete_action_example": any(
                f'{{"selected_action":"{action}"}}' in item.prompt for action in ("COMMIT", "PROBE_SIGN", "PROBE_BLUE")
            ),
            "image_reference": "<image>" in item.prompt.lower(),
            "prompt_chars": len(item.prompt),
        })
    # B1 and B3 deliberately retain the anchored JSON example.  The neutral arms
    # must contain no concrete action example at all.
    if any(row["concrete_action_example"] for row in trigger_checks if row["arm"] in ("B2", "B4")):
        raise ValueError("a concrete action example leaked into a neutral prompt")
    if any(row["image_reference"] for row in trigger_checks):
        raise ValueError("an image marker leaked into the text prompt")
    checkpoint_manifest = _snapshot_override(json.loads((frozen / "checkpoint_manifest.json").read_text(encoding="utf-8")))
    verify_checkpoint_manifest(checkpoint_manifest, hash_files=True)
    if sum(1 for item in plan.requests if item.arm == "B1") != 12:
        raise ValueError("B1 arm is incomplete")
    if sum(1 for item in plan.requests if item.arm == "B2") != 12:
        raise ValueError("B2 arm is incomplete")
    if sum(1 for item in plan.requests if item.arm == "B3") != 12:
        raise ValueError("B3 arm is incomplete")
    if sum(1 for item in plan.requests if item.arm == "B4") != 12:
        raise ValueError("B4 arm is incomplete")
    result = {
        "status": "GO_FOR_CPU_ONLY_DRY_RUN",
        "frozen": str(frozen),
        "protocol_sha256": freeze["protocol_sha256"],
        "freeze_sha256": (frozen / "freeze.sha256").read_text(encoding="utf-8").strip(),
        "request_count": len(plan.requests),
        "arms": {arm: sum(1 for item in plan.requests if item.arm == arm) for arm in ("B1", "B2", "B3", "B4")},
        "request_ids_unique": len({item.request_id for item in plan.requests}) == 48,
        "all_contexts_present": {item.context_id for item in plan.requests} == {item.context_id for item in plan.contexts},
        "checkpoint": {
            "snapshot_path": checkpoint_manifest.get("snapshot_path"),
            "file_count": checkpoint_manifest.get("file_count"),
            "total_bytes": checkpoint_manifest.get("total_bytes"),
            "aggregate_sha256": checkpoint_manifest.get("aggregate_sha256"),
        },
        "leakage": {
            "no_posterior_in_prior_arms": all(
                not (item.arm in ("B1", "B2") and "posterior" in item.public_payload) for item in plan.requests
            ),
            "posterior_in_both_explicit_arms": all(
                "posterior" in item.public_payload for item in plan.requests if item.arm in ("B3", "B4")
            ),
            "no_concrete_action_example_in_neutral": not any(
                row["concrete_action_example"] for row in trigger_checks if row["arm"] in ("B2", "B4")
            ),
            "no_image_marker": not any(row["image_reference"] for row in trigger_checks),
            "no_image_input": all(item.input_mode == "text" for item in plan.requests),
        },
        "prompt_chars": {
            "min": min(len(item.prompt) for item in plan.requests),
            "max": max(len(item.prompt) for item in plan.requests),
            "mean": sum(len(item.prompt) for item in plan.requests) / len(plan.requests),
        },
        "frozen_hashes": {key: freeze[key] for key in sorted(freeze) if key.endswith("_sha256")},
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "preflight.json", result)
    _write_json(out / "prompt_audit.json", {"requests": trigger_checks})
    (out / "PREFLIGHT.md").write_text(
        "# Prompt/Belief Diagnostic CPU Preflight\n\n"
        f"- Status: `{result['status']}`\n"
        f"- Requests: {result['request_count']}\n"
        f"- Arms: {result['arms']}\n"
        f"- Freeze: `{result['freeze_sha256']}`\n"
        f"- Protocol: `{result['protocol_sha256']}`\n"
        f"- Checkpoint aggregate: `{result['checkpoint']['aggregate_sha256']}`\n"
        "- No model was loaded and no model request was made.\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    try:
        result = preflight(args.frozen.resolve(), args.out.resolve(), args.protocol.resolve())
        print(f"{result['status']}: {result['request_count']} requests, 0 model calls")
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())