"""CPU preflight for the frozen 48-request VLM decision smoke.

This loads only the processor, never the model weights. It proves that the same
frozen RGB card reaches processor pixel tensors for A3/controls, while A1/A2
remain text-only, and that prompts/public payloads contain no evaluator labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ruleshift.agents.vlm_smoke_backend import FrozenQwenBackend
from ruleshift.evaluator.vlm_decision_visual import visual_digest
from ruleshift.evaluator.vlm_smoke import (
    FORBIDDEN_LABEL_KEYS,
    build_plan,
    image_png_bytes,
    manifest_record,
    render_prompt,
    sha256_json,
    sha256_text,
    verify_freeze,
    write_json,
)
from ruleshift.evaluator.vlm_smoke_checkpoint import verify_checkpoint_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = REPO_ROOT / "results" / "vlm_decision_smoke" / "frozen"
DEFAULT_OUT = REPO_ROOT / "results" / "vlm_decision_smoke" / "preflight"
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def walk_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(walk_keys(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for item in value:
            keys.extend(walk_keys(item, prefix))
    return keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args(argv)
    frozen = args.frozen_dir.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite non-empty preflight directory: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    try:
        freeze = verify_freeze(frozen, source_paths=[path.resolve() for path in source_paths()])
        checks["freeze_verified"] = True
        checks["freeze_sha256"] = (frozen / "freeze.sha256").read_text(encoding="utf-8").strip()
        plan = build_plan()
        manifest = json.loads((frozen / "request_manifest.json").read_text(encoding="utf-8"))
        expected = [manifest_record(item) for item in plan.requests]
        checks["manifest_exact"] = manifest == expected
        if not checks["manifest_exact"]:
            raise ValueError("frozen manifest differs from rebuilt manifest")
        checks["request_count"] = len(manifest)
        checks["arm_counts"] = {
            arm: sum(row["arm"] == arm for row in manifest)
            for arm in ("A1", "A2", "A3", "RGB_NUISANCE", "RGB_DIAGNOSTIC")
        }
        if checks["request_count"] != 48 or any(
            checks["arm_counts"][arm] != 12 for arm in ("A1", "A2", "A3")
        ):
            raise ValueError("unexpected request counts")
        if checks["arm_counts"]["RGB_NUISANCE"] != 6 or checks["arm_counts"]["RGB_DIAGNOSTIC"] != 6:
            raise ValueError("unexpected control counts")

        prompt_rows = [
            json.loads(line)
            for line in (frozen / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        public_rows = [
            json.loads(line)
            for line in (frozen / "public_payloads.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        checks["prompt_rows"] = len(prompt_rows)
        checks["public_payload_rows"] = len(public_rows)
        if len(prompt_rows) != 48 or len(public_rows) != 48:
            raise ValueError("prompt/public artifact row count mismatch")

        leakage: list[str] = []
        for row in prompt_rows:
            serialized = json.dumps(row["prompt"], ensure_ascii=True).lower()
            for key in FORBIDDEN_LABEL_KEYS:
                if key in serialized:
                    leakage.append(f"{row['request_id']}:{key}")
            if "posterior" in serialized and row["arm"] == "A1":
                leakage.append(f"{row['request_id']}:posterior_in_A1")
            if "posterior" in serialized and row["arm"] == "A3":
                leakage.append(f"{row['request_id']}:posterior_in_A3_text")
        for row in public_rows:
            for key in walk_keys(row["public_payload"]):
                if key in FORBIDDEN_LABEL_KEYS:
                    leakage.append(f"{row['request_id']}:public_key:{key}")
        checks["leakage_hits"] = leakage
        checks["leakage_zero"] = not leakage
        if leakage:
            raise ValueError(f"leakage detected: {leakage[:10]}")

        # Rebuild exact prompts, not just hashes, to prove the frozen bytes.
        rebuilt_prompt_mismatches = []
        for item, row in zip(plan.requests, prompt_rows):
            if item.request_id != row["request_id"] or sha256_text(item.prompt) != row["prompt_sha256"]:
                rebuilt_prompt_mismatches.append(item.request_id)
        checks["rebuilt_prompt_mismatches"] = rebuilt_prompt_mismatches
        checks["prompts_rebuilt_exactly"] = not rebuilt_prompt_mismatches
        if rebuilt_prompt_mismatches:
            raise ValueError("frozen prompts do not rebuild exactly")

        # Images: metadata, dimensions, uniqueness, and nuisance/diagnostic claims.
        images_dir = frozen / "images"
        image_checks: list[dict[str, Any]] = []
        for item, row in zip(plan.requests, manifest):
            if item.image is None:
                if row["image_pixel_digest"] is not None:
                    raise ValueError(f"text request has image digest: {item.request_id}")
                continue
            path = images_dir / f"{item.request_id}.png"
            png = path.read_bytes()
            if "sha256:" + hashlib.sha256(png).hexdigest() != row["image_png_sha256"]:
                raise ValueError(f"image PNG hash mismatch: {item.request_id}")
            array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            if array.shape != (320, 512, 3):
                raise ValueError(f"image shape mismatch: {item.request_id} {array.shape}")
            if visual_digest(array) != row["image_pixel_digest"]:
                raise ValueError(f"image pixel digest mismatch: {item.request_id}")
            if visual_digest(array) != visual_digest(item.image):
                raise ValueError(f"image differs from planned array: {item.request_id}")
            image_checks.append(
                {
                    "request_id": item.request_id,
                    "shape": list(array.shape),
                    "image_pixel_digest": row["image_pixel_digest"],
                    "pixel_digest": row["image_pixel_digest"],
                    "png_sha256": row["image_png_sha256"],
                    "image_png_sha256": row["image_png_sha256"],
                }
            )
        checks["images"] = image_checks
        checks["image_count"] = len(image_checks)
        base_digests = {row["image_pixel_digest"] for row in image_checks if row["request_id"].startswith("A3-")}
        nuisance_digests = {row["image_pixel_digest"] for row in image_checks if row["request_id"].startswith("N")}
        checks["base_images_unique"] = len(base_digests) == 12
        checks["nuisance_images_unique"] = len(nuisance_digests) == 6
        checks["base_nuisance_disjoint"] = not bool(base_digests & nuisance_digests)
        if not all((checks["base_images_unique"], checks["nuisance_images_unique"], checks["base_nuisance_disjoint"])):
            raise ValueError("base/nuisance image digest invariant failed")

        nuisance_facts = []
        for item in plan.nuisance_requests:
            base = next(context for context in plan.base_contexts if context.context_id == item.base_context_id)
            if item.label != next(label for label in plan.base_labels if label.context_id == item.base_context_id):
                raise ValueError(f"nuisance label changed: {item.request_id}")
            if item.public_payload != base.to_public_payload(include_posterior=True):
                raise ValueError(f"nuisance public facts changed: {item.request_id}")
            base_row = next(
                row
                for row in manifest
                if row["arm"] == "A3" and row["context_id"] == item.base_context_id
            )
            if base_row["image_pixel_digest"] == next(
                row for row in manifest if row["request_id"] == item.request_id
            )["image_pixel_digest"]:
                raise ValueError(f"nuisance card pixel-identical: {item.request_id}")
            nuisance_facts.append(item.request_id)
        checks["nuisance_facts_unchanged"] = len(nuisance_facts) == 6

        base_label_by_context = {label.context_id: label for label in plan.base_labels}
        diagnostic_checks = []
        for item in plan.diagnostic_requests:
            base_label = base_label_by_context[item.base_context_id]
            changed = set(base_label.optimal_actions) != set(item.label.optimal_actions)
            diagnostic_checks.append(
                {
                    "request_id": item.request_id,
                    "base_id": item.base_context_id,
                    "role": item.diagnostic_role,
                    "optimal_set_changed": changed,
                    "base_best": base_label.best_action,
                    "fresh_best": item.label.best_action,
                }
            )
        checks["diagnostics"] = diagnostic_checks
        checks["all_diagnostic_sets_changed"] = all(row["optimal_set_changed"] for row in diagnostic_checks)
        if not checks["all_diagnostic_sets_changed"]:
            raise ValueError("a diagnostic did not change the exact optimal set")

        checkpoint = json.loads((frozen / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        verify_checkpoint_manifest(checkpoint, hash_files=False)
        checks["checkpoint_manifest_verified"] = True
        checks["checkpoint_aggregate_sha256"] = checkpoint["aggregate_sha256"]
        checks["checkpoint_total_bytes"] = checkpoint["total_bytes"]

        # This is the actual processor path, without model weights.
        backend = FrozenQwenBackend(args.snapshot, device="cpu", dtype="bfloat16")
        backend.load_processor_only()
        probe_requests = {
            "A1-01": next(item for item in plan.base_requests if item.request_id == "A1-01"),
            "A2-01": next(item for item in plan.base_requests if item.request_id == "A2-01"),
            "A3-01": next(item for item in plan.base_requests if item.request_id == "A3-01"),
            "N1": plan.nuisance_requests[0],
            "D1": plan.diagnostic_requests[0],
        }
        from ruleshift.evaluator.vlm_smoke import make_model_request

        probes: dict[str, Any] = {}
        for request_id, item in probe_requests.items():
            prepared = backend.prepare(make_model_request(item))
            probes[request_id] = {
                "input_tokens": prepared["input_tokens"],
                "vision_tensor_sha256": prepared["vision_tensor_sha256"],
                "vision_tensor_shape": prepared["vision_tensor_shape"],
                "image_grid_thw": prepared["image_grid_thw"],
                "image_seen": prepared["vision_tensor_sha256"] is not None,
                "input_ids_sha256": prepared["input_ids_sha256"],
            }
        checks["processor_probes"] = probes
        if probes["A1-01"]["image_seen"] or probes["A2-01"]["image_seen"]:
            raise ValueError("text prompt unexpectedly reached an image tensor")
        if not probes["A3-01"]["image_seen"]:
            raise ValueError("A3 prompt did not reach an image tensor")
        if not probes["N1"]["image_seen"] or not probes["D1"]["image_seen"]:
            raise ValueError("a control did not reach an image tensor")
        if len({probes[key]["vision_tensor_sha256"] for key in ("A3-01", "N1", "D1")}) != 3:
            raise ValueError("base/nuisance/diagnostic vision tensors are not distinct")
        checks["actual_image_forwarding_verified"] = True
        checks["text_stayed_text_only"] = True
        checks["processor_only_no_model_call"] = True
        checks["status"] = "PASS"
    except Exception as exc:
        checks["status"] = "FAIL"
        checks["error"] = f"{type(exc).__name__}: {exc}"
        checks["traceback"] = traceback.format_exc()
        write_json(out / "preflight.json", checks)
        print(checks["error"], file=sys.stderr)
        traceback.print_exc()
        return 2
    write_json(out / "preflight.json", checks)
    checksums = {
        str(path.relative_to(out)): file_sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    write_json(out / "checksums.json", checksums)
    print(f"preflight PASS at {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
