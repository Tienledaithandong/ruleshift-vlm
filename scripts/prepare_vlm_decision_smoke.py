"""Prepare and freeze the authorized 48-request VLM decision smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ruleshift.evaluator.vlm_smoke import build_plan, freeze_artifacts, verify_freeze
from ruleshift.evaluator.vlm_smoke_checkpoint import build_checkpoint_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = Path(
    r"C:\Users\Lenovo\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct"
    r"\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3"
)
DEFAULT_OUT = REPO_ROOT / "results" / "vlm_decision_smoke" / "frozen"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite non-empty frozen directory: {out}", file=sys.stderr)
        return 2
    missing = [str(path) for path in source_paths() if not path.is_file()]
    if missing:
        print("missing source/doc files: " + ", ".join(missing), file=sys.stderr)
        return 2
    if not args.protocol.is_file():
        print(f"missing protocol: {args.protocol}", file=sys.stderr)
        return 2
    plan = build_plan()
    print(f"selected base contexts: {len(plan.base_contexts)}")
    print(f"planned requests: {len(plan.requests)}")
    print("hashing frozen checkpoint (this reads the local weights, not the model):")
    checkpoint = build_checkpoint_manifest(args.snapshot)
    print(f"checkpoint files: {checkpoint['file_count']} total_bytes={checkpoint['total_bytes']}")
    print(f"checkpoint aggregate: {checkpoint['aggregate_sha256']}")
    freeze = freeze_artifacts(
        plan,
        out,
        checkpoint_manifest=checkpoint,
        checkpoint_manifest_path=out / "checkpoint_manifest.json",
        protocol_path=args.protocol.resolve(),
        source_paths=[path.resolve() for path in source_paths()],
    )
    verify_freeze(out, source_paths=[path.resolve() for path in source_paths()])
    print(f"frozen at: {out}")
    print(f"request_manifest_sha256: {freeze['request_manifest_sha256']}")
    print(f"reference_labels_sha256: {freeze['reference_labels_sha256']}")
    print(f"freeze_sha256: {(out / 'freeze.sha256').read_text(encoding='utf-8').strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
