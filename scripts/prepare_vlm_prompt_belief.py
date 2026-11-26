"""Create the frozen 48-request prompt/belief diagnostic artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from ruleshift.evaluator.vlm_prompt_belief import build_plan, freeze_artifacts, source_paths, validate_design
from ruleshift.evaluator.vlm_smoke import build_plan as build_original_plan
from ruleshift.evaluator.vlm_smoke_checkpoint import verify_checkpoint_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results" / "vlm_prompt_belief_diagnostic" / "frozen"
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "VLM_PROMPT_BELIEF_DIAGNOSTIC_PROTOCOL.md"
DEFAULT_CHECKPOINT_MANIFEST = REPO_ROOT / "results" / "vlm_decision_smoke" / "frozen" / "checkpoint_manifest.json"


def prepare(out_dir: Path, protocol: Path, checkpoint_manifest_path: Path) -> dict:
    out_dir = Path(out_dir)
    protocol = Path(protocol)
    checkpoint_manifest_path = Path(checkpoint_manifest_path)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty frozen output: {out_dir}")
    if not protocol.is_file():
        raise FileNotFoundError(f"protocol not found: {protocol}")
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    verify_checkpoint_manifest(checkpoint_manifest, hash_files=True)
    plan = build_plan()
    original = build_original_plan()
    validate_design(plan, original)
    freeze = freeze_artifacts(
        plan,
        out_dir,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_manifest_path=checkpoint_manifest_path,
        protocol_path=protocol,
        source_paths=source_paths(protocol),
    )
    print(f"frozen {freeze['request_count']} requests at {out_dir}")
    print(f"freeze version: {freeze['version']}")
    return freeze


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        prepare(args.out.resolve(), args.protocol.resolve(), args.checkpoint_manifest.resolve())
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())