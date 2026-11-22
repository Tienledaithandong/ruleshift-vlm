"""Run the offline pilot with a REAL frozen VLM (setup for later).

This script NEVER falls back to the stub. Run `scripts/preflight.py` first.

Usage:
    python scripts/run_real_pilot.py --model-key qwen2.5-vl-3b \
        --out results/real --report reports/real_pilot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import ruleshift  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.model_check import check_model
from ruleshift.agents.vlm_backend import TransformersVLMBackend
from ruleshift.evaluator.offline_audit import build_offline_contexts, run_offline_audit
from ruleshift.evaluator.real_run import write_real_artifacts

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--contexts", type=int, default=81)
    parser.add_argument("--out", default="results/real")
    parser.add_argument("--report", default="reports/real_pilot.md")
    parser.add_argument("--config", default="configs/models.yaml")
    args = parser.parse_args()

    with (ROOT / args.config).open(encoding="utf-8") as handle:
        models = yaml.safe_load(handle)
    key = args.model_key or models["default"]
    model = models["models"][key]
    runtime = models["runtime"]
    model_id = model["hf_id"]

    readiness = check_model(
        model_id,
        local_files_only=runtime["local_files_only"],
        allow_download=runtime["allow_download"],
    )
    if not readiness.ready:
        for message in readiness.messages:
            print(message)
        raise SystemExit("no usable checkpoint; preflight failed (stub fallback disabled)")

    backend = TransformersVLMBackend(
        model_id=model_id,
        device=model["device"],
        revision=model["revision"],
    )
    contexts = build_offline_contexts(args.contexts)
    cache = ResponseCache(ROOT / args.out / "response_cache.jsonl")
    records, metrics = run_offline_audit(contexts, backend, cache)
    paths = write_real_artifacts(
        records,
        contexts,
        metrics,
        {
            "model_id": model_id,
            "device": model["device"],
            "revision": model["revision"],
        },
        ROOT / args.out,
        ROOT / args.report,
    )
    print(f"real pilot complete: {len(records)} decisions")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
