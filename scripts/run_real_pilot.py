"""Run the offline pilot with a REAL frozen VLM (setup for later).

This script NEVER falls back to the stub. Run `scripts/preflight.py` first.

Usage:
    python scripts/run_real_pilot.py --model-key qwen2.5-vl-3b \
        --out results/real --report reports/real_pilot
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.model_check import check_model, model_settings, readiness_errors
from ruleshift.agents.vlm_backend import TransformersVLMBackend
from ruleshift.evaluator.h1_contexts import build_h1_contexts
from ruleshift.evaluator.history_contexts import build_history_contexts
from ruleshift.evaluator.offline_audit import build_offline_contexts, run_offline_audit
from ruleshift.evaluator.provenance import collect_hashes, file_hash, implementation_digest
from ruleshift.evaluator.real_run import write_real_artifacts

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--contexts", type=int, default=81)
    parser.add_argument("--design", choices=("controlled", "history", "h1"), default="controlled",
                        help="public card lottery, histories, or the balanced H1 factorial")
    parser.add_argument("--method", choices=("direct", "m1"), default="direct",
                        help="direct action or structured M1 value selection")
    parser.add_argument("--out", default="results/real")
    parser.add_argument("--report", default="reports/real_pilot.md")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu")
    parser.add_argument("--allow-download", action="store_true",
                        help="explicitly allow fetching the pinned checkpoint on this run")
    parser.add_argument("--force", action="store_true",
                        help="reuse a non-empty output directory (not allowed for a fresh pilot)")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        parser.error(
            f"output directory is not empty: {out_dir}; choose a fresh --out path "
            "or pass --force for an intentional rerun"
        )

    with (ROOT / args.config).open(encoding="utf-8") as handle:
        models = yaml.safe_load(handle)
    key = args.model_key or models["default"]
    if key not in models["models"]:
        parser.error(f"unknown model key: {key}")
    if args.contexts < 1:
        parser.error("--contexts must be positive")
    settings = model_settings(models, key, args.device, args.allow_download)
    errors = readiness_errors(settings, "pilot", {})
    if errors:
        parser.error(", ".join(errors))

    readiness = check_model(**settings)
    if not readiness.ready:
        for message in readiness.messages:
            print(message)
        raise SystemExit("no usable checkpoint; preflight failed (stub fallback disabled)")

    loader_settings = {name: value for name, value in settings.items() if name != "allow_download"}
    model = models["models"][key]
    backend = TransformersVLMBackend(
        **loader_settings, dtype=model["dtype"], max_new_tokens=model["max_new_tokens"]
    )
    builder = {
        "controlled": build_offline_contexts,
        "history": build_history_contexts,
        "h1": build_h1_contexts,
    }[args.design]
    contexts = builder(args.contexts)
    cache = ResponseCache(out_dir / "response_cache.jsonl")
    records, metrics = run_offline_audit(contexts, backend, cache, method=args.method)
    paths = write_real_artifacts(
        records,
        contexts,
        metrics,
        {
            "model_id": backend.model_id,
            "device": backend.device,
            "revision": backend.revision,
            "dtype": backend.dtype,
            "max_new_tokens": backend.max_new_tokens,
            "input_mode": "P1_public_text",
            "context_design": args.design,
            "method": args.method,
            "implementation_digest": implementation_digest(),
            "prompt_sha256": {
                name: file_hash(ROOT / "src" / "ruleshift" / "prompts" / name)
                for name in ("direct_v1.txt", "outcomes_v1.txt")
            },
            "python": platform.python_version(),
            "config_sha256": file_hash(ROOT / args.config),
            "dev_config_sha256": file_hash(ROOT / "configs" / "dev.yaml"),
            "file_hashes": collect_hashes(ROOT),
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "gymnasium", "numpy", "Pillow")
            },
        },
        ROOT / args.out,
        ROOT / args.report,
    )
    print(f"real pilot complete: {len(records)} decisions")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
