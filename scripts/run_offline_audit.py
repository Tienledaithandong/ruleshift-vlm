"""Run the Week 4 offline pilot harness (SYNTHETIC dry run by default).

Usage:
    python scripts/run_offline_audit.py --contexts 81 \
        --out data --report reports

A real VLM is not bundled. Use StubVLMBackend -> TransformersVLMBackend in code
only when an actual checkpoint is available; this script never downloads one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # allow running straight from a source checkout
    import ruleshift  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.evaluator.offline_audit import (
    build_offline_contexts,
    run_offline_audit,
    write_taxonomy_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, default=81)
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--out", default="data")
    parser.add_argument("--report", default="reports")
    args = parser.parse_args()

    contexts = build_offline_contexts(args.contexts)
    data_dir = Path(args.out)
    report_dir = Path(args.report)
    records, metrics = run_offline_audit(
        contexts,
        backend=StubVLMBackend(model_id="stub-v0"),
        cache=ResponseCache(data_dir / "stub_cache.jsonl"),
    )
    write_taxonomy_report(
        metrics,
        records,
        contexts,
        data_dir / "offline_pilot_manifest.jsonl",
        data_dir / "offline_pilot_evaluator.jsonl",
        report_dir / "w4_failure_taxonomy.md",
    )
    print("SYNTHETIC DRY RUN — stub backend, no VLM was run")
    print(f"contexts={metrics.n_contexts} calls={metrics.call_count} "
          f"json_valid={metrics.json_valid_rate:.3f} "
          f"correct_argmax={metrics.correct_argmax_rate:.3f} "
          f"mean_regret={metrics.mean_decision_regret:.3f}")
    for stratum in metrics.strata:
        print(f"  {stratum.key}: probe {stratum.n_probes}/{stratum.n_contexts} "
              f"({stratum.probe_rate:.3f}) "
              f"argmax={stratum.correct_argmax_rate:.3f}")


if __name__ == "__main__":
    main()
