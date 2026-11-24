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

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.evaluator.h1_contexts import build_h1_contexts
from ruleshift.evaluator.history_contexts import build_history_contexts
from ruleshift.evaluator.offline_audit import (
    build_offline_contexts,
    run_offline_audit,
    write_taxonomy_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, default=81)
    parser.add_argument("--design", choices=("controlled", "history", "h1"), default="controlled",
                        help="public card lottery, histories, or the balanced H1 factorial")
    parser.add_argument("--method", choices=("direct", "m1"), default="direct",
                        help="direct action or structured M1 value selection")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--out", default="data")
    parser.add_argument("--report", default="reports")
    args = parser.parse_args()

    builder = {
        "controlled": build_offline_contexts,
        "history": build_history_contexts,
        "h1": build_h1_contexts,
    }[args.design]
    contexts = builder(args.contexts, Path(args.config))
    data_dir = Path(args.out)
    report_dir = Path(args.report)
    records, metrics = run_offline_audit(
        contexts,
        backend=StubVLMBackend(model_id="stub-v0"),
        cache=ResponseCache(data_dir / "stub_cache.jsonl"),
        method=args.method,
    )
    write_taxonomy_report(
        metrics,
        records,
        contexts,
        data_dir / "offline_pilot_manifest.jsonl",
        data_dir / "offline_pilot_evaluator.jsonl",
        report_dir / "w4_failure_taxonomy.md",
    )
    print(f"SYNTHETIC DRY RUN ({args.design}/{args.method}): stub backend")
    print(f"contexts={metrics.n_contexts} calls={metrics.call_count} "
          f"json_valid={metrics.json_valid_rate:.3f} "
          f"over_probe={metrics.over_probe_rate:.3f} "
          f"under_probe={metrics.under_probe_rate:.3f}")


if __name__ == "__main__":
    main()
