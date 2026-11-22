"""Aggregate per-scenario returns with paired bootstrap CIs.

Usage:
    python scripts/aggregate_results.py --input results/demo.csv \
        --method M1 --baseline B0 --out reports/paired_summary.json

The input CSV must have scenario_id, method and return columns, with exactly
one return per (scenario_id, method). The resampling unit is the scenario.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:  # allow running straight from a source checkout
    import ruleshift  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.evaluator.metrics_ci import summarize_paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--method", default="M1")
    parser.add_argument("--baseline", default="B0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    by_scenario: dict[str, dict[str, float]] = {}
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_scenario.setdefault(row["scenario_id"], {})[row["method"]] = float(row["return"])

    method_returns = [values[args.method] for values in by_scenario.values() if args.method in values]
    baseline_returns = [values[args.baseline] for values in by_scenario.values() if args.baseline in values]
    if len(method_returns) != len(baseline_returns):
        raise SystemExit("scenarios must contain returns for both methods")

    summary = summarize_paired(method_returns, baseline_returns, seed=args.seed)
    output = {
        "method": args.method,
        "baseline": args.baseline,
        "n_scenarios": summary.n_scenarios,
        "mean_difference": summary.mean_difference,
        "ci95_low": summary.low,
        "ci95_high": summary.high,
    }
    print(json.dumps(output, sort_keys=True))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
