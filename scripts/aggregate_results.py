"""Aggregate per-scenario returns with paired bootstrap CIs.

Usage:
    python scripts/aggregate_results.py --input results/demo.csv \
        --method M1 --baseline B0 --out reports/paired_summary.json

The input CSV must have scenario_id, method and return columns, with exactly
one paired return per (scenario_id, seed, method). The seed column is optional
for pre-aggregated input. The resampling unit is the scenario.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
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

    by_scenario: dict[str, dict[str, dict[str, float]]] = {}
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in (args.method, args.baseline):
                continue
            scenario = row["scenario_id"]
            seed = row.get("seed", "")
            pairs = by_scenario.setdefault(scenario, {}).setdefault(seed, {})
            if method in pairs:
                raise SystemExit(f"duplicate return for {scenario}, seed {seed}, {method}")
            pairs[method] = float(row["return"])

    if not by_scenario or args.method == args.baseline:
        raise SystemExit("two distinct methods with paired returns are required")
    method_returns = []
    baseline_returns = []
    for scenario, pairs in sorted(by_scenario.items()):
        for seed, values in pairs.items():
            if {args.method, args.baseline} != values.keys():
                raise SystemExit(f"missing paired return for {scenario}, seed {seed}")
        method_returns.append(sum(v[args.method] for v in pairs.values()) / len(pairs))
        baseline_returns.append(sum(v[args.baseline] for v in pairs.values()) / len(pairs))

    summary = summarize_paired(
        method_returns, baseline_returns, seed=args.seed,
        n_episodes=sum(len(pairs) for pairs in by_scenario.values()),
    )
    output = {
        "method": args.method,
        "baseline": args.baseline,
        "n_scenarios": summary.n_scenarios,
        "n_episode_pairs": summary.n_episodes,
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
