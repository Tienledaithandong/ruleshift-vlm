"""Generate Week 11 stress-test tables (SYNTHETIC stub runs)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import ruleshift  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.evaluator.heldout import build_heldout_family_a
from ruleshift.evaluator.stress import (
    paired_ci_across_boot_seeds,
    seed_return_variance,
    sweep_calibration,
    sweep_shift_prior,
)

METHODS = ("never_probe", "m1_stub")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/w11_stress.md")
    parser.add_argument("--csv", default="results/stress.csv")
    args = parser.parse_args()

    prior_rows = sweep_shift_prior((0.0, 0.05, 0.2), METHODS, seeds=(1,))
    noise_rows = sweep_calibration((0.0, 0.1, 0.25), METHODS, seeds=(1,))
    scenarios = build_heldout_family_a()[:4]
    variances = seed_return_variance(scenarios, METHODS, seeds=(1, 2))
    intervals = paired_ci_across_boot_seeds(
        scenarios, "m1_stub", "never_probe", seeds=(1,), boot_seeds=(0, 1, 2)
    )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "method", "mean_return", "std_return", "n"])
        for row in (*prior_rows, *noise_rows):
            writer.writerow(
                [row.parameter, row.value, row.method, f"{row.mean_return:.6f}", f"{row.std_return:.6f}", row.n]
            )

    lines = [
        "# Week 11 — Statistical stress tests (SYNTHETIC)",
        "",
        "**No VLM was run.** These sweeps stress the evaluation pipeline with "
        "deterministic stub methods; they are not empirical findings.",
        "",
        "## Shift-prior sensitivity",
        "",
        "| shift hazard | method | mean return | sd | n |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in prior_rows:
        lines.append(
            f"| {row.value:.2f} | {row.method} | {row.mean_return:+.3f} | "
            f"{row.std_return:.3f} | {row.n} |"
        )
    lines += [
        "",
        "## Feedback-calibration sensitivity",
        "",
        "| noise | method | mean return | sd | n |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in noise_rows:
        lines.append(
            f"| {row.value:.2f} | {row.method} | {row.mean_return:+.3f} | "
            f"{row.std_return:.3f} | {row.n} |"
        )
    lines += [
        "",
        "## Seed and bootstrap stability",
        "",
        "| method | return variance across scenarios/seeds |",
        "|---|---:|",
    ]
    for method, variance in variances.items():
        lines.append(f"| {method} | {variance:.3f} |")
    lines += ["", "Paired 95% CI (m1_stub − never_probe) by bootstrap seed:"]
    for boot_seed, (low, high) in zip((0, 1, 2), intervals):
        lines.append(f"- seed {boot_seed}: [{low:+.3f}, {high:+.3f}]")
    lines += [
        "",
        "Resampling is at the scenario level. A stable interval across bootstrap "
        "seeds is a pipeline property, not evidence that a real effect exists.",
        "",
    ]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path} and {path}")


if __name__ == "__main__":
    main()
