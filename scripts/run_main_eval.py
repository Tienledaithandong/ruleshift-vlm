"""Week 9 held-out main evaluation (SYNTHETIC stub run).

Usage:
    python scripts/run_main_eval.py --seeds 101 102 --out results --report reports

No VLM checkpoint is available here; the test configuration stays unfrozen.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from ruleshift.evaluator.family_b_reference import FamilyBCosts, delta_probe_family_b
from ruleshift.evaluator.heldout import (
    build_heldout_family_a,
    run_heldout_evaluation,
    write_result_files,
)

DEFAULT_METHODS = ("direct_stub", "diva_stub", "m1_stub", "immediate_stub")


def write_family_b_reference(results_dir: Path) -> Path:
    """Held-out analytical Family B cells (not VLM results)."""

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "main_family_B.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["safe_steps", "slip_stable", "slip_slippery", "probe_steps", "delta_probe", "preferred"]
        )
        for safe_steps in (6, 10, 14):
            for slip_slippery in (0.55, 0.75):
                config = FamilyBCosts(
                    safe_steps=safe_steps,
                    bridge_steps=1,
                    recovery_steps=12,
                    probe_steps=3,
                    slip_stable=0.02,
                    slip_slippery=slip_slippery,
                )
                delta = delta_probe_family_b(config, 0.5)
                writer.writerow(
                    [
                        safe_steps,
                        0.02,
                        slip_slippery,
                        3,
                        f"{delta:+.6f}",
                        "probe" if delta > 0 else "task",
                    ]
                )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--out", default="results")
    parser.add_argument("--report", default="reports")
    parser.add_argument("--config", default="configs/preregistered_test.yaml")
    parser.add_argument(
        "--call-cap",
        type=int,
        default=None,
        help="per-episode VLM call cap applied to every method",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    budget = config["call_and_step_budget"]
    scenarios = build_heldout_family_a(max_env_steps=budget["max_env_steps"])
    result = run_heldout_evaluation(
        scenarios,
        tuple(args.methods),
        tuple(args.seeds),
        max_vlm_calls_per_episode=(budget["max_vlm_calls_per_episode"]
                                   if args.call_cap is None else args.call_cap),
    )
    paths = write_result_files(result, Path(args.out), Path(args.report))
    family_b_path = write_family_b_reference(Path(args.out))

    print("SYNTHETIC held-out evaluation — stub methods, no VLM was run")
    print(f"episodes={len(result.episodes)} csv={paths['csv']} family_b={family_b_path}")
    for key, summary in result.summaries.items():
        print(
            f"{key[0]} - {key[1]}: {summary.mean_difference:+.3f} "
            f"[{summary.low:+.3f}, {summary.high:+.3f}]"
        )


if __name__ == "__main__":
    main()
