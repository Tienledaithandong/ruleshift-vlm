"""Run the Week 7 closed-loop Family A study (SYNTHETIC stub methods).

Usage:
    python scripts/run_closed_loop.py --seeds 1 2 --out reports --data data

No real VLM is used. The output validates the loop, cost accounting and trace
separation; it makes no claim about model behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.envs.config import load_dev_grid
from ruleshift.evaluator.closed_loop import (
    build_family_a_scenarios,
    run_closed_loop,
    write_episode_files,
    write_report,
)

DEFAULT_METHODS = (
    "never_probe",
    "random_probe",
    "direct_stub",
    "m1_stub",
    "diva_stub",
    "true_rule_oracle",
)
DEFAULT_COMPARISONS = (
    ("m1_stub", "direct_stub"),
    ("m1_stub", "diva_stub"),
    ("m1_stub", "never_probe"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--out", default="reports")
    parser.add_argument("--data", default="data")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument(
        "--call-cap",
        type=int,
        default=None,
        help="per-episode VLM call cap applied to every method",
    )
    args = parser.parse_args()

    scenarios = build_family_a_scenarios(Path(args.config))
    budget = load_dev_grid(Path(args.config))["model_call_cap"]["per_episode"]
    result = run_closed_loop(
        scenarios,
        DEFAULT_METHODS,
        tuple(args.seeds),
        DEFAULT_COMPARISONS,
        max_vlm_calls_per_episode=budget if args.call_cap is None else args.call_cap,
    )
    data_dir = Path(args.data)
    report_dir = Path(args.out)
    public_path, evaluator_path = write_episode_files(result, data_dir)
    report_path = write_report(result, report_dir / "w7_family_A.md")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(DEFAULT_METHODS)
    returns = [
        [ep.total_return for (_, _, name), ep in result.episodes.items() if name == method]
        for method in methods
    ]
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot(returns, showmeans=True)
    axis.set_xticks(range(1, len(methods) + 1), methods)
    axis.set_ylabel("episodic return (synthetic)")
    axis.set_title("Week 7 Family A closed loop — STUB METHODS, no VLM")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    image_path = report_dir / "w7_family_A_return.png"
    fig.savefig(image_path)
    plt.close(fig)

    print("SYNTHETIC closed-loop run — stub methods, no VLM was run")
    print(f"episodes={len(result.episodes)} public={public_path} evaluator={evaluator_path}")
    print(f"report={report_path} plot={image_path}")
    for key, summary in result.summaries.items():
        print(
            f"{key[0]} - {key[1]}: {summary.mean_difference:+.3f} "
            f"[{summary.low:+.3f}, {summary.high:+.3f}]"
        )


if __name__ == "__main__":
    main()
