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

import yaml

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.model_check import check_model, model_settings, readiness_errors
from ruleshift.agents.vlm_backend import TransformersVLMBackend
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
REAL_METHODS = (
    "never_probe",
    "random_probe",
    "direct",
    "m1",
    "diva",
    "true_rule_oracle",
)
REAL_COMPARISONS = (
    ("m1", "direct"),
    ("m1", "diva"),
    ("m1", "never_probe"),
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
    parser.add_argument(
        "--backend",
        choices=("stub", "transformers"),
        default="stub",
        help="keep the default stub unless a real frozen checkpoint is ready",
    )
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0")
    parser.add_argument(
        "--input-mode",
        choices=("text", "rgb"),
        default="text",
        help="P1 public text or the P0 public RGB frame",
    )
    parser.add_argument("--cache", default=None, help="response cache path for a real run")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="explicitly allow fetching the pinned checkpoint on this run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reuse an existing real-run output directory",
    )
    args = parser.parse_args()

    if args.backend == "stub" and args.input_mode != "text":
        parser.error("--input-mode rgb requires --backend transformers")
    if args.backend == "transformers":
        data_dir = Path(args.data)
        report_path = Path(args.out) / "w7_family_A.md"
        existing = [
            path
            for path in (
                data_dir / "episodes.jsonl",
                data_dir / "evaluator_only.jsonl",
                report_path,
            )
            if path.exists()
        ]
        if existing and not args.force:
            parser.error(
                "real-run outputs already exist ("
                + ", ".join(str(path) for path in existing)
                + "); choose fresh --data/--out paths or pass --force"
            )

    if args.backend == "transformers":
        with (Path(args.models_config)).open(encoding="utf-8") as handle:
            models = yaml.safe_load(handle)
        key = args.model_key or models["default"]
        if key not in models["models"]:
            parser.error(f"unknown model key: {key}")
        settings = model_settings(models, key, args.device, args.allow_download)
        errors = readiness_errors(settings, "pilot", {})
        if errors:
            parser.error(", ".join(errors))
        readiness = check_model(**settings)
        if not readiness.ready:
            for message in readiness.messages:
                print(message)
            raise SystemExit("no usable checkpoint; real closed loop refused")
        loader_settings = {
            name: value for name, value in settings.items() if name != "allow_download"
        }
        model = models["models"][key]
        backend: object | None = TransformersVLMBackend(
            **loader_settings,
            dtype=model["dtype"],
            max_new_tokens=model["max_new_tokens"],
            input_mode=args.input_mode,
        )
        cache_path = Path(args.cache) if args.cache else Path(args.data) / "response_cache.jsonl"
        cache: ResponseCache | None = ResponseCache(cache_path)
        methods = REAL_METHODS
        comparisons = REAL_COMPARISONS
    else:
        backend = None
        cache = None
        methods = DEFAULT_METHODS
        comparisons = DEFAULT_COMPARISONS

    scenarios = build_family_a_scenarios(Path(args.config))
    budget = load_dev_grid(Path(args.config))["model_call_cap"]["per_episode"]
    result = run_closed_loop(
        scenarios,
        methods,
        tuple(args.seeds),
        comparisons,
        backend=backend,
        cache=cache,
        max_vlm_calls_per_episode=budget if args.call_cap is None else args.call_cap,
    )
    data_dir = Path(args.data)
    report_dir = Path(args.out)
    public_path, evaluator_path = write_episode_files(result, data_dir)
    report_path = write_report(result, report_dir / "w7_family_A.md")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_list = list(methods)
    returns = [
        [ep.total_return for (_, _, name), ep in result.episodes.items() if name == method]
        for method in method_list
    ]
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot(returns, showmeans=True)
    axis.set_xticks(range(1, len(method_list) + 1), method_list)
    axis.set_ylabel("episodic return" + (" (synthetic)" if result.synthetic else ""))
    axis.set_title(
        "Week 7 Family A closed loop — "
        + ("STUB METHODS, no VLM" if result.synthetic else "real frozen backend")
    )
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    image_path = report_dir / "w7_family_A_return.png"
    fig.savefig(image_path)
    plt.close(fig)

    print(
        "SYNTHETIC closed-loop run — stub methods, no VLM was run"
        if result.synthetic
        else "REAL closed-loop run — frozen model backend"
    )
    print(f"episodes={len(result.episodes)} public={public_path} evaluator={evaluator_path}")
    print(f"report={report_path} plot={image_path}")
    for key, summary in result.summaries.items():
        print(
            f"{key[0]} - {key[1]}: {summary.mean_difference:+.3f} "
            f"[{summary.low:+.3f}, {summary.high:+.3f}]"
        )


if __name__ == "__main__":
    main()
