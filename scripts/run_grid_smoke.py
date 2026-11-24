"""Synthetic CPU smoke run for RuleShift-Grid.

This script is a plumbing check, never a source of scientific findings. It runs
the public-only baselines and the evaluator-only oracle on the development
config and writes a plainly labelled synthetic report.

Usage: python scripts/run_grid_smoke.py [--quick] [--maps N] [--seeds N]
                                       [--methods ...] [--config PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Always prefer this checkout's src over any installed copy: the installed
# site-packages build does not carry the grid modules or prompt data files.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ruleshift.agents.grid_baselines import (
    InfoGainPolicy,
    NoAdaptationPolicy,
    PublicAdvisorPolicy,
    RandomProbePolicy,
)
from ruleshift.envs.grid.config import load_grid_config
from ruleshift.evaluator.grid_metrics import paired_grid_ci, summarize_grid
from ruleshift.evaluator.grid_runner import (
    OraclePolicy,
    build_grid_scenarios,
    run_grid_benchmark,
)

METHODS = ("no_adaptation", "public_advisor", "random_probe", "info_gain", "oracle")
DISCLAIMER = "SYNTHETIC CPU SMOKE — NOT A SCIENTIFIC FINDING"
_BASELINE = "no_adaptation"
_KEY_METRICS = (
    "success_rate",
    "mean_return",
    "mean_total_cost",
    "false_alarm_rate",
    "detection_rate",
    "mean_adaptation_steps",
)


def build_methods() -> dict:
    """Factories for every smoke method, keyed by the brief's method names."""

    return {
        "no_adaptation": lambda config, env: NoAdaptationPolicy(config),
        "public_advisor": lambda config, env: PublicAdvisorPolicy(config),
        "random_probe": lambda config, env: RandomProbePolicy(config, seed=0),
        "info_gain": lambda config, env: InfoGainPolicy(config),
        "oracle": lambda config, env: OraclePolicy(env, config),
    }


def _markdown(payload: dict) -> str:
    lines = [
        f"# {DISCLAIMER}",
        "",
        "Synthetic CPU-only smoke run of the RuleShift-Grid plumbing. "
        "These numbers are **not** a benchmark result, a bound, or a scientific finding.",
        "",
        f"- Config: `{payload['config']}`",
        f"- Quick mode: {payload['quick']}",
        f"- Maps: {payload['maps']}; seeds: {payload['seeds']}; "
        f"scenarios: {payload['scenarios']}",
        f"- Methods: {', '.join(payload['methods'])}",
        "",
        "## Per-method summary",
        "",
    ]
    for name, summary in payload["summary"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for key in _KEY_METRICS:
            if key in summary:
                lines.append(f"| {key} | {summary[key]:.4f} |")
        lines.append("")

    lines.extend(
        [
            "## Paired cost vs no_adaptation (scenario-level bootstrap)",
            "",
            "| Method | Mean difference | Low | High | Scenarios |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, summary in payload["paired_vs_no_adaptation"].items():
        lines.append(
            f"| {name} | {summary['mean_difference']:+.4f} | {summary['low']:+.4f} "
            f"| {summary['high']:+.4f} | {summary['n_scenarios']} |"
        )
    lines.extend(
        [
            "",
            "Negative differences mean the method spent fewer total cost units. "
            "The scenario (not the seed or frame) is the resampling unit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--maps", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    parser.add_argument("--out", default="reports")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    maps = args.maps if args.maps is not None else (5 if args.quick else 100)
    seeds = args.seeds if args.seeds is not None else (2 if args.quick else 5)
    methods = tuple(args.methods) if args.methods is not None else METHODS

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    config = load_grid_config(config_path)
    scenarios = build_grid_scenarios(maps, seeds, config)

    factories = build_methods()
    episodes = run_grid_benchmark(
        {name: factories[name] for name in methods}, scenarios, config
    )

    summary = {
        name: summarize_grid([episode for episode in episodes if episode.method == name])
        for name in methods
    }
    paired = {}
    baseline = [episode for episode in episodes if episode.method == _BASELINE]
    if baseline and _BASELINE in methods:
        for name in methods:
            if name != _BASELINE:
                paired[name] = asdict(
                    paired_grid_ci(
                        [episode for episode in episodes if episode.method == name],
                        baseline,
                        n_boot=500,
                    )
                )

    payload = {
        "synthetic": True,
        "disclaimer": DISCLAIMER,
        "config": str(args.config),
        "quick": bool(args.quick),
        "maps": int(maps),
        "seeds": int(seeds),
        "scenarios": len(scenarios),
        "methods": list(methods),
        "summary": summary,
        "paired_vs_no_adaptation": paired,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid_smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "grid_smoke.md").write_text(_markdown(payload), encoding="utf-8")
    print(f"SYNTHETIC CPU SMOKE — NOT A SCIENTIFIC FINDING: {out / 'grid_smoke.md'}")
    print(f"scenarios={len(scenarios)} episodes={len(episodes)} methods={len(methods)}")


if __name__ == "__main__":
    main()
