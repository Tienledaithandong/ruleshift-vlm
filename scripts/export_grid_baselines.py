"""Export public CPU baselines for the full RuleShift-Grid scenario set.

The export deliberately excludes the oracle: it is a privileged diagnostic and
must never be merged into the public baseline comparison. Rows are JSONL and
contain only episode summaries, never public observations or evaluator traces.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ruleshift.agents.grid_baselines import (
    DiagnosticBeliefPolicy,
    InfoGainPolicy,
    NoAdaptationPolicy,
    PublicAdvisorPolicy,
    RandomProbePolicy,
)
from ruleshift.envs.grid.config import GridConfig, load_grid_config
from ruleshift.evaluator.grid_runner import build_grid_scenarios, run_grid_benchmark

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "no_adaptation",
    "public_advisor",
    "random_probe",
    "info_gain",
    "diagnostic_belief",
)
_ALIASES = {
    "NoAdaptationPolicy": "no_adaptation",
    "PublicAdvisorPolicy": "public_advisor",
    "RandomProbePolicy": "random_probe",
    "InfoGainPolicy": "info_gain",
    "DiagnosticBeliefPolicy": "diagnostic_belief",
}


def _factories() -> dict[str, Callable[[GridConfig, object], object]]:
    return {
        "no_adaptation": lambda config, env: NoAdaptationPolicy(config),
        "public_advisor": lambda config, env: PublicAdvisorPolicy(config),
        "random_probe": lambda config, env: RandomProbePolicy(config, seed=0),
        "info_gain": lambda config, env: InfoGainPolicy(config),
        "diagnostic_belief": lambda config, env: DiagnosticBeliefPolicy(config),
    }


def _normalize_methods(methods: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for method in methods:
        key = _ALIASES.get(str(method), str(method))
        if key not in METHODS:
            raise ValueError(f"unknown baseline method: {method!r}")
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("at least one baseline method is required")
    return tuple(normalized)


def export_grid_baselines(
    config: GridConfig,
    methods: Sequence[str],
    out_path: str | Path,
    *,
    scenarios: Sequence[object] | None = None,
) -> Path:
    """Run selected public baselines and write one JSONL row per episode."""

    selected = _normalize_methods(methods)
    available = _factories()
    episode_scenarios = (
        tuple(build_grid_scenarios(100, 5, config))
        if scenarios is None
        else tuple(scenarios)
    )
    episodes = run_grid_benchmark(
        {method: available[method] for method in selected},
        episode_scenarios,
        config,
    )
    rows = []
    for episode in episodes:
        rows.append(
            {
                "scenario_id": episode.scenario_id,
                "method": episode.method,
                "success": bool(episode.success),
                "total_return": float(episode.total_return),
                "env_steps": int(episode.env_steps),
                "tool_units": int(episode.tool_units),
                "advisor_calls": int(episode.advisor_calls),
                "total_cost": float(episode.total_cost),
                "adaptation_steps": episode.adaptation_steps,
                "false_alarm": bool(episode.false_alarm),
                "final_rule_estimate": episode.final_rule_estimate,
                "scenario_type": episode.scenario_type,
                "shift_detected": bool(episode.final_rule_estimate != "base"),
            }
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--out", default="results/grid_baselines/baselines.jsonl")
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="comma-separated public baseline names",
    )
    parser.add_argument(
        "--maps",
        type=int,
        default=100,
        help="number of maps (default: the full 100-map grid)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="paired run seeds per map (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_grid_config(ROOT / args.config)
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    scenarios = tuple(build_grid_scenarios(args.maps, args.seeds, config))
    out = export_grid_baselines(
        config, methods, ROOT / args.out, scenarios=scenarios
    )
    print(out)


if __name__ == "__main__":
    main()
