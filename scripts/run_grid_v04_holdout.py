"""Run the exploratory v0.4 base-policy holdout on fresh map layouts.

The holdout uses generator map indices 100-199, which are not part of the
v0.3 10-map study, the 100-map base-stress study, or the 100-map audit.
It compares the frozen v0.3 policy, the v0.4 candidate, and no adaptation on
the same 500 true-base episodes. This is an exploratory candidate evaluation,
not a gate result and not a Phase 1 authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ruleshift.agents.grid_baselines import (
    DiagnosticBeliefPolicy,
    DiagnosticBeliefPolicyV4,
    NoAdaptationPolicy,
)
from ruleshift.envs.grid.config import GridConfig, load_grid_config
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.evaluator.grid_base_stress import (
    analyze_base_stress,
    write_base_stress_report,
)
from ruleshift.evaluator.grid_runner import (
    GridEpisode,
    build_grid_scenarios,
    run_grid_benchmark,
)

ROOT = _ROOT

METHODS: tuple[str, ...] = (
    "diagnostic_belief_v3",
    "diagnostic_belief_v4",
    "no_adaptation",
)


def _factories() -> dict[str, Callable[[GridConfig, object], object]]:
    return {
        "diagnostic_belief_v3": lambda config, env: DiagnosticBeliefPolicy(config),
        "diagnostic_belief_v4": lambda config, env: DiagnosticBeliefPolicyV4(config),
        "no_adaptation": lambda config, env: NoAdaptationPolicy(config),
    }


def _holdout_scenarios(
    config: GridConfig, *, offset: int, maps: int, seeds: int
) -> tuple[GridScenario, ...]:
    total = offset + maps
    scenarios = build_grid_scenarios(total, seeds, config)
    selected = tuple(
        scenario
        for scenario in scenarios
        if scenario.scenario_type == "identification"
        and scenario.initial_rule == "base"
        and scenario.shift_rule is None
        and offset <= int(scenario.map_seed) < total
    )
    expected = maps * seeds
    if len(selected) != expected:
        raise ValueError(f"expected {expected} holdout episodes, found {len(selected)}")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_row(episode: GridEpisode) -> dict[str, object]:
    return {
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


def _write_rows(episodes: Sequence[GridEpisode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_episode_row(episode), sort_keys=True, ensure_ascii=False)
            + "\n"
            for episode in episodes
        ),
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _paired_success(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    left_index = {str(row["scenario_id"]): row for row in left}
    right_index = {str(row["scenario_id"]): row for row in right}
    if set(left_index) != set(right_index):
        raise ValueError("paired scenario ids differ")
    both = only_left = only_right = neither = 0
    for scenario_id in left_index:
        left_ok = bool(left_index[scenario_id]["success"])
        right_ok = bool(right_index[scenario_id]["success"])
        if left_ok and right_ok:
            both += 1
        elif left_ok:
            only_left += 1
        elif right_ok:
            only_right += 1
        else:
            neither += 1
    discordant = only_left + only_right
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(only_left, only_right) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "both_success": both,
        "only_left_success": only_left,
        "only_right_success": only_right,
        "neither_success": neither,
        "discordant": discordant,
        "exact_mcnemar_p": p_value,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--offset", type=int, default=100)
    parser.add_argument("--maps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--out",
        default="results/grid_phase0/v04_holdout_100x5.jsonl",
        help="raw JSONL output path",
    )
    parser.add_argument(
        "--analysis-dir",
        default="results/grid_phase0/v04_holdout_analysis",
        help="directory for v3/v4 analysis outputs",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.offset < 0 or args.maps < 1 or args.seeds < 1:
        raise ValueError("offset must be non-negative and maps/seeds positive")
    config = load_grid_config(ROOT / args.config)
    scenarios = _holdout_scenarios(
        config, offset=args.offset, maps=args.maps, seeds=args.seeds
    )
    episodes = run_grid_benchmark(_factories(), scenarios, config)
    raw_path = ROOT / args.out
    _write_rows(episodes, raw_path)
    rows = _read_rows(raw_path)
    by_method = {
        method: [row for row in rows if row["method"] == method]
        for method in METHODS
    }
    base_rows = by_method["no_adaptation"]
    raw_hash = _sha256(raw_path)
    analysis_dir = ROOT / args.analysis_dir
    payloads: dict[str, dict[str, object]] = {}
    for method, label in (
        ("diagnostic_belief_v3", "v3"),
        ("diagnostic_belief_v4", "v4"),
    ):
        payload = analyze_base_stress(
            by_method[method],
            base_rows,
            n_maps=args.maps,
            seeds=args.seeds,
            n_boot=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        payload["raw_path"] = str(raw_path.relative_to(ROOT))
        payload["raw_sha256"] = raw_hash
        payload["config"] = args.config
        payload["policy"] = method
        payload["map_offset"] = args.offset
        if label == "v4":
            payload["paired_vs_v3"] = _paired_success(
                by_method["diagnostic_belief_v4"],
                by_method["diagnostic_belief_v3"],
            )
        write_base_stress_report(payload, analysis_dir / label)
        payloads[label] = payload
    v3 = payloads["v3"]["methods"]["diagnostic_belief"]  # type: ignore[index]
    v4 = payloads["v4"]["methods"]["diagnostic_belief"]  # type: ignore[index]
    print(
        json.dumps(
            {
                "raw": str(raw_path),
                "analysis": str(analysis_dir),
                "raw_sha256": raw_hash,
                "v3_failure": v3["task_failure"],
                "v3_misdiagnosis": v3["misdiagnosis"],
                "v4_failure": v4["task_failure"],
                "v4_misdiagnosis": v4["misdiagnosis"],
                "v4_paired_vs_v3": payloads["v4"]["paired_vs_v3"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()