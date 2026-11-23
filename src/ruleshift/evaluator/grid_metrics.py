"""Aggregate metrics and paired scenario-level intervals for RuleShift-Grid.

An omitted key means "not measured" in a report: empty subgroups never produce
``nan``. Privileged labels are only read from an episode when a caller
explicitly attached them; the public decision rows are never mined for hidden
state.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.envs.grid.mapgen import all_hypotheses_solvable, generate_solvable_map
from ruleshift.evaluator.grid_reference import oracle_return
from ruleshift.evaluator.metrics_ci import PairedSummary, summarize_paired
from ruleshift.permissions import find_forbidden_fields, scan_package
from ruleshift.public_api.grid_rules import (
    GRID_HYPOTHESES,
    PublicGridMap,
    rules_consistent_with_all_clauses,
    rules_consistent_with_clause,
)
from ruleshift.public_api.grid_schemas import GridAction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruleshift.evaluator.grid_runner import GridEpisode

__all__ = ["audit_grid", "paired_grid_ci", "summarize_grid"]

_CALIBRATION_PROBABILITY_KEYS = (
    "shift_probability",
    "shift_trigger_probability",
    "change_point_probability",
)
_CALIBRATION_LABEL_KEYS = (
    "shift_label",
    "shift_occurred",
    "true_shift",
    "shifted",
)
_ECE_BINS = 10


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _first_attr(item: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(item, name):
            value = getattr(item, name)
            if value is not None:
                return value
    return None


def _first_key(row: Any, names: Sequence[str]) -> Any:
    if not isinstance(row, Mapping):
        return None
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return None


def _rows(episode: "GridEpisode") -> tuple[Mapping[str, Any], ...]:
    return (*tuple(getattr(episode, "evaluator_rows", ())),
            *tuple(getattr(episode, "public_rows", ())))


def _first_in_rows(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Any:
    for row in rows:
        value = _first_key(row, names)
        if value is not None:
            return value
    return None


def _expected_calibration_error(pairs: Sequence[tuple[float, float]]) -> float:
    bins: list[list[tuple[float, float]]] = [[] for _ in range(_ECE_BINS)]
    for probability, label in pairs:
        index = min(_ECE_BINS - 1, max(0, int(probability * _ECE_BINS)))
        bins[index].append((probability, label))
    total = len(pairs)
    return sum(
        (len(bucket) / total)
        * abs(_mean([p for p, _ in bucket]) - _mean([y for _, y in bucket]))
        for bucket in bins
        if bucket
    )


def _calibration_pairs(episodes: Sequence["GridEpisode"]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for episode in episodes:
        rows = _rows(episode)
        probability = _finite(
            _first_attr(episode, _CALIBRATION_PROBABILITY_KEYS)
        )
        if probability is None:
            probability = _finite(_first_in_rows(rows, _CALIBRATION_PROBABILITY_KEYS))
        raw_label = _first_attr(episode, _CALIBRATION_LABEL_KEYS)
        if raw_label is None:
            raw_label = _first_in_rows(rows, _CALIBRATION_LABEL_KEYS)
        if probability is None or raw_label is None:
            continue
        if 0.0 <= probability <= 1.0:
            pairs.append((probability, 1.0 if bool(raw_label) else 0.0))
    return pairs


def _tool_pr(episode: "GridEpisode") -> tuple[float, float] | None:
    precision = _finite(_first_attr(episode, ("tool_precision",)))
    recall = _finite(_first_attr(episode, ("tool_recall",)))
    if precision is not None and recall is not None:
        return precision, recall

    total = _finite(_first_attr(episode, ("total_tool_calls", "tool_calls_made")))
    useful = _finite(_first_attr(episode, ("useful_tool_calls", "informative_tool_calls")))
    needed = _finite(_first_attr(episode, ("needed_tool_calls", "required_tool_calls")))
    if any(value is None for value in (total, useful, needed)):
        calls = tuple(getattr(episode, "tool_calls", ()) or ())
        mappings = [call for call in calls if isinstance(call, Mapping)]
        if mappings:
            if total is None:
                total = float(len(mappings))
            if useful is None:
                useful = float(sum(1 for call in mappings if call.get("useful")))
            if needed is None:
                needed = float(
                    sum(1 for call in mappings if call.get("needed") or call.get("available"))
                )
    if any(value is None for value in (total, useful, needed)):
        return None
    return (
        useful / total if total else 0.0,  # type: ignore[operator]
        useful / needed if needed else 0.0,  # type: ignore[operator]
    )


def summarize_grid(episodes: Sequence["GridEpisode"]) -> dict[str, float]:
    """Summarize grid episodes, omitting every metric with no measured data."""

    rows = tuple(episodes)
    n = len(rows)
    summary: dict[str, float] = {"n_episodes": n}
    if n == 0:
        return summary

    summary["success_rate"] = sum(bool(row.success) for row in rows) / n
    summary["mean_return"] = _mean([float(row.total_return) for row in rows])
    summary["mean_env_steps"] = _mean([float(row.env_steps) for row in rows])
    summary["mean_tool_units"] = _mean([float(row.tool_units) for row in rows])
    summary["mean_total_cost"] = _mean([float(row.total_cost) for row in rows])

    adaptation = [
        float(value)
        for value in (getattr(row, "adaptation_steps", None) for row in rows)
        if value is not None
    ]
    if adaptation:
        summary["mean_adaptation_steps"] = _mean(adaptation)

    no_shift = [row for row in rows if row.scenario_type == "no_shift"]
    shifted = [row for row in rows if row.scenario_type == "change_detection"]
    if no_shift:
        summary["false_alarm_rate"] = sum(bool(row.false_alarm) for row in no_shift) / len(no_shift)
    if shifted:
        detected = sum(bool(row.shift_detected) for row in shifted)
        summary["detection_rate"] = detected / len(shifted)
        false_positives = sum(
            bool(row.false_alarm) or bool(row.shift_detected) for row in no_shift
        )
        if detected + false_positives:
            summary["detection_precision"] = detected / (detected + false_positives)

    regrets = [
        max(0.0, oracle - float(row.total_return))
        for row in rows
        for oracle in (oracle_return(row),)
        if oracle is not None
    ]
    if regrets:
        summary["mean_regret"] = _mean(regrets)

    pairs = _calibration_pairs(rows)
    if pairs:
        summary["brier_score"] = _mean(
            [(probability - label) ** 2 for probability, label in pairs]
        )
        summary["expected_calibration_error"] = _expected_calibration_error(pairs)

    # Only episodes whose policy actually measured expected information gain
    # contribute to these metrics. Omit the keys entirely when nobody did.
    measured_gain = [
        value
        for value in (
            _finite(getattr(row, "information_gain_bits", None)) for row in rows
        )
        if value is not None
    ]
    if measured_gain:
        information_gain = sum(measured_gain)
        # Calls, not tool units: INSPECT spends 2 units but is still one call.
        tool_calls = sum(int(getattr(row, "tool_calls_made", 0) or 0) for row in rows)
        total_cost = sum(float(row.total_cost) for row in rows)
        if tool_calls > 0:
            summary["information_gain_per_tool_call"] = information_gain / tool_calls
        if total_cost > 0.0:
            summary["information_gain_per_total_cost"] = information_gain / total_cost

    hazard_damage = []
    deaths = []
    for row in rows:
        observation_rows = _rows(row)
        damage = _finite(_first_attr(row, ("hazard_damage", "total_hazard_damage")))
        if damage is None:
            damage = _finite(_first_in_rows(observation_rows, ("hazard_damage",)))
        if damage is not None:
            hazard_damage.append(damage)
        death_count = _finite(_first_attr(row, ("deaths", "n_deaths", "died")))
        if death_count is None:
            death_count = _finite(_first_in_rows(observation_rows, ("deaths", "n_deaths", "died")))
        if death_count is not None:
            deaths.append(death_count)
    if hazard_damage:
        summary["mean_hazard_damage"] = _mean(hazard_damage)
    if deaths:
        summary["deaths"] = float(sum(deaths))

    tool_prs = [value for value in (_tool_pr(row) for row in rows) if value is not None]
    if tool_prs:
        summary["tool_precision"] = _mean([pair[0] for pair in tool_prs])
        summary["tool_recall"] = _mean([pair[1] for pair in tool_prs])

    summary["json_valid_rate"] = sum(bool(row.json_valid) for row in rows) / n
    summary["valid_action_rate"] = sum(bool(row.valid_action) for row in rows) / n
    summary["mean_output_tokens"] = _mean([float(row.output_tokens) for row in rows])

    latency = []
    for row in rows:
        value = _finite(_first_attr(row, ("latency_ms", "mean_latency_ms")))
        if value is None:
            value = _finite(_first_in_rows(_rows(row), ("latency_ms",)))
        if value is not None:
            latency.append(value)
    if latency:
        summary["mean_latency_ms"] = _mean(latency)

    summary["cache_hit_rate"] = sum(bool(row.cache_hit) for row in rows) / n
    return summary


def paired_grid_ci(
    method_episodes: Sequence["GridEpisode"],
    baseline_episodes: Sequence["GridEpisode"],
    n_boot: int = 2000,
) -> PairedSummary:
    """Paired cost interval with the *scenario* as the resampling unit.

    Episodes are grouped by ``scenario_id`` and their ``total_cost`` is averaged
    over ``run_seed`` first, so repeated seeds never inflate the sample size.
    The scenario-level means are then paired between method and baseline and
    passed to :func:`summarize_paired` (lower cost gives a negative difference).
    """

    def by_scenario(episodes: Sequence["GridEpisode"]) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for episode in episodes:
            grouped[str(episode.scenario_id)].append(float(episode.total_cost))
        return grouped

    method = by_scenario(method_episodes)
    baseline = by_scenario(baseline_episodes)
    scenario_ids = sorted(set(method) & set(baseline))
    if not scenario_ids:
        raise ValueError("method and baseline episodes share no scenarios")

    method_means = [_mean(method[scenario_id]) for scenario_id in scenario_ids]
    baseline_means = [_mean(baseline[scenario_id]) for scenario_id in scenario_ids]
    return summarize_paired(
        method_means,
        baseline_means,
        n_boot=n_boot,
        n_episodes=len(method_episodes) + len(baseline_episodes),
    )


def audit_grid(config: GridConfig, n_maps: int = 20) -> dict[str, Any]:
    """Run the structural RuleShift-Grid invariants as one deterministic audit.

    All checks run on the CPU and never use held-out data. The returned dict
    has ``passed``, ``checks`` (one boolean per invariant) and ``failures``
    (human-readable details for every failed invariant).
    """

    failures: list[str] = []
    checks: dict[str, bool] = {}

    def run(name: str, check: Any) -> None:
        before = len(failures)
        try:
            ok = bool(check())
        except Exception as exc:  # an audit must report, not crash
            failures.append(f"{name}: raised {type(exc).__name__}: {exc}")
            ok = False
        checks[name] = ok and len(failures) == before

    try:
        reference_map = generate_solvable_map(0, config).map.public
    except Exception as exc:  # keep auditing the other invariants
        failures.append(f"reference_map: raised {type(exc).__name__}: {exc}")
        reference_map = None

    def render_scenario(rule_id: str) -> GridScenario:
        return GridScenario(
            scenario_id=f"audit-render-{rule_id}",
            scenario_type="identification",
            initial_rule=rule_id,
            shift_rule=None,
            shift_time=None,
            map_seed=0,
            run_seed=0,
        )

    def generator_solvability() -> bool:
        for seed in range(int(n_maps)):
            grid_map = generate_solvable_map(seed, config).map
            if not all_hypotheses_solvable(
                grid_map.public, GRID_HYPOTHESES, config.door_required
            ):
                failures.append(
                    f"generator_solvability: seed {seed} is not solvable under every hypothesis"
                )
        return True

    def render_rule_independence() -> bool:
        frames = []
        for rule_id in ("base", "remap_swap"):
            env = GridEnv(config, observation_mode="visual")
            env.reset(
                seed=0,
                options={"map": reference_map, "scenario": render_scenario(rule_id)},
            )
            frames.append(env.render())
        if not np.array_equal(frames[0], frames[1]):
            failures.append(
                "render_rule_independence: identical public states rendered differently "
                "under different hidden rules"
            )
        return True

    def public_observation_no_leak() -> bool:
        env = GridEnv(config)
        observation, _ = env.reset(
            seed=0,
            options={"map": reference_map, "scenario": render_scenario("base")},
        )
        payload = observation if isinstance(observation, dict) else observation.to_payload()
        leaks = find_forbidden_fields(payload)
        if leaks:
            failures.append(f"public_observation_no_leak: privileged fields {leaks}")
        return True

    def tool_accounting() -> bool:
        tool_config = replace(
            config,
            tool_budget=3,
            inspect_cost=2,
            read_sign_cost=1,
            inspect_range=2,
        )
        tool_map = PublicGridMap(
            width=7,
            height=7,
            walls=frozenset(),
            start=(3, 3),
            markers={"red": (3, 2), "blue": (2, 3), "green": (4, 3)},
            key=None,
            door=None,
            signpost=(3, 4),
        )
        scenario = GridScenario(
            scenario_id="audit-tool-accounting",
            scenario_type="no_shift",
            initial_rule="base",
            shift_rule=None,
            shift_time=None,
            map_seed=0,
            run_seed=0,
        )

        def expect(label: str, actual: Any, expected: Any) -> None:
            if actual != expected:
                failures.append(
                    f"tool_accounting: {label} was {actual!r}, expected {expected!r}"
                )

        env = GridEnv(tool_config)
        _, info = env.reset(seed=0, options={"map": tool_map, "scenario": scenario})
        expect("initial budget", info["tool_budget_remaining"], 3)
        expect("initial env steps", info["env_steps"], 0)

        _, _, _, _, info = env.step(GridAction("INSPECT", (3, 2)))
        expect("budget after INSPECT", info["tool_budget_remaining"], 1)
        expect("env steps after INSPECT", info["env_steps"], 0)

        _, _, _, _, info = env.step(GridAction("INSPECT", (3, 2)))
        expect("rejected second INSPECT", info.get("tool_rejected"), "insufficient_budget")
        expect("budget after rejected INSPECT", info["tool_budget_remaining"], 1)

        _, _, _, _, info = env.step(GridAction("READ_SIGN"))
        expect("budget after READ_SIGN", info["tool_budget_remaining"], 0)
        expect("env steps after READ_SIGN", info["env_steps"], 0)

        _, _, _, _, info = env.step(GridAction("READ_SIGN"))
        expect("rejected second READ_SIGN", info.get("tool_rejected"), "insufficient_budget")
        expect("final budget", info["tool_budget_remaining"], 0)
        expect("final env steps", info["env_steps"], 0)
        return True

    def sign_partiality() -> bool:
        for rule in GRID_HYPOTHESES:
            for side, clause in enumerate(rule.sign_clauses):
                if len(rules_consistent_with_clause(side, clause)) < 2:
                    failures.append(
                        f"sign_partiality: side {side} of {rule.rule_id} identifies one rule"
                    )
            if len(rules_consistent_with_all_clauses(rule.sign_clauses)) < 2:
                failures.append(
                    f"sign_partiality: the full sign of {rule.rule_id} identifies one rule"
                )
        return True

    def permissions_scan() -> bool:
        package_root = Path(__file__).resolve().parents[1]
        for name in ("agents", "public_api"):
            path = package_root / name
            if not path.exists():
                path = Path("src/ruleshift") / name
            violations = scan_package(path)
            if violations:
                failures.append(
                    f"permissions_scan: {name} has {len(violations)} violation(s): "
                    f"{violations[:3]}"
                )
        return True

    run("generator_solvability", generator_solvability)
    run("render_rule_independence", render_rule_independence)
    run("public_observation_no_leak", public_observation_no_leak)
    run("tool_accounting", tool_accounting)
    run("sign_partiality", sign_partiality)
    run("permissions_scan", permissions_scan)

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": failures,
    }
