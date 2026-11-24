"""Week 11 statistical stress tests on synthetic closed-loop runs.

Sweeps vary the hidden shift prior and feedback calibration; seed/boot checks
show how much conclusions move with seeds. These are pipeline stress tests on
stub methods, not empirical findings about a VLM.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import pstdev, pvariance
from typing import Sequence

from ruleshift.envs.config import EnvConfig
from ruleshift.evaluator.closed_loop import run_closed_loop
from ruleshift.evaluator.closed_loop import build_family_a_scenarios
from ruleshift.evaluator.metrics_ci import summarize_paired

Scenario = tuple[str, EnvConfig]


@dataclass(frozen=True)
class SensitivityRow:
    parameter: str
    value: float
    method: str
    mean_return: float
    std_return: float
    n: int


def _with_parameter(
    scenarios: Sequence[Scenario], field: str, value: float
) -> tuple[Scenario, ...]:
    return tuple(
        (
            scenario_id,
            replace(config, **{field: value}),
        )
        for scenario_id, config in scenarios
    )


def _sweep(
    parameter: str,
    values: Sequence[float],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[SensitivityRow, ...]:
    base = tuple(item for item in build_family_a_scenarios() if not item[1].disable_shift)[:4]
    rows: list[SensitivityRow] = []
    for value in values:
        scenarios = _with_parameter(base, parameter, value)
        result = run_closed_loop(scenarios, methods, seeds, ())
        for method in methods:
            returns = [
                episode.total_return
                for (_, _, name), episode in result.episodes.items()
                if name == method
            ]
            rows.append(
                SensitivityRow(
                    parameter=parameter,
                    value=float(value),
                    method=method,
                    mean_return=sum(returns) / len(returns),
                    std_return=pstdev(returns),
                    n=len(returns),
                )
            )
    return tuple(rows)


def sweep_shift_prior(
    values: Sequence[float],
    methods: tuple[str, ...] = ("never_probe", "m1_stub"),
    seeds: tuple[int, ...] = (1,),
) -> tuple[SensitivityRow, ...]:
    return _sweep("shift_hazard", values, methods, seeds)


def sweep_calibration(
    values: Sequence[float],
    methods: tuple[str, ...] = ("never_probe", "m1_stub"),
    seeds: tuple[int, ...] = (1,),
) -> tuple[SensitivityRow, ...]:
    return _sweep("feedback_noise", values, methods, seeds)


def seed_return_variance(
    scenarios: Sequence[Scenario],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, float]:
    """Per-method variance of returns across scenarios and seeds."""

    result = run_closed_loop(tuple(scenarios), methods, seeds, ())
    variances: dict[str, float] = {}
    for method in methods:
        returns = [
            episode.total_return
            for (_, _, name), episode in result.episodes.items()
            if name == method
        ]
        variances[method] = pvariance(returns)
    return variances


def paired_ci_across_boot_seeds(
    scenarios: Sequence[Scenario],
    method: str,
    baseline: str,
    seeds: tuple[int, ...] = (1,),
    boot_seeds: tuple[int, ...] = (0, 1, 2),
) -> tuple[tuple[float, float], ...]:
    """Show how a paired CI moves when only the bootstrap seed changes."""

    result = run_closed_loop(tuple(scenarios), (method, baseline), seeds, ())
    by_scenario: dict[str, dict[str, list[float]]] = {}
    for (scenario_id, _, name), episode in result.episodes.items():
        by_scenario.setdefault(scenario_id, {}).setdefault(name, []).append(
            episode.total_return
        )

    method_returns = [
        sum(values[method]) / len(values[method]) for values in by_scenario.values()
    ]
    baseline_returns = [
        sum(values[baseline]) / len(values[baseline])
        for values in by_scenario.values()
    ]
    return tuple(
        (
            summary.low,
            summary.high,
        )
        for summary in (
            summarize_paired(method_returns, baseline_returns, seed=boot_seed)
            for boot_seed in boot_seeds
        )
    )
