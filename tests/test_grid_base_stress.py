from __future__ import annotations

import pytest

from ruleshift.envs.grid.config import GridConfig
from ruleshift.evaluator.grid_base_stress import (
    analyze_base_stress,
    build_base_stress_scenarios,
    map_cluster_bootstrap,
    wilson_interval,
)


def _row(scenario_id: str, *, success: bool, estimate: str) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "success": success,
        "final_rule_estimate": estimate,
        "env_steps": 10,
        "tool_units": 0,
        "total_return": 1.0 if success else 0.0,
    }


def test_build_base_stress_scenarios_filters_only_shift_free_base_episodes():
    scenarios = build_base_stress_scenarios(3, 2, GridConfig())
    assert len(scenarios) == 6
    assert len({scenario.scenario_id for scenario in scenarios}) == 6
    assert all(scenario.scenario_type == "identification" for scenario in scenarios)
    assert all(scenario.initial_rule == "base" for scenario in scenarios)
    assert all(scenario.shift_rule is None for scenario in scenarios)


def test_wilson_interval_bounds_and_known_values():
    low, high = wilson_interval(0, 50)
    assert low == 0.0
    assert 0.0 < high < 0.1
    low, high = wilson_interval(49, 50)
    assert 0.85 < low < 0.95
    assert 0.99 < high < 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_map_cluster_bootstrap_is_deterministic_and_resamples_maps():
    rows = [
        _row("m0-s0", success=True, estimate="base"),
        _row("m0-s1", success=False, estimate="base"),
        _row("m1-s0", success=True, estimate="base"),
        _row("m1-s1", success=True, estimate="base"),
    ]
    first = map_cluster_bootstrap(
        rows, lambda row: not bool(row["success"]), n_boot=100, seed=7
    )
    second = map_cluster_bootstrap(
        rows, lambda row: not bool(row["success"]), n_boot=100, seed=7
    )
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_analyze_base_stress_reports_failures_and_misdiagnoses():
    diagnostic = [
        _row("m0-s0", success=True, estimate="base"),
        _row("m0-s1", success=True, estimate="slippery"),
        _row("m1-s0", success=True, estimate="base"),
        _row("m1-s1", success=False, estimate="slippery"),
    ]
    baseline = [
        _row("m0-s0", success=True, estimate="base"),
        _row("m0-s1", success=True, estimate="base"),
        _row("m1-s0", success=True, estimate="base"),
        _row("m1-s1", success=True, estimate="base"),
    ]
    result = analyze_base_stress(
        diagnostic, baseline, n_maps=2, seeds=2, n_boot=100
    )
    diag = result["methods"]["diagnostic_belief"]
    assert diag["task_failure"]["events"] == 1
    assert diag["misdiagnosis"]["events"] == 2
    assert diag["final_estimate_counts"] == {"base": 2, "slippery": 2}
    assert diag["misdiagnosis"]["map_cluster_degenerate"] is True
    assert result["paired"]["both"] == 3
    assert result["paired"]["only_diagnostic"] == 0
    assert result["paired"]["only_baseline"] == 1
    assert result["paired"]["neither"] == 0
    assert result["paired"]["mcnemar_exact_p"] == 1.0
    assert [row["scenario_id"] for row in result["diagnostic_failures"]] == ["m1-s1"]
    assert [row["scenario_id"] for row in result["diagnostic_misdiagnoses"]] == [
        "m0-s1",
        "m1-s1",
    ]
    assert [
        row["scenario_id"] for row in result["diagnostic_misdiagnoses_with_success"]
    ] == ["m0-s1"]


def test_analyze_base_stress_rejects_mismatched_scenario_sets():
    diagnostic = [_row("m0-s0", success=True, estimate="base")]
    baseline = [_row("m0-s1", success=True, estimate="base")]
    with pytest.raises(ValueError, match="scenario ids differ"):
        analyze_base_stress(
            diagnostic, baseline, n_maps=1, seeds=1, n_boot=10
        )