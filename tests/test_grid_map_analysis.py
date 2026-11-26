"""Tests for map-clustered paired analysis and the Holm family."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ruleshift.evaluator.grid_identification import EvaluationError, PairingError
from ruleshift.evaluator.grid_map_analysis import (
    holm_adjust,
    map_family_summary,
    map_level_paired,
    map_sign_test,
    wilson_interval,
)


def _record(
    method: str,
    map_seed: int,
    index: int,
    *,
    correct: bool = True,
    scenario_type: str = "identification",
    declaration_available: bool = True,
    commitment_correct: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=f"m{map_seed}-s0-{scenario_type}-x-{index}",
        method=method,
        map_seed=map_seed,
        scenario_type=scenario_type,
        map_correct=correct,
        task_success=True,
        task_and_map_correct=correct,
        declaration_available=declaration_available,
        commitment_correct=commitment_correct,
    )


def _method_records(method: str, values: dict[int, list[bool]]) -> list[SimpleNamespace]:
    records = []
    for map_seed, outcomes in values.items():
        for index, correct in enumerate(outcomes):
            records.append(_record(method, map_seed, index, correct=correct))
    return records


def test_map_level_identical_and_all_discordant():
    same = _method_records("a", {0: [True, False], 1: [True, True], 2: [False, False]})
    result = map_level_paired(same, same, bootstrap_samples=500, permutation_samples=500)
    assert result["maps"] == 3
    assert result["mean_difference"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["p_one_sided_positive"] == pytest.approx(1.0)

    left = _method_records("a", {0: [True, True], 1: [True, True], 2: [True, True]})
    right = _method_records("b", {0: [False, False], 1: [False, False], 2: [False, False]})
    result = map_level_paired(left, right, bootstrap_samples=500, permutation_samples=500)
    assert result["mean_difference"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0
    assert result["maps_positive"] == 3
    assert result["p_one_sided_positive"] < 0.2


def test_map_level_equal_weights_not_episode_weights():
    left = [
        _record("a", 0, 0, correct=True),
        _record("a", 1, 0, correct=True),
        _record("a", 1, 1, correct=True),
    ]
    right = [
        _record("b", 0, 0, correct=False),
        _record("b", 1, 0, correct=True),
        _record("b", 1, 1, correct=True),
    ]
    result = map_level_paired(left, right, bootstrap_samples=500, permutation_samples=500)
    assert result["mean_difference"] == pytest.approx(0.5)
    assert result["episodes_per_map"] != 3


def test_map_level_rejects_non_static_or_mismatched_sets():
    left = _method_records("a", {0: [True], 1: [True]})
    right = _method_records("b", {0: [False], 1: [False]})
    shifted = _record("a", 0, 0, correct=True, scenario_type="change_detection")
    with pytest.raises(EvaluationError):
        map_level_paired([left[0], shifted, left[1]], right)
    with pytest.raises(PairingError):
        map_level_paired(left, right + [_record("b", 1, 9, correct=False)])
    with pytest.raises(EvaluationError):
        map_level_paired(
            [_record("a", 0, 0, declaration_available=False)],
            [_record("b", 0, 0, declaration_available=False)],
            metric="commitment_correct",
        )


def test_wilson_interval_and_exact_design_checks():
    interval = wilson_interval(300, 300)
    assert interval["estimate"] == 1.0
    assert interval["lower"] < 1.0
    assert interval["upper"] > 0.9999
    assert wilson_interval(0, 150)["upper"] < 0.03
    with pytest.raises(ValueError):
        wilson_interval(1, 0)

    left = _method_records("a", {0: [True, True], 1: [True, True]})
    right = _method_records("b", {0: [False, False], 1: [False, False]})
    with pytest.raises(EvaluationError):
        map_level_paired(
            left,
            right,
            bootstrap_samples=200,
            permutation_samples=200,
            expected_episodes_per_map=3,
        )


def test_holm_and_sign_test_are_deterministic():
    assert holm_adjust({"primary": 0.01, "secondary": 0.04}) == {
        "primary": pytest.approx(0.02),
        "secondary": pytest.approx(0.04),
    }
    assert holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5}) == {
        "a": pytest.approx(0.003),
        "b": pytest.approx(0.04),
        "c": pytest.approx(0.5),
    }
    assert holm_adjust({}) == {}
    result = map_sign_test([1.0, -1.0, 0.0])
    assert result["positive"] == 1
    assert result["negative"] == 1
    assert result["zero"] == 1
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_map_family_summary_is_two_one_sided_hypotheses():
    baseline = _method_records("no_adaptation", {0: [False, False], 1: [False, False], 2: [False, False]})
    historical = _method_records("diagnostic_belief_v3", {0: [True, True], 1: [True, True], 2: [True, True]})
    candidate = _method_records("diagnostic_belief_v4", {0: [True, True], 1: [True, True], 2: [True, True]})
    result = map_family_summary(
        baseline + historical + candidate,
        seed=7,
    )
    assert result["family_size"] == 2
    assert [item["id"] for item in result["hypotheses"]] == ["H1", "H2"]
    assert result["primary_endpoint"] == "terminal_map_accuracy"
    assert result["descriptive_candidate_vs_historical"]["not_in_confirmatory_family"]
    assert result["supplementary_episode_mcnemar"]["episodes"] == 6
    assert result["hypotheses"][0]["minimum_effect"] == 0.05
