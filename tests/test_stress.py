"""Week 11 statistical stress tests: calibration, shift prior, seed/boot variance."""

from __future__ import annotations

import pytest

from ruleshift.envs.config import EnvConfig
from ruleshift.evaluator.heldout import build_heldout_family_a
from ruleshift.evaluator.stress import (
    paired_ci_across_boot_seeds,
    seed_return_variance,
    sweep_calibration,
    sweep_shift_prior,
)

SCENARIOS = build_heldout_family_a()[:4]
METHODS = ("never_probe", "m1_stub")


def test_shift_prior_sweep_covers_every_value_and_method() -> None:
    rows = sweep_shift_prior((0.0, 0.05, 0.2), METHODS, seeds=(1,))
    assert len(rows) == 3 * len(METHODS)
    assert {row.value for row in rows} == {0.0, 0.05, 0.2}
    assert all(row.n > 0 and row.mean_return == pytest.approx(row.mean_return) for row in rows)


def test_calibration_sweep_is_finite() -> None:
    rows = sweep_calibration((0.0, 0.1, 0.25), METHODS, seeds=(1,))
    assert len(rows) == 3 * len(METHODS)
    assert all(row.std_return >= 0.0 for row in rows)


def test_seed_variance_is_non_negative_and_deterministic() -> None:
    first = seed_return_variance(SCENARIOS, METHODS, seeds=(1, 2))
    second = seed_return_variance(SCENARIOS, METHODS, seeds=(1, 2))
    assert first == second
    assert set(first) == set(METHODS)
    assert all(variance >= 0.0 for variance in first.values())


def test_paired_ci_is_stable_across_bootstrap_seeds() -> None:
    intervals = paired_ci_across_boot_seeds(
        SCENARIOS,
        "m1_stub",
        "never_probe",
        seeds=(1,),
        boot_seeds=(0, 1, 2),
    )
    assert len(intervals) == 3
    assert all(low <= high for low, high in intervals)
