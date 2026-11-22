"""Paired bootstrap intervals resampled at the scenario level."""

from __future__ import annotations

import pytest

from ruleshift.evaluator.metrics_ci import paired_bootstrap_ci, summarize_paired


def test_constant_positive_effect_has_degenerate_ci() -> None:
    low, high = paired_bootstrap_ci([2.0] * 6, seed=0, n_boot=200)
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(2.0)


def test_ci_stays_inside_the_observed_range() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = paired_bootstrap_ci(values, seed=1, n_boot=500)
    assert min(values) <= low <= high <= max(values)


def test_summarize_paired_returns_mean_and_interval() -> None:
    summary = summarize_paired([10.0, 12.0, 9.0], [8.0, 9.0, 10.0], seed=2)
    assert summary.mean_difference == pytest.approx(4.0 / 3.0)
    assert summary.low <= summary.mean_difference <= summary.high
    assert summary.n_scenarios == 3


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_paired([1.0, 2.0], [1.0], seed=0)


# ---------------------------------------------------------------------------
# Input hygiene: CIs must never be computed from non-finite or degenerate input.
# ---------------------------------------------------------------------------
def test_nan_and_infinite_differences_are_rejected() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0, bad])


def test_single_scenario_is_allowed_and_brackets_the_point() -> None:
    low, high = paired_bootstrap_ci([2.5])
    assert low == pytest.approx(2.5)
    assert high == pytest.approx(2.5)


def test_invalid_alpha_and_n_boot_are_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], alpha=0.0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], n_boot=0)


def test_summary_records_episode_count_separately_from_scenarios() -> None:
    summary = summarize_paired([1.0, 2.0], [0.0, 0.0], n_episodes=6)
    assert summary.n_scenarios == 2
    assert summary.n_episodes == 6
