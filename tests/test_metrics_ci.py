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
