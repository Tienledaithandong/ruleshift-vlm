"""Paired bootstrap intervals with the scenario as the resampling unit.

Callers must pass ONE difference per independent scenario/layout/seed cluster,
never one per frame or decision. The interval is percentile-based.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class PairedSummary:
    mean_difference: float
    low: float
    high: float
    n_scenarios: int


def paired_bootstrap_ci(
    scenario_differences: Sequence[float],
    seed: int = 0,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return a percentile CI for the mean of scenario-level differences."""

    values = tuple(float(value) for value in scenario_differences)
    if not values:
        raise ValueError("at least one scenario difference is required")
    rng = random.Random(seed)
    n = len(values)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    low_index = int((alpha / 2.0) * n_boot)
    high_index = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot) - 1)
    return boot_means[low_index], boot_means[high_index]


def summarize_paired(
    method_returns: Sequence[float],
    baseline_returns: Sequence[float],
    seed: int = 0,
    n_boot: int = 2000,
) -> PairedSummary:
    """Summarise ``method - baseline`` paired by scenario."""

    if len(method_returns) != len(baseline_returns):
        raise ValueError("paired return vectors must have equal length")
    differences = [m - b for m, b in zip(method_returns, baseline_returns)]
    low, high = paired_bootstrap_ci(differences, seed=seed, n_boot=n_boot)
    return PairedSummary(mean(differences), low, high, len(differences))
