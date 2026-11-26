"""Paired bootstrap intervals with the scenario as the resampling unit.

Callers must pass ONE difference per independent scenario/layout/seed cluster,
never one per frame or decision. The interval is percentile-based.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class PairedSummary:
    mean_difference: float
    low: float
    high: float
    #: Independent units (scenarios/layout clusters), never frames or seeds.
    n_scenarios: int
    #: Episodes that were averaged into those units.
    n_episodes: int = 0


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
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scenario differences must be finite")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if min(values) == max(values):
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(math.fsum(sample) / n)
    boot_means.sort()
    low_index = int((alpha / 2.0) * n_boot)
    high_index = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot) - 1)
    return boot_means[low_index], boot_means[high_index]


def summarize_paired(
    method_returns: Sequence[float],
    baseline_returns: Sequence[float],
    seed: int = 0,
    n_boot: int = 2000,
    n_episodes: int | None = None,
) -> PairedSummary:
    """Summarise ``method - baseline`` paired by scenario.

    ``method_returns`` and ``baseline_returns`` must contain exactly one value
    per independent scenario, in the same scenario order. ``n_episodes`` only
    records how many underlying episodes were averaged into those units.
    """

    if len(method_returns) != len(baseline_returns):
        raise ValueError("paired return vectors must have equal length")
    differences = [m - b for m, b in zip(method_returns, baseline_returns)]
    low, high = paired_bootstrap_ci(differences, seed=seed, n_boot=n_boot)
    return PairedSummary(
        mean(differences),
        low,
        high,
        len(differences),
        n_episodes if n_episodes is not None else len(differences),
    )
