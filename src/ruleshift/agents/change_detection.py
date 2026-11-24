"""Public-signal change detection for optional slow-reasoner gating.

This module deliberately knows nothing about the environment or evaluator.
Callers supply prediction residuals produced by an agent-side model and
calibrate thresholds on no-shift development episodes. It is a small CPU-only
building block; it does not train a world model or imply that a detector has
been validated for either RuleShift family.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def standardized_residual_score(
    actual: Sequence[float],
    predicted_mean: Sequence[float],
    predicted_variance: Sequence[float],
    *,
    variance_floor: float = 1e-8,
) -> float:
    """Return RMS standardized error for one model transition prediction.

    ``actual``, ``predicted_mean`` and ``predicted_variance`` must describe the
    same non-empty vector. The variance floor prevents a zero-variance model
    from causing division by zero. The result is a surprise score, not a
    calibrated probability; calibrate the sequential threshold separately.
    """

    if not (len(actual) == len(predicted_mean) == len(predicted_variance)):
        raise ValueError("actual, predicted_mean and predicted_variance must have equal lengths")
    if len(actual) == 0:
        raise ValueError("transition prediction vectors must not be empty")
    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")

    squared_z: list[float] = []
    for observed, mean, variance in zip(actual, predicted_mean, predicted_variance):
        values = (float(observed), float(mean), float(variance))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("prediction values must be finite")
        observed, mean, variance = values
        if variance < 0.0:
            raise ValueError("predicted variances must be non-negative")
        z = (observed - mean) / math.sqrt(max(variance, variance_floor))
        squared_z.append(z * z)
    return math.sqrt(sum(squared_z) / len(squared_z))


@dataclass
class CUSUMDetector:
    """One-sided cumulative surprise detector with an explicit call gate.

    ``drift`` is the expected score level to ignore on each observation.
    ``threshold`` must be calibrated from no-shift development sequences for
    the chosen score and drift. ``update`` returns true when the cumulative
    statistic exceeds that threshold.
    """

    drift: float
    threshold: float
    statistic: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.drift) or self.drift < 0.0:
            raise ValueError("drift must be finite and non-negative")
        if math.isnan(self.threshold) or self.threshold < 0.0:
            raise ValueError("threshold must be non-negative and not NaN")
        if not math.isfinite(self.statistic) or self.statistic < 0.0:
            raise ValueError("statistic must be finite and non-negative")

    def update(self, score: float) -> bool:
        """Accumulate one finite, non-negative surprise score."""

        value = float(score)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("score must be finite and non-negative")
        self.statistic = max(0.0, self.statistic + value - self.drift)
        return self.statistic > self.threshold

    def reset(self) -> None:
        """Clear accumulated evidence at an episode boundary or after review."""

        self.statistic = 0.0


def calibrate_cusum_threshold(
    no_shift_score_sequences: Iterable[Sequence[float]],
    *,
    drift: float,
    false_alarm_rate: float = 0.05,
) -> float:
    """Choose an episode-level threshold from no-shift development runs.

    The calibration statistic is the maximum CUSUM value in each supplied
    episode. The finite-sample order statistic is conservative: when there
    are too few calibration episodes to support the requested rate, the
    function returns ``math.inf`` (a detector that never triggers) instead of
    claiming an unsupported false-alarm guarantee.
    """

    if not math.isfinite(drift) or drift < 0.0:
        raise ValueError("drift must be finite and non-negative")
    if not math.isfinite(false_alarm_rate) or not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must be between 0 and 1")

    maxima: list[float] = []
    for sequence in no_shift_score_sequences:
        statistic = 0.0
        maximum = 0.0
        for score in sequence:
            value = float(score)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("calibration scores must be finite and non-negative")
            statistic = max(0.0, statistic + value - drift)
            maximum = max(maximum, statistic)
        maxima.append(maximum)

    if not maxima:
        raise ValueError("at least one no-shift calibration episode is required")

    # Split-conformal style upper order statistic for an episode-level FAR.
    rank = math.ceil((len(maxima) + 1) * (1.0 - false_alarm_rate))
    if rank > len(maxima):
        return math.inf
    return sorted(maxima)[rank - 1]

