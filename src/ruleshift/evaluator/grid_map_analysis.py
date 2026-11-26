"""Map-clustered paired analysis for the static identification endpoint.

The design has 30 maps, five run seeds and six initial rules.  Episodes that
share a map share geometry, so treating all 900 episodes as independent would
understate uncertainty.  The primary paired estimand here is the equal-map mean
of the within-map terminal-MAP accuracy difference.  Uncertainty is a percentile
bootstrap over maps.  The p-value is a deterministic Monte-Carlo sign-flip
randomization test over the same map-level differences.  Episode-level McNemar
tests live in :mod:`grid_identification` and are supplementary only.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from ruleshift.evaluator.grid_identification import (
    EvaluationError,
    PairingError,
    paired_summary,
)

__all__ = [
    "MAP_METRIC_NAMES",
    "holm_adjust",
    "map_family_summary",
    "map_level_paired",
    "map_sign_test",
    "wilson_interval",
]

MAP_METRIC_NAMES = (
    "map_correct",
    "task_success",
    "task_and_map_correct",
    "commitment_correct",
)


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float]:
    """Two-sided Wilson score interval for a binomial proportion."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "estimate": proportion,
        "lower": max(0.0, centre - half_width),
        "upper": min(1.0, centre + half_width),
        "z": z,
    }


def _metric_value(record: Any, metric: str) -> float:
    if metric not in MAP_METRIC_NAMES:
        raise EvaluationError(f"unknown map-level metric {metric!r}")
    if metric == "commitment_correct" and not bool(
        getattr(record, "declaration_available", False)
    ):
        raise EvaluationError(
            "commitment_correct is unavailable for a policy without a declaration API"
        )
    value = getattr(record, metric)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise EvaluationError(f"metric {metric!r} is not numeric")


def _unique_by_id(records: Sequence[Any], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        scenario_id = str(record.scenario_id)
        if scenario_id in result:
            raise PairingError(f"duplicate scenario id in {label}: {scenario_id}")
        result[scenario_id] = record
    return result


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise EvaluationError("cannot take a percentile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _group_by_map(records: Iterable[Any]) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for record in records:
        grouped[int(record.map_seed)].append(record)
    return dict(grouped)


def map_sign_test(differences: Sequence[float]) -> dict[str, Any]:
    """Exact sign test over map-level differences; zeros are discarded."""

    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    zero = len(differences) - positive - negative
    n = positive + negative
    if n == 0:
        return {
            "positive": 0,
            "negative": 0,
            "zero": zero,
            "p_one_sided_positive": 1.0,
            "p_two_sided": 1.0,
        }
    denominator = 2**n
    lower = sum(math.comb(n, index) for index in range(0, positive + 1))
    upper = sum(math.comb(n, index) for index in range(positive, n + 1))
    p_one_sided = upper / denominator
    p_two_sided = min(1.0, 2.0 * min(lower, upper) / denominator)
    return {
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "p_one_sided_positive": p_one_sided,
        "p_two_sided": p_two_sided,
    }


def map_level_paired(
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    metric: str = "map_correct",
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 20_000,
    seed: int = 20_261_125,
    expected_episodes_per_map: int | None = None,
) -> dict[str, Any]:
    """Paired map-clustered comparison of two methods on static episodes.

    The map is the resampling unit.  Each method must contain exactly the same
    unique static scenario ids, and each map must be balanced across methods.
    """

    if bootstrap_samples < 100 or permutation_samples < 100:
        raise ValueError("at least 100 bootstrap and permutation samples are required")
    left_by_id = _unique_by_id(left, "left")
    right_by_id = _unique_by_id(right, "right")
    if set(left_by_id) != set(right_by_id):
        missing = sorted(set(left_by_id) - set(right_by_id))
        extra = sorted(set(right_by_id) - set(left_by_id))
        raise PairingError(
            f"paired methods have different scenario ids; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    for record in list(left_by_id.values()) + list(right_by_id.values()):
        if str(record.scenario_type) != "identification":
            raise EvaluationError(
                "map-level static analysis received a non-identification episode"
            )
    left_maps = _group_by_map(left_by_id.values())
    right_maps = _group_by_map(right_by_id.values())
    if set(left_maps) != set(right_maps):
        raise PairingError("paired methods have different map ids")
    map_ids = sorted(left_maps)
    if len(map_ids) < 2:
        raise EvaluationError("map-level analysis requires at least two maps")
    differences: list[float] = []
    for map_id in map_ids:
        left_rows = left_maps[map_id]
        right_rows = right_maps[map_id]
        if len(left_rows) != len(right_rows):
            raise PairingError(f"map {map_id} is unbalanced across methods")
        if not left_rows:
            raise PairingError(f"map {map_id} has no episodes")
        if (
            expected_episodes_per_map is not None
            and len(left_rows) != expected_episodes_per_map
        ):
            raise EvaluationError(
                f"map {map_id} has {len(left_rows)} episodes, expected "
                f"{expected_episodes_per_map}"
            )
        left_mean = sum(_metric_value(row, metric) for row in left_rows) / len(left_rows)
        right_mean = sum(_metric_value(row, metric) for row in right_rows) / len(right_rows)
        differences.append(left_mean - right_mean)
    observed = sum(differences) / len(differences)

    rng = random.Random(seed)
    bootstrap: list[float] = []
    n = len(differences)
    for _ in range(bootstrap_samples):
        total = 0.0
        for _ in range(n):
            total += differences[rng.randrange(n)]
        bootstrap.append(total / n)
    bootstrap.sort()
    ci_low = _percentile(bootstrap, 0.025)
    ci_high = _percentile(bootstrap, 0.975)

    rng = random.Random(seed + 1)
    hits = 0
    for _ in range(permutation_samples):
        total = 0.0
        for difference in differences:
            total += difference if rng.random() < 0.5 else -difference
        if total / n >= observed - 1e-15:
            hits += 1
    p_one_sided = (hits + 1) / (permutation_samples + 1)
    hits_two = 0
    for _ in range(permutation_samples):
        total = 0.0
        for difference in differences:
            total += difference if rng.random() < 0.5 else -difference
        if abs(total / n) >= abs(observed) - 1e-15:
            hits_two += 1
    p_two_sided = (hits_two + 1) / (permutation_samples + 1)

    sign = map_sign_test(differences)
    return {
        "metric": metric,
        "maps": len(map_ids),
        "episodes_left": len(left_by_id),
        "episodes_right": len(right_by_id),
        "episodes_per_map": len(left_by_id) // len(map_ids),
        "mean_difference": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": 0.95,
        "ci_method": "percentile cluster bootstrap over maps",
        "bootstrap_samples": bootstrap_samples,
        "p_one_sided_positive": p_one_sided,
        "p_two_sided": p_two_sided,
        "permutation_samples": permutation_samples,
        "permutation_seed": seed,
        "maps_positive": sign["positive"],
        "maps_negative": sign["negative"],
        "maps_zero": sign["zero"],
        "sign_test_p_one_sided_positive": sign["p_one_sided_positive"],
        "sign_test_p_two_sided": sign["p_two_sided"],
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, returned in input order."""

    if not p_values:
        return {}
    for label, value in p_values.items():
        if not 0.0 <= float(value) <= 1.0 or not math.isfinite(float(value)):
            raise ValueError(f"invalid p-value for {label}: {value!r}")
    ordered = sorted(p_values.items(), key=lambda item: (float(item[1]), item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (label, value) in enumerate(ordered):
        candidate = (count - index) * float(value)
        running = max(running, min(1.0, candidate))
        adjusted[label] = running
    return {label: adjusted[label] for label in p_values}


def _static_method_records(records: Sequence[Any]) -> dict[str, list[Any]]:
    by_method: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        if str(record.scenario_type) == "identification":
            by_method[str(record.method)].append(record)
    return dict(by_method)


def map_family_summary(
    records: Sequence[Any],
    *,
    candidate: str = "diagnostic_belief_v4",
    historical: str = "diagnostic_belief_v3",
    baseline: str = "no_adaptation",
    minimum_effect: float = 0.05,
    seed: int = 20_261_125,
    expected_maps: int | None = None,
    expected_episodes_per_map: int | None = None,
) -> dict[str, Any]:
    """Return the prespecified two-hypothesis family plus descriptive contrast.

    H1 and H2 are one-sided positive map-level terminal-MAP comparisons.  The
    candidate-versus-historical contrast is descriptive and is not part of the
    Holm rejection family.
    """

    if not 0.0 < minimum_effect < 1.0:
        raise ValueError("minimum_effect must lie strictly inside (0, 1)")
    by_method = _static_method_records(records)
    required = {candidate, historical, baseline}
    missing = sorted(required - set(by_method))
    if missing:
        raise EvaluationError(f"missing methods for map-level family: {missing}")

    h1 = map_level_paired(
        by_method[candidate],
        by_method[baseline],
        metric="map_correct",
        seed=seed,
        expected_episodes_per_map=expected_episodes_per_map,
    )
    h2 = map_level_paired(
        by_method[historical],
        by_method[baseline],
        metric="map_correct",
        seed=seed,
        expected_episodes_per_map=expected_episodes_per_map,
    )
    descriptive = map_level_paired(
        by_method[candidate],
        by_method[historical],
        metric="map_correct",
        seed=seed,
        expected_episodes_per_map=expected_episodes_per_map,
    )
    if expected_maps is not None and h1["maps"] != expected_maps:
        raise EvaluationError(
            f"family contains {h1['maps']} maps, expected {expected_maps}"
        )
    raw = {"H1": h1["p_one_sided_positive"], "H2": h2["p_one_sided_positive"]}
    adjusted = holm_adjust(raw)

    hypotheses: list[dict[str, Any]] = []
    for label, result in (("H1", h1), ("H2", h2)):
        hypotheses.append(
            {
                "id": label,
                "comparison": (
                    f"{candidate} - {baseline}"
                    if label == "H1"
                    else f"{historical} - {baseline}"
                ),
                "direction": "greater",
                "metric": "terminal_map_accuracy",
                "estimand": "equal-map mean paired risk difference",
                "mean_difference": result["mean_difference"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "p_one_sided": raw[label],
                "p_holm": adjusted[label],
                "minimum_effect": minimum_effect,
                "reject": bool(
                    adjusted[label] < 0.05
                    and result["mean_difference"] >= minimum_effect
                    and result["ci_low"] > 0.0
                ),
            }
        )

    paired_episode = paired_summary(
        by_method[candidate],
        by_method[historical],
        metric="map_correct",
    )
    return {
        "primary_endpoint": "terminal_map_accuracy",
        "family_size": 2,
        "alpha": 0.05,
        "expected_maps": expected_maps,
        "expected_episodes_per_map": expected_episodes_per_map,
        "holm": adjusted,
        "hypotheses": hypotheses,
        "descriptive_candidate_vs_historical": {
            "mean_difference": descriptive["mean_difference"],
            "ci_low": descriptive["ci_low"],
            "ci_high": descriptive["ci_high"],
            "p_two_sided": descriptive["p_two_sided"],
            "not_in_confirmatory_family": True,
        },
        "supplementary_episode_mcnemar": paired_episode,
    }
