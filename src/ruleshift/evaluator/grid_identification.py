"""Independent, versioned scoring for RuleShift-Grid identification episodes.

This module deliberately does not reuse the legacy Phase-0 gate helpers. The
legacy helpers score task completion and a joint success/estimate match; those
are useful historical labels but they are not independent identification
accuracy. Here task success, terminal MAP accuracy, explicit commitment status,
posterior quality, tool spend and shift timing are separate fields.

Explicit commitments are only scored when the policy exposes a declaration
API. A policy without that API is reported as commitment_unavailable; this
module never invents a declaration from a terminal MAP or from task success.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS, GRID_TOOL_ACTIONS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruleshift.evaluator.grid_runner import GridEpisode

EVALUATION_VERSION = "grid-identification-v1"
TOP_THRESHOLD = 0.90
MARGIN_THRESHOLD = 0.10
TIE_EPS = 1e-6

_RULE_ORDER = {rule: index for index, rule in enumerate(GRID_RULE_IDS)}

__all__ = [
    "DeclaredCommit",
    "EpisodeRecord",
    "EvaluationError",
    "PairingError",
    "build_episode_record",
    "classify_posterior",
    "is_settled_posterior",
    "mcnemar_exact",
    "paired_summary",
    "posterior_brier",
    "posterior_nll",
    "summarize_records",
    "shift_summary",
    "validate_posterior",
    "write_jsonl",
]


class EvaluationError(ValueError):
    """A record or posterior violates the evaluator contract."""


class PairingError(EvaluationError):
    """Paired methods do not contain exactly the same unique episode ids."""


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationError(f"expected a finite number, got {value!r}")
    return number


def validate_posterior(
    posterior: Mapping[str, float] | None,
    *,
    rules: Sequence[str] = GRID_RULE_IDS,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Validate a full-menu terminal posterior and return a plain dict."""

    if posterior is None:
        raise EvaluationError("posterior is missing")
    if not isinstance(posterior, Mapping):
        raise EvaluationError("posterior must be a mapping")
    expected = set(rules)
    actual = set(posterior)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvaluationError(
            f"posterior keys mismatch; missing={missing}, extra={extra}"
        )
    normalized: dict[str, float] = {}
    for rule in rules:
        probability = _finite_number(posterior[rule])
        if probability < 0.0:
            raise EvaluationError(f"negative probability for {rule}")
        normalized[rule] = probability
    total = sum(normalized.values())
    if abs(total - 1.0) > tolerance:
        raise EvaluationError(f"posterior mass is {total}, expected 1.0")
    return normalized


def classify_posterior(
    posterior: Mapping[str, float],
) -> dict[str, Any]:
    """Return MAP, margin, settlement and tie status for one posterior."""

    normalized = validate_posterior(posterior)
    ordered = sorted(
        normalized.items(),
        key=lambda item: (-item[1], _RULE_ORDER.get(item[0], 999)),
    )
    top_rule, top_probability = ordered[0]
    second_probability = ordered[1][1]
    margin = top_probability - second_probability
    tie = margin <= TIE_EPS
    settled = top_probability >= TOP_THRESHOLD and margin >= MARGIN_THRESHOLD
    return {
        "top_rule": top_rule,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "margin": margin,
        "tie": bool(tie),
        "settled": bool(settled and not tie),
    }


def is_settled_posterior(posterior: Mapping[str, float]) -> bool:
    return bool(classify_posterior(posterior)["settled"])


def posterior_brier(
    posterior: Mapping[str, float],
    true_rule: str,
    *,
    rules: Sequence[str] = GRID_RULE_IDS,
) -> float:
    normalized = validate_posterior(posterior, rules=rules)
    if true_rule not in normalized:
        raise EvaluationError(f"true rule {true_rule!r} is not in the menu")
    # Conventional multiclass Brier score: sum of squared class errors.
    return sum(
        (normalized[rule] - (1.0 if rule == true_rule else 0.0)) ** 2
        for rule in rules
    )


def posterior_nll(
    posterior: Mapping[str, float],
    true_rule: str,
    *,
    rules: Sequence[str] = GRID_RULE_IDS,
) -> float:
    normalized = validate_posterior(posterior, rules=rules)
    if true_rule not in normalized:
        raise EvaluationError(f"true rule {true_rule!r} is not in the menu")
    return -math.log(max(normalized[true_rule], 1e-12))


@dataclass(frozen=True)
class DeclaredCommit:
    """An explicit, pre-terminal rule declaration emitted by a policy."""

    step: int
    declared_rule: str
    confidence: float
    posterior_margin: float
    source: str = "policy"

    def validate(self, *, terminal_step: int) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise EvaluationError("declaration step must be an integer")
        if self.step < 0 or self.step > terminal_step:
            raise EvaluationError(
                f"declaration step {self.step} is outside 0..{terminal_step}"
            )
        if self.declared_rule not in _RULE_ORDER:
            raise EvaluationError(f"unknown declared rule {self.declared_rule!r}")
        _finite_number(self.confidence)
        _finite_number(self.posterior_margin)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise EvaluationError("declaration confidence must lie in [0, 1]")
        if not 0.0 <= float(self.posterior_margin) <= 1.0:
            raise EvaluationError("declaration posterior margin must lie in [0, 1]")


@dataclass(frozen=True)
class EpisodeRecord:
    """One scored episode. Task, identification and commitment stay separate."""

    evaluation_version: str
    scenario_id: str
    method: str
    scenario_type: str
    map_seed: int
    run_seed: int
    layout_sha256: str
    initial_rule: str
    terminal_true_rule: str
    shift_rule: str | None
    scheduled_shift_time: int | None
    actual_shift_time: int | None
    shift_censored: bool
    belief_shift_event_time: int | None
    belief_shift_false_alarm: bool
    task_success: bool
    env_steps: int
    tool_units: int
    budget_cap: int
    terminated: bool
    truncated: bool
    termination_reason: str
    terminal_posterior: dict[str, float] | None
    posterior_valid: bool
    posterior_error: str | None
    terminal_map: str
    terminal_top_probability: float | None
    terminal_margin: float | None
    terminal_tie: bool
    map_correct: bool
    task_and_map_correct: bool
    successful_misdiagnosis: bool
    task_failure_map_correct: bool
    declaration_available: bool
    declared_rule: str | None
    declared_step: int | None
    declared_confidence: float | None
    declared_margin: float | None
    commitment_status: str
    settled_declaration: bool
    commitment_correct: bool
    earlier_wrong_commitments: int
    earlier_wrong_commitment_rules: tuple[str, ...]
    brier_score: float | None
    nll: float | None
    tool_events: tuple[dict[str, Any], ...] = ()
    code_commit: str = "unknown"
    config_sha256: str | None = None
    manifest_sha256: str = "unknown"

    @property
    def zero_tool_success(self) -> bool:
        return bool(self.task_success and self.tool_units == 0)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["zero_tool_success"] = self.zero_tool_success
        return row


def _tool_events(
    episode: "GridEpisode", config: GridConfig
) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for row in episode.public_rows:
        action = row.get("chosen_action")
        if action not in GRID_TOOL_ACTIONS:
            continue
        if action == "READ_SIGN":
            cost = int(getattr(config, "read_sign_cost", 1))
        elif action == "INSPECT":
            cost = int(getattr(config, "inspect_cost", 2))
        else:
            cost = 0
        value = row.get("last_tool_value")
        fallback = bool(row.get("fallback", False))
        events.append(
            {
                "action": str(action),
                "step": int(row.get("env_steps", 0)),
                "target": row.get("tool_target"),
                "cost": cost,
                "successful": bool(value is not None and not fallback),
                "value": value,
                "fallback": fallback,
            }
        )
    return tuple(events)


def build_episode_record(
    episode: "GridEpisode",
    scenario: GridScenario,
    config: GridConfig,
    *,
    layout_sha256_value: str,
    declarations: Sequence[DeclaredCommit] | None = None,
    declaration_available: bool = False,
    code_commit: str = "unknown",
    config_sha256: str | None = None,
    manifest_sha256: str = "unknown",
) -> EpisodeRecord:
    """Build one independent record without reading hidden state in the policy."""

    if declaration_available and declarations is None:
        declarations = ()
    if not declaration_available and declarations:
        raise EvaluationError(
            "declarations supplied but declaration_available is false"
        )
    true_terminal_rule = str(scenario.initial_rule)
    if episode.evaluator_rows:
        true_terminal_rule = str(
            episode.evaluator_rows[-1].get("true_rule", true_terminal_rule)
        )
    terminal_step = int(episode.env_steps)

    posterior: dict[str, float] | None = None
    posterior_valid = False
    posterior_error: str | None = None
    terminal_map = str(episode.final_rule_estimate)
    terminal_top: float | None = None
    terminal_margin: float | None = None
    terminal_tie = False
    try:
        posterior = validate_posterior(episode.terminal_posterior)
        summary = classify_posterior(posterior)
        posterior_valid = True
        terminal_map = str(summary["top_rule"])
        terminal_top = float(summary["top_probability"])
        terminal_margin = float(summary["margin"])
        terminal_tie = bool(summary["tie"])
    except EvaluationError as exc:
        posterior_error = str(exc)

    map_correct = terminal_map == true_terminal_rule
    task_success = bool(episode.success)
    valid_events: list[DeclaredCommit] = []
    if declaration_available:
        for event in declarations or ():
            event.validate(terminal_step=terminal_step)
            valid_events.append(event)
        valid_events.sort(key=lambda item: item.step)

    declared_rule: str | None = None
    declared_step: int | None = None
    declared_confidence: float | None = None
    declared_margin: float | None = None
    settled_declaration = False
    earlier_wrong_rules: list[str] = []
    if declaration_available and valid_events:
        for event in valid_events[:-1]:
            if (
                event.declared_rule != scenario.initial_rule
                and float(event.confidence) >= TOP_THRESHOLD
                and float(event.posterior_margin) >= MARGIN_THRESHOLD
            ):
                earlier_wrong_rules.append(event.declared_rule)
    if not declaration_available:
        commitment_status = "unavailable"
    elif not valid_events:
        commitment_status = "unresolved"
    else:
        final = valid_events[-1]
        declared_rule = final.declared_rule
        declared_step = final.step
        declared_confidence = float(final.confidence)
        declared_margin = float(final.posterior_margin)
        settled_declaration = bool(
            declared_confidence >= TOP_THRESHOLD
            and declared_margin >= MARGIN_THRESHOLD
        )
        if not settled_declaration:
            commitment_status = "unresolved"
        elif declared_rule == scenario.initial_rule:
            commitment_status = "correct_commit"
        else:
            commitment_status = "wrong_commit"

    brier_score = (
        posterior_brier(posterior, true_terminal_rule)
        if posterior_valid and posterior is not None
        else None
    )
    nll = (
        posterior_nll(posterior, true_terminal_rule)
        if posterior_valid and posterior is not None
        else None
    )
    return EpisodeRecord(
        evaluation_version=EVALUATION_VERSION,
        scenario_id=scenario.scenario_id,
        method=str(episode.method),
        scenario_type=str(scenario.scenario_type),
        map_seed=int(scenario.map_seed),
        run_seed=int(scenario.run_seed),
        layout_sha256=str(layout_sha256_value),
        initial_rule=str(scenario.initial_rule),
        terminal_true_rule=true_terminal_rule,
        shift_rule=scenario.shift_rule,
        scheduled_shift_time=scenario.shift_time,
        actual_shift_time=episode.actual_shift_time,
        shift_censored=bool(episode.shift_censored),
        belief_shift_event_time=episode.belief_shift_event_time,
        belief_shift_false_alarm=bool(episode.belief_shift_false_alarm),
        task_success=task_success,
        env_steps=terminal_step,
        tool_units=int(episode.tool_units),
        budget_cap=int(config.tool_budget),
        terminated=bool(episode.terminated),
        truncated=bool(episode.truncated),
        termination_reason=str(episode.termination_reason),
        terminal_posterior=posterior,
        posterior_valid=posterior_valid,
        posterior_error=posterior_error,
        terminal_map=terminal_map,
        terminal_top_probability=terminal_top,
        terminal_margin=terminal_margin,
        terminal_tie=terminal_tie,
        map_correct=bool(map_correct),
        task_and_map_correct=bool(task_success and map_correct),
        successful_misdiagnosis=bool(task_success and not map_correct),
        task_failure_map_correct=bool(not task_success and map_correct),
        declaration_available=bool(declaration_available),
        declared_rule=declared_rule,
        declared_step=declared_step,
        declared_confidence=declared_confidence,
        declared_margin=declared_margin,
        commitment_status=commitment_status,
        settled_declaration=bool(settled_declaration),
        commitment_correct=commitment_status == "correct_commit",
        earlier_wrong_commitments=len(earlier_wrong_rules),
        earlier_wrong_commitment_rules=tuple(earlier_wrong_rules),
        brier_score=brier_score,
        nll=nll,
        tool_events=_tool_events(episode, config),
        code_commit=str(code_commit),
        config_sha256=config_sha256,
        manifest_sha256=str(manifest_sha256),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(
    records: Sequence[EpisodeRecord], *, score_commitment: bool = True
) -> dict[str, Any]:
    n = len(records)
    successes = sum(record.task_success for record in records)
    map_correct = sum(record.map_correct for record in records)
    joint = sum(record.task_and_map_correct for record in records)
    misdiagnosis = sum(record.successful_misdiagnosis for record in records)
    task_failure_correct = sum(record.task_failure_map_correct for record in records)
    ties = sum(record.terminal_tie for record in records)
    posterior_valid = sum(record.posterior_valid for record in records)
    availability = sum(record.declaration_available for record in records)
    statuses = {
        status: (
            sum(record.commitment_status == status for record in records)
            if score_commitment
            else 0
        )
        for status in ("correct_commit", "wrong_commit", "unresolved", "unavailable")
    }
    settled = statuses["correct_commit"] + statuses["wrong_commit"]
    commitment_accuracy: float | None = None
    if score_commitment and availability == n and n:
        commitment_accuracy = statuses["correct_commit"] / n
    selective_error: float | None = None
    if score_commitment and settled:
        selective_error = statuses["wrong_commit"] / settled
    tools = [record.tool_units for record in records]
    brier = [record.brier_score for record in records if record.brier_score is not None]
    nll = [record.nll for record in records if record.nll is not None]
    return {
        "episodes": n,
        "task_successes": successes,
        "task_success_rate": _rate(successes, n),
        "map_correct": map_correct,
        "map_accuracy": _rate(map_correct, n),
        "task_and_map_correct": joint,
        "task_and_map_rate": _rate(joint, n),
        "successful_misdiagnosis": misdiagnosis,
        "task_failure_map_correct": task_failure_correct,
        "posterior_available": posterior_valid,
        "posterior_coverage": _rate(posterior_valid, n),
        "terminal_ties": ties,
        "declaration_available_episodes": availability,
        "declaration_coverage": _rate(availability, n),
        "commitment_scope": "static_identification" if score_commitment else "not_scored",
        "commitment_accuracy": commitment_accuracy,
        "commitment_correct": statuses["correct_commit"],
        "commitment_wrong": statuses["wrong_commit"],
        "commitment_unresolved": statuses["unresolved"],
        "commitment_unavailable": statuses["unavailable"],
        "episodes_with_earlier_wrong_commitment": sum(
            record.earlier_wrong_commitments > 0 for record in records
        ),
        "earlier_wrong_commitments_total": sum(
            record.earlier_wrong_commitments for record in records
        ),
        "selective_error": selective_error,
        "brier_score_mean": _mean(brier),
        "nll_mean": _mean(nll),
        "tool_units_total": sum(tools),
        "tool_units_mean": _mean([float(value) for value in tools]),
        "tool_units_max": max(tools) if tools else None,
        "tool_budget_violations": sum(
            record.tool_units > record.budget_cap for record in records
        ),
        "zero_tool_episodes": sum(record.tool_units == 0 for record in records),
        "zero_tool_successes": sum(record.zero_tool_success for record in records),
    }


def summarize_records(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """Aggregate by method and static cohort without dropping any episode."""

    if not records:
        raise EvaluationError("cannot summarize an empty record set")
    by_method: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_method[record.method].append(record)
    methods: dict[str, Any] = {}
    for method, method_records in sorted(by_method.items()):
        static = [
            record
            for record in method_records
            if record.scenario_type == "identification"
        ]
        shift = [
            record
            for record in method_records
            if record.shift_rule is not None
        ]
        no_shift = [
            record
            for record in method_records
            if record.scenario_type == "no_shift"
        ]
        base = [
            record
            for record in static
            if record.initial_rule == "base"
        ]
        by_rule: dict[str, Any] = {}
        for rule in GRID_RULE_IDS:
            subset = [record for record in static if record.initial_rule == rule]
            if subset:
                by_rule[rule] = _aggregate(subset)
        methods[method] = {
            "all": _aggregate(method_records, score_commitment=False),
            "static_identification": _aggregate(static),
            "base_identification": _aggregate(base),
            "shift": shift_summary(shift),
            "no_shift": _aggregate(no_shift, score_commitment=False),
            "by_initial_rule": by_rule,
        }
    return {
        "evaluation_version": EVALUATION_VERSION,
        "methods": methods,
        "paired": {},
    }


def mcnemar_exact(
    only_a: int,
    only_b: int,
    *,
    alternative: str = "two-sided",
) -> float:
    """Exact McNemar p value with no normal approximation."""

    if only_a < 0 or only_b < 0:
        raise ValueError("discordant counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    total = 2**n

    def upper(k: int) -> int:
        return sum(math.comb(n, i) for i in range(k, n + 1))

    if alternative in ("a>b", "greater"):
        return upper(only_a) / total
    if alternative == "less":
        return sum(math.comb(n, i) for i in range(0, only_a + 1)) / total
    if alternative != "two-sided":
        raise ValueError(f"unknown alternative {alternative!r}")
    low = sum(math.comb(n, i) for i in range(0, only_a + 1))
    high = upper(only_a)
    return min(1.0, 2.0 * min(low, high) / total)


def _index_unique(
    records: Sequence[EpisodeRecord], label: str
) -> dict[str, EpisodeRecord]:
    result: dict[str, EpisodeRecord] = {}
    for record in records:
        if record.scenario_id in result:
            raise PairingError(
                f"{label} contains duplicate scenario_id {record.scenario_id!r}"
            )
        result[record.scenario_id] = record
    return result


def paired_summary(
    left: Sequence[EpisodeRecord],
    right: Sequence[EpisodeRecord],
    *,
    metric: str = "map_correct",
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Pair two methods by exact scenario id; fail on missing or duplicate ids."""

    if metric not in ("task_success", "map_correct", "task_and_map_correct"):
        raise ValueError(f"unsupported paired metric {metric!r}")
    left_index = _index_unique(left, "left")
    right_index = _index_unique(right, "right")
    left_ids = set(left_index)
    right_ids = set(right_index)
    if left_ids != right_ids:
        missing = sorted(right_ids - left_ids)
        extra = sorted(left_ids - right_ids)
        raise PairingError(
            f"paired sets differ; missing_left={missing[:5]}, "
            f"extra_left={extra[:5]}"
        )
    both = only_left = only_right = neither = 0
    for scenario_id in sorted(left_ids):
        a = bool(getattr(left_index[scenario_id], metric))
        b = bool(getattr(right_index[scenario_id], metric))
        if a and b:
            both += 1
        elif a:
            only_left += 1
        elif b:
            only_right += 1
        else:
            neither += 1
    n = len(left_ids)
    return {
        "metric": metric,
        "episodes": n,
        "both": both,
        "only_left": only_left,
        "only_right": only_right,
        "neither": neither,
        "left_rate": (both + only_left) / n if n else None,
        "right_rate": (both + only_right) / n if n else None,
        "mcnemar_p_two_sided": mcnemar_exact(
            only_left, only_right, alternative="two-sided"
        ),
        "mcnemar_p_left_gt_right": mcnemar_exact(
            only_left, only_right, alternative="a>b"
        ),
    }


def shift_summary(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """Summarize shift timing separately from static identification accuracy."""

    n = len(records)
    actual = [record for record in records if record.actual_shift_time is not None]
    censored = [record for record in records if record.shift_censored]
    events = [
        record
        for record in records
        if record.belief_shift_event_time is not None
    ]
    false_alarms = sum(record.belief_shift_false_alarm for record in records)
    post_shift_events = [
        record
        for record in events
        if record.actual_shift_time is not None
        and record.belief_shift_event_time is not None
        and record.belief_shift_event_time >= record.actual_shift_time
    ]
    latencies = [
        int(record.belief_shift_event_time) - int(record.actual_shift_time)
        for record in post_shift_events
        if record.belief_shift_event_time is not None
        and record.actual_shift_time is not None
    ]
    return {
        "episodes": n,
        "scheduled_shift_episodes": sum(
            record.scheduled_shift_time is not None for record in records
        ),
        "actual_shift_episodes": len(actual),
        "censored_before_shift": len(censored),
        "policy_shift_event_episodes": len(events),
        "pre_shift_false_alarm_episodes": false_alarms,
        "post_shift_event_episodes": len(post_shift_events),
        "latency_steps": latencies,
        "median_latency_steps": (
            sorted(latencies)[len(latencies) // 2] if latencies else None
        ),
        "detection_basis": "policy_belief_shift_probability_proxy",
        "detection_note": (
            "This is the policy's shift-probability threshold crossing, not an "
            "explicit detection declaration; static identification never uses it."
        ),
    }


def write_jsonl(records: Iterable[EpisodeRecord], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_row(), sort_keys=True) + "\n")
    return out
