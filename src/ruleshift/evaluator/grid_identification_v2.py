"""Versioned v2 scoring for separated epistemic/control decisions.

This module is deliberately separate from grid_identification (v1). v1 is
frozen historical scoring; v2 adds explicit declaration, abstention, request,
revision and final-observation accounting without retroactively rescoring v1.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.evaluator.grid_identification import (
    EvaluationError,
    PairingError,
    classify_posterior,
    mcnemar_exact,
    validate_posterior,
)
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS, GRID_TOOL_ACTIONS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruleshift.evaluator.grid_runner import GridEpisode

EVALUATION_VERSION = "grid-identification-v2"
TOP_THRESHOLD = 0.90
MARGIN_THRESHOLD = 0.10
TIE_EPS = 1e-6
DECLARATION_CAP = 6
MAX_DECISIONS = 200

_EVENT_KINDS = frozenset({"DECLARE_RULE", "ABSTAIN", "REQUEST_EVIDENCE"})
_DECLARATION_SOURCES = frozenset({"pre_action", "final_observation"})
_ABSTAIN_REASONS = frozenset(
    {
        "insufficient_evidence",
        "ambiguous",
        "noisy_evidence_unconfirmed",
        "budget_exhausted",
        "terminal_only",
        "evidence_requested",
    }
)
_REQUEST_CHANNELS = frozenset({"SIGN", "BLUE"})
_REQUEST_STATUSES = frozenset(
    {
        "pending",
        "executed",
        "unmet",
        "expired",
        "superseded",
    }
)
_RULE_ORDER = {rule: index for index, rule in enumerate(GRID_RULE_IDS)}

__all__ = [
    "EVALUATION_VERSION",
    "V2EpisodeRecord",
    "build_v2_episode_record",
    "paired_transitions_v2",
    "summarize_records_v2",
    "summarize_v2",
    "validate_development_slice",
    "validate_epistemic_events",
    "write_jsonl_v2",
]


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationError(f"expected a finite number, got {value!r}")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{name} must be an integer")
    return int(value)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0


def _copy_events(
    events: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    copied: list[dict[str, Any]] = []
    for event in events:
        clone = dict(event)
        if "evidence_refs" in clone:
            clone["evidence_refs"] = list(clone["evidence_refs"])
        copied.append(clone)
    return tuple(copied)


def validate_epistemic_events(
    events: Sequence[Mapping[str, Any]] | None,
    *,
    terminal_step: int,
    max_decisions: int = MAX_DECISIONS,
) -> tuple[dict[str, Any], ...]:
    """Validate the public v2 event ledger and return plain copies."""

    if events is None:
        return ()
    if not isinstance(events, Sequence):
        raise EvaluationError("epistemic_events must be a sequence")
    validated: list[dict[str, Any]] = []
    last_decision = -1
    declarations = 0
    declarations_by_decision: set[int] = set()
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"event {index} is not a mapping")
        event = dict(raw)
        kind = event.get("kind")
        if kind not in _EVENT_KINDS:
            raise EvaluationError(f"event {index} has unknown kind {kind!r}")
        step = _integer(event.get("step"), f"event {index} step")
        if not 0 <= step <= terminal_step:
            raise EvaluationError(
                f"event {index} step {step} is outside 0..{terminal_step}"
            )
        decision_index = _integer(
            event.get("decision_index"), f"event {index} decision_index"
        )
        # The final observation is not a pre-action control decision, so it
        # may carry the terminal index one past the pre-action bound.
        if decision_index < 0 or decision_index > max_decisions + 1:
            raise EvaluationError(
                f"event {index} decision_index is outside 0..{max_decisions + 1}"
            )
        if decision_index < last_decision:
            raise EvaluationError("epistemic events are not in decision order")
        last_decision = decision_index

        if kind == "DECLARE_RULE":
            declarations += 1
            if declarations > DECLARATION_CAP:
                raise EvaluationError("too many declaration events")
            if decision_index in declarations_by_decision:
                raise EvaluationError(
                    "more than one declaration at the same decision"
                )
            declarations_by_decision.add(decision_index)
            rule = event.get("rule_id", event.get("declared_rule"))
            if rule not in _RULE_ORDER:
                raise EvaluationError(f"unknown declared rule {rule!r}")
            confidence = _finite_number(event.get("confidence"))
            margin = _finite_number(event.get("posterior_margin"))
            if not 0.0 <= confidence <= 1.0:
                raise EvaluationError("declaration confidence is outside [0, 1]")
            if not 0.0 <= margin <= 1.0:
                raise EvaluationError("declaration margin is outside [0, 1]")
            if confidence + TIE_EPS < TOP_THRESHOLD:
                raise EvaluationError("declaration is below the confidence gate")
            if margin + TIE_EPS < MARGIN_THRESHOLD:
                raise EvaluationError("declaration is below the margin gate")
            source = event.get("source")
            if source not in _DECLARATION_SOURCES:
                raise EvaluationError(f"unknown declaration source {source!r}")
            refs = event.get("evidence_refs")
            if not isinstance(refs, (list, tuple)) or not refs:
                raise EvaluationError("declaration needs evidence_refs")
            normalized_refs: list[int] = []
            for ref in refs:
                ref_id = _integer(ref, "evidence ref")
                if ref_id < 0:
                    raise EvaluationError("evidence ref must be non-negative")
                normalized_refs.append(ref_id)
            if len(set(normalized_refs)) != len(normalized_refs):
                raise EvaluationError("duplicate evidence refs")
            event["rule_id"] = str(rule)
            event["declared_rule"] = str(rule)
            event["evidence_refs"] = normalized_refs
            event["terminal"] = bool(event.get("terminal", False))
            if source == "final_observation" and not event["terminal"]:
                raise EvaluationError(
                    "final_observation declaration must be terminal"
                )
        elif kind == "ABSTAIN":
            if event.get("reason") not in _ABSTAIN_REASONS:
                raise EvaluationError(
                    f"unknown abstention reason {event.get('reason')!r}"
                )
            event["terminal"] = bool(event.get("terminal", False))
        else:
            if event.get("channel") not in _REQUEST_CHANNELS:
                raise EvaluationError(
                    f"unknown request channel {event.get('channel')!r}"
                )
            status = event.get("status")
            if status not in _REQUEST_STATUSES:
                raise EvaluationError(f"unknown request status {status!r}")
        validated.append(event)
    return tuple(validated)


def _tool_events(
    episode: GridEpisode, config: GridConfig
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


@dataclass(frozen=True)
class V2EpisodeRecord:
    """One v2 episode with explicit epistemic and control accounting."""

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
    task_success: bool
    env_steps: int
    tool_units: int
    budget_cap: int
    decision_count: int
    terminated: bool
    truncated: bool
    termination_reason: str
    total_return: float
    total_cost: float
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
    final_declared_rule: str | None
    final_declared_step: int | None
    final_declared_confidence: float | None
    final_declared_margin: float | None
    final_declaration_source: str | None
    final_declaration_terminal: bool
    final_declaration_correct: bool
    final_declaration_wrong: bool
    unresolved: bool
    declaration_coverage: bool
    declaration_event_count: int
    wrong_declaration_events: int
    wrong_declaration_rules: tuple[str, ...]
    any_wrong_declaration: bool
    strict_clean_correct: bool
    revision_count: int
    revision_episode: bool
    first_correct_declaration_step: int | None
    first_correct_declaration_decision: int | None
    time_to_first_correct_declaration: int | None
    abstention_count: int
    abstention_reasons: tuple[str, ...]
    request_count: int
    request_status_counts: tuple[tuple[str, int], ...]
    event_count: int
    declaration_events: tuple[dict[str, Any], ...]
    abstention_events: tuple[dict[str, Any], ...]
    request_events: tuple[dict[str, Any], ...]
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


def build_v2_episode_record(
    episode: GridEpisode,
    scenario: GridScenario,
    config: GridConfig,
    *,
    layout_sha256_value: str,
    events: Sequence[Mapping[str, Any]] | None = None,
    declaration_available: bool = True,
    code_commit: str = "unknown",
    config_sha256: str | None = None,
    manifest_sha256: str = "unknown",
    max_decisions: int | None = None,
) -> V2EpisodeRecord:
    """Build one v2 record without converting terminal MAP into a declaration."""

    true_terminal_rule = str(scenario.initial_rule)
    if episode.evaluator_rows:
        true_terminal_rule = str(
            episode.evaluator_rows[-1].get("true_rule", true_terminal_rule)
        )
    terminal_step = int(episode.env_steps)
    max_decision_bound = int(
        max_decisions
        if max_decisions is not None
        else getattr(config, "max_decisions", MAX_DECISIONS)
    )

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

    validated_events = validate_epistemic_events(
        episode.epistemic_events if events is None else events,
        terminal_step=terminal_step,
        max_decisions=max_decision_bound,
    )
    declarations = tuple(
        event for event in validated_events if event["kind"] == "DECLARE_RULE"
    )
    abstentions = tuple(
        event for event in validated_events if event["kind"] == "ABSTAIN"
    )
    requests = tuple(
        event for event in validated_events if event["kind"] == "REQUEST_EVIDENCE"
    )

    final = declarations[-1] if declarations else None
    final_rule = None if final is None else str(final["rule_id"])
    final_step = None if final is None else int(final["step"])
    final_confidence = None if final is None else float(final["confidence"])
    final_margin = None if final is None else float(final["posterior_margin"])
    final_source = None if final is None else str(final["source"])
    final_terminal = bool(final is not None and final.get("terminal", False))
    final_correct = bool(final_rule is not None and final_rule == true_terminal_rule)
    final_wrong = bool(final_rule is not None and final_rule != true_terminal_rule)
    unresolved = final is None
    wrong_labels = tuple(
        str(event["rule_id"])
        for event in declarations
        if str(event["rule_id"]) != true_terminal_rule
    )
    first_correct = next(
        (
            (int(event["step"]), int(event["decision_index"]))
            for event in declarations
            if str(event["rule_id"]) == true_terminal_rule
        ),
        None,
    )
    revision_count = max(0, len(declarations) - 1)
    request_status_counts = tuple(
        sorted(Counter(str(event["status"]) for event in requests).items())
    )
    map_correct = terminal_map == true_terminal_rule
    task_success = bool(episode.success)
    return V2EpisodeRecord(
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
        task_success=task_success,
        env_steps=terminal_step,
        tool_units=int(episode.tool_units),
        budget_cap=int(config.tool_budget),
        decision_count=len(episode.public_rows),
        terminated=bool(episode.terminated),
        truncated=bool(episode.truncated),
        termination_reason=str(episode.termination_reason),
        total_return=float(episode.total_return),
        total_cost=float(episode.total_cost),
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
        final_declared_rule=final_rule,
        final_declared_step=final_step,
        final_declared_confidence=final_confidence,
        final_declared_margin=final_margin,
        final_declaration_source=final_source,
        final_declaration_terminal=final_terminal,
        final_declaration_correct=final_correct,
        final_declaration_wrong=final_wrong,
        unresolved=unresolved,
        declaration_coverage=not unresolved,
        declaration_event_count=len(declarations),
        wrong_declaration_events=len(wrong_labels),
        wrong_declaration_rules=wrong_labels,
        any_wrong_declaration=bool(wrong_labels),
        strict_clean_correct=bool(final_correct and not wrong_labels),
        revision_count=revision_count,
        revision_episode=revision_count > 0,
        first_correct_declaration_step=(
            None if first_correct is None else first_correct[0]
        ),
        first_correct_declaration_decision=(
            None if first_correct is None else first_correct[1]
        ),
        time_to_first_correct_declaration=(
            None if first_correct is None else first_correct[0]
        ),
        abstention_count=len(abstentions),
        abstention_reasons=tuple(str(event["reason"]) for event in abstentions),
        request_count=len(requests),
        request_status_counts=request_status_counts,
        event_count=len(validated_events),
        declaration_events=declarations,
        abstention_events=abstentions,
        request_events=requests,
        tool_events=_tool_events(episode, config),
        code_commit=str(code_commit),
        config_sha256=config_sha256,
        manifest_sha256=str(manifest_sha256),
    )


def _aggregate_v2(records: Sequence[V2EpisodeRecord]) -> dict[str, Any]:
    n = len(records)
    correct = sum(record.final_declaration_correct for record in records)
    wrong = sum(record.final_declaration_wrong for record in records)
    unresolved = sum(record.unresolved for record in records)
    coverage = sum(record.declaration_coverage for record in records)
    strict = sum(record.strict_clean_correct for record in records)
    any_wrong = sum(record.any_wrong_declaration for record in records)
    wrong_events = sum(record.wrong_declaration_events for record in records)
    revision_episodes = sum(record.revision_episode for record in records)
    revisions = sum(record.revision_count for record in records)
    task = sum(record.task_success for record in records)
    terminal_map = sum(record.map_correct for record in records)
    posterior_valid = sum(record.posterior_valid for record in records)
    zero_tool_success = sum(record.zero_tool_success for record in records)
    zero_tool_hazard_sign = sum(
        record.zero_tool_success
        for record in records
        if record.terminal_true_rule in {"hazard_goal_swap", "sign_composition"}
    )
    first_steps = [
        record.time_to_first_correct_declaration
        for record in records
        if record.time_to_first_correct_declaration is not None
    ]
    tool_values = [record.tool_units for record in records]
    decision_values = [record.decision_count for record in records]
    return_values = [record.total_return for record in records]
    cost_values = [record.total_cost for record in records]
    abstention_reasons = Counter(
        reason for record in records for reason in record.abstention_reasons
    )
    request_statuses = Counter(
        status
        for record in records
        for status, count in record.request_status_counts
        for _ in range(count)
    )
    request_count = sum(record.request_count for record in records)
    declaration_events = sum(record.declaration_event_count for record in records)
    return {
        "episodes": n,
        "task_successes": task,
        "task_success_rate": _rate(task, n),
        "terminal_map_correct": terminal_map,
        "terminal_map_accuracy": _rate(terminal_map, n),
        "posterior_valid": posterior_valid,
        "declaration_available": sum(record.declaration_available for record in records),
        "final_declaration_correct": correct,
        "final_declaration_wrong": wrong,
        "unresolved": unresolved,
        "declaration_coverage": coverage,
        "declaration_coverage_rate": _rate(coverage, n),
        "unconditional_declaration_accuracy": _rate(correct, n),
        "final_declaration_error_rate": _rate(wrong, n),
        "final_declaration_precision": _rate(correct, correct + wrong),
        "strict_clean_correct": strict,
        "strict_clean_accuracy": _rate(strict, n),
        "any_wrong_declaration_episodes": any_wrong,
        "any_wrong_rate": _rate(any_wrong, n),
        "wrong_declaration_events": wrong_events,
        "declaration_event_count": declaration_events,
        "revision_episodes": revision_episodes,
        "revision_count": revisions,
        "revision_rate": _rate(revision_episodes, n),
        "time_to_first_correct_mean_steps": _mean(first_steps),
        "time_to_first_correct_median_steps": _median(first_steps),
        "time_to_first_correct_available": len(first_steps),
        "tool_units_total": sum(tool_values),
        "tool_units_mean": _mean(tool_values),
        "tool_units_max": max(tool_values) if tool_values else None,
        "tool_budget_respected": all(
            record.tool_units <= record.budget_cap for record in records
        ),
        "zero_tool_successes": zero_tool_success,
        "zero_tool_hazard_sign_successes": zero_tool_hazard_sign,
        "decision_count_mean": _mean(decision_values),
        "decision_count_max": max(decision_values) if decision_values else None,
        "total_return_mean": _mean(return_values),
        "total_cost_mean": _mean(cost_values),
        "abstention_events": sum(record.abstention_count for record in records),
        "abstention_reasons": dict(sorted(abstention_reasons.items())),
        "request_events": request_count,
        "request_episodes": sum(record.request_count > 0 for record in records),
        "request_statuses": dict(sorted(request_statuses.items())),
        "successful_misdiagnoses": sum(
            record.successful_misdiagnosis for record in records
        ),
        "task_failure_map_correct": sum(
            record.task_failure_map_correct for record in records
        ),
        "task_and_map_correct": sum(record.task_and_map_correct for record in records),
    }


def summarize_v2(records: Sequence[V2EpisodeRecord]) -> dict[str, Any]:
    """Summarize one method with per-rule, per-map and readiness fields."""

    overall = _aggregate_v2(records)
    by_rule: dict[str, Any] = {}
    by_map: dict[str, Any] = {}
    rules = sorted({record.terminal_true_rule for record in records})
    maps = sorted({record.map_seed for record in records})
    for rule in rules:
        subset = [record for record in records if record.terminal_true_rule == rule]
        by_rule[rule] = _aggregate_v2(subset)
    for map_seed in maps:
        subset = [record for record in records if record.map_seed == map_seed]
        by_map[f"m{map_seed}"] = _aggregate_v2(subset)
    base = by_rule.get("base", {})
    hazard = by_rule.get("hazard_goal_swap", {})
    sign = by_rule.get("sign_composition", {})
    readiness = {
        "overall_declaration_accuracy_ge_95": (
            overall["unconditional_declaration_accuracy"] is not None
            and overall["unconditional_declaration_accuracy"] >= 0.95
        ),
        "hazard_declaration_accuracy_ge_90": (
            hazard.get("unconditional_declaration_accuracy") is not None
            and hazard["unconditional_declaration_accuracy"] >= 0.90
        ),
        "sign_declaration_accuracy_ge_90": (
            sign.get("unconditional_declaration_accuracy") is not None
            and sign["unconditional_declaration_accuracy"] >= 0.90
        ),
        "base_task_success_100": (
            base.get("task_successes") == base.get("episodes")
            and base.get("episodes", 0) > 0
        ),
        "zero_tool_hazard_sign_success": (
            overall["zero_tool_hazard_sign_successes"] == 0
        ),
        "tool_budget_respected": overall["tool_budget_respected"],
    }
    readiness["all_pass"] = all(readiness.values())
    return {
        "evaluation_version": EVALUATION_VERSION,
        "overall": overall,
        "by_rule": by_rule,
        "by_map": by_map,
        "readiness": readiness,
    }


def summarize_records_v2(
    records: Sequence[V2EpisodeRecord],
) -> dict[str, Any]:
    methods = sorted({record.method for record in records})
    return {
        "evaluation_version": EVALUATION_VERSION,
        "methods": {
            method: summarize_v2(
                [record for record in records if record.method == method]
            )
            for method in methods
        },
    }


def _status(record: V2EpisodeRecord) -> str:
    if not record.declaration_available:
        return "unavailable"
    if record.unresolved:
        return "unresolved"
    if record.final_declaration_correct:
        return "correct"
    return "wrong"


def _transition_table(
    left: Sequence[V2EpisodeRecord],
    right: Sequence[V2EpisodeRecord],
    key: Any,
) -> dict[str, Any]:
    left_by = {record.scenario_id: record for record in left}
    right_by = {record.scenario_id: record for record in right}
    if set(left_by) != set(right_by):
        raise PairingError("paired methods do not contain the same scenario ids")
    table: Counter[tuple[str, str]] = Counter()
    for scenario_id in sorted(left_by):
        table[(str(key(left_by[scenario_id])), str(key(right_by[scenario_id])))] += 1
    return {
        "n": len(left_by),
        "table": {f"{a}->{b}": count for (a, b), count in sorted(table.items())},
    }


def paired_transitions_v2(
    left: Sequence[V2EpisodeRecord],
    right: Sequence[V2EpisodeRecord],
) -> dict[str, Any]:
    """Paired status/task/map/strict transitions for identical scenario ids."""

    status = _transition_table(left, right, _status)
    task = _transition_table(
        left, right, lambda record: "success" if record.task_success else "failure"
    )
    terminal_map = _transition_table(
        left, right, lambda record: "correct" if record.map_correct else "wrong"
    )
    strict = _transition_table(
        left,
        right,
        lambda record: "strict_clean" if record.strict_clean_correct else "not_strict",
    )
    any_wrong = _transition_table(
        left,
        right,
        lambda record: "any_wrong" if record.any_wrong_declaration else "no_wrong",
    )
    left_by = {record.scenario_id: record for record in left}
    right_by = {record.scenario_id: record for record in right}
    left_correct = {
        sid for sid, record in left_by.items() if _status(record) == "correct"
    }
    right_correct = {
        sid for sid, record in right_by.items() if _status(record) == "correct"
    }
    discordant_left_only = len(left_correct - right_correct)
    discordant_right_only = len(right_correct - left_correct)
    return {
        "n": status["n"],
        "status": status["table"],
        "task": task["table"],
        "terminal_map": terminal_map["table"],
        "strict_clean": strict["table"],
        "any_wrong": any_wrong["table"],
        "declaration_correct": {
            "left_only": discordant_left_only,
            "right_only": discordant_right_only,
            "mcnemar_p": mcnemar_exact(discordant_left_only, discordant_right_only),
        },
    }


def validate_development_slice(
    manifest: Any,
    *,
    development_maps: Sequence[int] = tuple(range(10)),
    development_seeds: Sequence[int] = tuple(range(5)),
    forbidden_maps: Sequence[int] = tuple(range(210, 240)),
) -> None:
    """Reject a manifest outside the authorized exposed development slice."""

    maps = tuple(int(value) for value in manifest.map_ids)
    seeds = tuple(int(value) for value in manifest.run_seeds)
    if maps != tuple(int(value) for value in development_maps):
        raise EvaluationError(f"development maps mismatch: {maps!r}")
    if seeds != tuple(int(value) for value in development_seeds):
        raise EvaluationError(f"development seeds mismatch: {seeds!r}")
    if any(value in set(forbidden_maps) for value in maps):
        raise EvaluationError("manifest contains reserved maps 210-239")
    if len(manifest.entries) != len(maps) * len(seeds) * len(GRID_RULE_IDS):
        raise EvaluationError("manifest does not contain the 300 static scenarios")
    def field(entry: Any, name: str) -> Any:
        if isinstance(entry, Mapping):
            return entry.get(name)
        return getattr(entry, name, None)

    if any(field(entry, "scenario_type") != "identification" for entry in manifest.entries):
        raise EvaluationError("manifest contains non-identification scenarios")
    if any(field(entry, "shift_rule") is not None for entry in manifest.entries):
        raise EvaluationError("static manifest contains a shift rule")


def write_jsonl_v2(
    records: Sequence[V2EpisodeRecord], path: str | Path
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_row(), sort_keys=True) + "\n")
