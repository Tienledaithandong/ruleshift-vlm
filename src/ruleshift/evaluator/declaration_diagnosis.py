"""Pure helpers for the Phase 0.5 declaration-failure diagnosis.

The helpers in this module only classify already-collected, exposed
development traces. They do not run a policy, read hidden state, or convert a
terminal MAP estimate into a declaration.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

DECLARATION_STATUSES = ("correct_commit", "wrong_commit", "unresolved")

UNRESOLVED_PRIMARY_CAUSES = (
    "early_success_stability_gate",
    "early_success_terminal_only_evidence",
    "early_success_insufficient_evidence",
    "instrumentation_gap",
    "max_decisions_insufficient_evidence",
    "declared_but_marked_unresolved",
)

INCORRECT_PRIMARY_CAUSES = (
    "inspect_noise_confident_error",
    "premature_commit_ambiguous_evidence",
)


def confidence_criterion(decision: Mapping[str, Any]) -> bool:
    """Return whether the public confidence and margin rule is met."""

    return float(decision["top"]) >= 0.90 and float(decision["margin"]) >= 0.10


def _event_decision_index(
    trace: Mapping[str, Any], event: Mapping[str, Any] | None = None
) -> int | None:
    events = list(trace.get("events") or [])
    if event is None:
        if not events:
            return None
        event = events[-1]
    decisions = list(trace.get("decisions") or [])
    for index, decision in enumerate(decisions):
        if decision.get("event") == event:
            return index
    return None


def classify_unresolved_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one unresolved episode from its decision trace.

    The primary labels are mutually exclusive. A trace with a declaration event
    cannot be classified as unresolved; that is a data-integrity failure.
    """

    decisions = list(trace.get("decisions") or [])
    events = list(trace.get("events") or [])
    terminal = dict(trace.get("terminal") or {})
    if events:
        label = "declared_but_marked_unresolved"
        return {
            "primary": label,
            "secondary": [],
            "criterion_decisions": [],
            "eligible_decisions": [],
            "evidence": "declaration event present despite unresolved status",
        }

    qualified = [i for i, d in enumerate(decisions) if confidence_criterion(d)]
    eligible = [i for i, d in enumerate(decisions) if d.get("safe_commit_eligible")]
    terminal_qualified = (
        float(terminal.get("top") or 0.0) >= 0.90
        and float(terminal.get("margin") or 0.0) >= 0.10
    )
    secondary: list[str] = []
    if int(terminal.get("run") or 0) == 1:
        secondary.append("single_leader_observation")
    if terminal.get("tool_units") == 0:
        secondary.append("no_tool_evidence")
    if terminal.get("termination_reason") != "success":
        secondary.append("non_success_termination")

    if eligible:
        label = "instrumentation_gap"
        evidence = "safe-commit branch was eligible but no event was recorded"
    elif qualified:
        label = (
            "early_success_stability_gate"
            if terminal.get("termination_reason") == "success"
            else "max_decisions_insufficient_evidence"
        )
        evidence = (
            "posterior met the confidence rule at a decision, but the "
            "policy did not have a safe declaration opportunity"
        )
    elif terminal.get("termination_reason") != "success":
        label = "max_decisions_insufficient_evidence"
        evidence = "no qualified posterior before the decision limit"
    elif terminal_qualified:
        label = "early_success_terminal_only_evidence"
        evidence = (
            "the qualified posterior appeared after the last decision and the "
            "episode ended on the goal"
        )
    else:
        label = "early_success_insufficient_evidence"
        evidence = "terminal posterior never met the confidence/margin rule"

    return {
        "primary": label,
        "secondary": secondary,
        "criterion_decisions": qualified,
        "eligible_decisions": eligible,
        "evidence": evidence,
    }


def classify_incorrect_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Classify an incorrect declaration using pre-declaration trace evidence."""

    decisions = list(trace.get("decisions") or [])
    events = list(trace.get("events") or [])
    if not events:
        raise ValueError("incorrect declaration trace has no declaration event")
    event = events[-1]
    event_index = _event_decision_index(trace, event)
    if event_index is None:
        raise ValueError("declaration event is not attached to a decision")
    prior = decisions[: event_index + 1]
    contradictions = [
        d
        for d in prior
        if d.get("action") == "INSPECT"
        and d.get("true_tool_value") is not None
        and d.get("tool_value") != d.get("true_tool_value")
    ]
    tool_count = sum(
        1 for d in prior if d.get("action") in {"INSPECT", "READ_SIGN"}
    )
    confidence = float(event.get("confidence") or 0.0)
    secondary: list[str] = []
    if confidence < 0.95:
        secondary.append("confidence_near_threshold")
    if tool_count == 0:
        secondary.append("no_tool_evidence")
    if len(contradictions) > 1:
        secondary.append("multiple_contradictory_inspections")
    if contradictions:
        primary = "inspect_noise_confident_error"
        evidence = (
            "an INSPECT result contradicted the true role and the posterior "
            "was still above the confidence threshold"
        )
    else:
        primary = "premature_commit_ambiguous_evidence"
        evidence = (
            "declaration was wrong without a recorded contradictory inspection; "
            "the posterior crossed the threshold on ambiguous movement or sign "
            "evidence; terminal-MAP recovery, if any, is recorded separately"
        )
    if (
        str(trace.get("terminal_map")) == str(trace.get("true_rule"))
        and str(event.get("declared_rule")) != str(trace.get("true_rule"))
    ):
        secondary.append("terminal_map_recovered")
    return {
        "primary": primary,
        "secondary": secondary,
        "contradictory_inspections": [
            {
                "decision_index": index,
                "step": d.get("step"),
                "target": d.get("target"),
                "observed": d.get("tool_value"),
                "true": d.get("true_tool_value"),
            }
            for index, d in enumerate(prior)
            if d.get("action") == "INSPECT"
            and d.get("true_tool_value") is not None
            and d.get("tool_value") != d.get("true_tool_value")
        ],
        "evidence": evidence,
    }


def classify_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch a trace by its stored declaration status."""

    status = str(trace.get("status"))
    if status == "unresolved":
        return classify_unresolved_trace(trace)
    if status == "wrong_commit":
        return classify_incorrect_trace(trace)
    if status == "correct_commit":
        return {"primary": "correct_commit", "secondary": []}
    raise ValueError(f"unsupported declaration status: {status!r}")


def outcome_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count declaration and task outcomes for one method."""

    rows = list(records)
    statuses = Counter(str(r["commitment_status"]) for r in rows)
    return {
        "episodes": len(rows),
        "statuses": {key: int(statuses.get(key, 0)) for key in DECLARATION_STATUSES},
        "task_success": sum(bool(r["task_success"]) for r in rows),
        "map_correct": sum(bool(r["map_correct"]) for r in rows),
        "wrong_commitments": int(statuses.get("wrong_commit", 0)),
    }


def paired_transition(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    """Build scenario-level paired transition matrices for two methods."""

    left_by = {str(r["scenario_id"]): r for r in left}
    right_by = {str(r["scenario_id"]): r for r in right}
    if set(left_by) != set(right_by):
        raise ValueError("paired methods do not share the same scenario set")
    status_matrix: Counter[tuple[str, str]] = Counter()
    task_matrix: Counter[tuple[bool, bool]] = Counter()
    map_matrix: Counter[tuple[bool, bool]] = Counter()
    for scenario_id in sorted(left_by):
        a, b = left_by[scenario_id], right_by[scenario_id]
        status_matrix[(str(a["commitment_status"]), str(b["commitment_status"]))] += 1
        task_matrix[(bool(a["task_success"]), bool(b["task_success"]))] += 1
        map_matrix[(bool(a["map_correct"]), bool(b["map_correct"]))] += 1
    def matrices(counter: Counter) -> dict[str, int]:
        return {f"{a}->{b}": int(n) for (a, b), n in sorted(counter.items())}

    return {
        "left": left_name,
        "right": right_name,
        "n": len(left_by),
        "declaration_status": matrices(status_matrix),
        "task_success": matrices(task_matrix),
        "map_correct": matrices(map_matrix),
    }


def aggregate_categories(traces: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate primary and secondary diagnosis labels."""

    rows = list(traces)
    primary = Counter()
    secondary = Counter()
    for trace in rows:
        result = classify_trace(trace)
        primary[str(result["primary"])] += 1
        for item in result.get("secondary", []):
            secondary[str(item)] += 1
    return {
        "episodes": len(rows),
        "primary": dict(sorted(primary.items())),
        "secondary": dict(sorted(secondary.items())),
    }


__all__ = [
    "DECLARATION_STATUSES",
    "INCORRECT_PRIMARY_CAUSES",
    "UNRESOLVED_PRIMARY_CAUSES",
    "aggregate_categories",
    "classify_incorrect_trace",
    "classify_trace",
    "classify_unresolved_trace",
    "confidence_criterion",
    "outcome_counts",
    "paired_transition",
]