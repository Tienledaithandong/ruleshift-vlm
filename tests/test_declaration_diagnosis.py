"""Tests for declaration-failure classification.

The classifiers are deliberately pure: they consume already-collected public
traces and never run a policy or read hidden state. These tests pin the
mutually exclusive labels and the paired transition table.
"""

from __future__ import annotations

from ruleshift.evaluator.declaration_diagnosis import (
    aggregate_categories,
    classify_incorrect_trace,
    classify_unresolved_trace,
    confidence_criterion,
    outcome_counts,
    paired_transition,
)


def _decision(
    *,
    top: float,
    margin: float,
    safe: bool = False,
    action: str = "MOVE",
    **extra,
):
    row = {
        "top": top,
        "margin": margin,
        "safe_commit_eligible": safe,
        "action": action,
        "step": 0,
        "target": None,
        "tool_value": None,
        "true_tool_value": None,
    }
    row.update(extra)
    return row


def _trace(*, decisions, terminal=None, events=None, status="unresolved"):
    return {
        "scenario_id": "m0-s0-identification-base-None",
        "status": status,
        "true_rule": "base",
        "terminal_map": "base",
        "decisions": decisions,
        "events": events or [],
        "terminal": terminal
        or {
            "top": 0.95,
            "margin": 0.8,
            "run": 1,
            "tool_units": 0,
            "termination_reason": "success",
        },
    }


def test_confidence_criterion_uses_both_thresholds():
    assert confidence_criterion({"top": 0.90, "margin": 0.10})
    assert not confidence_criterion({"top": 0.899, "margin": 0.50})
    assert not confidence_criterion({"top": 0.95, "margin": 0.099})


def test_unresolved_stability_gate_is_not_terminal_evidence():
    trace = _trace(
        decisions=[_decision(top=0.95, margin=0.80, safe=False, leader_run=1)],
        terminal={
            "top": 0.95,
            "margin": 0.80,
            "run": 1,
            "tool_units": 0,
            "termination_reason": "success",
        },
    )
    result = classify_unresolved_trace(trace)
    assert result["primary"] == "early_success_stability_gate"
    assert result["criterion_decisions"] == [0]
    assert result["eligible_decisions"] == []
    assert "single_leader_observation" in result["secondary"]


def test_unresolved_terminal_only_is_not_retroactive_declaration():
    trace = _trace(
        decisions=[_decision(top=0.69, margin=0.54)],
        terminal={
            "top": 0.95,
            "margin": 0.80,
            "run": 1,
            "tool_units": 0,
            "termination_reason": "success",
        },
    )
    result = classify_unresolved_trace(trace)
    assert result["primary"] == "early_success_terminal_only_evidence"
    assert result["criterion_decisions"] == []


def test_unresolved_insufficient_evidence():
    trace = _trace(
        decisions=[_decision(top=0.50, margin=0.0)],
        terminal={
            "top": 0.60,
            "margin": 0.20,
            "run": 2,
            "tool_units": 0,
            "termination_reason": "success",
        },
    )
    result = classify_unresolved_trace(trace)
    assert result["primary"] == "early_success_insufficient_evidence"
    assert "single_leader_observation" not in result["secondary"]


def test_unresolved_safe_branch_without_event_is_instrumentation_gap():
    trace = _trace(
        decisions=[_decision(top=0.95, margin=0.80, safe=True)],
        terminal={
            "top": 0.95,
            "margin": 0.80,
            "run": 2,
            "tool_units": 0,
            "termination_reason": "success",
        },
    )
    result = classify_unresolved_trace(trace)
    assert result["primary"] == "instrumentation_gap"


def test_unresolved_at_decision_limit_is_not_called_early_success():
    trace = _trace(
        decisions=[_decision(top=0.50, margin=0.0)],
        terminal={
            "top": 0.60,
            "margin": 0.20,
            "run": 2,
            "tool_units": 0,
            "termination_reason": "max_decisions",
        },
    )
    result = classify_unresolved_trace(trace)
    assert result["primary"] == "max_decisions_insufficient_evidence"


def test_incorrect_declaration_with_contradictory_inspect():
    trace = _trace(
        status="wrong_commit",
        decisions=[
            _decision(
                top=0.50,
                margin=0.0,
                action="INSPECT",
                target=[1, 1],
                tool_value="goal",
                true_tool_value="safe",
            ),
            _decision(
                top=0.974,
                margin=0.949,
                safe=True,
                event={
                    "step": 1,
                    "declared_rule": "hazard_goal_swap",
                    "confidence": 0.974,
                    "posterior_margin": 0.949,
                },
            ),
        ],
        events=[
            {
                "step": 1,
                "declared_rule": "hazard_goal_swap",
                "confidence": 0.974,
                "posterior_margin": 0.949,
            }
        ],
        terminal={
            "top": 0.974,
            "margin": 0.949,
            "run": 3,
            "tool_units": 3,
            "termination_reason": "max_decisions",
        },
    )
    result = classify_incorrect_trace(trace)
    assert result["primary"] == "inspect_noise_confident_error"
    assert result["contradictory_inspections"][0]["observed"] == "goal"
    assert result["contradictory_inspections"][0]["true"] == "safe"


def test_incorrect_declaration_without_contradiction_is_ambiguous_evidence():
    trace = _trace(
        status="wrong_commit",
        decisions=[
            _decision(top=0.60, margin=0.2),
            _decision(
                top=0.914,
                margin=0.829,
                safe=True,
                event={
                    "step": 1,
                    "declared_rule": "slippery",
                    "confidence": 0.914,
                    "posterior_margin": 0.829,
                },
            ),
        ],
        events=[
            {
                "step": 1,
                "declared_rule": "slippery",
                "confidence": 0.914,
                "posterior_margin": 0.829,
            }
        ],
        terminal={
            "top": 0.928,
            "margin": 0.855,
            "run": 1,
            "tool_units": 1,
            "termination_reason": "success",
        },
    )
    result = classify_incorrect_trace(trace)
    assert result["primary"] == "premature_commit_ambiguous_evidence"
    assert "confidence_near_threshold" in result["secondary"]
    assert "terminal_map_recovered" in result["secondary"]


def test_paired_transition_matrix_and_outcome_counts():
    left = [
        {"scenario_id": "a", "commitment_status": "correct_commit", "task_success": True, "map_correct": True},
        {"scenario_id": "b", "commitment_status": "correct_commit", "task_success": False, "map_correct": True},
        {"scenario_id": "c", "commitment_status": "wrong_commit", "task_success": False, "map_correct": False},
    ]
    right = [
        {"scenario_id": "a", "commitment_status": "correct_commit", "task_success": True, "map_correct": True},
        {"scenario_id": "b", "commitment_status": "unresolved", "task_success": True, "map_correct": True},
        {"scenario_id": "c", "commitment_status": "wrong_commit", "task_success": True, "map_correct": False},
    ]
    result = paired_transition(left, right, left_name="v3", right_name="v4")
    assert result["declaration_status"] == {
        "correct_commit->correct_commit": 1,
        "correct_commit->unresolved": 1,
        "wrong_commit->wrong_commit": 1,
    }
    assert result["task_success"] == {"False->True": 2, "True->True": 1}
    assert outcome_counts(left)["statuses"]["wrong_commit"] == 1


def test_aggregate_categories_counts_primary_and_secondary():
    traces = [
        _trace(
            decisions=[_decision(top=0.95, margin=0.8)],
            terminal={
                "top": 0.95,
                "margin": 0.8,
                "run": 1,
                "tool_units": 0,
                "termination_reason": "success",
            },
        ),
        _trace(
            decisions=[_decision(top=0.5, margin=0.0)],
            terminal={
                "top": 0.6,
                "margin": 0.2,
                "run": 2,
                "tool_units": 0,
                "termination_reason": "success",
            },
        ),
    ]
    result = aggregate_categories(traces)
    assert result["primary"]["early_success_stability_gate"] == 1
    assert result["primary"]["early_success_insufficient_evidence"] == 1
    assert result["secondary"]["no_tool_evidence"] == 2


def test_incorrect_declaration_uses_last_event_window():
    first = {
        "step": 0,
        "declared_rule": "slippery",
        "confidence": 0.70,
        "posterior_margin": 0.20,
    }
    final = {
        "step": 1,
        "declared_rule": "hazard_goal_swap",
        "confidence": 0.974,
        "posterior_margin": 0.949,
    }
    trace = _trace(
        status="wrong_commit",
        decisions=[
            _decision(
                top=0.70,
                margin=0.20,
                action="INSPECT",
                target=[1, 1],
                tool_value="goal",
                true_tool_value="safe",
                event=first,
            ),
            _decision(
                top=0.974,
                margin=0.949,
                safe=True,
                event=final,
            ),
        ],
        events=[first, final],
        terminal={
            "top": 0.974,
            "margin": 0.949,
            "run": 2,
            "tool_units": 2,
            "termination_reason": "max_decisions",
        },
    )
    result = classify_incorrect_trace(trace)
    assert result["primary"] == "inspect_noise_confident_error"
    assert "confidence_near_threshold" not in result["secondary"]
