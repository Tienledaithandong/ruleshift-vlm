"""Value-aware selection with one-step continuation, using only public costs."""

from __future__ import annotations

import pytest

from ruleshift.agents.value_selector import (
    ImmediateOnlySelector,
    PublicCostModel,
    ValueAwareSelector,
    q_hat_from_prediction,
)

BELIEF = {"red": 0.5, "blue": 0.5, "green": 0.0}


def prediction() -> dict:
    """Hand-specified VLM-style predictions with noise-free probe feedback."""

    return {
        "candidate_probabilities": {"red": 0.5, "blue": 0.5, "green": 0.0},
        "action_predictions": [
            {
                "action_id": "PROBE_RED",
                "under_rules": [
                    {"rule_id": "red", "possible_outcomes": [{"observation": "accept", "p": 1.0}]},
                    {"rule_id": "blue", "possible_outcomes": [{"observation": "reject", "p": 1.0}]},
                ],
                "estimated_future_best_action": "COMMIT_RED",
            },
            {
                "action_id": "COMMIT_RED",
                "under_rules": [
                    {"rule_id": "red", "possible_outcomes": [{"observation": "accept", "p": 1.0}]},
                    {"rule_id": "blue", "possible_outcomes": [{"observation": "reject", "p": 1.0}]},
                ],
                "estimated_future_best_action": "COMMIT_RED",
            },
        ],
        "selected_action": "PROBE_RED",
    }


def costly_lockout() -> PublicCostModel:
    return PublicCostModel(
        step_cost=-0.05,
        success_reward=10.0,
        probe_duration=1,
        commit_duration=1,
        lockout_steps=8,
        remaining_horizon=3,
        learning_rate=1.0,
    )


def test_q_values_match_hand_calculation() -> None:
    q = q_hat_from_prediction(prediction(), BELIEF, costly_lockout())
    # probe: -0.05 + 9.95 continuation in both branches = 9.90
    # commit: .5*(9.95+9.95) + .5*(-0.45 + 0 no continuation) = 9.725
    assert q["PROBE_RED"] == pytest.approx(9.90)
    assert q["COMMIT_RED"] == pytest.approx(9.725)


def test_value_aware_selector_prefers_probe_when_lockout_expensive() -> None:
    assert ValueAwareSelector().choose(prediction(), BELIEF, costly_lockout()) == "PROBE_RED"


def test_immediate_only_selector_undervalues_probe() -> None:
    q = q_hat_from_prediction(
        prediction(), BELIEF, costly_lockout(), account_continuation=False
    )
    assert q["PROBE_RED"] == pytest.approx(-0.05)
    assert q["COMMIT_RED"] == pytest.approx(4.75)
    assert ImmediateOnlySelector().choose(prediction(), BELIEF, costly_lockout()) == "COMMIT_RED"


def test_higher_public_probe_cost_flips_choice_to_task() -> None:
    expensive = PublicCostModel(
        step_cost=-0.05,
        success_reward=10.0,
        probe_duration=8,
        commit_duration=1,
        lockout_steps=8,
        remaining_horizon=10,
        learning_rate=1.0,
    )
    assert ValueAwareSelector().choose(prediction(), BELIEF, costly_lockout()) == "PROBE_RED"
    assert ValueAwareSelector().choose(prediction(), BELIEF, expensive) == "COMMIT_RED"


def test_selector_falls_back_to_wait_without_predictions() -> None:
    empty = {
        "candidate_probabilities": {"red": 1.0, "blue": 0.0, "green": 0.0},
        "action_predictions": [],
        "selected_action": "WAIT",
    }
    assert ValueAwareSelector().choose(empty, BELIEF, costly_lockout()) == "WAIT"
