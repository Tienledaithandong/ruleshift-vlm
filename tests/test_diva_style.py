"""DIVA-style adapted baseline with physical probe costs (not a reproduction)."""

from __future__ import annotations

import pytest

from ruleshift.agents.diva_style import DivaStyleSelector, diva_decision_value
from ruleshift.agents.value_selector import PublicCostModel

BELIEF = {"red": 0.5, "blue": 0.5, "green": 0.0}


def prediction() -> dict:
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


def costly() -> PublicCostModel:
    return PublicCostModel(
        step_cost=-0.05,
        success_reward=10.0,
        probe_duration=1,
        commit_duration=1,
        lockout_steps=8,
        remaining_horizon=3,
        learning_rate=1.0,
    )


def test_decision_value_is_hand_computed() -> None:
    values = diva_decision_value(prediction(), BELIEF, costly())
    # expected posterior correctness 1.0 vs baseline 0.5 => 5.0 gain,
    # minus a 1-step probe cost of 0.05 => 4.95.
    assert values["PROBE_RED"] == pytest.approx(4.95)


def test_selector_prefers_probe_when_check_is_worth_its_cost() -> None:
    assert DivaStyleSelector().choose(prediction(), BELIEF, costly()) == "PROBE_RED"


def test_selector_stops_when_probe_cost_exceeds_decision_value() -> None:
    expensive = PublicCostModel(
        step_cost=-0.05,
        success_reward=10.0,
        probe_duration=8,
        commit_duration=1,
        lockout_steps=8,
        remaining_horizon=10,
        learning_rate=1.0,
    )
    assert DivaStyleSelector().choose(prediction(), BELIEF, expensive) == "COMMIT_RED"


def test_baseline_is_labelled_as_adapted() -> None:
    assert DivaStyleSelector.label == "DIVA_STYLE_ADAPTED_PHYSICAL_COST"
