"""Week 5 ablations are all public-only and return valid macros."""

from __future__ import annotations

import pytest

from ruleshift.agents.value_selector import (
    ABLATION_NAMES,
    PublicCostModel,
    ablation_choose,
)
from ruleshift.public_api.schemas import MACRO_ACTIONS_FROZEN

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
            }
        ],
        "selected_action": "PROBE_RED",
    }


def cost() -> PublicCostModel:
    return PublicCostModel(lockout_steps=8, remaining_horizon=3, learning_rate=1.0)


def test_all_named_ablations_return_valid_macros() -> None:
    assert "M1" in ABLATION_NAMES
    assert len(ABLATION_NAMES) == 6
    for name in ABLATION_NAMES:
        action = ablation_choose(name, prediction(), BELIEF, cost())
        assert action in MACRO_ACTIONS_FROZEN


def test_no_selector_ablation_uses_raw_vlm_choice() -> None:
    assert ablation_choose("M1_minus_selector", prediction(), BELIEF, cost()) == "PROBE_RED"


def test_unknown_ablation_is_rejected() -> None:
    with pytest.raises(ValueError):
        ablation_choose("M1_minus_magic", prediction(), BELIEF, cost())
