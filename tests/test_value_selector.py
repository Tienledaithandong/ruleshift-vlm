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


# ---------------------------------------------------------------------------
# Cost-sign contract: public_costs publishes magnitudes, the model is signed.
# ---------------------------------------------------------------------------
from ruleshift.agents.value_selector import (  # noqa: E402
    _duration,
    _immediate_reward,
    public_macro_durations,
)

PUBLIC_COSTS = {
    "primitive_step": 0.05,
    "probe_detour": 2,
    "commit_travel": 4,
    "lockout": 8,
    "delivery_reward": 10.0,
}


def test_cost_model_from_public_costs_negates_the_published_magnitude() -> None:
    model = PublicCostModel.from_public_costs(
        PUBLIC_COSTS, probe_duration=5, commit_duration=5, remaining_horizon=10
    )
    assert model.step_cost == pytest.approx(-0.05)
    # A 5-step probe costs 0.25 and a rejected 5-step commit plus 8 lockout
    # steps costs 0.65; the old code charged these as *rewards*.
    assert _immediate_reward("PROBE_RED", "reject", model) == pytest.approx(-0.25)
    assert _immediate_reward("COMMIT_RED", "reject", model) == pytest.approx(-0.65)
    assert _immediate_reward("COMMIT_RED", "accept", model) == pytest.approx(9.75)


def test_positive_step_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="step_cost"):
        PublicCostModel(step_cost=0.05)


def test_public_macro_durations_match_the_executor_formula() -> None:
    assert public_macro_durations({"probe_detour": 3, "commit_travel": 4}) == (7, 5)
    assert public_macro_durations({"probe_detour": 0, "commit_travel": 0}) == (1, 1)


def test_longer_macro_lowers_the_estimate() -> None:
    payload = prediction()
    short = PublicCostModel(
        step_cost=-0.05, success_reward=10.0, probe_duration=1,
        commit_duration=5, lockout_steps=8, remaining_horizon=10, learning_rate=0.0,
    )
    long = PublicCostModel(
        step_cost=-0.05, success_reward=10.0, probe_duration=9,
        commit_duration=5, lockout_steps=8, remaining_horizon=10, learning_rate=0.0,
    )
    short_q = q_hat_from_prediction(payload, BELIEF, short, account_continuation=False)
    long_q = q_hat_from_prediction(payload, BELIEF, long, account_continuation=False)
    assert long_q["PROBE_RED"] < short_q["PROBE_RED"]
    assert _duration("PROBE_RED", "accept", long) == 9


def test_one_step_continuation_cancels_outside_a_truncated_horizon() -> None:
    """Characterises the current estimator: lookahead is only one step.

    With a generous remaining horizon every branch gets the same continuation,
    so ``Q(COMMIT_RED) - Q(PROBE_RED)`` is exactly the difference of their
    immediate rewards and the probe cannot win on information value alone.
    This is pinned deliberately: if the continuation is deepened to a
    multi-step ``Vhat``, this test should change with it.
    """

    generous = PublicCostModel(
        step_cost=-0.05,
        success_reward=10.0,
        probe_duration=1,
        commit_duration=1,
        lockout_steps=8,
        remaining_horizon=60,
        learning_rate=1.0,
    )
    q = q_hat_from_prediction(prediction(), BELIEF, generous)
    commit_immediate = 0.5 * 9.95 + 0.5 * -0.45
    assert q["COMMIT_RED"] - q["PROBE_RED"] == pytest.approx(
        commit_immediate - (-0.05)
    )
    assert ValueAwareSelector().choose(prediction(), BELIEF, generous) == "COMMIT_RED"
