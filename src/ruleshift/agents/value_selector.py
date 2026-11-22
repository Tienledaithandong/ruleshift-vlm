"""Value-aware macro selection from VLM predictions and PUBLIC costs only.

No simulator likelihood, Bayes Q*, oracle belief or hidden transition is used.
The selector adds a one-step continuation value so that a probe, whose
immediate task reward is zero, can be valued by how it improves the next
decision. All arithmetic uses public cost descriptions and VLM-provided
predictions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from ruleshift.public_api.schemas import COMMIT_ACTIONS

from .consistency import Belief, HistoryConsistencyUpdater


@dataclass(frozen=True)
class PublicCostModel:
    """Costs and budget that are publicly described to every agent."""

    step_cost: float = -0.05
    success_reward: float = 10.0
    probe_duration: int = 1
    commit_duration: int = 1
    lockout_steps: int = 0
    remaining_horizon: int = 3
    learning_rate: float = 0.3


def _duration(action: str, outcome: str, cost: PublicCostModel) -> int:
    if action.startswith("COMMIT_"):
        return cost.commit_duration + (cost.lockout_steps if outcome == "reject" else 0)
    if action.startswith("PROBE_"):
        return cost.probe_duration
    return 1


def _immediate_reward(action: str, outcome: str, cost: PublicCostModel) -> float:
    reward = _duration(action, outcome, cost) * cost.step_cost
    if action.startswith("COMMIT_") and outcome == "accept":
        reward += cost.success_reward
    return reward


def _continuation(belief: Belief, cost: PublicCostModel) -> float:
    """Best one-step task value after an outcome, from public rule semantics."""

    best = 0.0
    for action in COMMIT_ACTIONS:
        color = action.removeprefix("COMMIT_").lower()
        expected = sum(
            probability
            * _immediate_reward(
                action, "accept" if rule == color else "reject", cost
            )
            for rule, probability in belief.items()
        )
        best = max(best, expected)
    return best


def q_hat_from_prediction(
    payload: dict,
    belief: Belief,
    cost: PublicCostModel,
    account_continuation: bool = True,
) -> dict[str, float]:
    """Estimate Q-hat per predicted macro from VLM outcomes and public costs."""

    updater = HistoryConsistencyUpdater(
        rules=tuple(belief.keys()), learning_rate=cost.learning_rate
    )
    estimates: dict[str, float] = {}

    for prediction in payload.get("action_predictions", []):
        action = prediction.get("action_id")
        if not isinstance(action, str):
            continue
        total = 0.0
        for rule_prediction in prediction.get("under_rules", []):
            rule = rule_prediction.get("rule_id")
            rule_probability = belief.get(rule, 0.0)
            if rule_probability <= 0.0:
                continue
            for outcome in rule_prediction.get("possible_outcomes", []):
                observation = outcome.get("observation")
                probability = float(outcome.get("p", 0.0))
                duration = _duration(action, observation, cost)
                reward = _immediate_reward(action, observation, cost)
                if account_continuation and duration < cost.remaining_horizon:
                    posterior = updater.update(belief, action, observation)
                    reward += _continuation(posterior, cost)
                total += rule_probability * probability * reward
        estimates[action] = total
    return estimates


class ValueAwareSelector:
    """Choose the macro with the highest continuation-aware estimate."""

    def choose(
        self, payload: dict, belief: Belief, cost: PublicCostModel
    ) -> str:
        estimates = q_hat_from_prediction(payload, belief, cost)
        return max(estimates, key=estimates.get) if estimates else "WAIT"


class ImmediateOnlySelector:
    """Naive baseline that counts only immediate reward (undervalues probes)."""

    def choose(
        self, payload: dict, belief: Belief, cost: PublicCostModel
    ) -> str:
        estimates = q_hat_from_prediction(
            payload, belief, cost, account_continuation=False
        )
        return max(estimates, key=estimates.get) if estimates else "WAIT"


#: Ablation menu used by the Week 6 comparison matrix.
ABLATION_NAMES: tuple[str, ...] = (
    "M1",
    "M1_minus_belief",
    "M1_minus_outcome",
    "M1_minus_cost",
    "M1_minus_consistency",
    "M1_minus_selector",
)


def ablation_choose(
    name: str, payload: dict, belief: Belief, cost: PublicCostModel
) -> str:
    """Run one named ablation. All inputs remain public."""

    if name not in ABLATION_NAMES:
        raise ValueError(f"unknown ablation: {name}")
    if name == "M1_minus_selector":
        return payload.get("selected_action", "WAIT")

    if name == "M1_minus_cost":
        free_cost = replace(
            cost, step_cost=0.0, probe_duration=0, commit_duration=0, lockout_steps=0
        )
        return ValueAwareSelector().choose(payload, belief, free_cost)

    if name == "M1_minus_belief":
        uniform = {rule: 1.0 / len(belief) for rule in belief}
        return ValueAwareSelector().choose(payload, uniform, cost)

    if name == "M1_minus_consistency":
        return ValueAwareSelector().choose(
            payload, belief, replace(cost, learning_rate=0.0)
        )

    if name == "M1_minus_outcome":
        # Replace VLM outcome predictions with public rule semantics: a card of
        # colour c is accepted iff the candidate rule says it accepts c.
        deterministic = copy.deepcopy(payload)
        for prediction in deterministic.get("action_predictions", []):
            action = prediction.get("action_id", "")
            color = action.split("_", 1)[1].lower() if "_" in action else ""
            for rule_prediction in prediction.get("under_rules", []):
                rule_prediction["possible_outcomes"] = [
                    {
                        "observation": "accept" if rule_prediction.get("rule_id") == color else "reject",
                        "p": 1.0,
                    }
                ]
        return ValueAwareSelector().choose(deterministic, belief, cost)

    return ValueAwareSelector().choose(payload, belief, cost)
