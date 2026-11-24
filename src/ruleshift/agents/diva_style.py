"""DIVA-style decision-informed information seeking, adapted to physical probes.

This is an ADAPTATION, not a reproduction of DIVA: the check is a macro that
spends primitive steps in the shared reward stream and can only be valued by
how much it improves the subsequent action. It uses public costs and VLM
predictions only.
"""

from __future__ import annotations

from dataclasses import dataclass

from ruleshift.public_api.schemas import COMMIT_ACTIONS

from .value_selector import Belief, PublicCostModel, _immediate_reward, predicted_outcome_masses


def diva_decision_value(
    payload: dict, belief: Belief, cost: PublicCostModel
) -> dict[str, float]:
    """Net decision value of each imagined probe, minus its physical cost."""

    best_rule = max(belief, key=belief.get)
    baseline_correct = belief[best_rule]
    net_values: dict[str, float] = {}

    for prediction in payload.get("action_predictions", []):
        action = prediction.get("action_id")
        if not isinstance(action, str) or not action.startswith("PROBE_"):
            continue
        # E_o[max_m P(m|o)] = sum_o max_m P(m,o). This values the best
        # decision after observing evidence, even with noisy predictions.
        masses = predicted_outcome_masses(prediction, belief)
        expected_correct = sum(max(by_rule.values()) for by_rule in masses.values())
        following_commit = cost.commit_duration if cost.start_commit_duration is None else cost.start_commit_duration
        if cost.probe_duration + following_commit > cost.remaining_horizon:
            expected_correct = 0.0
        improvement = (expected_correct - baseline_correct) * cost.success_reward
        probe_cost = min(cost.probe_duration, cost.remaining_horizon) * abs(cost.step_cost)
        net_values[action] = improvement - probe_cost
    return net_values


@dataclass(frozen=True)
class DivaStyleSelector:
    """Acquire a probe only while its net decision value beats acting now."""

    label: str = "DIVA_STYLE_ADAPTED_PHYSICAL_COST"

    def choose(
        self, payload: dict, belief: Belief, cost: PublicCostModel
    ) -> str:
        if not belief or not set(belief) <= {"red", "blue", "green"}:
            raise ValueError("the adapted DIVA selector requires Family A card rules")
        values = diva_decision_value(payload, belief, cost)
        if not values:
            return payload.get("selected_action", "WAIT")
        best_probe = max(values, key=values.get)
        if values[best_probe] > 0.0:
            return best_probe
        return max(
            COMMIT_ACTIONS,
            key=lambda action: sum(
                probability
                * _immediate_reward(
                    action,
                    "accept" if rule == action.removeprefix("COMMIT_").lower() else "reject",
                    cost,
                )
                for rule, probability in belief.items()
            ),
        )
