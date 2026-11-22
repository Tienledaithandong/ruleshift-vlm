"""DIVA-style decision-informed information seeking, adapted to physical probes.

This is an ADAPTATION, not a reproduction of DIVA: the check is a macro that
spends primitive steps in the shared reward stream and can only be valued by
how much it improves the subsequent action. It uses public costs and VLM
predictions only.
"""

from __future__ import annotations

from dataclasses import dataclass

from ruleshift.public_api.schemas import COMMIT_ACTIONS

from .value_selector import Belief, PublicCostModel, _immediate_reward


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
        expected_correct = 0.0
        for rule_prediction in prediction.get("under_rules", []):
            rule = rule_prediction.get("rule_id")
            rule_probability = belief.get(rule, 0.0)
            if rule_probability <= 0.0:
                continue
            for outcome in rule_prediction.get("possible_outcomes", []):
                joint = rule_probability * float(outcome.get("p", 0.0))
                if joint <= 0.0:
                    continue
                total = sum(
                    belief.get(other.get("rule_id"), 0.0)
                    * float(
                        next(
                            (
                                item.get("p", 0.0)
                                for item in other.get("possible_outcomes", [])
                                if item.get("observation") == outcome.get("observation")
                            ),
                            0.0,
                        )
                    )
                    for other in prediction.get("under_rules", [])
                )
                if total > 0.0:
                    expected_correct += joint * (joint / total)

        improvement = (expected_correct - baseline_correct) * cost.success_reward
        probe_cost = cost.probe_duration * abs(cost.step_cost)
        net_values[action] = improvement - probe_cost
    return net_values


def _task_value(belief: Belief, cost: PublicCostModel) -> float:
    best = 0.0
    for action in COMMIT_ACTIONS:
        color = action.removeprefix("COMMIT_").lower()
        expected = sum(
            probability
            * _immediate_reward(action, "accept" if rule == color else "reject", cost)
            for rule, probability in belief.items()
        )
        best = max(best, expected)
    return best


@dataclass(frozen=True)
class DivaStyleSelector:
    """Acquire a probe only while its net decision value beats acting now."""

    label: str = "DIVA_STYLE_ADAPTED_PHYSICAL_COST"

    def choose(
        self, payload: dict, belief: Belief, cost: PublicCostModel
    ) -> str:
        values = diva_decision_value(payload, belief, cost)
        if not values:
            return payload.get("selected_action", "WAIT")
        best_probe = max(values, key=values.get)
        if values[best_probe] > 0.0 and values[best_probe] > _task_value(belief, cost):
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
