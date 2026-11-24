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
from typing import Mapping

from ruleshift.public_api.schemas import COMMIT_ACTIONS

from .consistency import Belief


@dataclass(frozen=True)
class PublicCostModel:
    """Costs and budget that are publicly described to every agent.

    ``step_cost`` is a **signed** reward delta: a cost must be negative so that
    a longer macro lowers its estimate. The public observation only publishes
    non-negative cost *magnitudes* (see ``schemas.PUBLIC_COST_KEYS``), so
    consumers must build this model through :meth:`from_public_costs` rather
    than copying ``public_costs["primitive_step"]`` straight into
    ``step_cost``. Constructing a model with a positive ``step_cost`` is
    rejected because it would reward taking more steps.
    """

    step_cost: float = -0.05
    success_reward: float = 10.0
    probe_duration: int = 1
    commit_duration: int = 1
    lockout_steps: int = 0
    remaining_horizon: int = 3
    learning_rate: float = 0.3
    start_commit_duration: int | None = None

    def __post_init__(self) -> None:
        if self.step_cost > 0.0:
            raise ValueError(
                "step_cost must be zero or negative (a cost, not a reward); "
                "use PublicCostModel.from_public_costs to convert the published "
                "non-negative magnitude"
            )
        if self.success_reward < 0.0:
            raise ValueError("success_reward must be non-negative")
        for name in ("probe_duration", "commit_duration", "lockout_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.remaining_horizon < 1:
            raise ValueError("remaining_horizon must be at least 1 primitive step")
        if not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")

    @classmethod
    def from_public_costs(
        cls,
        public_costs: Mapping[str, float],
        *,
        probe_duration: int,
        commit_duration: int,
        remaining_horizon: int,
        learning_rate: float = 0.3,
    ) -> "PublicCostModel":
        """Build a signed cost model from an agent-visible ``public_costs`` map.

        ``public_costs`` publishes ``primitive_step`` as a non-negative
        magnitude, so it is negated exactly once here.
        """

        magnitude = abs(float(public_costs.get("primitive_step", 0.05)))
        return cls(
            step_cost=-magnitude,
            success_reward=float(public_costs.get("delivery_reward", 10.0)),
            probe_duration=int(probe_duration),
            commit_duration=int(commit_duration),
            start_commit_duration=int(public_costs["start_to_gate"]) + 1
            if "start_to_gate" in public_costs else None,
            lockout_steps=int(public_costs.get("lockout", 0)),
            remaining_horizon=max(1, int(remaining_horizon)),
            learning_rate=float(learning_rate),
        )


def public_macro_durations(
    public_costs: Mapping[str, float],
) -> tuple[int, int]:
    """Public primitive duration of a probe macro and of a commit macro.

    A probe is ``station_travel + 1 + detour`` steps (out, test, back) and a commit is
    ``public travel to the gate + 1`` steps, matching ``envs.macro``. These are
    derived from public cost descriptions only.
    """

    detour = float(public_costs.get("probe_detour", 0.0))
    probe_duration = int(float(public_costs.get("station_travel", detour)) + 1 + detour)
    commit_duration = int(public_costs.get("commit_travel", 0)) + 1
    return probe_duration, commit_duration


def _duration(action: str, outcome: str, cost: PublicCostModel) -> int:
    if action.startswith("COMMIT_"):
        return cost.commit_duration + (cost.lockout_steps if outcome == "reject" else 0)
    if action.startswith("PROBE_"):
        return cost.probe_duration
    return 1


def _immediate_reward(action: str, outcome: str, cost: PublicCostModel) -> float:
    reward = min(_duration(action, outcome, cost), cost.remaining_horizon) * cost.step_cost
    if action.startswith("COMMIT_") and outcome == "accept" and cost.commit_duration <= cost.remaining_horizon:
        reward += cost.success_reward
    return reward


def _continuation(belief: Belief, cost: PublicCostModel) -> float:
    """Best one-step task value after an outcome, from public rule semantics."""

    best = float("-inf")
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
    """Estimate Q-hat per predicted macro from VLM outcomes and public costs.

    For each possible public outcome, the VLM supplies its likelihood under
    each candidate rule. Condition the current belief on that prediction,
    temper the change by learning_rate, and score one subsequent commitment
    within the remaining primitive-step budget. This one-step approximation
    uses public travel distances after the return or rejection.
    """

    estimates: dict[str, float] = {}

    for prediction in payload.get("action_predictions", []):
        action = prediction.get("action_id")
        if not isinstance(action, str):
            continue
        total = 0.0
        for observation, masses in predicted_outcome_masses(prediction, belief).items():
            probability = sum(masses.values())
            if probability <= 0.0:
                continue
            duration = _duration(action, observation, cost)
            reward = _immediate_reward(action, observation, cost)
            if account_continuation and duration < cost.remaining_horizon:
                eta = cost.learning_rate
                posterior = {
                    rule: (1.0 - eta) * prior + eta * masses.get(rule, 0.0) / probability
                    for rule, prior in belief.items()
                }
                next_commit = cost.commit_duration
                if cost.start_commit_duration is not None:
                    if action.startswith("PROBE_") or (action.startswith("COMMIT_") and observation == "accept"):
                        next_commit = cost.start_commit_duration
                    elif action.startswith("COMMIT_") and observation == "reject":
                        next_commit = min(1, cost.commit_duration)
                remaining = replace(cost, remaining_horizon=cost.remaining_horizon - duration,
                                    commit_duration=next_commit)
                reward += _continuation(posterior, remaining)
            total += probability * reward
        estimates[action] = total
    return estimates


def predicted_outcome_masses(prediction: dict, belief: Belief) -> dict[str, Belief]:
    """Joint rule/outcome mass computed from the agent's predictions."""

    masses: dict[str, Belief] = {}
    for row in prediction.get("under_rules", []):
        rule = row["rule_id"]
        for outcome in row["possible_outcomes"]:
            by_rule = masses.setdefault(outcome["observation"], {})
            by_rule[rule] = belief.get(rule, 0.0) * outcome["p"]
    return masses


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
            cost, step_cost=0.0, probe_duration=0, commit_duration=0,
            start_commit_duration=0, lockout_steps=0
        )
        return ValueAwareSelector().choose(payload, belief, free_cost)

    if name == "M1_minus_belief":
        uniform = {rule: 1.0 / len(belief) for rule in belief}
        return ValueAwareSelector().choose(payload, uniform, cost)

    if name == "M1_minus_consistency":
        # Remove the history correction but keep reasoning about future evidence.
        return ValueAwareSelector().choose(
            payload, payload["candidate_probabilities"], cost
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
