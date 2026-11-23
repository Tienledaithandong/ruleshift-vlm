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

from .consistency import Belief, HistoryConsistencyUpdater


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
    noise_floor: float = 0.02
    relevance_decay: float = 0.01

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
            lockout_steps=int(public_costs.get("lockout", 0)),
            remaining_horizon=max(1, int(remaining_horizon)),
            learning_rate=float(learning_rate),
        )


def public_macro_durations(
    public_costs: Mapping[str, float],
) -> tuple[int, int]:
    """Public primitive duration of a probe macro and of a commit macro.

    A probe is ``2 * detour + 1`` steps (out, test, back) and a commit is
    ``public travel to the gate + 1`` steps, matching ``envs.macro``. These are
    derived from public cost descriptions only.
    """

    detour = float(public_costs.get("probe_detour", 0.0))
    probe_duration = int(2 * detour + 1)
    commit_duration = int(public_costs.get("commit_travel", 0)) + 1
    return probe_duration, commit_duration


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
    """Estimate Q-hat per predicted macro from VLM outcomes and public costs.

    Lookahead depth (known limitation, documented rather than hidden): the
    continuation is a *one-step* value, ``max_a rhat(a | posterior)``. It is
    added to every branch, and the posterior map depends only on the card
    colour and the observed outcome — not on whether the outcome came from a
    probe or from a commit. When both actions' durations fit inside
    ``remaining_horizon``, the continuation therefore cancels between
    ``PROBE_c`` and ``COMMIT_c`` and the comparison reduces to the probe's cost
    versus the commit's expected immediate reward. Since a commit normally has
    positive expected reward, the selector prefers committing; a probe is only
    chosen when the commit's lockout branch is truncated by the remaining
    horizon (which is what ``test_value_aware_selector_prefers_probe_*``
    exercises). Valuing evidence that pays off across several future deliveries
    requires a deeper ``Vhat`` and is a method decision, not a cost-accounting
    fix.
    """

    updater = HistoryConsistencyUpdater(
        rules=tuple(belief.keys()),
        learning_rate=cost.learning_rate,
        noise_floor=cost.noise_floor,
        relevance_decay=cost.relevance_decay,
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
    "M1_multi_step",
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

    if name == "M1_multi_step":
        return MultiStepValueAwareSelector().choose(payload, belief, cost)

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


# ---------------------------------------------------------------------------
# Multi-step Vhat: finite-horizon recursion over public predictions only.
#
# This is the deeper ``Vhat`` mentioned in the docstring of
# ``q_hat_from_prediction``. It recurses over the remaining horizon: a probe
# updates the public belief and consumes ``probe_duration`` steps; a commit
# ends the current delivery but the agent may continue with the steps left.
# No simulator, no oracle, no hidden transitions are used.
# ---------------------------------------------------------------------------


def multi_step_q_hat_from_prediction(
    payload: dict,
    belief: Belief,
    cost: PublicCostModel,
) -> dict[str, float]:
    """Finite-horizon multi-step Vhat using only public costs and VLM
    predictions.

    Recurses over the remaining horizon. At each step the agent may probe any
    predicted action or commit any commit action. Probe feedback updates the
    public belief (via :class:`HistoryConsistencyUpdater`) and consumes
    ``probe_duration`` steps; a commit consumes ``commit_duration`` (plus
    lockout on reject) and ends the current delivery, but the agent may keep
    acting while steps remain.
    """

    from functools import lru_cache

    # Index VLM predictions by action -> rule -> {observation: p}.
    action_models: dict[str, dict[str, dict[str, float]]] = {}
    for prediction in payload.get("action_predictions", []):
        action = prediction.get("action_id")
        if not isinstance(action, str):
            continue
        rule_models: dict[str, dict[str, float]] = {}
        for rule_pred in prediction.get("under_rules", []):
            rule = rule_pred.get("rule_id")
            obs_map: dict[str, float] = {}
            for oc in rule_pred.get("possible_outcomes", []):
                obs_map[oc.get("observation")] = float(oc.get("p", 0.0))
            rule_models[rule] = obs_map
        action_models[action] = rule_models

    actions = list(action_models.keys())
    updater = HistoryConsistencyUpdater(
        rules=tuple(belief.keys()),
        learning_rate=cost.learning_rate,
        noise_floor=cost.noise_floor,
        relevance_decay=cost.relevance_decay,
    )

    @lru_cache(maxsize=None)
    def V(
        belief_tuple: tuple[tuple[str, float], ...], steps_left: int
    ) -> float:
        if steps_left <= 0:
            return 0.0
        b = dict(belief_tuple)
        best = float("-inf")
        for action in actions:
            total = 0.0
            feasible = False
            for rule, rule_prob in b.items():
                if rule_prob <= 0.0:
                    continue
                for obs, p in action_models[action].get(rule, {}).items():
                    if p <= 0.0:
                        continue
                    duration = _duration(action, obs, cost)
                    if duration > steps_left:
                        continue
                    feasible = True
                    reward = _immediate_reward(action, obs, cost)
                    posterior = updater.update(b, action, obs)
                    future = V(
                        tuple(sorted((rule, round(p, 6)) for rule, p in posterior.items())),
                        steps_left - duration,
                    )
                    total += rule_prob * p * (reward + future)
            if feasible:
                best = max(best, total)
        return best if best != float("-inf") else 0.0

    estimates: dict[str, float] = {}
    horizon = cost.remaining_horizon
    for prediction in payload.get("action_predictions", []):
        action = prediction.get("action_id")
        if not isinstance(action, str):
            continue
        total = 0.0
        for rule_pred in prediction.get("under_rules", []):
            rule = rule_pred.get("rule_id")
            rule_prob = belief.get(rule, 0.0)
            if rule_prob <= 0.0:
                continue
            for oc in rule_pred.get("possible_outcomes", []):
                obs = oc.get("observation")
                p = float(oc.get("p", 0.0))
                if p <= 0.0:
                    continue
                duration = _duration(action, obs, cost)
                if duration > horizon:
                    continue
                reward = _immediate_reward(action, obs, cost)
                posterior = updater.update(belief, action, obs)
                future = V(
                    tuple(sorted((rule, round(p, 6)) for rule, p in posterior.items())),
                    horizon - duration,
                )
                total += rule_prob * p * (reward + future)
        estimates[action] = total
    return estimates


class MultiStepValueAwareSelector:
    """Choose the macro with the highest multi-step Vhat estimate.

    Unlike :class:`ValueAwareSelector` (one-step continuation), this selector
    recurses over the finite horizon, so a probe whose information pays off
    across several future decisions is valued correctly rather than being
    crowded out by an immediate commit.
    """

    def choose(
        self, payload: dict, belief: Belief, cost: PublicCostModel
    ) -> str:
        estimates = multi_step_q_hat_from_prediction(payload, belief, cost)
        return max(estimates, key=estimates.get) if estimates else "WAIT"

