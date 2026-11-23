"""Microtask diagnostics for the M1 value-aware selector.

These tests compute exact answers by hand on tiny, fully-specified microtask
instances. They are *diagnostics*, not evidence that M1 is a good method.

Key known limitation (documented in value_selector.q_hat_from_prediction):
the one-step continuation ``max_a rhat(a | posterior)`` is added to *every*
outcome branch, including successful commits. That double-counts future
commits and therefore over-values immediate commit relative to a probe whose
value only pays off across several future decisions. When that matters, M1
one-step disagrees with the exact multi-step optimum.

The belief updater (consistency.HistoryConsistencyUpdater) is also a soft
heuristic: it does not model feedback noise, does not know when a shift makes
old evidence stale, and never decays probability on rules that have simply not
been observed for a while.
"""

from __future__ import annotations

import pytest

from ruleshift.agents.value_selector import (
    MultiStepValueAwareSelector,
    PublicCostModel,
    ValueAwareSelector,
    q_hat_from_prediction,
)
from ruleshift.agents.consistency import HistoryConsistencyUpdater

COMMIT_ACTIONS = ("COMMIT_RED", "COMMIT_BLUE", "COMMIT_GREEN")
PROBE_ACTIONS = ("PROBE_RED", "PROBE_BLUE", "PROBE_GREEN")


def _fixed_payload() -> dict:
    """Deterministic VLM payload: probe/commit colour c yields accept iff the
    active rule equals c; reject otherwise. No noise, no hidden dynamics."""
    predictions = []
    for action in (*PROBE_ACTIONS, *COMMIT_ACTIONS):
        color = action.split("_", 1)[1].lower()
        under_rules = []
        for rule in ("red", "blue", "green"):
            observation = "accept" if rule == color else "reject"
            under_rules.append(
                {"rule_id": rule, "possible_outcomes": [{"observation": observation, "p": 1.0}]}
            )
        predictions.append(
            {
                "action_id": action,
                "under_rules": under_rules,
                "estimated_future_best_action": f"COMMIT_{color}",
            }
        )
    return {
        "candidate_probabilities": {"red": 1 / 3, "blue": 1 / 3, "green": 1 / 3},
        "action_predictions": predictions,
        "selected_action": "COMMIT_RED",
    }


# ---------------------------------------------------------------------------
# Exact multi-step optimum by brute force over short horizons.
#
# State = (belief, steps_left). At each step the agent may probe any colour or
# commit any colour. Probe: cost probe_duration, outcome accept/reject tells
# whether rule == colour, then belief updates by the same consistency rule.
# Commit: cost commit_duration (+ lockout on reject), success_reward on accept,
# episode ends (delivery complete). We take the max expected total reward.
# ---------------------------------------------------------------------------


def _posterior(belief: dict[str, float], action: str, observation: str, eta: float) -> dict[str, float]:
    """Mirror of HistoryConsistencyUpdater.update for exact computation."""
    color = action.split("_", 1)[1].lower()
    updated = dict(belief)
    if action.startswith(("PROBE_", "COMMIT_")) and observation in ("accept", "reject") and color in updated:
        if observation == "accept":
            updated[color] *= 1.0 + eta
            for rule in updated:
                if rule != color:
                    updated[rule] *= 1.0 - eta
        else:
            updated[color] *= 1.0 - eta
    cleaned = {rule: max(0.0, updated.get(rule, 0.0)) for rule in updated}
    total = sum(cleaned.values())
    if total <= 0.0:
        return dict(belief)
    return {rule: value / total for rule, value in cleaned.items()}


def _immediate_reward(action: str, observation: str, cost: PublicCostModel) -> float:
    if action.startswith("COMMIT_"):
        duration = cost.commit_duration + (cost.lockout_steps if observation == "reject" else 0)
    elif action.startswith("PROBE_"):
        duration = cost.probe_duration
    else:
        duration = 1
    reward = duration * cost.step_cost
    if action.startswith("COMMIT_") and observation == "accept":
        reward += cost.success_reward
    return reward


def _outcome_prob(action: str, observation: str, rule: str) -> float:
    color = action.split("_", 1)[1].lower()
    expected_obs = "accept" if rule == color else "reject"
    return 1.0 if observation == expected_obs else 0.0


def multi_step_optimal(
    belief: dict[str, float],
    cost: PublicCostModel,
    horizon: int,
) -> tuple[str, float]:
    """Exact finite-horizon optimum over {probe any colour, commit any colour}.

    Returns (best_action, best_value) for the first decision.
    """

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def value(belief_tuple: tuple[tuple[str, float], ...], steps_left: int) -> float:
        belief = dict(belief_tuple)
        if steps_left <= 0:
            return 0.0
        best = float("-inf")
        for action in (*PROBE_ACTIONS, *COMMIT_ACTIONS):
            total = 0.0
            for rule, rule_prob in belief.items():
                if rule_prob <= 0.0:
                    continue
                for observation in ("accept", "reject"):
                    p = _outcome_prob(action, observation, rule)
                    if p <= 0.0:
                        continue
                    duration = (
                        cost.probe_duration
                        if action.startswith("PROBE_")
                        else cost.commit_duration + (cost.lockout_steps if observation == "reject" else 0)
                    )
                    if duration > steps_left:
                        continue
                    reward = _immediate_reward(action, observation, cost)
                    if action.startswith("COMMIT_"):
                        future = 0.0
                    else:
                        posterior = _posterior(belief, action, observation, cost.learning_rate)
                        future = value(tuple(sorted(posterior.items())), steps_left - duration)
                    total += rule_prob * p * (reward + future)
            best = max(best, total)
        return best if best != float("-inf") else 0.0

    belief_tuple = tuple(sorted(belief.items()))
    # First-action choice: maximise one-step lookahead from the current belief.
    best_action = "WAIT"
    best_value = float("-inf")
    for action in (*PROBE_ACTIONS, *COMMIT_ACTIONS):
        total = 0.0
        for rule, rule_prob in belief.items():
            if rule_prob <= 0.0:
                continue
            for observation in ("accept", "reject"):
                p = _outcome_prob(action, observation, rule)
                if p <= 0.0:
                    continue
                duration = (
                    cost.probe_duration
                    if action.startswith("PROBE_")
                    else cost.commit_duration + (cost.lockout_steps if observation == "reject" else 0)
                )
                if duration > horizon:
                    continue
                reward = _immediate_reward(action, observation, cost)
                if action.startswith("COMMIT_"):
                    future = 0.0
                else:
                    posterior = _posterior(belief, action, observation, cost.learning_rate)
                    future = value(tuple(sorted(posterior.items())), horizon - duration)
                total += rule_prob * p * (reward + future)
        if total > best_value:
            best_value = total
            best_action = action
    return best_action, best_value

# ---------------------------------------------------------------------------
# Microtask A: probe is valuable across many future decisions (long horizon,
# cheap probe, expensive lockout). Exact optimum = probe. M1 one-step = commit.
# ---------------------------------------------------------------------------

def test_microtask_A_probe_valuable_but_m1_picks_commit() -> None:
    """Long horizon: probing RED identifies the rule and avoids costly lockouts.

    Exact multi-step optimum chooses PROBE_RED. M1 one-step continuation is
    added to the successful-commit branch too, so it double-counts future
    commits and over-values COMMIT_RED. This gap is the documented M1
    limitation; M1 is therefore not yet a trustworthy probe-vs-commit method.
    """
    belief = {"red": 1 / 3, "blue": 1 / 3, "green": 1 / 3}
    cost = PublicCostModel(
        step_cost=-0.05,
        success_reward=10.0,
        probe_duration=1,
        commit_duration=1,
        lockout_steps=3,
        remaining_horizon=10,
        learning_rate=0.3,
    )
    payload = _fixed_payload()

    exact_action, exact_value = multi_step_optimal(belief, cost, horizon=10)
    assert exact_action == "PROBE_RED", f"exact optimum should probe, got {exact_action}"

    estimates = q_hat_from_prediction(payload, belief, cost, account_continuation=True)
    m1_choice = ValueAwareSelector().choose(payload, belief, cost)

    # Documented gap: M1 does not pick the exact-optimal probe; it over-values
    # some immediate commit (continuation is double-counted on success).
    assert m1_choice != "PROBE_RED", f"M1 should not pick the probe here, got {m1_choice}"
    commit_estimates = {k: v for k, v in estimates.items() if k.startswith("COMMIT_")}
    assert max(commit_estimates.values()) > estimates["PROBE_RED"], estimates

    # Multi-step Vhat recovers the information value: it picks the probe.
    ms_choice = MultiStepValueAwareSelector().choose(payload, belief, cost)
    assert ms_choice == "PROBE_RED", f"multi-step Vhat should pick the probe, got {ms_choice}"


# ---------------------------------------------------------------------------
# Microtask B: probe is not worth the cost (belief already concentrated on
# red, short horizon). Both exact optimum and M1 should commit immediately.
# ---------------------------------------------------------------------------

def test_microtask_B_probe_not_worth_it() -> None:
    """Strong prior + expensive probe + short horizon: commit RED now, probing
    only wastes steps. Both exact optimum and M1 agree on immediate commit."""
    belief = {"red": 0.99, "blue": 0.005, "green": 0.005}
    cost = PublicCostModel(
        step_cost=-0.05,
        success_reward=10.0,
        probe_duration=3,
        commit_duration=1,
        lockout_steps=5,
        remaining_horizon=4,
        learning_rate=0.3,
    )
    payload = _fixed_payload()

    exact_action, _ = multi_step_optimal(belief, cost, horizon=4)
    assert exact_action == "COMMIT_RED", f"exact optimum should commit, got {exact_action}"

    m1_choice = ValueAwareSelector().choose(payload, belief, cost)
    assert m1_choice == "COMMIT_RED", f"M1 should commit here too, got {m1_choice}"


# ---------------------------------------------------------------------------
# Belief updater: no decay when evidence simply ages (known heuristic gap).
# ---------------------------------------------------------------------------

def test_belief_updater_has_no_staleness_decay() -> None:
    """HistoryConsistencyUpdater never lowers probability on unobserved rules
    merely because time passed or because a shift may have made them stale.
    This is a known limitation of the heuristic belief model."""
    updater = HistoryConsistencyUpdater(rules=("red", "blue", "green"), learning_rate=0.3)
    belief = {"red": 0.9, "blue": 0.05, "green": 0.05}
    unchanged = updater.update(belief, "WAIT", "none")
    assert unchanged == belief, "non-probe/commit actions must leave belief untouched"
    for _ in range(10):
        unchanged = updater.update(unchanged, "WAIT", "none")
        assert unchanged == belief


def test_belief_updater_noise_floor() -> None:
    """noise_floor moves a fraction of mass toward the uniform prior.

    With noise_floor = 0.5, an accept on RED is treated as partly unreliable:
    half the mass is redistributed toward uniform, so blue/green recover some
    probability that a hard update would have drained.
    """
    updater = HistoryConsistencyUpdater(
        rules=("red", "blue", "green"), learning_rate=0.3, noise_floor=0.5
    )
    belief = {"red": 1 / 3, "blue": 1 / 3, "green": 1 / 3}
    updated = updater.update(belief, "PROBE_RED", "accept")

    assert updated["red"] == pytest.approx(0.4035, abs=1e-3)
    assert updated["blue"] == pytest.approx(0.2982, abs=1e-3)
    assert updated["green"] == pytest.approx(0.2982, abs=1e-3)
    assert sum(updated.values()) == pytest.approx(1.0)


def test_belief_updater_relevance_decay() -> None:
    """relevance_decay decays rules not confirmed by the latest outcome.

    With relevance_decay = 0.5, an accept on RED confirms RED but blue and
    green (not observed) lose half their mass. This models staleness: evidence
    that has not been refreshed loses relevance over time.
    """
    updater = HistoryConsistencyUpdater(
        rules=("red", "blue", "green"), learning_rate=0.3, relevance_decay=0.5
    )
    belief = {"red": 1 / 3, "blue": 1 / 3, "green": 1 / 3}
    updated = updater.update(belief, "PROBE_RED", "accept")

    # red confirmed: 1/3 * 1.3 = 0.4333; blue/green decayed by 0.5 after the
    # base update (1/3 * 0.7 = 0.2333, then * 0.5 = 0.1167).
    assert updated["red"] == pytest.approx(0.65, abs=1e-3)
    assert updated["blue"] == pytest.approx(0.175, abs=1e-3)
    assert updated["green"] == pytest.approx(0.175, abs=1e-3)
    assert sum(updated.values()) == pytest.approx(1.0)

