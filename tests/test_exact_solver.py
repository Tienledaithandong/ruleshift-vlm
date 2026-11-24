"""Exact DP versus an independent forward brute-force policy enumeration.

The brute-force checker below does NOT perform a max-at-each-node recursion.
It enumerates every deterministic policy tree explicitly and evaluates each one
by forward path expansion, then takes the maximum over the whole policy set.
It shares only the generative model (macro_branches), never the DP algorithm.
Expected numeric values are hand-derived literals.
"""

from __future__ import annotations

import itertools

import pytest

from ruleshift.evaluator.bayes_filter import RuleShiftFilter
from ruleshift.evaluator.exact_solver import (
    COMMIT_ACTIONS,
    PROBE_ACTIONS,
    Branch,
    ExactSolver,
    MicrotaskConfig,
    branch_reward,
    freeze_belief,
    macro_branches,
)

ACTIONS = ("COMMIT_RED", "COMMIT_BLUE", "PROBE_RED")
OUTCOMES = ("accept", "reject")


def make_config(
    lockout: int = 0, hazard: float = 0.2, noise: float = 0.1
) -> MicrotaskConfig:
    return MicrotaskConfig(
        rules=("red", "blue", "green"),
        initial_rule="red",
        shift_hazard=hazard,
        shift_targets=("blue", "green"),
        feedback_noise=noise,
        reward_success=10.0,
        step_cost=-0.05,
        lockout_steps=lockout,
        durations={action: {1: 1.0} for action in ACTIONS},
        observation_steps={action: 0 for action in ACTIONS},
    )


def uncertain_belief() -> dict[tuple[str, bool], float]:
    return {("red", False): 0.5, ("blue", True): 0.5}


# ---------------------------------------------------------------------------
# Independent brute force: enumerate policies, evaluate forward.
# ---------------------------------------------------------------------------
def _histories(horizon: int) -> list[tuple[str, ...]]:
    return [
        tuple(bits)
        for depth in range(horizon)
        for bits in itertools.product(OUTCOMES, repeat=depth)
    ]


def evaluate_policy(
    config: MicrotaskConfig,
    belief: dict[tuple[str, bool], float],
    horizon: int,
    policy: dict[tuple[str, ...], str],
) -> float:
    total = 0.0
    # Primitive steps consumed are tracked separately from decision depth:
    # a rejected commit spends extra lockout steps without extra decisions.
    stack: list[tuple[float, dict, float, tuple[str, ...], int]] = [
        (1.0, dict(belief), 0.0, (), 0)
    ]
    while stack:
        probability, current, accumulated, history, steps_used = stack.pop()
        remaining = horizon - steps_used
        if remaining <= 0:
            total += probability * accumulated
            continue
        action = policy[history]
        for branch in macro_branches(config, current, action):
            if branch.probability <= 0.0:
                continue
            executed = min(branch.steps, remaining)
            reward = branch_reward(config, branch, executed)
            stack.append(
                (
                    probability * branch.probability,
                    branch.posterior,
                    accumulated + reward,
                    history + (branch.observation,),
                    steps_used + branch.steps,
                )
            )
    return total


def brute_force_value(
    config: MicrotaskConfig,
    belief: dict[tuple[str, bool], float],
    horizon: int,
    forced_action: str | None = None,
) -> float:
    histories = _histories(horizon)
    best = float("-inf")
    if forced_action is None:
        policies = (
            dict(zip(histories, choices))
            for choices in itertools.product(ACTIONS, repeat=len(histories))
        )
    else:
        later_histories = [h for h in histories if h != ()]
        policies = (
            {**dict(zip(later_histories, choices)), (): forced_action}
            for choices in itertools.product(ACTIONS, repeat=len(later_histories))
        )
    for policy in policies:
        best = max(best, evaluate_policy(config, belief, horizon, policy))
    return best


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_dp_value_matches_brute_force(horizon: int) -> None:
    config = make_config(lockout=2)
    belief = uncertain_belief()
    solver = ExactSolver(config, actions=ACTIONS)
    assert solver.value(belief, horizon) == pytest.approx(
        brute_force_value(config, belief, horizon), abs=1e-8
    )


@pytest.mark.parametrize("action", ACTIONS)
def test_dp_q_value_matches_brute_force_with_forced_action(action: str) -> None:
    config = make_config(lockout=2)
    belief = uncertain_belief()
    solver = ExactSolver(config, actions=ACTIONS)
    q_values = dict(solver.q_values(freeze_belief(belief), 3))
    assert q_values[action] == pytest.approx(
        brute_force_value(config, belief, 3, forced_action=action), abs=1e-8
    )


def test_zero_horizon_value_is_zero() -> None:
    solver = ExactSolver(make_config(), actions=ACTIONS)
    assert solver.value(uncertain_belief(), 0) == 0.0


def test_branch_probabilities_sum_to_one_for_every_action() -> None:
    config = make_config(lockout=2)
    for action in ACTIONS:
        branches = macro_branches(config, uncertain_belief(), action)
        assert sum(branch.probability for branch in branches) == pytest.approx(1.0)


def test_delta_probe_is_positive_when_lockout_is_expensive() -> None:
    config = make_config(lockout=2, hazard=0.0, noise=0.0)
    solver = ExactSolver(config, actions=ACTIONS)
    # Hand check at h=3: commit = .5*29.85 + .5*(-0.15) = 14.85;
    # probe = .5*19.85 + .5*19.85 = 19.85; delta = +5.0.
    assert solver.delta_probe(uncertain_belief(), 3) == pytest.approx(5.0)


def test_delta_probe_is_negative_when_lockout_is_cheap() -> None:
    config = make_config(lockout=0, hazard=0.0, noise=0.0)
    solver = ExactSolver(config, actions=ACTIONS)
    # With no lockout a wrong commit is informative and cheap: commit = 24.85,
    # probe = 19.85, delta = -5.0.
    assert solver.delta_probe(uncertain_belief(), 3) == pytest.approx(-5.0)


def test_in_macro_shift_is_visible_to_the_observation() -> None:
    config = MicrotaskConfig(
        shift_hazard=1.0,
        shift_targets=("blue",),
        feedback_noise=0.0,
        durations={"COMMIT_RED": {2: 1.0}},
        observation_steps={"COMMIT_RED": 1},
    )
    belief = {("red", False): 1.0}
    branches = macro_branches(config, belief, "COMMIT_RED")
    # Shift happens at step 0, before the observation at step 1, so the
    # reject likelihood is evaluated against blue and the posterior is shifted.
    rejects = [branch for branch in branches if branch.observation == "reject"]
    assert sum(branch.probability for branch in rejects) == pytest.approx(1.0)
    assert all(branch.posterior[("blue", True)] == pytest.approx(1.0) for branch in rejects)
    assert all(isinstance(branch, Branch) for branch in branches)


def test_delta_probe_requires_one_probe_and_one_task_action() -> None:
    """Restricted action sets must error, not silently mislabel the delta."""

    task_only = ExactSolver(make_config(), ("COMMIT_RED", "COMMIT_BLUE"))
    with pytest.raises(ValueError, match="delta_probe"):
        task_only.delta_probe(uncertain_belief(), 3)

    probe_only = ExactSolver(make_config(), ("PROBE_RED", "WAIT"))
    with pytest.raises(ValueError, match="delta_probe"):
        probe_only.delta_probe(uncertain_belief(), 3)


def test_family_a_microtask_durations_match_the_macro_executor() -> None:
    from ruleshift.evaluator.exact_solver import family_a_microtask
    from ruleshift.envs.macro import Geometry, MacroExecutor

    config = family_a_microtask(
        probe_detour_steps=3,
        commit_travel_steps=4,
        feedback_noise=0.0,
        lockout_steps=0,
    )
    executor = MacroExecutor(
        Geometry(start=(0, 0), gate=(4, 0), station=(3, 0), grid=(5, 5))
    )
    assert next(iter(config.durations["PROBE_BLUE"])) == len(
        executor.plan("PROBE_BLUE", (0, 0)).ops
    )
    assert next(iter(config.durations["COMMIT_RED"])) == len(
        executor.plan("COMMIT_RED", (0, 0)).ops
    )
    # The test outcome is observed after the outbound leg, not at the start.
    assert config.observation_steps["PROBE_BLUE"] == 3
    assert config.observation_steps["COMMIT_RED"] == 4


def test_grid_reference_matches_env_after_rejected_commit_and_station_move() -> None:
    from ruleshift.envs.config import EnvConfig
    from ruleshift.envs.gate_lock import RuleShiftEnv
    from ruleshift.evaluator.exact_solver import family_a_microtask

    env = RuleShiftEnv(EnvConfig(
        probe_detour_steps=2, lockout_steps=2, disable_shift=True, feedback_noise=0.0
    ))
    env.reset(seed=1)
    config = family_a_microtask(
        probe_detour_steps=2, commit_travel_steps=4, feedback_noise=0, lockout_steps=2
    )
    belief = {("red", False): 1.0}
    pos, locked = env.geometry.start, 0
    for action in ("COMMIT_BLUE", "COMMIT_RED", "GO_TO_STATION", "GO_TO_STATION",
                   "PROBE_RED", "COMMIT_BLUE", "WAIT", "WAIT", "COMMIT_RED"):
        (branch,) = macro_branches(config, belief, action, pos, locked)
        before = env.evaluator_state()["steps_taken"]
        _, reward, _, _, _ = env.step(action)
        state = env.evaluator_state()
        assert branch_reward(config, branch, branch.steps) == pytest.approx(reward)
        assert branch.steps == state["steps_taken"] - before
        assert branch.position == state["position"]
        assert branch.lockout_remaining == state["lockout_remaining"]
        assert env.public_view().public_text.endswith("last=" + branch.observation)
        pos, locked = branch.position, branch.lockout_remaining


def test_abstract_lockout_steps_advance_the_hidden_hazard() -> None:
    config = MicrotaskConfig(shift_hazard=0.5, shift_targets=("blue",), lockout_steps=2)
    (branch,) = macro_branches(config, {("red", False): 1.0}, "COMMIT_GREEN")
    assert branch.steps == 3
    assert branch.posterior[("red", False)] == pytest.approx(0.5**3)


def test_grid_probe_models_shift_before_test_and_during_return_trip() -> None:
    from ruleshift.evaluator.exact_solver import family_a_microtask

    config = family_a_microtask(
        probe_detour_steps=1, commit_travel_steps=4, feedback_noise=0,
        lockout_steps=0, shift_hazard=0.5,
    )
    branches = macro_branches(config, {("red", False): 1.0}, "PROBE_RED")
    accepted = next(branch for branch in branches if branch.observation == "accept")
    # No shift on the two steps up to and including the test. A shift is
    # still possible on the one-step return trip even after a clean accept.
    assert accepted.probability == pytest.approx(0.25)
    assert accepted.posterior[("red", False)] == pytest.approx(0.5)
    assert accepted.posterior[("blue", True)] == pytest.approx(0.25)
