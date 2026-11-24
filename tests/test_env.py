"""Core Family A environment tests (master plan critical tests 1-4, 7, 14)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ruleshift.envs.config import EnvConfig
from ruleshift.public_api.schemas import FAMILY_A_ACTIONS
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.permissions import PRIVILEGED_FIELDS
from ruleshift.public_api.schemas import validate_public_observation


def make_env(**kwargs: object) -> RuleShiftEnv:
    return RuleShiftEnv(config=EnvConfig(**kwargs))


def test_reset_returns_rgb_and_public_info() -> None:
    env = make_env()
    obs, info = env.reset(seed=7)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 3
    assert obs.shape[2] == 3
    assert obs.shape[0] >= 5 and obs.shape[1] >= 5
    assert obs.dtype == np.uint8
    assert set(info).isdisjoint(PRIVILEGED_FIELDS)


def test_family_a_observation_advertises_only_family_a_macros() -> None:
    env = make_env()
    env.reset(seed=7)
    assert set(env.public_view().available_actions) == set(FAMILY_A_ACTIONS)


def test_public_view_validates_and_evaluator_state_is_separate() -> None:
    env = make_env()
    env.reset(seed=7)
    assert validate_public_observation(env.public_view().to_payload()) == []
    public_keys = set(env.public_view().to_payload())
    private_keys = set(env.evaluator_state())
    assert public_keys.isdisjoint(private_keys)
    assert {"mechanic_id", "shift_time", "true_rule"} <= private_keys


def test_same_seed_reproduces_public_and_evaluator_traces() -> None:
    actions = ["PROBE_RED", "WAIT", "COMMIT_RED", "COMMIT_BLUE", "WAIT"]

    def run() -> tuple[list[object], list[dict], list[float]]:
        env = make_env(feedback_noise=0.10, lockout_steps=3, shift_hazard=0.5)
        env.reset(seed=123)
        public: list[object] = []
        private: list[dict] = []
        rewards: list[float] = []
        for action in actions:
            _, reward, terminated, truncated, _ = env.step(action)
            public.append(env.public_view().to_payload())
            private.append(env.evaluator_state())
            rewards.append(reward)
            if terminated or truncated:
                break
        return public, private, rewards

    assert run() == run()


def test_shift_alone_does_not_change_public_view() -> None:
    shared = dict(initial_mechanic="red", feedback_noise=0.0)
    no_shift = make_env(disable_shift=True, **shared)
    forced_shift = make_env(disable_shift=False, shift_hazard=1.0, **shared)
    no_shift.reset(seed=11)
    forced_shift.reset(seed=11)

    no_shift.step("WAIT")
    forced_shift.step("WAIT")

    assert no_shift.public_view().to_payload() == forced_shift.public_view().to_payload()
    assert no_shift.evaluator_state()["mechanic_id"] == "red"
    assert forced_shift.evaluator_state()["mechanic_id"] != "red"


@pytest.mark.parametrize("detour", [0, 2, 5])
@pytest.mark.parametrize(
    "macro, expected",
    [("PROBE_BLUE", "accept"), ("PROBE_RED", "reject")],
)
def test_probe_feedback_comes_from_active_mechanic(
    detour: int, macro: str, expected: str
) -> None:
    """The probe result must survive the walk back to the start tile.

    With ``detour=0`` the return leg is empty, which previously hid the bug:
    every real cell uses ``detour >= 1`` and lost the feedback to a ``move``.
    """

    env = make_env(
        initial_mechanic="blue",
        disable_shift=True,
        feedback_noise=0.0,
        probe_detour_steps=detour,
    )
    env.reset(seed=3)
    env.step(macro)
    public = env.public_view()
    assert expected in public.public_text
    # The decision-boundary observation is what an agent actually receives.
    assert env._last_outcome == expected
    assert public.public_text.endswith(f"last={expected}")


def test_commit_travel_is_public_and_matches_the_executor() -> None:
    for detour in (0, 2, 5):
        env = make_env(probe_detour_steps=detour)
        env.reset(seed=1)
        costs = env.public_view().public_costs
        plan = env.executor.plan("COMMIT_RED", env.geometry.start)
        assert int(costs["commit_travel"]) + 1 == len(plan.ops)
        probe_plan = env.executor.plan("PROBE_RED", env.geometry.start)
        assert 2 * int(costs["probe_detour"]) + 1 == len(probe_plan.ops)


def test_failed_task_action_leaves_public_reject_feedback() -> None:
    env = make_env(
        initial_mechanic="blue",
        disable_shift=True,
        feedback_noise=0.0,
        probe_detour_steps=0,
        lockout_steps=0,
    )
    env.reset(seed=4)
    env.step("COMMIT_RED")
    assert "reject" in env.public_view().public_text
    assert env.evaluator_state()["deliveries"] == 0


def test_lockout_ticks_consume_one_step_each_then_allow_commit() -> None:
    env = make_env(
        initial_mechanic="blue",
        disable_shift=True,
        feedback_noise=0.0,
        lockout_steps=2,
    )
    env.reset(seed=4)
    _, first_reward, _, _, _ = env.step("COMMIT_RED")
    assert env.evaluator_state()["steps_taken"] == 5
    assert first_reward == pytest.approx(-5 * 0.05)
    assert env.evaluator_state()["lockout_remaining"] == 2
    # The remaining lockout is public state, not only an evaluator field.
    assert "lockout=2" in env.public_view().public_text

    for expected_remaining in (1, 0):
        _, reward, _, _, _ = env.step("WAIT")
        assert reward == pytest.approx(-0.05)
        assert env.evaluator_state()["lockout_remaining"] == expected_remaining
        assert f"lockout={expected_remaining}" in env.public_view().public_text

    # A rejected commit leaves the agent at the gate (no teleport), so the
    # follow-up commit costs a single step: 5 + 2 + 1 = 8 primitive steps.
    _, reward, _, _, _ = env.step("COMMIT_BLUE")
    assert env.evaluator_state()["steps_taken"] == 8
    assert reward == pytest.approx(10 - 0.05)  # this macro: 1 step + delivery
    assert env.evaluator_state()["deliveries"] == 1


def test_gymnasium_env_checker() -> None:
    env = gym.make("RuleShift-GateLock-v0")
    try:
        check_env(env.unwrapped, skip_render_check=False)
    finally:
        env.close()


def test_public_travel_costs_and_history_follow_the_current_position() -> None:
    env = make_env(probe_detour_steps=2, disable_shift=True)
    env.reset(seed=9)
    env.step("COMMIT_BLUE")
    public = env.public_view()
    assert public.public_costs["commit_travel"] == 0
    assert public.public_costs["station_travel"] == 2
    assert "COMMIT_BLUE outcome=reject" in public.public_history
    env.step("GO_TO_STATION")
    assert env.public_view().public_costs["station_travel"] == 0
    steps = env.evaluator_state()["steps_taken"]
    env.step("GO_TO_STATION")
    assert env.evaluator_state()["steps_taken"] == steps + 1


def test_registered_env_uses_primitive_step_limit_for_long_episodes() -> None:
    env = gym.make("RuleShift-GateLock-v0", config=EnvConfig(max_env_steps=62))
    try:
        env.reset(seed=1)
        for i in range(62):
            _, _, _, truncated, info = env.step("WAIT")
            assert info["physical_steps_elapsed"] == i + 1
            assert truncated == (i == 61)
    finally:
        env.close()
