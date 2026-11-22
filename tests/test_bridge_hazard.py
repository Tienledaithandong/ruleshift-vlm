"""Family B: transition law changes between regimes, not just reward labels."""

from __future__ import annotations

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

from ruleshift.envs.bridge_hazard import BridgeConfig, BridgeHazardEnv
from ruleshift.permissions import PRIVILEGED_FIELDS
from ruleshift.public_api.schemas import validate_public_observation


def env_with(**overrides: object) -> BridgeHazardEnv:
    return BridgeHazardEnv(BridgeConfig(**overrides))


def crossing_success_rate(regime: str, count: int = 200) -> float:
    env = env_with(
        initial_regime=regime,
        disable_shift=True,
        slip_stable=0.05,
        slip_slippery=0.40,
    )
    successes = 0
    for seed in range(count):
        env.reset(seed=seed)
        env.step("CROSS_BRIDGE")
        successes += env.evaluator_state()["side"] == "goal"
    return successes / count


def test_slip_probabilities_differ_by_regime_and_match_configuration() -> None:
    stable = crossing_success_rate("stable")
    slippery = crossing_success_rate("slippery")
    assert stable == pytest.approx(0.95, abs=0.05)
    assert slippery == pytest.approx(0.60, abs=0.06)
    assert stable > slippery


def test_slip_costs_recovery_steps_and_no_reward() -> None:
    failed = env_with(
        initial_regime="stable",
        disable_shift=True,
        slip_stable=1.0,
        bridge_steps=3,
        recovery_steps=4,
    )
    failed.reset(seed=1)
    _, reward, _, _, info = failed.step("CROSS_BRIDGE")
    assert info["physical_steps_elapsed"] == 7
    assert reward == pytest.approx(-7 * 0.05)
    assert failed.evaluator_state()["side"] == "start"

    clean = env_with(
        initial_regime="stable",
        disable_shift=True,
        slip_stable=0.0,
        bridge_steps=3,
    )
    clean.reset(seed=1)
    _, reward, _, _, info = clean.step("CROSS_BRIDGE")
    assert info["physical_steps_elapsed"] == 3
    assert reward == pytest.approx(10 - 3 * 0.05)
    assert clean.evaluator_state()["side"] == "goal"


def test_safe_route_is_deterministic_success() -> None:
    env = env_with(disable_shift=True, safe_steps=8)
    env.reset(seed=3)
    _, reward, _, _, info = env.step("GO_SAFE")
    assert info["physical_steps_elapsed"] == 8
    assert reward == pytest.approx(10 - 8 * 0.05)
    assert env.evaluator_state()["side"] == "goal"


@pytest.mark.parametrize(
    "macro, expected",
    [("PROBE_STABLE", "safe"), ("PROBE_SLIPPERY", "slippery")],
)
def test_probe_feedback_follows_active_regime(macro: str, expected: str) -> None:
    env = env_with(
        initial_regime="stable", disable_shift=True, feedback_noise=0.0
    )
    env.reset(seed=4)
    env.step(macro)
    assert expected in env.public_view().public_text


def test_shift_alone_does_not_change_public_view() -> None:
    base = dict(initial_regime="stable")
    no_shift = env_with(disable_shift=True, **base)
    forced = env_with(disable_shift=False, shift_hazard=1.0, **base)
    no_shift.reset(seed=11)
    forced.reset(seed=11)
    no_shift.step("WAIT")
    forced.step("WAIT")
    assert no_shift.public_view().to_payload() == forced.public_view().to_payload()
    assert no_shift.evaluator_state()["mechanic_id"] == "stable"
    assert forced.evaluator_state()["mechanic_id"] == "slippery"


def test_public_view_and_info_have_no_privileged_fields() -> None:
    env = env_with()
    _, info = env.reset(seed=2)
    assert validate_public_observation(env.public_view().to_payload()) == []
    assert set(info).isdisjoint(PRIVILEGED_FIELDS)
    assert set(env.evaluator_state()) & PRIVILEGED_FIELDS


def test_held_out_maps_change_pixels_but_not_semantics() -> None:
    first = env_with(map_id="A", distractor_seed=1)
    held_out = env_with(map_id="held-out-7", distractor_seed=1)
    first.reset(seed=5)
    held_out.reset(seed=5)
    assert not (first.render() == held_out.render()).all()
    assert first.public_view().public_text == held_out.public_view().public_text
    assert first.public_view().public_costs == held_out.public_view().public_costs


def test_gymnasium_env_checker() -> None:
    env = gym.make("RuleShift-BridgeHazard-v0")
    try:
        check_env(env.unwrapped, skip_render_check=False)
    finally:
        env.close()
