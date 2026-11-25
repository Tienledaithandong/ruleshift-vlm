"""The full-environment reference must be labelled approximate, never exact."""

from __future__ import annotations

import math

import pytest
from ruleshift.evaluator.approx_reference import TruncatedRolloutReference
from ruleshift.envs.config import EnvConfig


def test_approx_reference_is_labelled_and_deterministic() -> None:
    config = EnvConfig(
        shift_hazard=0.0,
        disable_shift=True,
        initial_mechanic="red",
        probe_detour_steps=0,
        lockout_steps=0,
    )
    first = TruncatedRolloutReference(config, horizon=5, samples=2, seed=7)
    second = TruncatedRolloutReference(config, horizon=5, samples=2, seed=7)
    result = first.q_values()
    assert "APPROXIMATE" in result.label
    assert result == second.q_values()
    assert all(math.isfinite(value) for value in result.values.values())


def test_approx_q_value_on_deterministic_micro_horizon() -> None:
    config = EnvConfig(
        shift_hazard=0.0,
        disable_shift=True,
        initial_mechanic="red",
        probe_detour_steps=0,
        lockout_steps=0,
    )
    # The commit macro from the start tile is exactly 5 primitive steps.
    result = TruncatedRolloutReference(config, horizon=5, samples=1, seed=1).q_values()
    assert result.values["COMMIT_RED"] == 10.0 - 5 * 0.05


def test_approx_rollout_starts_from_supplied_belief_and_only_offers_gate_actions() -> None:
    reference = TruncatedRolloutReference(EnvConfig(shift_hazard=1.0), horizon=5, samples=2)
    result = reference.q_values({("green", True): 1.0})
    assert result.values["COMMIT_GREEN"] == pytest.approx(9.75)
    assert result.values["COMMIT_RED"] == pytest.approx(-0.25)
    assert "CROSS_BRIDGE" not in result.values
