"""Small analytical CPU reference for Family B's changed transition law."""

from __future__ import annotations

import pytest

from ruleshift.evaluator.family_b_reference import (
    FamilyBCosts,
    crossing_value,
    delta_probe_family_b,
    family_b_values,
)


def costs() -> FamilyBCosts:
    return FamilyBCosts(
        reward=10.0,
        step_cost=-0.05,
        safe_steps=12,
        bridge_steps=1,
        recovery_steps=10,
        probe_steps=5,
        slip_stable=0.0,
        slip_slippery=0.9,
    )


def test_crossing_value_uses_the_regime_transition_probability() -> None:
    config = costs()
    assert crossing_value(config, 0.0) == pytest.approx(10 - 1 * 0.05)
    # slip: 11 steps of cost and no delivery reward
    assert crossing_value(config, 1.0) == pytest.approx(-11 * 0.05)


def test_family_b_values_are_hand_computed() -> None:
    config = costs()
    values = family_b_values(config, q_stable=0.5)
    assert values["GO_SAFE"] == pytest.approx(10 - 12 * 0.05)
    assert values["CROSS_BRIDGE"] == pytest.approx(0.5 * 9.95 + 0.5 * 0.5)
    assert values["PROBE_STABLE"] == pytest.approx(-0.25 + 0.5 * 9.95 + 0.5 * 9.4)


def test_probe_becomes_worthwhile_when_slip_risk_differs_strongly() -> None:
    assert delta_probe_family_b(costs(), q_stable=0.5) > 0
    safe = FamilyBCosts(
        reward=10.0,
        step_cost=-0.05,
        safe_steps=2,
        bridge_steps=1,
        recovery_steps=0,
        probe_steps=5,
        slip_stable=0.0,
        slip_slippery=0.9,
    )
    assert delta_probe_family_b(safe, q_stable=0.5) < 0
