"""Small analytical CPU reference for the Family B one-decision microtask.

Evaluator side only. It uses the true regime transition probabilities to show
where a probe's value comes from in a *dynamics* shift, as a contrast with the
Family A lockout story. This is an exact calculation for this tiny abstract
decision, not a solution of the full gridworld.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyBCosts:
    reward: float = 10.0
    step_cost: float = -0.05
    safe_steps: int = 8
    bridge_steps: int = 1
    recovery_steps: int = 10
    probe_steps: int = 5
    slip_stable: float = 0.0
    slip_slippery: float = 0.9


def crossing_value(config: FamilyBCosts, slip_probability: float) -> float:
    """Expected return of the shortcut under one slip probability."""

    success = config.reward + config.bridge_steps * config.step_cost
    failure = (config.bridge_steps + config.recovery_steps) * config.step_cost
    return (1.0 - slip_probability) * success + slip_probability * failure


def family_b_values(config: FamilyBCosts, q_stable: float) -> dict[str, float]:
    """Return values for safe, bridge and a noise-free revealing probe."""

    safe = config.reward + config.safe_steps * config.step_cost
    bridge = (
        q_stable * crossing_value(config, config.slip_stable)
        + (1.0 - q_stable) * crossing_value(config, config.slip_slippery)
    )
    stable_best = max(
        safe, crossing_value(config, config.slip_stable)
    )
    slippery_best = max(
        safe, crossing_value(config, config.slip_slippery)
    )
    probe_cost = config.probe_steps * abs(config.step_cost)
    probe = -probe_cost + q_stable * stable_best + (1.0 - q_stable) * slippery_best
    return {
        "GO_SAFE": safe,
        "CROSS_BRIDGE": bridge,
        "PROBE_STABLE": probe,
        "PROBE_SLIPPERY": probe,
    }


def delta_probe_family_b(config: FamilyBCosts, q_stable: float) -> float:
    """Probe value minus the best immediate task value."""

    values = family_b_values(config, q_stable)
    probe_best = max(values["PROBE_STABLE"], values["PROBE_SLIPPERY"])
    task_best = max(values["GO_SAFE"], values["CROSS_BRIDGE"])
    return probe_best - task_best
