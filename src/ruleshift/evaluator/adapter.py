"""Separate evaluation-only trace container.

This module holds privileged labels in a structure that cannot be passed to an
agent by construction: no public API imports it, and the permissions tests
forbid agent-side imports of this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluatorTrace:
    """Privileged per-episode record. No solver values are computed yet."""

    scenario_id: str
    true_rule_before: str
    true_rule_after: str
    shift_time: int | None
    bayes_belief: dict[str, float] | None = None
    q_star: dict[str, float] | None = None
    delta_probe: float | None = None
    actual_noise_outcome: str | None = None
    privileged_oracle_return: float | None = None


def trace_from_env(env: Any) -> EvaluatorTrace:
    """Build a privileged trace from an environment's evaluator state."""

    state = env.evaluator_state()
    rule = str(state["mechanic_id"])
    return EvaluatorTrace(
        scenario_id=str(state.get("scenario_id", "dev-default")),
        true_rule_before="red",
        true_rule_after=rule,
        shift_time=state["shift_time"],
    )
