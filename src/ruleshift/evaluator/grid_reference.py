"""Evaluator-only reference computations for RuleShift-Grid.

Everything in this module is a *diagnostic*, not a policy. In particular
:func:`q_star_microtask` is a depth-limited exhaustive search over public
dynamics hypotheses; it may only be used to score how close an agent came to a
bounded reference. It is never imported by ``agents/`` or ``public_api/`` and is
never exposed in an agent-visible row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ruleshift.envs.grid.config import GridConfig
from ruleshift.public_api.grid_planner import GridPlannerState, HypothesisDynamics
from ruleshift.public_api.grid_rules import RULE_BY_ID, cell_role
from ruleshift.public_api.grid_schemas import GridAction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruleshift.agents.grid_belief import GridBelief
    from ruleshift.evaluator.grid_runner import GridEpisode
    from ruleshift.public_api.grid_rules import PublicGridMap

__all__ = ["oracle_return", "q_star_microtask"]

# Public cost constants, taken from the development GridConfig defaults so the
# diagnostic stays in lockstep with the simulator's public reward schedule.
_COSTS = GridConfig()
_ACTIONS = tuple(GridAction(kind) for kind in ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT"))


def _landing_reward(rule, public_map, coord):
    """Public immediate reward for landing on ``coord`` and whether it terminates."""

    color = next(
        (name for name, marker in public_map.markers.items() if tuple(marker) == tuple(coord)),
        None,
    )
    role = cell_role(rule, color) if color is not None else "safe"
    if role == "goal":
        return float(_COSTS.goal_reward), True
    if role == "hazard":
        return float(_COSTS.soft_hazard_damage), False
    return float(_COSTS.step_cost), False


def _search(
    state: GridPlannerState,
    remaining: int,
    model: HypothesisDynamics,
    rule,
    public_map,
    memo: dict[tuple[GridPlannerState, int], float],
) -> float:
    """Best achievable public return within ``remaining`` physical steps."""

    if remaining <= 0:
        return 0.0
    key = (state, remaining)
    if key in memo:
        return memo[key]

    best = float("-inf")
    for action in _ACTIONS:
        distribution = model.predict_proba(state, action)
        if not distribution:
            continue
        # The most likely outcome is the same convention used by public A*.
        successor = max(distribution.items(), key=lambda item: item[1])[0]
        reward, terminal = _landing_reward(rule, public_map, successor.position)
        value = reward if terminal else reward + _search(
            successor, remaining - 1, model, rule, public_map, memo
        )
        best = max(best, value)

    result = best if best != float("-inf") else 0.0
    memo[key] = result
    return result


def q_star_microtask(
    belief: "GridBelief",
    public_map: "PublicGridMap",
    position: tuple[int, int],
    horizon: int = 2,
) -> float:
    """Belief-weighted depth-limited upper bound on the reachable return.

    For each rule with positive belief mass, one :class:`HypothesisDynamics`
    model is built from public helpers and searched exhaustively to ``horizon``
    using the most likely outcome of every action. The per-rule optima are
    averaged under the rule marginal. This is an evaluator-only calibration
    diagnostic: it assumes the true rule is known when choosing the action, so
    it is an upper bound rather than an agent policy.
    """

    horizon = int(horizon)
    if horizon <= 0:
        return 0.0
    marginal = belief.marginal()
    if not marginal:
        return 0.0

    start = GridPlannerState(tuple(position), False, False)
    total = 0.0
    mass = 0.0
    for rule_id, probability in marginal.items():
        if probability <= 0.0 or rule_id not in RULE_BY_ID:
            continue
        rule = RULE_BY_ID[rule_id]
        model = HypothesisDynamics(rule, public_map)
        total += float(probability) * _search(start, horizon, model, rule, public_map, {})
        mass += float(probability)
    return total / mass if mass > 0.0 else 0.0


def oracle_return(episode: "GridEpisode") -> float | None:
    """Return the privileged oracle return when a caller supplied one."""

    value = getattr(episode, "oracle_return", None)
    return None if value is None else float(value)
