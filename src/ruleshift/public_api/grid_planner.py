"""Dependency-free public planning helpers for RuleShift-Grid.

This module is additive and stays inside the public boundary: it imports only
the public grid rule/schema modules and the standard library. It never imports
the simulator or the privileged evaluator, so agents may import it freely.

``DynamicsModel`` is a small pluggable protocol. :class:`HypothesisDynamics`
is the reference implementation: it turns a *hypothesized* public rule into a
one-step distribution, which :func:`astar` consumes through the protocol. A*
is therefore only one possible planner behind the protocol, not a privileged
solver.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass
from typing import Protocol

from .grid_rules import Coord, GridRule, PublicGridMap, effective_map, movement_outcomes
from .grid_schemas import GridAction

__all__ = [
    "DynamicsModel",
    "GridPlannerState",
    "HypothesisDynamics",
    "astar",
    "plan_to_goal",
]

#: Movement kinds the planner knows how to push through the dynamics model.
_MOVE_KINDS: tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
#: All physical action kinds explored by the reference A* planner.
_ACTION_KINDS: tuple[str, ...] = _MOVE_KINDS + ("INTERACT",)


@dataclass(frozen=True)
class GridPlannerState:
    """The small, fully public state an agent reasons over while planning."""

    position: Coord
    has_key: bool
    door_open: bool


class DynamicsModel(Protocol):
    """One-step dynamics interface that any public planner may consume."""

    def predict_proba(
        self, state: GridPlannerState, action: GridAction
    ) -> dict[GridPlannerState, float]: ...

    def sample(
        self, state: GridPlannerState, action: GridAction, rng: random.Random
    ) -> GridPlannerState: ...


@dataclass(frozen=True)
class HypothesisDynamics:
    """Public one-step dynamics under a single hypothesized rule."""

    rule: GridRule
    public_map: PublicGridMap

    def predict_proba(
        self, state: GridPlannerState, action: GridAction
    ) -> dict[GridPlannerState, float]:
        """Return the public distribution over successor planner states."""
        if action.kind == "INTERACT":
            return self._interact(state)

        blocked_map = effective_map(self.public_map, state.door_open)

        outcomes = movement_outcomes(
            self.rule, state.position, action.kind, blocked_map
        )
        result: dict[GridPlannerState, float] = {}
        for coord, probability in outcomes.items():
            if probability <= 0.0:
                continue
            successor = GridPlannerState(coord, state.has_key, state.door_open)
            result[successor] = result.get(successor, 0.0) + probability
        return result

    def _interact(
        self, state: GridPlannerState
    ) -> dict[GridPlannerState, float]:
        """Deterministic key pickup / door opening, else a self-loop."""
        key = self.public_map.key
        door = self.public_map.door

        if key is not None and state.position == key and not state.has_key:
            successor = GridPlannerState(state.position, True, state.door_open)
            return {successor: 1.0}

        if (
            door is not None
            and _adjacent(state.position, door)
            and state.has_key
            and not state.door_open
        ):
            successor = GridPlannerState(state.position, state.has_key, True)
            return {successor: 1.0}

        return {state: 1.0}

    def sample(
        self, state: GridPlannerState, action: GridAction, rng: random.Random
    ) -> GridPlannerState:
        """Draw one successor using ``rng`` and the predicted distribution."""
        distribution = self.predict_proba(state, action)
        if not distribution:
            return state
        threshold = rng.random()
        cumulative = 0.0
        fallback = state
        for successor, probability in distribution.items():
            fallback = successor
            cumulative += probability
            if threshold < cumulative:
                return successor
        return fallback


def _adjacent(a: Coord, b: Coord) -> bool:
    """Whether two cells share an edge (Manhattan distance of one)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct(
    start: GridPlannerState,
    goal_state: GridPlannerState,
    came_from: dict[GridPlannerState, tuple[GridPlannerState, GridAction]],
) -> list[GridAction]:
    plan: list[GridAction] = []
    cursor = goal_state
    while cursor != start:
        previous, action = came_from[cursor]
        plan.append(action)
        cursor = previous
    plan.reverse()
    return plan


def astar(
    start: GridPlannerState,
    goal: Coord,
    model: DynamicsModel,
    max_expansions: int = 10000,
) -> list[GridAction]:
    """Find a shortest action plan from ``start`` to cell ``goal``.

    Successors come from ``model.predict_proba`` using the most likely outcome
    of each action; self-loops are ignored and each edge costs one. Returns an
    empty list when the goal is unreachable or the expansion budget is spent.
    """
    actions = tuple(GridAction(kind) for kind in _ACTION_KINDS)
    counter = itertools.count()
    heuristic = lambda state: _manhattan(state.position, goal)
    open_set: list[tuple[int, int, int, GridPlannerState]] = [
        (heuristic(start), 0, next(counter), start)
    ]
    came_from: dict[GridPlannerState, tuple[GridPlannerState, GridAction]] = {}
    best_g: dict[GridPlannerState, int] = {start: 0}
    expansions = 0

    while open_set:
        _, g, _, state = heapq.heappop(open_set)
        if g > best_g.get(state, math.inf):
            continue  # stale heap entry
        if state.position == goal:
            return _reconstruct(start, state, came_from)

        expansions += 1
        if expansions > max_expansions:
            return []

        for action in actions:
            distribution = model.predict_proba(state, action)
            if not distribution:
                continue
            successor = max(distribution.items(), key=lambda kv: kv[1])[0]
            if successor == state:
                continue
            tentative_g = g + 1
            if tentative_g < best_g.get(successor, math.inf):
                best_g[successor] = tentative_g
                came_from[successor] = (state, action)
                heapq.heappush(
                    open_set,
                    (
                        tentative_g + heuristic(successor),
                        tentative_g,
                        next(counter),
                        successor,
                    ),
                )

    return []


def plan_to_goal(
    rule: GridRule,
    public_map: PublicGridMap,
    start: Coord,
    has_key: bool = False,
    door_open: bool = False,
) -> list[GridAction]:
    """Plan to the marker matching ``rule.goal_color`` under ``rule``."""
    goal = public_map.markers[rule.goal_color]
    model = HypothesisDynamics(rule, public_map)
    return astar(GridPlannerState(start, has_key, door_open), goal, model)
