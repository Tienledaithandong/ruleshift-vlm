"""Dependency-free public rule hypotheses and map geometry for RuleShift-Grid.

This module is additive: it neither imports the simulator/evaluator nor uses
``numpy``/``gymnasium``. It models only what an agent is allowed to know: the
geometry of the public map, a menu of candidate rules stated in plain language,
and deterministic helpers for reasoning about those candidates. The true rule
for an episode is never stored here -- ``GRID_HYPOTHESES`` is a *public* menu,
and the signpost clauses are deliberately insufficient to single one out.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

#: An (x, y) cell in the public grid.
Coord = tuple[int, int]

# Direction arithmetic. x grows to the right, y grows downward.
_DELTAS: Mapping[str, Coord] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
_PERPENDICULARS: Mapping[str, tuple[str, str]] = {
    "UP": ("LEFT", "RIGHT"),
    "DOWN": ("LEFT", "RIGHT"),
    "LEFT": ("UP", "DOWN"),
    "RIGHT": ("UP", "DOWN"),
}

_IDENTITY_MOVE_MAP: dict[str, str] = {
    "UP": "UP",
    "DOWN": "DOWN",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
}
_SWAP_MOVE_MAP: dict[str, str] = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}
_ROTATE_MOVE_MAP: dict[str, str] = {
    "UP": "RIGHT",
    "RIGHT": "DOWN",
    "DOWN": "LEFT",
    "LEFT": "UP",
}

# Shared signpost clauses. The clause-sharing table is load-bearing: no single
# side -- and no complete three-clause set -- may identify one rule, so the
# agent must still use transition evidence to narrow the menu.
_SHARED_KEY_CLAUSE = "The green key opens the green door."
_RED_NOT_HAZARD = "The red tile is not the hazard."
_RED_NOT_GOAL = "The red tile is not the goal."
_MAY_SLIP = "Movement may slip."
_MAY_REMAP = "Movement may be remapped."


@dataclass(frozen=True)
class PublicGridMap:
    """The complete agent-visible grid geometry. No hidden transitions here."""

    width: int
    height: int
    walls: frozenset[Coord]
    start: Coord
    markers: Mapping[str, Coord]
    key: Coord
    door: Coord
    signpost: Coord

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable, deterministic public dict."""

        return {
            "width": self.width,
            "height": self.height,
            "walls": [list(coord) for coord in sorted(self.walls)],
            "start": list(self.start),
            "markers": {
                color: list(coord) for color, coord in sorted(self.markers.items())
            },
            "key": list(self.key),
            "door": list(self.door),
            "signpost": list(self.signpost),
        }


@dataclass(frozen=True)
class GridRule:
    """One public candidate rule, stated without any hidden likelihood table."""

    rule_id: str
    description: str
    move_map: Mapping[str, str]
    slip_prob: float
    slide_two_prob: float
    goal_color: str
    hazard_color: str
    safe_color: str
    key_color: str
    door_color: str
    sign_clauses: tuple[str, ...]


BASE_RULE = GridRule(
    rule_id="base",
    description="Movement follows the chosen direction; the red tile is the goal.",
    move_map=dict(_IDENTITY_MOVE_MAP),
    slip_prob=0.10,
    slide_two_prob=0.0,
    goal_color="red",
    hazard_color="blue",
    safe_color="green",
    key_color="green",
    door_color="green",
    sign_clauses=(_RED_NOT_HAZARD, _MAY_SLIP, _SHARED_KEY_CLAUSE),
)

_REMAP_SWAP_RULE = replace(
    BASE_RULE,
    rule_id="remap_swap",
    description="Up/down and left/right are swapped; the red tile is the goal.",
    move_map=dict(_SWAP_MOVE_MAP),
    sign_clauses=(_RED_NOT_HAZARD, _MAY_REMAP, _SHARED_KEY_CLAUSE),
)
_REMAP_ROTATE_RULE = replace(
    BASE_RULE,
    rule_id="remap_rotate",
    description="Movement is rotated clockwise; the red tile is the goal.",
    move_map=dict(_ROTATE_MOVE_MAP),
    sign_clauses=(_RED_NOT_HAZARD, _MAY_REMAP, _SHARED_KEY_CLAUSE),
)
_SLIPPERY_RULE = replace(
    BASE_RULE,
    rule_id="slippery",
    description="Movement often slips and may slide two cells; red is the goal.",
    move_map=dict(_IDENTITY_MOVE_MAP),
    slip_prob=0.80,
    slide_two_prob=0.5,
)
_HAZARD_GOAL_SWAP_RULE = replace(
    BASE_RULE,
    rule_id="hazard_goal_swap",
    description="The blue tile is the goal and the red tile is a hazard.",
    move_map=dict(_IDENTITY_MOVE_MAP),
    goal_color="blue",
    hazard_color="red",
    sign_clauses=(_RED_NOT_GOAL, _MAY_SLIP, _SHARED_KEY_CLAUSE),
)
_SIGN_COMPOSITION_RULE = replace(
    BASE_RULE,
    rule_id="sign_composition",
    description="Green is the goal and red is a hazard; movement is ordinary.",
    move_map=dict(_IDENTITY_MOVE_MAP),
    goal_color="green",
    hazard_color="red",
    safe_color="blue",
    sign_clauses=(_RED_NOT_GOAL, _MAY_SLIP, _SHARED_KEY_CLAUSE),
)

#: The public hypothesis menu, in the same order as ``GRID_RULE_IDS``.
GRID_HYPOTHESES: tuple[GridRule, ...] = (
    BASE_RULE,
    _REMAP_SWAP_RULE,
    _REMAP_ROTATE_RULE,
    _SLIPPERY_RULE,
    _HAZARD_GOAL_SWAP_RULE,
    _SIGN_COMPOSITION_RULE,
)
RULE_BY_ID: dict[str, GridRule] = {rule.rule_id: rule for rule in GRID_HYPOTHESES}


def cell_role(rule: GridRule, color: str) -> str:
    """Return the public role of a color under ``rule``."""

    if color == rule.goal_color:
        return "goal"
    if color == rule.hazard_color:
        return "hazard"
    return "safe"


def key_opens(rule: GridRule, key_color: str, door_color: str) -> bool:
    """Whether a carried key color opens the given door color under ``rule``."""

    return key_color == rule.key_color and door_color == rule.door_color


def sign_clause(rule: GridRule, side: int) -> str:
    """The clause written on the given side of the rule's signpost."""

    return rule.sign_clauses[side]


def rules_consistent_with_clause(side: int, clause: str) -> tuple[str, ...]:
    """Rule ids whose ``side`` clause matches ``clause`` exactly."""

    return tuple(
        rule.rule_id
        for rule in GRID_HYPOTHESES
        if sign_clause(rule, side) == clause
    )


def rules_consistent_with_all_clauses(clauses: tuple[str, ...]) -> tuple[str, ...]:
    """Rule ids matching every (side, clause) pair in ``clauses``."""

    return tuple(
        rule.rule_id
        for rule in GRID_HYPOTHESES
        if all(sign_clause(rule, side) == clause for side, clause in enumerate(clauses))
    )


def _blocked(coord: Coord, public_map: PublicGridMap) -> bool:
    x, y = coord
    return (
        coord in public_map.walls
        or not (0 <= x < public_map.width and 0 <= y < public_map.height)
    )


def _step(coord: Coord, delta: Coord) -> Coord:
    return (coord[0] + delta[0], coord[1] + delta[1])


def _resolve_one(
    position: Coord, direction: str, public_map: PublicGridMap
) -> Coord:
    target = _step(position, _DELTAS[direction])
    return position if _blocked(target, public_map) else target


def _resolve_two(
    position: Coord, direction: str, public_map: PublicGridMap
) -> Coord:
    delta = _DELTAS[direction]
    first = _step(position, delta)
    if _blocked(first, public_map):
        return position
    second = _step(first, delta)
    if _blocked(second, public_map):
        return first
    return second


def movement_outcomes(
    rule: GridRule,
    position: Coord,
    action: str,
    public_map: PublicGridMap,
) -> dict[Coord, float]:
    """Public distribution over next cells for one intended movement.

    The intended direction passes through ``rule.move_map``; with probability
    ``slip_prob`` the agent instead slides two cells or drifts to one of the two
    perpendicular cells. Each move is resolved against walls and bounds, and
    equal landing cells are aggregated. Zero-probability entries are dropped.
    """

    actual = rule.move_map.get(action, action)
    slip = rule.slip_prob
    slide = rule.slide_two_prob
    left_dir, right_dir = _PERPENDICULARS[actual]
    drift = slip * (1.0 - slide) / 2.0

    weighted_moves = (
        (actual, 1.0 - slip, False),
        (actual, slip * slide, True),
        (left_dir, drift, False),
        (right_dir, drift, False),
    )

    outcomes: dict[Coord, float] = {}
    for direction, probability, is_slide in weighted_moves:
        if not probability:
            continue
        landing = (
            _resolve_two(position, direction, public_map)
            if is_slide
            else _resolve_one(position, direction, public_map)
        )
        outcomes[landing] = outcomes.get(landing, 0.0) + probability
    return outcomes


__all__ = [
    "BASE_RULE",
    "GRID_HYPOTHESES",
    "RULE_BY_ID",
    "Coord",
    "GridRule",
    "PublicGridMap",
    "cell_role",
    "key_opens",
    "movement_outcomes",
    "rules_consistent_with_all_clauses",
    "rules_consistent_with_clause",
    "sign_clause",
]
