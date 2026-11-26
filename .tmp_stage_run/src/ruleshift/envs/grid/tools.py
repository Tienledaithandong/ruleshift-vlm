"""Epistemic tools for RuleShift-Grid: INSPECT and READ_SIGN.

This module is additive. It never reads simulator-private state: every value it
returns is derived from the public map and a public rule hypothesis, and the
caller supplies the stochastic draws explicitly. A tool call is one decision;
it spends budget but never advances the physical environment step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ruleshift.public_api.grid_rules import (
    Coord,
    GridRule,
    PublicGridMap,
    cell_role,
    sign_clause,
)

from .config import GridConfig

#: The three public roles a marker cell can take under any rule hypothesis.
#: An erroneous INSPECT returns one of the *other* roles, indexed by wrong_index.
_ROLES: tuple[str, ...] = ("goal", "hazard", "safe")


@dataclass(frozen=True)
class ToolResult:
    """One tool call's public outcome. No hidden state is carried here."""

    kind: str
    target: Coord | None
    value: str | None
    cost: int
    rejected: bool
    reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable, deterministic public dict."""

        def cell(coord: Coord | None) -> list[int] | None:
            return list(coord) if coord is not None else None

        return {
            "kind": self.kind,
            "target": cell(self.target),
            "value": self.value,
            "cost": self.cost,
            "rejected": self.rejected,
            "reason": self.reason,
        }


def tool_cost(kind: str, config: GridConfig) -> int:
    """Public budget cost of one tool action."""

    if kind == "INSPECT":
        return config.inspect_cost
    if kind == "READ_SIGN":
        return config.read_sign_cost
    raise ValueError(f"unknown tool action {kind!r}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_coord(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_is_int(part) for part in value)
    )


def _in_bounds(coord: Coord, public_map: PublicGridMap) -> bool:
    x, y = coord
    return 0 <= x < public_map.width and 0 <= y < public_map.height


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _is_inspect_target(public_map: PublicGridMap, target: Coord) -> bool:
    """Only marker cells, the key cell and the door cell can be inspected."""

    if tuple(target) in public_map.walls:
        return False
    if any(tuple(coord) == tuple(target) for coord in public_map.markers.values()):
        return True
    if public_map.key is not None and tuple(public_map.key) == tuple(target):
        return True
    if public_map.door is not None and tuple(public_map.door) == tuple(target):
        return True
    return False


def _marker_color_at(public_map: PublicGridMap, target: Coord) -> str | None:
    return next(
        (
            color
            for color, coord in public_map.markers.items()
            if tuple(coord) == tuple(target)
        ),
        None,
    )


def _true_inspect_value(
    public_map: PublicGridMap, rule: GridRule, target: Coord
) -> str:
    color = _marker_color_at(public_map, target)
    if color is not None:
        return cell_role(rule, color)
    if public_map.key is not None and tuple(public_map.key) == tuple(target):
        return "key"
    # The only remaining valid target is the door. In v1 every rule has the
    # same key/door mapping, so this is truthful but rule-uninformative.
    return "required_key:" + rule.door_color


def inspect_cell(
    public_map: PublicGridMap,
    rule: GridRule,
    target: Coord,
    draw: float,
    wrong_index: int,
    config: GridConfig,
) -> ToolResult:
    """Inspect one cell, with a configurable chance of a wrong role.

    ``draw`` and ``wrong_index`` are supplied by the caller so the result is a
    deterministic function of explicit arguments. Walls and floor are rejected;
    the cost is charged only for accepted calls (enforced by the caller).
    """

    if not _is_coord(target):
        return ToolResult("INSPECT", None, None, 0, True, "invalid_target")
    target = (int(target[0]), int(target[1]))
    if not _in_bounds(target, public_map) or not _is_inspect_target(public_map, target):
        return ToolResult("INSPECT", target, None, 0, True, "invalid_target")

    true_value = _true_inspect_value(public_map, rule, target)
    value = true_value
    if draw < config.inspect_error:
        alternatives = tuple(role for role in _ROLES if role != true_value)
        if alternatives:
            value = alternatives[int(wrong_index) % len(alternatives)]

    return ToolResult("INSPECT", target, value, config.inspect_cost, False, None)


def read_sign(rule: GridRule, side: int, config: GridConfig) -> ToolResult:
    """Read one side of the signpost; exactly one clause, with no noise in v1."""

    if not _is_int(side) or not 0 <= side < len(rule.sign_clauses):
        raise ValueError(f"invalid sign side {side!r}")
    return ToolResult(
        "READ_SIGN",
        None,
        sign_clause(rule, side),
        config.read_sign_cost,
        False,
        None,
    )


def sign_side(position: Coord, signpost: Coord) -> int:
    """Which face of the signpost an adjacent agent is standing at.

    ``0`` for the south/west faces, ``1`` for east and ``2`` for north. Raises
    ``ValueError`` unless ``position`` is orthogonally adjacent to ``signpost``.
    """

    dx, dy = position[0] - signpost[0], position[1] - signpost[1]
    if (dx, dy) == (0, -1):
        return 2
    if (dx, dy) == (1, 0):
        return 1
    if (dx, dy) in ((0, 1), (-1, 0)):
        return 0
    raise ValueError("position must be orthogonally adjacent to the signpost")


def validate_tool_action(
    kind: str,
    target: Coord | None,
    position: Coord,
    public_map: PublicGridMap,
    budget: int,
    config: GridConfig,
) -> list[str]:
    """Return all public reasons why a tool action is currently illegal."""

    errors: list[str] = []
    if kind == "INSPECT":
        if not _is_coord(target):
            return ["missing_target"]
        coord = (int(target[0]), int(target[1]))
        if not _in_bounds(coord, public_map) or _manhattan(position, coord) > config.inspect_range:
            errors.append("out_of_range")
        if not _is_inspect_target(public_map, coord):
            errors.append("invalid_target")
        if budget < config.inspect_cost:
            errors.append("insufficient_budget")
    elif kind == "READ_SIGN":
        signpost = public_map.signpost
        if signpost is None:
            errors.append("no_signpost")
        elif _manhattan(position, signpost) != 1:
            errors.append("not_adjacent")
        if budget < config.read_sign_cost:
            errors.append("insufficient_budget")
    return errors


__all__ = [
    "ToolResult",
    "inspect_cell",
    "read_sign",
    "sign_side",
    "tool_cost",
    "validate_tool_action",
]
