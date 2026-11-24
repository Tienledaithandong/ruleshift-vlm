"""Macro-to-primitive expansion using only public map geometry.

This planner never sees the active mechanic, the shift state or any transition
probability. It is a pure function of a public geometry and a public position.
"""

from __future__ import annotations

from dataclasses import dataclass

from ruleshift.public_api.schemas import MACRO_ACTIONS_FROZEN

Coord = tuple[int, int]


@dataclass(frozen=True)
class Geometry:
    """Public fixed map geometry shared by every compared agent."""

    start: Coord
    gate: Coord
    station: Coord
    grid: Coord  # (width, height)


@dataclass(frozen=True)
class PrimitiveOp:
    """One primitive step inside a macro."""

    kind: str  # "move" | "move_back" | "commit" | "test" | "wait"
    target: Coord | None = None
    color: str | None = None


@dataclass(frozen=True)
class MacroPlan:
    """The primitive expansion of one public macro id."""

    macro_id: str
    ops: tuple[PrimitiveOp, ...]


def _route(source: Coord, target: Coord, kind: str = "move") -> list[PrimitiveOp]:
    """Deterministic horizontal-then-vertical public route.

    ``kind="move_back"`` marks the return leg of a probe. The environment keeps
    the probe feedback as the last public outcome across those steps, so the
    agent can still see what the test reported after travelling home.
    """

    x, y = source
    tx, ty = target
    ops: list[PrimitiveOp] = []
    while x != tx:
        x += 1 if tx > x else -1
        ops.append(PrimitiveOp(kind, (x, y)))
    while y != ty:
        y += 1 if ty > y else -1
        ops.append(PrimitiveOp(kind, (x, y)))
    return ops


class MacroExecutor:
    """Expand public macro ids into primitive operations."""

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    def plan(self, macro_id: str, position: Coord) -> MacroPlan:
        """Return the primitive plan for *macro_id* from public *position*."""

        geometry = self.geometry
        if macro_id not in MACRO_ACTIONS_FROZEN:
            macro_id = "WAIT"

        if macro_id.startswith("COMMIT_"):
            color = macro_id.removeprefix("COMMIT_").lower()
            ops = [*_route(position, geometry.gate), PrimitiveOp("commit", color=color)]
        elif macro_id.startswith("PROBE_"):
            color = macro_id.removeprefix("PROBE_").lower()
            ops = [
                *_route(position, geometry.station),
                PrimitiveOp("test", color=color),
                *_route(geometry.station, geometry.start, kind="move_back"),
            ]
        elif macro_id == "GO_TO_STATION":
            ops = _route(position, geometry.station) or [PrimitiveOp("wait")]
        else:  # WAIT
            ops = [PrimitiveOp("wait")]

        return MacroPlan(macro_id=macro_id, ops=tuple(ops))
