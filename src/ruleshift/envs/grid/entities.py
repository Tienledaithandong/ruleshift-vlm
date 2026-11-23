"""Simulator-side grid entities.

These wrap the public map with privileged bookkeeping (the map seed and the
door requirement). No hidden transition rules live here -- the public geometry
is the complete agent-visible payload.
"""

from __future__ import annotations

from dataclasses import dataclass

from ruleshift.public_api.grid_rules import PublicGridMap


@dataclass(frozen=True)
class GridMap:
    """A public grid map plus simulator-only generation metadata."""

    public: PublicGridMap
    map_seed: int
    door_required: bool = False


@dataclass(frozen=True)
class MapGenResult:
    """A generated map and the number of candidate attempts it took."""

    map: GridMap
    attempts: int
