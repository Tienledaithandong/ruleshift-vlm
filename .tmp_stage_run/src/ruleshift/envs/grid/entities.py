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


@dataclass(frozen=True)
class GridScenario:
    """One episode specification.

    ``map_seed`` drives map generation while ``run_seed`` drives noise draws;
    keeping them separate means changing the noise seed never changes the map.
    The trailing default keeps the original six-argument constructions valid.
    """

    scenario_id: str
    scenario_type: str
    initial_rule: str
    shift_rule: str | None
    shift_time: int | None
    map_seed: int
    run_seed: int = 0
