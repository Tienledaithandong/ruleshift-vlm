"""Deterministic grid map generation with an all-hypothesis solvability gate.

A candidate is kept only when the goal of every public rule hypothesis is
reachable while that rule's hazard marker is treated as blocked. This lets the
agent, in principle, win under any rule in the public menu -- an invariant the
generator enforces rather than hoping for.
"""

from __future__ import annotations

import random
from collections import deque

from ruleshift.public_api.grid_rules import (
    GRID_HYPOTHESES,
    Coord,
    GridRule,
    PublicGridMap,
    key_opens,
)

from .config import GridConfig
from .entities import GridMap, MapGenResult

_DELTAS: tuple[Coord, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))
_MARKER_COLORS: tuple[str, ...] = ("red", "blue", "green")
_SPECIAL_CELL_COUNT = 7  # start + three markers + key + door + signpost
_MIN_SEPARATION = 2  # Manhattan distance between any two special cells


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reachable(public_map: PublicGridMap, start: Coord, blocked: set[Coord]) -> set[Coord]:
    """Return the cells reachable from ``start`` avoiding ``blocked`` and bounds."""

    if start in blocked:
        return set()
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _DELTAS:
            nxt = (x + dx, y + dy)
            nx, ny = nxt
            if nxt in seen or nxt in blocked:
                continue
            if not (0 <= nx < public_map.width and 0 <= ny < public_map.height):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def is_solvable(
    public_map: PublicGridMap, rule: GridRule, door_required: bool = False
) -> bool:
    """Whether ``rule``'s goal is winnable under the simulator's dynamics.

    Walls, out-of-bounds cells and ``rule``'s hazard marker are blocked; the
    other markers stay walkable (they may be hazards under a different rule).

    The simulator blocks a closed door for *every* movement and opens it only
    via ``INTERACT`` from an adjacent cell while carrying the matching key.
    It does this even when ``door_required`` is false, so whenever the map has
    a door the gate enforces the same three conditions unconditionally:

    1. the key is reachable while the door is still closed;
    2. a cell adjacent to the door is reachable while it is still closed (so
       the agent can stand there and open it);
    3. the goal is reachable once the door is open.

    ``door_required`` is kept for callers that pass it explicitly; the door
    logic above applies whenever ``public_map.door`` exists, regardless of the
    flag, so the gate can never accept a map the simulator makes unwinnable.
    """

    blocked: set[Coord] = set(public_map.walls)
    hazard = public_map.markers.get(rule.hazard_color)
    if hazard is not None:
        blocked.add(hazard)
    goal = public_map.markers[rule.goal_color]

    if public_map.door is not None:
        if not key_opens(rule, rule.key_color, rule.door_color):
            return False
        door_closed = blocked | {public_map.door}
        reachable_closed = _reachable(public_map, public_map.start, door_closed)
        if public_map.key not in reachable_closed:
            return False
        door_x, door_y = public_map.door
        if not any(
            0 <= door_x + dx < public_map.width
            and 0 <= door_y + dy < public_map.height
            and (door_x + dx, door_y + dy) in reachable_closed
            for dx, dy in _DELTAS
        ):
            return False
        # With the key held and the door opened, the goal must be reachable.
        return goal in _reachable(public_map, public_map.start, blocked)

    return goal in _reachable(public_map, public_map.start, blocked)


def all_hypotheses_solvable(
    public_map: PublicGridMap,
    rules: tuple[GridRule, ...] = GRID_HYPOTHESES,
    door_required: bool = False,
) -> bool:
    """Whether the map is solvable under every rule in ``rules``."""

    return all(is_solvable(public_map, rule, door_required) for rule in rules)


def _pick_special_cells(rng: random.Random, open_cells: list[Coord]) -> list[Coord] | None:
    """Greedily choose pairwise-separated cells, or ``None`` if impossible."""

    shuffled = list(open_cells)
    rng.shuffle(shuffled)
    chosen: list[Coord] = []
    for cell in shuffled:
        if all(_manhattan(cell, other) >= _MIN_SEPARATION for other in chosen):
            chosen.append(cell)
            if len(chosen) == _SPECIAL_CELL_COUNT:
                return chosen
    return None


def _generate_candidate(rng: random.Random, config: GridConfig) -> PublicGridMap | None:
    """Draw one candidate public map; ``None`` if special cells cannot be placed."""

    walls: set[Coord] = set()
    for x in range(config.width):
        walls.add((x, 0))
        walls.add((x, config.height - 1))
    for y in range(config.height):
        walls.add((0, y))
        walls.add((config.width - 1, y))

    interior = [
        (x, y)
        for x in range(1, config.width - 1)
        for y in range(1, config.height - 1)
    ]
    for cell in interior:
        if rng.random() < config.wall_density:
            walls.add(cell)

    open_cells = [cell for cell in interior if cell not in walls]
    chosen = _pick_special_cells(rng, open_cells)
    if chosen is None:
        return None

    start = chosen[0]
    markers = dict(zip(_MARKER_COLORS, chosen[1:4]))
    return PublicGridMap(
        width=config.width,
        height=config.height,
        walls=frozenset(walls),
        start=start,
        markers=markers,
        key=chosen[4],
        door=chosen[5],
        signpost=chosen[6],
    )


def generate_solvable_map(
    seed: int,
    config: GridConfig,
    rules: tuple[GridRule, ...] = GRID_HYPOTHESES,
    max_attempts: int = 200,
) -> MapGenResult:
    """Generate a map solvable under every rule, deterministically per seed.

    Each attempt uses its own ``random.Random(seed + attempt)`` so a given seed
    always yields the same map and the same attempt count. A ``RuntimeError`` is
    raised if no candidate passes the gate within ``max_attempts``.
    """

    for attempt in range(max_attempts):
        public_map = _generate_candidate(random.Random(seed + attempt), config)
        if public_map is None:
            continue
        if not all_hypotheses_solvable(public_map, rules, config.door_required):
            continue
        return MapGenResult(
            map=GridMap(
                public=public_map,
                map_seed=seed,
                door_required=config.door_required,
            ),
            attempts=attempt + 1,
        )
    raise RuntimeError(f"no solvable grid map found after {max_attempts} attempts")
