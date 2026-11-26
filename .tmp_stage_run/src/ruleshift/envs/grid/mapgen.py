"""Deterministic grid map generation with an all-hypothesis solvability gate.

A candidate is kept only when the goal of every public rule hypothesis is
reachable while that rule's hazard marker and the signpost are treated as
blocked. This lets the agent, in principle, win under any rule in the public
menu -- an invariant the generator enforces rather than hoping for.
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
    sign_clause,
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


def _sign_face_cells(
    public_map: PublicGridMap, signpost: Coord, side: int
) -> list[Coord]:
    """Free cells from which the public ``sign_side`` mapping reads ``side``."""

    if side == 0:
        candidates = [
            (signpost[0], signpost[1] + 1),
            (signpost[0] - 1, signpost[1]),
        ]
    elif side == 1:
        candidates = [(signpost[0] + 1, signpost[1])]
    elif side == 2:
        candidates = [(signpost[0], signpost[1] - 1)]
    else:
        return []
    return [
        cell
        for cell in candidates
        if 0 <= cell[0] < public_map.width
        and 0 <= cell[1] < public_map.height
        and cell not in public_map.walls
    ]


def informative_sign_sides(
    rules: tuple[GridRule, ...] = GRID_HYPOTHESES,
) -> tuple[int, ...]:
    """Sign faces whose clause separates at least two public hypotheses."""

    return tuple(
        side
        for side in (0, 1, 2)
        if len({sign_clause(rule, side) for rule in rules}) > 1
    )


def _readable_face_cells(public_map: PublicGridMap, side: int) -> set[Coord]:
    """Free face cells of ``side`` reachable without crossing a marker or door."""

    signpost = public_map.signpost
    if signpost is None:
        return set()
    blocked = set(public_map.walls) | set(public_map.markers.values()) | {signpost}
    if public_map.door is not None:
        blocked.add(public_map.door)
    reachable = _reachable(public_map, public_map.start, blocked)
    return {
        cell
        for cell in _sign_face_cells(public_map, signpost, side)
        if cell in reachable
    }


def has_readable_informative_face(
    public_map: PublicGridMap,
    rules: tuple[GridRule, ...] = GRID_HYPOTHESES,
) -> bool:
    """Whether a free informative sign face is reachable without crossing a marker.

    This is the broad audit predicate: side 1 is informative about movement
    remaps even though it does not separate ``base`` from the hazard/sign
    hypotheses. The generator uses :func:`has_readable_base_diagnostic_face`
    because identification of ``base`` specifically requires side 0.
    """

    signpost = public_map.signpost
    if signpost is None:
        return False
    return any(
        _readable_face_cells(public_map, side)
        for side in informative_sign_sides(rules)
    )


def has_readable_base_diagnostic_face(
    public_map: PublicGridMap,
    rules: tuple[GridRule, ...] = GRID_HYPOTHESES,
) -> bool:
    """Whether side 0 is readable without crossing a marker or the closed door.

    Side 0 separates ``base`` ("red is not the hazard") from
    ``hazard_goal_swap`` and ``sign_composition`` ("red is not the goal"). Side
    1 says "movement may slip" under all three of those hypotheses, so a map
    with only side 1 cannot support the base-identification gate. Such a map
    must not enter the benchmark even though side 1 is technically
    informative about the remap hypotheses.
    """

    if 0 not in informative_sign_sides(rules):
        return False
    return bool(_readable_face_cells(public_map, 0))


def is_solvable(
    public_map: PublicGridMap, rule: GridRule, door_required: bool = False
) -> bool:
    """Whether ``rule``'s goal is winnable under the simulator's dynamics.

    Walls, out-of-bounds cells, ``rule``'s hazard marker and the signpost are
    blocked; the other markers stay walkable (they may be hazards under a
    different rule). The simulator blocks the signpost for every movement, so
    the gate must block it too or it can accept a map whose only route runs
    through the signpost.

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
    if public_map.signpost is not None:
        blocked.add(public_map.signpost)
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

    Retry streams are keyed by both ``seed`` and ``attempt`` with a disjoint
    integer encoding, so adjacent seeds cannot share an attempt stream. A
    candidate must also expose a reachable side-0 sign face; otherwise it
    cannot separate ``base`` from the hazard/sign hypotheses and cannot support
    the active rule-identification protocol. A ``RuntimeError`` is raised if no
    candidate passes both gates within ``max_attempts``.
    """

    for attempt in range(max_attempts):
        # Unique for all realistic attempt counts; unlike seed + attempt, this
        # cannot alias seed=1/attempt=1 with seed=2/attempt=0.
        retry_seed = seed * 1_000_003 + attempt
        public_map = _generate_candidate(random.Random(retry_seed), config)
        if public_map is None:
            continue
        if not all_hypotheses_solvable(public_map, rules, config.door_required):
            continue
        if not has_readable_base_diagnostic_face(public_map, rules):
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
