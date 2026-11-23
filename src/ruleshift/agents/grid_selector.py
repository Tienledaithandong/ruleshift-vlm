"""Public information-gain selector and advisor for RuleShift-Grid.

Agent-side only. This module imports the public rule, schema and planner
helpers, never the simulator or the privileged evaluator. It scores candidate
actions by expected information gain *per unit cost*, using a copy of the
public change-point belief, and falls back to the public reference planner when
no informative action is available.

Costs are public: a physical move (and INTERACT) spends one environment step;
INSPECT spends ``inspect_cost`` tool units and READ_SIGN spends
``read_sign_cost``. The config object is consumed structurally (via attribute
access) so this module keeps no import dependency on the scenario package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ruleshift.agents.grid_belief import GridBelief
from ruleshift.public_api.grid_planner import plan_to_goal
from ruleshift.public_api.grid_rules import (
    Coord,
    PublicGridMap,
    RULE_BY_ID,
    cell_role,
    movement_outcomes,
)
from ruleshift.public_api.grid_schemas import GridAction

__all__ = ["GridSelectorConfig", "InfoGainSelector", "PublicAdvisor"]

_MOVE_KINDS: frozenset[str] = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})
_ROLES: tuple[str, ...] = ("goal", "hazard", "safe")


class GridSelectorConfig(Protocol):
    """The public config fields the selector reads (structural typing)."""

    inspect_cost: int
    read_sign_cost: int
    inspect_range: int
    inspect_error: float


@dataclass(frozen=True)
class _Candidate:
    """One legal decision together with its public cost and mask position."""

    action: GridAction
    cost: int
    order: int
    color: str | None = None
    side: int | None = None


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _adjacent_side(position: Coord, signpost: Coord | None) -> int | None:
    """Which face of the signpost an adjacent agent stands at, else ``None``."""

    if signpost is None:
        return None
    dx, dy = position[0] - signpost[0], position[1] - signpost[1]
    if (dx, dy) == (0, -1):
        return 2
    if (dx, dy) == (1, 0):
        return 1
    if (dx, dy) in ((0, 1), (-1, 0)):
        return 0
    return None


class InfoGainSelector:
    """Choose the legal action with the best expected IG per unit cost."""

    def __init__(self, config: GridSelectorConfig) -> None:
        self.config = config

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default)

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _inspectable(
        self, public_map: PublicGridMap, position: Coord, target: Coord
    ) -> bool:
        coord = tuple(target)
        if coord in public_map.walls:
            return False
        x, y = coord
        if not (0 <= x < public_map.width and 0 <= y < public_map.height):
            return False
        return _manhattan(tuple(position), coord) <= int(
            self._cfg("inspect_range", 2)
        )

    def _candidates(
        self,
        action_mask: tuple[str, ...],
        public_map: PublicGridMap,
        position: Coord,
        tool_budget: int,
    ) -> list[_Candidate]:
        inspect_cost = int(self._cfg("inspect_cost", 2))
        sign_cost = int(self._cfg("read_sign_cost", 1))
        result: list[_Candidate] = []

        for order, kind in enumerate(action_mask):
            if kind in _MOVE_KINDS or kind == "INTERACT":
                result.append(_Candidate(GridAction(kind), 1, order))
            elif kind == "INSPECT":
                if tool_budget < inspect_cost:
                    continue
                for color, coord in sorted(public_map.markers.items()):
                    if self._inspectable(public_map, position, coord):
                        result.append(
                            _Candidate(
                                GridAction("INSPECT", tuple(coord)),
                                inspect_cost,
                                order,
                                color=color,
                            )
                        )
                # The key and door are inspectable but rule-uninformative in v1;
                # they are kept so the candidate set mirrors the public tools.
                for coord, tag in ((public_map.key, "key"), (public_map.door, "door")):
                    if coord is not None and self._inspectable(public_map, position, coord):
                        result.append(
                            _Candidate(
                                GridAction("INSPECT", tuple(coord)),
                                inspect_cost,
                                order,
                                color=tag,
                            )
                        )
            elif kind == "READ_SIGN":
                if tool_budget < sign_cost:
                    continue
                side = _adjacent_side(tuple(position), public_map.signpost)
                if side is not None:
                    result.append(
                        _Candidate(GridAction("READ_SIGN"), sign_cost, order, side=side)
                    )
        return result

    # ------------------------------------------------------------------
    # Expected information gain
    # ------------------------------------------------------------------

    def _move_hazard_probability(
        self,
        belief: GridBelief,
        public_map: PublicGridMap,
        position: Coord,
        kind: str,
    ) -> float:
        color_at = {tuple(coord): color for color, coord in public_map.markers.items()}
        probability = 0.0
        for rule_id, rule_mass in belief.marginal().items():
            if rule_mass <= 0.0:
                continue
            rule = RULE_BY_ID[rule_id]
            for landing, mass in movement_outcomes(rule, position, kind, public_map).items():
                color = color_at.get(tuple(landing))
                if color is not None and cell_role(rule, color) == "hazard":
                    probability += rule_mass * mass
        return probability

    def _interact_outcome(
        self,
        public_map: PublicGridMap,
        position: Coord,
        inventory: tuple[str, ...],
        door_open: bool,
    ) -> str:
        key = public_map.key
        if key is not None and tuple(position) == tuple(key) and not inventory:
            return "key_picked"
        door = public_map.door
        if (
            door is not None
            and not door_open
            and _manhattan(tuple(position), tuple(door)) == 1
            and inventory
        ):
            return "door_opened"
        return "nothing"

    def _expected_ig(
        self,
        belief: GridBelief,
        public_map: PublicGridMap,
        position: Coord,
        candidate: _Candidate,
        step: int,
        inventory: tuple[str, ...],
        door_open: bool,
    ) -> float:
        """Compute ``H(before) - E[H(after)]`` for one candidate action."""

        kind = candidate.action.kind
        marginal = belief.marginal()
        before = belief.entropy()
        outcomes: dict[Any, float] = {}

        if kind in _MOVE_KINDS:
            for rule_id, rule_mass in marginal.items():
                if rule_mass <= 0.0:
                    continue
                for landing, mass in movement_outcomes(
                    RULE_BY_ID[rule_id], position, kind, public_map
                ).items():
                    outcomes[tuple(landing)] = (
                        outcomes.get(tuple(landing), 0.0) + rule_mass * mass
                    )

            def apply(branch: GridBelief, outcome: Any) -> None:
                branch.update_move(position, kind, outcome, public_map, step)

        elif kind == "INSPECT":
            color = candidate.color
            # Key/door cells are truthful under every public rule in v1.
            if color is None or color in {"key", "door"}:
                return 0.0
            error = float(self._cfg("inspect_error", 0.05))
            for rule_id, rule_mass in marginal.items():
                if rule_mass <= 0.0:
                    continue
                true_role = cell_role(RULE_BY_ID[rule_id], color)
                outcomes[true_role] = outcomes.get(true_role, 0.0) + rule_mass * (
                    1.0 - error
                )
                for other in _ROLES:
                    if other != true_role:
                        outcomes[other] = outcomes.get(other, 0.0) + rule_mass * (
                            error / 2.0
                        )

            def apply(branch: GridBelief, outcome: Any) -> None:
                branch.update_inspect(color, outcome, step)

        elif kind == "READ_SIGN":
            side = candidate.side
            if side is None:
                return 0.0
            for rule_id, rule_mass in marginal.items():
                if rule_mass <= 0.0:
                    continue
                clause = RULE_BY_ID[rule_id].sign_clauses[side]
                outcomes[clause] = outcomes.get(clause, 0.0) + rule_mass

            def apply(branch: GridBelief, outcome: Any) -> None:
                branch.update_sign(side, outcome, step)

        elif kind == "INTERACT":
            outcomes[self._interact_outcome(public_map, position, inventory, door_open)] = 1.0

            def apply(branch: GridBelief, outcome: Any) -> None:
                branch.update_interact(outcome, step)

        else:
            return 0.0

        probe = belief.copy()
        probe.inspect_error = float(self._cfg("inspect_error", probe.inspect_error))
        expected_after = 0.0
        for outcome, probability in outcomes.items():
            if probability <= 0.0:
                continue
            branch = probe.copy()
            apply(branch, outcome)
            expected_after += probability * branch.entropy()
        return max(0.0, before - expected_after)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def choose(
        self,
        belief: GridBelief,
        public_map: PublicGridMap,
        *,
        position: Coord,
        inventory: tuple[str, ...] = (),
        door_open: bool = False,
        tool_budget: int,
        action_mask: tuple[str, ...],
        step: int,
        safety_threshold: float | None = None,
    ) -> GridAction:
        """Return one legal decision with maximal expected IG per unit cost."""

        threshold = 1.0 if safety_threshold is None else float(safety_threshold)
        candidates = self._candidates(action_mask, public_map, position, tool_budget)

        best_key: tuple[float, float, int] | None = None
        best: _Candidate | None = None
        for candidate in candidates:
            kind = candidate.action.kind
            if kind in _MOVE_KINDS and self._move_hazard_probability(
                belief, public_map, position, kind
            ) > threshold:
                continue
            information_gain = self._expected_ig(
                belief, public_map, position, candidate, step, inventory, door_open
            )
            if information_gain <= 0.0:
                continue
            cost = max(1, candidate.cost)
            # Highest IG/cost, then lower raw cost, then earlier mask entry.
            key = (information_gain / cost, -float(cost), -candidate.order)
            if best_key is None or key > best_key:
                best_key, best = key, candidate

        if best is not None:
            return best.action

        # No positive-IG candidate: fall back to the public reference planner.
        rule = RULE_BY_ID.get(belief.most_likely())
        if rule is not None:
            plan = plan_to_goal(
                rule,
                public_map,
                tuple(position),
                has_key=bool(inventory),
                door_open=door_open,
            )
            if plan:
                return plan[0]
        for kind in action_mask:
            if kind in _MOVE_KINDS:
                return GridAction(kind)
        return GridAction("INTERACT")


class PublicAdvisor:
    """A public-only advisor that plans under the most likely hypothesis.

    It never updates the belief, never reads the true rule, and returns exactly
    one public action.
    """

    def __init__(self, config: GridSelectorConfig) -> None:
        self.config = config

    def advise(
        self,
        belief: GridBelief,
        public_map: PublicGridMap,
        *,
        position: Coord,
        inventory: tuple[str, ...] = (),
        door_open: bool = False,
    ) -> GridAction:
        """Return the first planned action toward the believed goal marker."""

        rule = RULE_BY_ID.get(belief.most_likely())
        if rule is None:
            return GridAction("INTERACT")

        plan = plan_to_goal(
            rule,
            public_map,
            tuple(position),
            has_key=bool(inventory),
            door_open=door_open,
        )
        if plan:
            return plan[0]

        # When no motion plan exists, an interaction at the key or the door is
        # the only useful public action left.
        key = public_map.key
        door = public_map.door
        at_key = key is not None and tuple(position) == tuple(key)
        at_door = door is not None and _manhattan(tuple(position), tuple(door)) == 1
        if at_key or at_door:
            return GridAction("INTERACT")
        return GridAction("UP")
