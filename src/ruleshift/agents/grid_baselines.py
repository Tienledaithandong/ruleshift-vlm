"""Public-only baseline policies for RuleShift-Grid.

This module is purely agent-side: it imports only the standard library, the
public API and other agent modules. It never imports the simulator or the
privileged evaluator, it never reads the true rule, and every decision is made
from the public observation and the already-executed outcome passed to
``record``.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, Protocol

from ruleshift.agents.grid_belief import GridBelief
from ruleshift.agents.grid_selector import InfoGainSelector, PublicAdvisor
from ruleshift.public_api.grid_planner import (
    GridPlannerState,
    HypothesisDynamics,
    astar,
    plan_to_goal,
)
from ruleshift.public_api.grid_rules import (
    BASE_RULE,
    RULE_BY_ID,
    PublicGridMap,
    cell_role,
    sign_clause,
    sign_side,
)
from ruleshift.public_api.grid_schemas import (
    GRID_PHYSICAL_ACTIONS,
    GRID_TOOL_ACTIONS,
    GridAction,
)

__all__ = [
    "DiagnosticBeliefPolicy",
    "DiagnosticBeliefPolicyV4",
    "DiagnosticBeliefPolicyV5",
    "GridPolicy",
    "InfoGainPolicy",
    "NoAdaptationPolicy",
    "PublicAdvisorPolicy",
    "RandomProbePolicy",
    "as_observation",
]

_MOVE_KINDS: frozenset[str] = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})


# ---------------------------------------------------------------------------
# Observation adapter
# ---------------------------------------------------------------------------


def _map_from_payload(payload: dict[str, Any]) -> PublicGridMap:
    """Rebuild the public geometry from a serialized observation payload."""

    def cell(value: Any) -> tuple[int, int] | None:
        return tuple(value) if value is not None else None

    return PublicGridMap(
        width=int(payload["width"]),
        height=int(payload["height"]),
        walls=frozenset(tuple(wall) for wall in payload["walls"]),
        start=tuple(payload["start"]),
        markers={color: tuple(coord) for color, coord in payload["markers"].items()},
        key=cell(payload["key"]),
        door=cell(payload["door"]),
        signpost=cell(payload["signpost"]),
    )


class _ObservationView:
    """Attribute-accessible view over a public observation payload.

    The environment returns the canonical public payload dict. Policies consume
    the same fields by attribute, so this thin adapter lets both shapes work
    without adding or hiding any state. ``to_payload`` returns the original
    public payload so it can be logged verbatim.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.public_map = _map_from_payload(payload["public_map"])
        self.agent_position = tuple(payload["agent_position"])
        self.inventory = tuple(payload.get("inventory", ()))
        self.door_open = bool(payload["door_open"])
        self.steps = int(payload["steps"])
        self.action_mask = tuple(payload.get("action_mask", ()))
        self.tool_budget_remaining = int(payload["tool_budget_remaining"])
        self.image_hash = str(payload.get("image_hash", ""))
        # Public pixels are allowed as agent input, but must never be copied
        # into a logged public decision row (only their digest is).
        self.image = payload.get("image")

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self._payload.items() if key != "image"}


def as_observation(observation: Any) -> Any:
    """Return an attribute-accessible view of a public grid observation."""

    if hasattr(observation, "public_map") and hasattr(observation, "to_payload"):
        return observation
    return _ObservationView(observation)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class GridPolicy(Protocol):
    """The decision contract every grid policy implements."""

    def reset(self, observation: Any, info: dict[str, Any]) -> None: ...

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction: ...

    def record(
        self,
        action: GridAction,
        observation: Any,
        reward: float,
        info: dict[str, Any],
    ) -> None: ...

    def most_likely_rule(self) -> str: ...

    def shift_detected(self) -> bool: ...


# ---------------------------------------------------------------------------
# Shared bookkeeping
# ---------------------------------------------------------------------------


class _BasePolicy:
    """Shared observation, planning and belief bookkeeping for baselines.

    ``use_belief=False`` is supported for scripted policies that plan from a
    fixed rule menu: those still get ``previous_observation`` and ``base_plan``
    but carry no change-point filter.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        use_belief: bool = True,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.use_belief = bool(use_belief)
        self.rng = random.Random(seed)
        self.previous_observation: Any = None
        self.previous_action: GridAction | None = None
        self.previous_info: dict[str, Any] = {}
        self.advisor_calls = 0
        self.belief: GridBelief | None = self._make_belief() if use_belief else None

    def _make_belief(self) -> GridBelief:
        cfg = self.config
        # GridBelief's p_shift is a *per-step* hazard. GridConfig's
        # shift_probability is the episode-level chance of a shift scenario,
        # so it is deliberately not reused here.
        return GridBelief(
            shift_window=tuple(getattr(cfg, "shift_window", (0, 60))),
            inspect_error=float(getattr(cfg, "inspect_error", 0.05)),
        )

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        observation = as_observation(observation)
        self.previous_observation = observation
        self.previous_action = None
        self.previous_info = dict(info or {})
        self.advisor_calls = 0
        if self.use_belief:
            self.belief = self._make_belief()

    # -- public helpers for scripted policies -------------------------------

    def base_plan(self, rule_id: str, observation: Any) -> list[GridAction]:
        """Plan toward ``rule_id``'s goal using only the public observation."""

        observation = as_observation(observation)
        return plan_to_goal(
            RULE_BY_ID[rule_id],
            observation.public_map,
            tuple(observation.agent_position),
            has_key=bool(observation.inventory),
            door_open=bool(observation.door_open),
        )

    @staticmethod
    def _first_plan_action(plan: list[GridAction], observation: Any) -> GridAction:
        """First planned action, or the first legal physical action."""

        if plan:
            return plan[0]
        mask = tuple(as_observation(observation).action_mask)
        for kind in GRID_PHYSICAL_ACTIONS:
            if kind in mask:
                return GridAction(kind)
        return GridAction("INTERACT")

    @staticmethod
    def _inspect_color(public_map: PublicGridMap, target: Any) -> str:
        """Map an INSPECT target to the public color whose role it reveals."""

        if target is None:
            return ""
        coord = tuple(target)
        for color, marker in public_map.markers.items():
            if tuple(marker) == coord:
                return color
        if public_map.key is not None and tuple(public_map.key) == coord:
            return BASE_RULE.key_color
        if public_map.door is not None and tuple(public_map.door) == coord:
            return BASE_RULE.door_color
        return ""

    # -- belief conditioning ------------------------------------------------

    def _update_belief(
        self,
        action: GridAction,
        observation: Any,
        info: dict[str, Any],
    ) -> None:
        belief = self.belief
        if belief is None:
            return
        observation = as_observation(observation)
        step = int(info.get("env_steps", 0))
        if self.previous_observation is None:
            previous = observation
        else:
            previous = as_observation(self.previous_observation)
        previous_position = tuple(previous.agent_position)
        public_map = previous.public_map
        kind = action.kind

        if kind == "INSPECT":
            value = info.get("last_tool_value")
            if value is not None:
                color = self._inspect_color(public_map, action.target)
                if color:
                    belief.update_inspect(color, str(value), step)
        elif kind == "READ_SIGN":
            value = info.get("last_tool_value")
            signpost = public_map.signpost
            if value is not None and signpost is not None:
                try:
                    side = sign_side(previous_position, signpost)
                except ValueError:
                    side = None
                if side is not None:
                    belief.update_sign(side, str(value), step)
        elif kind in _MOVE_KINDS:
            # The hazard is per physical step; tools at the same step do not
            # advance ``env_steps`` and must not re-apply it.
            previous_steps = int(self.previous_info.get("env_steps", 0))
            if step > previous_steps:
                belief.update_move(
                    previous_position,
                    kind,
                    tuple(observation.agent_position),
                    public_map,
                    step,
                    door_open=bool(getattr(previous, "door_open", False)),
                )
        elif kind == "INTERACT":
            outcome = info.get("last_outcome")
            if outcome is not None:
                belief.update_interact(str(outcome), step)

    def record(
        self,
        action: GridAction,
        observation: Any,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        # Update before the previous observation is overwritten.
        if self.use_belief:
            self._update_belief(action, observation, info)
        self.previous_observation = as_observation(observation)
        self.previous_action = action
        self.previous_info = dict(info or {})

    # -- queries ------------------------------------------------------------

    def most_likely_rule(self) -> str:
        if self.belief is None:
            return "base"
        return self.belief.most_likely()

    def shift_detected(self) -> bool:
        return self.belief is not None and self.belief.shift_probability() > 0.5


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class NoAdaptationPolicy(_BasePolicy):
    """Replan toward the base goal every decision; never adapt."""

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config, use_belief=False)

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        observation = as_observation(observation)
        return self._first_plan_action(self.base_plan("base", observation), observation)

    def most_likely_rule(self) -> str:
        return "base"

    def shift_detected(self) -> bool:
        return False


class PublicAdvisorPolicy(_BasePolicy):
    """Follows the base plan until a public discrepancy triggers the advisor."""

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)
        self.advisor = PublicAdvisor(config)

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        observation = as_observation(observation)
        if self.belief is not None and self.belief.shift_probability() > 0.5:
            # advise() is read-only: it never mutates this policy's belief.
            action = self.advisor.advise(
                self.belief,
                observation.public_map,
                position=tuple(observation.agent_position),
                inventory=tuple(observation.inventory),
                door_open=bool(observation.door_open),
            )
            self.advisor_calls += 1
            return action
        return self._first_plan_action(self.base_plan("base", observation), observation)

    def most_likely_rule(self) -> str:
        return "base"

    def shift_detected(self) -> bool:
        return self.advisor_calls > 0


class RandomProbePolicy(_BasePolicy):
    """On a public discrepancy, choose a random affordable, legal tool."""

    def __init__(self, config: Any | None = None, seed: int = 0) -> None:
        super().__init__(config, seed=seed)

    @staticmethod
    def _inspect_targets(observation: Any, reach: int) -> list[tuple[int, int]]:
        public_map = observation.public_map
        position = tuple(observation.agent_position)
        coords = [tuple(coord) for coord in public_map.markers.values()]
        coords.extend(
            tuple(coord) for coord in (public_map.key, public_map.door)
            if coord is not None
        )
        return [
            coord
            for coord in coords
            if coord not in public_map.walls
            and abs(coord[0] - position[0]) + abs(coord[1] - position[1]) <= reach
        ]

    def _random_probe(self, observation: Any) -> GridAction | None:
        cfg = self.config
        budget = int(observation.tool_budget_remaining)
        inspect_cost = int(getattr(cfg, "inspect_cost", 2))
        sign_cost = int(getattr(cfg, "read_sign_cost", 1))
        tools: list[str] = []
        if "INSPECT" in observation.action_mask and budget >= inspect_cost:
            tools.append("INSPECT")
        if "READ_SIGN" in observation.action_mask and budget >= sign_cost:
            tools.append("READ_SIGN")
        if not tools:
            return None
        kind = self.rng.choice(tools)
        if kind == "READ_SIGN":
            return GridAction("READ_SIGN")
        reach = int(getattr(cfg, "inspect_range", 2))
        targets = self._inspect_targets(observation, reach)
        if targets:
            return GridAction("INSPECT", self.rng.choice(targets))
        if "READ_SIGN" in tools:
            return GridAction("READ_SIGN")
        return None

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        observation = as_observation(observation)
        if self.belief is not None and self.belief.shift_probability() > 0.5:
            probe = self._random_probe(observation)
            if probe is not None:
                return probe
        return self._first_plan_action(self.base_plan("base", observation), observation)


class InfoGainPolicy(_BasePolicy):
    """Chooses with the public information-gain selector."""

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)
        self.selector = InfoGainSelector(config)
        self.information_gain_bits: float | None = None
        self.tool_information_gain_bits: float | None = None
        self.move_information_gain_bits: float | None = None

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        super().reset(observation, info)
        self.information_gain_bits = None
        self.tool_information_gain_bits = None
        self.move_information_gain_bits = None

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        observation = as_observation(observation)
        assert self.belief is not None
        action = self.selector.choose(
            self.belief,
            observation.public_map,
            position=tuple(observation.agent_position),
            inventory=tuple(observation.inventory),
            door_open=bool(observation.door_open),
            tool_budget=int(observation.tool_budget_remaining),
            action_mask=tuple(observation.action_mask),
            step=int(info.get("env_steps", observation.steps)),
        )
        # Accumulate the selector's expected IG for this decision, keeping tool
        # and move information separate so the evaluator never divides move IG
        # by the tool-call count.
        chosen_ig = self.selector.last_information_gain_bits
        if isinstance(chosen_ig, (int, float)) and not isinstance(chosen_ig, bool):
            gain = float(chosen_ig)
            self.information_gain_bits = float(self.information_gain_bits or 0.0) + gain
            if action.kind in GRID_TOOL_ACTIONS:
                self.tool_information_gain_bits = (
                    float(self.tool_information_gain_bits or 0.0) + gain
                )
            else:
                self.move_information_gain_bits = (
                    float(self.move_information_gain_bits or 0.0) + gain
                )
        return action


class DiagnosticBeliefPolicy(_BasePolicy):
    """Run a fixed two-tool diagnostic, then commit to the MAP goal.

    ``base``, ``hazard_goal_swap`` and ``sign_composition`` have the same
    movement distribution, so movement alone cannot separate them.
    ``hazard_goal_swap`` and ``sign_composition`` also share their sign
    clauses. The protocol is therefore:

    1. ``READ_SIGN`` on an informative face (side 0 preferred, side 1 when
       side 0 is walled off). ``base`` reports that red is not the hazard;
       the other two report that red is not the goal.
    2. When red is not the goal, ``INSPECT`` blue. It is the goal under
       ``hazard_goal_swap`` and safe under ``sign_composition``.

    Tool navigation treats every marker as blocked: any marker can be the
    goal under some hypothesis, so crossing one can end the episode before
    the evidence is collected. The two tools cost 3 units in total, exactly
    the public budget. The policy stops probing as soon as the posterior has
    a decisive gap, so movement-identifiable rules are committed without
    spending tools.
    """

    _probe_leaders_only = False
    _reserve_blue_budget = False

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)
        self.sign_observed = False
        self.blue_observed = False
        self.tool_attempts = 0

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        super().reset(observation, info)
        self.sign_observed = False
        self.blue_observed = False
        self.tool_attempts = 0

    def record(
        self,
        action: GridAction,
        observation: Any,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        super().record(action, observation, reward, info)
        payload = info or {}
        if action.kind == "READ_SIGN" and payload.get("last_tool_value") is not None:
            self.sign_observed = True
        elif action.kind == "INSPECT" and payload.get("last_tool_value") is not None:
            self.blue_observed = True

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        observation = as_observation(observation)
        if self._commit_is_safe():
            return self._commit(observation)
        action = self._diagnostic_action(observation)
        if action is not None:
            return action
        return self._commit(observation)

    # -- posterior readout -------------------------------------------------

    def _marginal(self) -> dict[str, float]:
        if self.belief is None:
            return {}
        return self.belief.marginal()

    def _top_two(self) -> tuple[float, float]:
        values = sorted(self._marginal().values(), reverse=True)
        if not values:
            return 0.0, 0.0
        if len(values) == 1:
            return values[0], 0.0
        return values[0], values[1]

    def _three_mass(self) -> float:
        marginal = self._marginal()
        return sum(
            marginal.get(rule, 0.0)
            for rule in ("base", "hazard_goal_swap", "sign_composition")
        )

    def _commit_is_safe(self) -> bool:
        top, second = self._top_two()
        return top >= 0.90 and (top - second) >= 0.10

    # -- diagnostic stages -------------------------------------------------

    def _diagnostic_action(self, observation: Any) -> GridAction | None:
        if self.tool_attempts >= 4:
            return None
        marginal = self._marginal()
        budget = int(observation.tool_budget_remaining)
        sign_cost = int(getattr(self.config, "read_sign_cost", 1))
        inspect_cost = int(getattr(self.config, "inspect_cost", 2))

        if not self.sign_observed and budget >= sign_cost:
            # The sign is also the first navigation target: the walk there
            # supplies movement evidence that can rule out remaps/slippery.
            action = self._go_read_sign(observation)
            if action is not None:
                return action

        if (
            self.sign_observed
            and not self.blue_observed
            and budget >= inspect_cost
            and (
                marginal.get("hazard_goal_swap", 0.0)
                + marginal.get("sign_composition", 0.0)
            )
            >= 0.05
        ):
            action = self._go_inspect_blue(observation)
            if action is not None:
                return action
        return None

    def _go_read_sign(self, observation: Any) -> GridAction | None:
        signpost = observation.public_map.signpost
        if signpost is None:
            return None
        signpost = tuple(signpost)
        current = tuple(observation.agent_position)
        # Side 0 is the only face that separates base from hazard/sign; side 1
        # is a fallback only when side 0 is walled off, not merely nearer.
        for side in self._informative_sides():
            cells = self._sign_cells(observation.public_map, signpost, side)
            if current in cells:
                if (
                    "READ_SIGN" in observation.action_mask
                    and int(observation.tool_budget_remaining)
                    >= int(getattr(self.config, "read_sign_cost", 1))
                ):
                    self.tool_attempts += 1
                    return GridAction("READ_SIGN")
                return None
            plan = self._shortest_plan(observation, cells)
            if plan is not None:
                return self._first_legal(plan, observation)
        return None

    def _go_inspect_blue(self, observation: Any) -> GridAction | None:
        blue = observation.public_map.markers.get("blue")
        if blue is None:
            return None
        blue = tuple(blue)
        current = tuple(observation.agent_position)
        reach = max(0, int(getattr(self.config, "inspect_range", 2)))
        if self._manhattan(current, blue) <= reach:
            if (
                "INSPECT" in observation.action_mask
                and int(observation.tool_budget_remaining)
                >= int(getattr(self.config, "inspect_cost", 2))
            ):
                self.tool_attempts += 1
                return GridAction("INSPECT", blue)
            return None

        targets = self._inspect_cells(observation.public_map, blue, reach, current)
        plan = self._shortest_plan(observation, targets)
        if plan is None:
            return None
        return self._first_legal(plan, observation)

    # -- navigation --------------------------------------------------------

    def _navigation_rule(self) -> str:
        if self._three_mass() >= 0.50:
            return "base"
        return self.belief.most_likely() if self.belief is not None else "base"

    def _shortest_plan(
        self, observation: Any, targets: list[tuple[int, int]]
    ) -> list[GridAction] | None:
        current = tuple(observation.agent_position)
        candidates: list[tuple[int, int, list[GridAction]]] = []
        for target in targets:
            plan = self._plan_to(observation, target)
            if plan:
                candidates.append((len(plan), self._manhattan(current, target), plan))
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1], item[2][0].kind))[2]

    def _plan_to(self, observation: Any, target: tuple[int, int]) -> list[GridAction]:
        state = GridPlannerState(
            tuple(observation.agent_position),
            bool(observation.inventory),
            bool(observation.door_open),
        )
        # Any marker can be the goal under some candidate rule. When walking
        # to a tool, do not cross one: otherwise the episode can terminate as
        # a success (or take hazard damage) before the evidence is collected.
        planning_map = replace(
            observation.public_map,
            walls=observation.public_map.walls
            | frozenset(observation.public_map.markers.values()),
        )
        rule_id = self._navigation_rule()
        model = HypothesisDynamics(RULE_BY_ID[rule_id], planning_map)
        plan = astar(state, target, model)
        if not plan and rule_id != "base":
            model = HypothesisDynamics(BASE_RULE, planning_map)
            plan = astar(state, target, model)
        return plan

    def _first_legal(self, plan: list[GridAction], observation: Any) -> GridAction:
        if plan and plan[0].kind in observation.action_mask:
            return plan[0]
        return self._first_plan_action([], observation)

    def _commit(self, observation: Any) -> GridAction:
        rule_id = self.belief.most_likely() if self.belief is not None else "base"
        return self._first_plan_action(self.base_plan(rule_id, observation), observation)

    # -- geometry ----------------------------------------------------------

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _informative_sides() -> list[int]:
        """Sign faces whose clause separates at least two hypotheses."""

        sides: list[int] = []
        for side in (0, 1, 2):
            clauses = {sign_clause(rule, side) for rule in RULE_BY_ID.values()}
            if len(clauses) > 1:
                sides.append(side)
        return sides

    @staticmethod
    def _sign_cells(
        public_map: PublicGridMap,
        signpost: tuple[int, int],
        side: int,
    ) -> list[tuple[int, int]]:
        """Free cells from which ``READ_SIGN`` reads ``side``."""

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
            coord
            for coord in candidates
            if 0 <= coord[0] < public_map.width
            and 0 <= coord[1] < public_map.height
            and coord not in public_map.walls
        ]

    @classmethod
    def _inspect_cells(
        cls,
        public_map: PublicGridMap,
        blue: tuple[int, int],
        reach: int,
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        cells = [
            (x, y)
            for x in range(public_map.width)
            for y in range(public_map.height)
            if (x, y) not in public_map.walls
            and cls._manhattan((x, y), blue) <= reach
        ]
        return sorted(cells, key=lambda coord: cls._manhattan(current, coord))


class DiagnosticBeliefPolicyV4(DiagnosticBeliefPolicy):
    """Diagnostic policy with safe planning and posterior-conditioned probing.

    This is an exploratory v0.4 candidate, kept separate from
    :class:`DiagnosticBeliefPolicy` so the v0.3 benchmark stays reproducible.
    It changes three decision rules:

    1. an empty MAP plan falls back to a ``base`` plan instead of the first
       legal physical action;
    2. sign evidence is stored per face, and an unread face is chosen by its
       expected information gain over the current posterior; a face that
       gives both leading rules the same clause is skipped;
    3. a rule must lead on two consecutive decisions before it is eligible
       for commitment (the probability thresholds themselves are unchanged);
    4. when only the two-unit inspection budget remains, a blue inspection
       that separates the leading rules is preferred over another sign read.

    The policy remains public-only: it reads no simulator state and no hidden
    rule, and the true rule is never available to it.
    """

    _probe_leaders_only = True
    _reserve_blue_budget = True

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)
        self.sign_sides_observed: set[int] = set()
        self.plan_fallback_count = 0
        self._last_top_rule: str | None = None
        self._top_rule_count = 0

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        super().reset(observation, info)
        self.sign_sides_observed = set()
        self.plan_fallback_count = 0
        self._last_top_rule = None
        self._top_rule_count = 0

    def record(
        self,
        action: GridAction,
        observation: Any,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        payload = info or {}
        previous = self.previous_observation
        if (
            action.kind == "READ_SIGN"
            and payload.get("last_tool_value") is not None
            and previous is not None
        ):
            signpost = previous.public_map.signpost
            if signpost is not None:
                try:
                    side = sign_side(
                        tuple(previous.agent_position), tuple(signpost)
                    )
                except ValueError:
                    side = None
                if side is not None:
                    self.sign_sides_observed.add(int(side))
                    self.sign_observed = True
        super().record(action, observation, reward, info)

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        self._observe_top_rule()
        observation = as_observation(observation)
        if self._commit_is_safe():
            return self._commit(observation)
        action = self._diagnostic_action(observation)
        if action is not None:
            return action
        return self._commit(observation)

    def _observe_top_rule(self) -> None:
        rule = self.belief.most_likely() if self.belief is not None else "base"
        if rule == self._last_top_rule:
            self._top_rule_count += 1
        else:
            self._last_top_rule = rule
            self._top_rule_count = 1

    def _commit_is_safe(self) -> bool:
        if self._top_rule_count < 2:
            return False
        return super()._commit_is_safe()

    def _diagnostic_action(self, observation: Any) -> GridAction | None:
        if self.tool_attempts >= 4:
            return None
        marginal = self._marginal()
        budget = int(observation.tool_budget_remaining)
        sign_cost = int(getattr(self.config, "read_sign_cost", 1))
        inspect_cost = int(getattr(self.config, "inspect_cost", 2))

        # At exactly the inspection cost, a sign read would strand the
        # remaining budget and make the blue face impossible to inspect.
        # Prefer the blue face when it separates the two leading rules.
        if (
            self._reserve_blue_budget
            and budget == inspect_cost
            and self.sign_observed
            and not self.blue_observed
            and self._blue_separates_leaders()
        ):
            action = self._go_inspect_blue(observation)
            if action is not None:
                return action

        if budget >= sign_cost:
            action = self._go_read_sign(observation)
            if action is not None:
                return action

        if (
            self.sign_observed
            and not self.blue_observed
            and budget >= inspect_cost
            and (
                marginal.get("hazard_goal_swap", 0.0)
                + marginal.get("sign_composition", 0.0)
            )
            >= 0.05
        ):
            action = self._go_inspect_blue(observation)
            if action is not None:
                return action
        return None

    def _go_read_sign(self, observation: Any) -> GridAction | None:
        signpost = observation.public_map.signpost
        if signpost is None:
            return None
        signpost = tuple(signpost)
        current = tuple(observation.agent_position)
        sign_cost = int(getattr(self.config, "read_sign_cost", 1))
        candidates: list[tuple[float, int, int, GridAction]] = []
        leaders = sorted(
            self._marginal().items(), key=lambda item: (-item[1], item[0])
        )[:2]
        for side in self._informative_sides():
            if side in self.sign_sides_observed:
                continue
            # A face that gives both leading rules the same clause cannot
            # change which one is on top.  Reading it would spend a tool
            # unit that may be needed for a blue inspection.
            if (
                self._probe_leaders_only
                and len(leaders) == 2
                and sign_clause(RULE_BY_ID[leaders[0][0]], side)
                == sign_clause(RULE_BY_ID[leaders[1][0]], side)
            ):
                continue
            gain = self._sign_side_gain(side)
            if gain <= 1e-9:
                continue
            cells = self._sign_cells(observation.public_map, signpost, side)
            if not cells:
                continue
            if current in cells:
                if (
                    "READ_SIGN" in observation.action_mask
                    and int(observation.tool_budget_remaining) >= sign_cost
                ):
                    candidates.append((gain, 0, side, GridAction("READ_SIGN")))
                continue
            plan = self._shortest_plan(observation, cells)
            if plan is not None:
                candidates.append(
                    (gain, len(plan), side, self._first_legal(plan, observation))
                )
        if not candidates:
            return None
        _, _, _, action = min(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        )
        if action.kind == "READ_SIGN":
            self.tool_attempts += 1
        return action

    def _blue_separates_leaders(self) -> bool:
        """Whether a blue inspection could distinguish the leading rules."""

        leaders = sorted(
            self._marginal().items(), key=lambda item: (-item[1], item[0])
        )[:2]
        if len(leaders) < 2:
            return False
        roles = {cell_role(RULE_BY_ID[rule], "blue") for rule, _ in leaders}
        return len(roles) > 1

    def _sign_side_gain(self, side: int) -> float:
        """Expected entropy reduction from reading one sign face."""

        marginal = self._marginal()
        if not marginal:
            return 0.0
        by_clause: dict[str, list[float]] = {}
        for rule, probability in marginal.items():
            if probability <= 0.0:
                continue
            clause = sign_clause(RULE_BY_ID[rule], side)
            by_clause.setdefault(clause, []).append(float(probability))
        if len(by_clause) <= 1:
            return 0.0
        before = -sum(
            float(probability) * math.log2(float(probability))
            for probability in marginal.values()
            if probability > 0.0
        )
        after = 0.0
        for weights in by_clause.values():
            mass = sum(weights)
            if mass <= 0.0:
                continue
            after += mass * -sum(
                (weight / mass) * math.log2(weight / mass)
                for weight in weights
            )
        return max(0.0, before - after)

    def _commit(self, observation: Any) -> GridAction:
        rule_id = self.belief.most_likely() if self.belief is not None else "base"
        if rule_id == "slippery":
            # The slippery MAP plan can cycle under its own high slip rate.
            # Base shares the same movement support, so use its plan as a
            # stable fallback without changing the goal geometry.
            plan = self.base_plan("base", observation)
            if plan:
                return self._first_plan_action(plan, observation)
        plan = self.base_plan(rule_id, observation)
        if not plan and rule_id != "base":
            fallback = self.base_plan("base", observation)
            if fallback:
                self.plan_fallback_count += 1
            plan = fallback
        return self._first_plan_action(plan, observation)


class DiagnosticBeliefPolicyV5(DiagnosticBeliefPolicyV4):
    """Diagnostic policy with direct cell-role feedback.

    v0.5 keeps the v0.4 probing and planning changes and adds one general
    observation channel. When a physical move ends on a coloured marker cell,
    the public outcome identifies that cell's role without tool noise:

    * ``move`` means the marker was traversable and safe;
    * ``hazard`` means the marker was a hazard;
    * ``goal`` ends the episode successfully.

    The first two outcomes are deterministic evidence about the rule, not model
    output or hidden state. Conditioning on them removes the failure mode where
    a single noisy INSPECT led the policy to occupy a non-goal marker and then
    loop on an empty plan. It also lets a stale post-shift observation create
    the ambiguity needed to re-probe after the environment has changed.

    This is a separate class so v0.3 and v0.4 remain reproducible. The policy
    is public-only: it reads the observed position and outcome, never the true
    rule.
    """

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)
        self.direct_role_observations = 0

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        super().reset(observation, info)
        self.direct_role_observations = 0

    def _update_belief(
        self,
        action: GridAction,
        observation: Any,
        info: dict[str, Any],
    ) -> None:
        super()._update_belief(action, observation, info)
        self._record_direct_cell_role(action, observation, info)

    def _record_direct_cell_role(
        self,
        action: GridAction,
        observation: Any,
        info: dict[str, Any],
    ) -> None:
        if (
            action.kind not in _MOVE_KINDS
            or self.belief is None
            or self.previous_observation is None
        ):
            return
        observation = as_observation(observation)
        position = tuple(observation.agent_position)
        color = next(
            (
                marker_color
                for marker_color, coord in observation.public_map.markers.items()
                if tuple(coord) == position
            ),
            None,
        )
        if color is None:
            return
        outcome = str(info.get("last_outcome", ""))
        role = {"move": "safe", "hazard": "hazard"}.get(outcome)
        if role is None:
            return
        step = int(info.get("env_steps", 0))
        self.belief.update_role(color, role, step)
        self.direct_role_observations += 1
