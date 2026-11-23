"""Public-only baseline policies for RuleShift-Grid.

This module is purely agent-side: it imports only the standard library, the
public API and other agent modules. It never imports the simulator or the
privileged evaluator, it never reads the true rule, and every decision is made
from the public observation and the already-executed outcome passed to
``record``.
"""

from __future__ import annotations

import random
from typing import Any, Protocol

from ruleshift.agents.grid_belief import GridBelief
from ruleshift.agents.grid_selector import InfoGainSelector, PublicAdvisor
from ruleshift.public_api.grid_planner import plan_to_goal
from ruleshift.public_api.grid_rules import (
    BASE_RULE,
    RULE_BY_ID,
    PublicGridMap,
    sign_side,
)
from ruleshift.public_api.grid_schemas import GRID_PHYSICAL_ACTIONS, GridAction

__all__ = [
    "GridPolicy",
    "NoAdaptationPolicy",
    "PublicAdvisorPolicy",
    "RandomProbePolicy",
    "InfoGainPolicy",
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

    def to_payload(self) -> dict[str, Any]:
        return self._payload


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

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        observation = as_observation(observation)
        assert self.belief is not None
        return self.selector.choose(
            self.belief,
            observation.public_map,
            position=tuple(observation.agent_position),
            inventory=tuple(observation.inventory),
            door_open=bool(observation.door_open),
            tool_budget=int(observation.tool_budget_remaining),
            action_mask=tuple(observation.action_mask),
            step=int(info.get("env_steps", observation.steps)),
        )
