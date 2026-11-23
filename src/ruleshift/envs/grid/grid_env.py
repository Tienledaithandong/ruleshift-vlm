"""RuleShift-Grid environment: physical dynamics and hidden rule shifts.

Only public geometry, public rule hypotheses and public costs are ever exposed
through :meth:`GridEnv.public_view` / :meth:`GridEnv._obs`. The simulator's true
rule, shift schedule and per-episode configuration live exclusively in
:meth:`GridEnv.evaluator_state`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from ruleshift.envs.events import SemanticRNG
from ruleshift.envs.observations import digest
from ruleshift.public_api.grid_rules import (
    RULE_BY_ID,
    Coord,
    PublicGridMap,
    cell_role,
    key_opens,
    movement_outcomes,
)
from ruleshift.public_api.grid_schemas import (
    GRID_ACTIONS,
    GRID_ACTIONS_FROZEN,
    GRID_PHYSICAL_ACTIONS,
    GRID_RULE_IDS,
    GRID_TOOL_ACTIONS,
    GridAction,
    GridPublicObservation,
)
from ruleshift.public_api.interface import CandidateRule

from .config import GridConfig
from .entities import GridMap, GridScenario
from .mapgen import generate_solvable_map
from .noise import draw_move
from .rules import sample_scenario

_PIXELS_PER_CELL = 16


@dataclass
class GridState:
    """Mutable, simulator-private episode state."""

    position: Coord
    health: int
    inventory: tuple[str, ...]
    steps: int
    tool_budget: int
    recovery: int
    door_open: bool
    terminated: bool
    truncated: bool
    success: bool
    last_outcome: str


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GridEnv(Env):
    """Small grid world with a hidden movement/role rule that may shift."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: GridConfig | None = None,
        render_mode: str = "rgb_array",
        observation_mode: str = "symbolic",
    ) -> None:
        self.config = config or GridConfig()
        self.render_mode = render_mode
        self.observation_mode = observation_mode

        cfg = self.config
        self.action_space = spaces.Dict(
            {
                "kind": spaces.Discrete(len(GRID_ACTIONS)),
                "target": spaces.MultiDiscrete([cfg.width, cfg.height]),
            }
        )
        observation_space: dict[str, Any] = {
            "public_text": spaces.Text(max_length=4096),
            "action_mask": spaces.MultiBinary(len(GRID_ACTIONS)),
            "tool_budget_remaining": spaces.Discrete(cfg.tool_budget + 1),
        }
        if observation_mode == "visual":
            observation_space["image"] = spaces.Box(
                low=0,
                high=255,
                shape=(
                    cfg.height * _PIXELS_PER_CELL,
                    cfg.width * _PIXELS_PER_CELL,
                    3,
                ),
                dtype=np.uint8,
            )
        self.observation_space = spaces.Dict(observation_space)

        self._grid_map: GridMap | None = None
        self._scenario: GridScenario | None = None
        self._episode_config = self.config
        self._state: GridState | None = None
        self._rng: SemanticRNG | None = None
        self._history: list[tuple[str, str | None, str]] = []
        self._shifted = False
        self._active_rule = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        opts = dict(options or {})

        config = self.config
        known_config = set(GridConfig.__dataclass_fields__)
        config_overrides = {
            key: value for key, value in opts.items() if key in known_config
        }
        if config_overrides:
            config = replace(config, **config_overrides)
        self._episode_config = config

        scenario = opts.get("scenario")
        if not isinstance(scenario, GridScenario):
            scenario = sample_scenario(
                seed or 0, config, opts.get("scenario_type")
            )
        for key in ("scenario_type", "initial_rule", "shift_rule", "shift_time"):
            if key in opts:
                scenario = replace(scenario, **{key: opts[key]})
        self._scenario = scenario

        map_option = opts.get("map")
        if isinstance(map_option, GridMap):
            grid_map = map_option
        elif isinstance(map_option, PublicGridMap):
            grid_map = GridMap(public=map_option, map_seed=scenario.map_seed)
        else:
            grid_map = generate_solvable_map(scenario.map_seed, config).map
        self._grid_map = grid_map

        self._rng = SemanticRNG(scenario.run_seed, scenario.scenario_id)
        self._state = GridState(
            position=grid_map.public.start,
            health=config.max_health,
            inventory=(),
            steps=0,
            tool_budget=config.tool_budget,
            recovery=0,
            door_open=False,
            terminated=False,
            truncated=False,
            success=False,
            last_outcome="none",
        )
        self._history = []
        self._shifted = False
        self._active_rule = scenario.initial_rule
        return self._obs(), self._info()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self._state.terminated or self._state.truncated:
            raise RuntimeError("episode has ended; call reset()")

        grid_action = self._resolve_action(action)
        if (
            grid_action is None
            or not isinstance(grid_action.kind, str)
            or grid_action.kind not in GRID_ACTIONS_FROZEN
        ):
            self._state.last_outcome = "invalid_action"
            return self._obs(), 0.0, self._state.terminated, self._state.truncated, self._info()

        if grid_action.kind in GRID_TOOL_ACTIONS:
            # Task 5 implements the epistemic tools; for now the action is
            # rejected without consuming a physical step.
            self._state.last_outcome = "tool_not_available"
            return self._obs(), 0.0, self._state.terminated, self._state.truncated, self._info()

        if grid_action.kind not in self._action_mask():
            self._state.last_outcome = "invalid_action"
            return self._obs(), 0.0, self._state.terminated, self._state.truncated, self._info()

        return self._apply_physical(grid_action)

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_action(action: Any) -> GridAction | None:
        if isinstance(action, GridAction):
            return action
        if isinstance(action, str):
            return GridAction(kind=action.upper())
        if isinstance(action, dict):
            kind = action.get("kind")
            if isinstance(kind, int) and not isinstance(kind, bool):
                if 0 <= kind < len(GRID_ACTIONS):
                    kind = GRID_ACTIONS[kind]
                else:
                    return None
            target = action.get("target")
            if isinstance(target, (list, tuple)) and len(target) == 2:
                target = (int(target[0]), int(target[1]))
            else:
                target = None
            return GridAction(kind=kind, target=target)
        return None

    def _rule_object(self, rule_id: str):
        """Public rule with the simulator's configured slip probabilities."""

        config = self._episode_config
        slip = (
            config.slippery_slip if rule_id == "slippery" else config.baseline_slip
        )
        return replace(RULE_BY_ID[rule_id], slip_prob=slip)

    def _apply_physical(self, action: GridAction) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        config = self._episode_config
        state = self._state
        public = self._grid_map.public
        reward = config.step_cost

        if state.recovery > 0:
            state.recovery -= 1
            state.last_outcome = "recovering"
            state.steps += 1
            self._maybe_shift()
            self._check_truncation()
            return self._finish(action.kind, reward)

        rule = self._rule_object(self._active_rule)

        if action.kind == "INTERACT":
            self._interact(rule, public, state)
        else:
            blocked = set(public.walls)
            if public.door is not None and not state.door_open:
                blocked.add(public.door)
            if public.signpost is not None:
                blocked.add(public.signpost)
            blocked_map = replace(public, walls=frozenset(blocked))
            outcomes = movement_outcomes(
                rule, state.position, action.kind, blocked_map
            )
            state.position = self._sample_outcome(
                outcomes, draw_move(self._rng, state.steps)
            )
            reward += self._apply_cell_effect(rule, public, state)

        state.steps += 1
        self._maybe_shift()
        self._check_truncation()
        return self._finish(action.kind, reward)

    @staticmethod
    def _sample_outcome(outcomes: dict[Coord, float], draw: float) -> Coord:
        items = sorted(outcomes.items(), key=lambda item: item[0])
        cumulative = 0.0
        for coord, probability in items:
            cumulative += probability
            if draw < cumulative:
                return coord
        return items[-1][0] if items else (0, 0)

    @staticmethod
    def _interact(rule, public: PublicGridMap, state: GridState) -> None:
        key_color = rule.key_color
        if (
            public.key is not None
            and state.position == public.key
            and key_color not in state.inventory
        ):
            state.inventory = state.inventory + (key_color,)
            state.last_outcome = "key_picked_up"
            return
        if (
            public.door is not None
            and not state.door_open
            and _manhattan(state.position, public.door) == 1
        ):
            if any(
                key_opens(rule, carried, rule.door_color)
                for carried in state.inventory
            ):
                state.door_open = True
                state.last_outcome = "door_opened"
            else:
                state.last_outcome = "door_locked"
            return
        state.last_outcome = "interact"

    def _apply_cell_effect(
        self, rule, public: PublicGridMap, state: GridState
    ) -> float:
        config = self._episode_config
        color = next(
            (color for color, coord in public.markers.items() if coord == state.position),
            None,
        )
        role = cell_role(rule, color) if color is not None else "safe"
        if role == "goal":
            state.success = True
            state.terminated = True
            state.last_outcome = "goal"
            return config.goal_reward
        if role == "hazard":
            state.health = max(0, state.health + int(config.soft_hazard_damage))
            state.recovery = int(config.soft_hazard_recovery_steps)
            state.last_outcome = "hazard"
            return config.soft_hazard_damage
        state.last_outcome = "move"
        return 0.0

    def _maybe_shift(self) -> None:
        scenario = self._scenario
        if (
            scenario.shift_time is not None
            and self._state.steps >= scenario.shift_time
            and not self._shifted
        ):
            self._active_rule = scenario.shift_rule
            self._shifted = True

    def _check_truncation(self) -> None:
        if self._state.steps >= self._episode_config.max_env_steps:
            self._state.truncated = True

    def _finish(
        self, action_kind: str, reward: float
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._history.append(
            (action_kind, None, self._state.last_outcome)
        )
        return (
            self._obs(),
            float(reward),
            self._state.terminated,
            self._state.truncated,
            self._info(),
        )

    # ------------------------------------------------------------------
    # Observation / info surfaces
    # ------------------------------------------------------------------

    def _action_mask(self) -> tuple[str, ...]:
        # Task 5 enables INSPECT/READ_SIGN; Task 4 exposes physical actions
        # (including INTERACT) only.
        return tuple(GRID_PHYSICAL_ACTIONS)

    def _obs(self) -> Any:
        view = self.public_view()
        payload = view.to_payload()
        if self.observation_mode == "visual":
            payload["image"] = np.zeros(
                self.observation_space["image"].shape, dtype=np.uint8
            )
        return payload

    def _info(self) -> dict[str, Any]:
        state = self._state
        return {
            "env_steps": state.steps,
            "success": state.success,
            "last_outcome": state.last_outcome,
            "tool_budget_remaining": state.tool_budget,
        }

    def public_view(self) -> GridPublicObservation:
        """Build the complete agent-visible observation (no hidden state)."""

        if self._state is None or self._grid_map is None:
            raise RuntimeError("call reset() before public_view()")
        state = self._state
        config = self._episode_config

        visible_state = {
            "position": list(state.position),
            "health": state.health,
            "inventory": list(state.inventory),
            "door_open": state.door_open,
            "steps": state.steps,
            "tool_budget": state.tool_budget,
        }
        history_lines = [
            f"Step {index}: {action.upper()}"
            + (f" {detail.upper()}" if detail else "")
            + f" -> {outcome or 'none'}"
            for index, (action, detail, outcome) in enumerate(self._history, 1)
        ]
        public_history = "; ".join(history_lines) if history_lines else "no prior steps"
        public_text = (
            f"step={state.steps} pos=({state.position[0]},{state.position[1]}) "
            f"health={state.health} inventory={','.join(state.inventory) or 'none'} "
            f"door={'open' if state.door_open else 'closed'} "
            f"budget={state.tool_budget} last={state.last_outcome}"
        )
        public_costs = {
            "step_cost": config.step_cost,
            "goal_reward": config.goal_reward,
            "inspect_cost": config.inspect_cost,
            "read_sign_cost": config.read_sign_cost,
            "advisor_cost": config.advisor_cost,
            "tool_budget": config.tool_budget,
        }
        return GridPublicObservation(
            observation_id=f"grid-obs-{state.steps}",
            public_obs_hash=digest(visible_state),
            public_history_hash=digest(list(self._history)),
            public_history=public_history,
            public_text=public_text,
            available_actions=tuple(GRID_ACTIONS),
            candidate_rules=tuple(
                CandidateRule(rule_id=rule_id, description=RULE_BY_ID[rule_id].description)
                for rule_id in GRID_RULE_IDS
            ),
            public_costs=public_costs,
            public_map=self._grid_map.public,
            agent_position=tuple(state.position),
            health=state.health,
            inventory=tuple(state.inventory),
            door_open=state.door_open,
            steps=state.steps,
            action_mask=self._action_mask(),
            tool_budget_remaining=state.tool_budget,
            image=None,
            image_hash="",
        )

    def evaluator_state(self) -> dict[str, Any]:
        """Privileged evaluator view; never call from agent code."""

        if self._state is None or self._scenario is None:
            raise RuntimeError("call reset() before evaluator_state()")
        state = self._state
        scenario = self._scenario
        return {
            "scenario_id": scenario.scenario_id,
            "true_rule": self._active_rule,
            "initial_rule": scenario.initial_rule,
            "shift_time": scenario.shift_time,
            "scenario_type": scenario.scenario_type,
            "map_seed": scenario.map_seed,
            "position": state.position,
            "health": state.health,
            "inventory": state.inventory,
            "steps": state.steps,
            "tool_budget": state.tool_budget,
            "recovery": state.recovery,
            "door_open": state.door_open,
            "success": state.success,
            "terminated": state.terminated,
            "truncated": state.truncated,
        }

    def render(self) -> None:
        # Task 6 implements the 224x224 renderer.
        return None


__all__ = ["GridEnv", "GridState"]
