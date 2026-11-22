"""Family A Gate/Lock Gymnasium environment.

Hidden state and transition logic live here, never in the agents package. The
public view is assembled separately in :mod:`ruleshift.envs.observations`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from ruleshift.public_api.schemas import FAMILY_A_ACTIONS, FAMILY_A_ACTIONS_FROZEN

from .config import EnvConfig, config_with_overrides
from .events import SemanticRNG
from .macro import Coord, Geometry, MacroExecutor, PrimitiveOp
from .observations import (
    RenderState,
    build_public_observation,
    render_rgb,
)


@dataclass
class GateState:
    """Full hidden state; the public projection lives in observations.py."""

    position: Coord
    mechanic: str
    shifted: bool
    lockout_remaining: int
    deliveries: int
    steps_taken: int


def _station_for_detour(detour: int, width: int, height: int) -> Coord:
    """Place the station at public Manhattan distance *detour* from start."""

    if detour <= 0:
        return (0, 0)
    x = min(detour, width - 1)
    y = detour - x
    if y > height - 1:
        y = height - 1
        x = min(detour - y, width - 1)
    return (x, y)


class RuleShiftEnv(Env):
    """Small gridworld with one hidden mechanic and at most one shift."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self, config: EnvConfig | None = None, render_mode: str = "rgb_array"
    ) -> None:
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self._episode_config = self.config
        self.geometry = self._build_geometry(self.config)
        self.executor = MacroExecutor(self.geometry)

        height = self.config.grid_height * 16
        width = self.config.grid_width * 16
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(FAMILY_A_ACTIONS))

        self._state: GateState | None = None
        self._rng: SemanticRNG | None = None
        self._history: list[tuple[str, str | None, str]] = []
        self._last_outcome: str | None = None
        self._shift_time: int | None = None

    @staticmethod
    def _build_geometry(config: EnvConfig) -> Geometry:
        return Geometry(
            start=(0, 0),
            gate=(config.grid_width - 1, 0),
            station=_station_for_detour(
                config.probe_detour_steps, config.grid_width, config.grid_height
            ),
            grid=(config.grid_width, config.grid_height),
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        opts = dict(options or {})
        master_seed = int(opts.pop("master_seed", 20261121))
        scenario_id = str(opts.pop("scenario_id", f"scn-{seed if seed is not None else 0}"))
        config = config_with_overrides(self.config, opts)
        self._episode_config = replace(config, scenario_id=scenario_id)
        self.geometry = self._build_geometry(self._episode_config)
        self.executor = MacroExecutor(self.geometry)

        self._rng = SemanticRNG(master_seed=master_seed, scenario_id=scenario_id)
        self._state = GateState(
            position=self.geometry.start,
            mechanic=self._episode_config.initial_mechanic,
            shifted=False,
            lockout_remaining=0,
            deliveries=0,
            steps_taken=0,
        )
        self._history = []
        self._last_outcome = None
        self._shift_time = None
        return self._obs(), self._info()

    def step(
        self, action: str | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")

        config = self._episode_config
        macro_id = self._resolve_action(action)
        plan = self.executor.plan(macro_id, self._state.position)
        total_reward = 0.0
        truncated = False

        for op in plan.ops:
            if self._state.steps_taken >= config.max_env_steps:
                truncated = True
                break
            was_locked = self._state.lockout_remaining > 0
            total_reward += self._primitive_step(op, config, was_locked)
            self._state.steps_taken += 1
            if was_locked:
                self._state.lockout_remaining -= 1

        if self._state.steps_taken >= config.max_env_steps:
            truncated = True
        return self._obs(), float(total_reward), False, truncated, self._info()

    def render(self) -> np.ndarray:
        return self._obs()

    # ------------------------------------------------------------------
    # Public / privileged adapters
    # ------------------------------------------------------------------
    def public_view(self) -> Any:
        """The only observation object an agent is allowed to receive."""

        if self._state is None:
            raise RuntimeError("call reset() before public_view()")
        config = self._episode_config
        return build_public_observation(
            geometry=self.geometry,
            state=self._render_state(),
            history=self._history,
            last_outcome=self._last_outcome,
            candidate_rules=config.candidate_rules,
            public_costs={
                "primitive_step": abs(config.step_cost),
                "probe_detour": float(config.probe_detour_steps),
                "commit_travel": float(self._gate_travel()),
                "lockout": float(config.lockout_steps),
                "delivery_reward": float(config.delivery_reward),
            },
            available_actions=FAMILY_A_ACTIONS,
        )

    def evaluator_state(self) -> dict[str, Any]:
        """Privileged trace; never passed to agent code or prompts."""

        if self._state is None:
            raise RuntimeError("call reset() before evaluator_state()")
        state = self._state
        return {
            "mechanic_id": state.mechanic,
            "active_mechanic": state.mechanic,
            "true_rule": state.mechanic,
            "shift_time": self._shift_time,
            "shift_schedule": None,
            "shift_hazard": self._episode_config.shift_hazard,
            "c_shifted": state.shifted,
            "lockout_remaining": state.lockout_remaining,
            "position": state.position,
            "deliveries": state.deliveries,
            "steps_taken": state.steps_taken,
            "transition_kernel": None,
            "oracle_belief": None,
            "q_star": None,
            "delta_probe": None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_action(action: str | int) -> str:
        if isinstance(action, str):
            return action if action in FAMILY_A_ACTIONS_FROZEN else "WAIT"
        try:
            index = int(action)
        except (TypeError, ValueError):
            return "WAIT"
        return FAMILY_A_ACTIONS[index] if 0 <= index < len(FAMILY_A_ACTIONS) else "WAIT"

    def _gate_travel(self) -> int:
        """Public Manhattan distance from the start tile to the gate tile."""

        (sx, sy), (gx, gy) = self.geometry.start, self.geometry.gate
        return abs(gx - sx) + abs(gy - sy)

    def _render_state(self) -> RenderState:
        assert self._state is not None
        return RenderState(
            position=self._state.position,
            gate_locked=self._state.lockout_remaining > 0,
            deliveries=self._state.deliveries,
            steps=self._state.steps_taken,
        )

    def _obs(self) -> np.ndarray:
        return render_rgb(self.geometry, self._render_state())

    def _info(self) -> dict[str, Any]:
        public = self.public_view()
        return {
            "public_obs_hash": public.public_obs_hash,
            "public_history_hash": public.public_history_hash,
            "physical_steps_elapsed": self._state.steps_taken if self._state else 0,
            "max_env_steps": self._episode_config.max_env_steps,
            "available_actions": list(FAMILY_A_ACTIONS),
        }

    def _maybe_shift(self, config: EnvConfig) -> None:
        assert self._state is not None and self._rng is not None
        if config.disable_shift or self._state.shifted:
            return
        step = self._state.steps_taken
        if self._rng.draw("SHIFT_TRIGGER", step, 0) < config.shift_hazard:
            self._state.mechanic = self._rng.choice(
                "SHIFT_TARGET", step, 0, config.shift_targets
            )
            self._state.shifted = True
            self._shift_time = step

    def _primitive_step(
        self, op: PrimitiveOp, config: EnvConfig, was_locked: bool
    ) -> float:
        assert self._state is not None and self._rng is not None
        self._maybe_shift(config)
        reward = config.step_cost
        step = self._state.steps_taken

        if was_locked and op.kind == "commit":
            op = PrimitiveOp("wait")

        if op.kind == "move":
            self._state.position = op.target or self._state.position
            self._last_outcome = "none"
            self._history.append(("move", None, "none"))
        elif op.kind == "move_back":
            # Return leg of a probe: move without erasing the test feedback, so
            # the decision-boundary public text still reports accept/reject.
            self._state.position = op.target or self._state.position
            self._history.append(("move", None, self._last_outcome or "none"))
        elif op.kind == "commit":
            color = op.color or ""
            if color == self._state.mechanic:
                reward += config.delivery_reward
                self._state.deliveries += 1
                self._state.position = self.geometry.start
                self._last_outcome = "accept"
            else:
                self._state.lockout_remaining = config.lockout_steps
                self._last_outcome = "reject"
            self._history.append(("commit", color, self._last_outcome))
        elif op.kind == "test":
            color = op.color or ""
            correct = color == self._state.mechanic
            flipped = (
                self._rng.draw("PROBE_FEEDBACK", step, 0) < config.feedback_noise
            )
            accepted = (not correct) if flipped else correct
            self._last_outcome = "accept" if accepted else "reject"
            self._history.append(("test", color, self._last_outcome))
        else:  # wait
            self._last_outcome = "none"
            self._history.append(("wait", None, "none"))

        return reward
