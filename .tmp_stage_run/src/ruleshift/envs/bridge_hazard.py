"""Family B — Hazard/Bridge environment with a genuinely switching transition law.

The shortcut crossing has a regime-dependent slip probability; the safe route
is deterministic. A sensing-station probe costs real primitive steps and
tests a candidate regime with symmetric noise. Held-out map ids vary the
decorative distractors on the fixed public layout.

Evaluator only. No agent module may import this file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from ruleshift.public_api.interface import CandidateRule
from ruleshift.public_api.schemas import FAMILY_B_ACTIONS, FAMILY_B_ACTIONS_FROZEN

from .events import SemanticRNG
from .observations import RenderState, build_public_observation, decision_event

REGIME_DESCRIPTIONS = {
    "stable": "The bridge is currently stable; a crossing usually succeeds.",
    "slippery": "The bridge may be slippery; a crossing can fail and cost recovery time.",
}
PROBE_ONE_WAY = 2
CELL = 16
GRID_WIDTH = 9
GRID_HEIGHT = 5


@dataclass(frozen=True)
class BridgeConfig:
    max_env_steps: int = 60
    safe_steps: int = 8
    bridge_steps: int = 3
    recovery_steps: int = 4
    slip_stable: float = 0.05
    slip_slippery: float = 0.40
    feedback_noise: float = 0.10
    shift_hazard: float = 0.03
    disable_shift: bool = False
    initial_regime: str = "stable"
    shift_targets: tuple[str, ...] = ("slippery",)
    map_id: str = "A"
    distractor_seed: int = 0
    delivery_reward: float = 10.0
    step_cost: float = -0.05
    scenario_id: str = "dev-bridge"

    def __post_init__(self) -> None:
        if min(self.max_env_steps, self.safe_steps, self.bridge_steps) < 1:
            raise ValueError("horizon and route durations must be positive")
        if self.recovery_steps < 0:
            raise ValueError("recovery steps must be non-negative")
        if any(not 0 <= p <= 1 for p in (
            self.slip_stable, self.slip_slippery, self.feedback_noise, self.shift_hazard
        )):
            raise ValueError("slip, feedback noise and hazard must be probabilities")
        if self.initial_regime not in REGIME_DESCRIPTIONS or not self.shift_targets:
            raise ValueError("bridge configuration needs a valid initial regime and shift targets")
        if any(rule not in REGIME_DESCRIPTIONS for rule in self.shift_targets):
            raise ValueError("unknown bridge shift target")


@dataclass
class BridgeState:
    regime: str
    shifted: bool
    side: str
    steps: int
    last_outcome: str


class BridgeHazardEnv(Env):
    """Small route-selection env with two hidden hazard regimes."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: BridgeConfig | None = None,
        render_mode: str = "rgb_array",
    ) -> None:
        self.config = config or BridgeConfig()
        self.render_mode = render_mode
        self._episode_config = self.config
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(GRID_HEIGHT * CELL, GRID_WIDTH * CELL, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(FAMILY_B_ACTIONS))
        self._state: BridgeState | None = None
        self._rng: SemanticRNG | None = None
        self._shift_time: int | None = None
        self._history: list[str] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        opts = dict(options or {})
        master_seed = int(opts.pop("master_seed", 20261121))
        scenario_id = str(opts.pop("scenario_id", f"bridge-{seed if seed is not None else 0}"))
        known = {field for field in BridgeConfig.__dataclass_fields__}
        config = replace(
            self.config,
            **{key: value for key, value in opts.items() if key in known and key != "scenario_id"},
        )
        self._episode_config = replace(config, scenario_id=scenario_id)
        self._rng = SemanticRNG(master_seed, scenario_id)
        self._state = BridgeState(
            regime=config.initial_regime,
            shifted=False,
            side="start",
            steps=0,
            last_outcome="none",
        )
        self._shift_time = None
        self._history = []
        return self._obs(), self._info()

    def step(
        self, action: str | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        macro = self._resolve_action(action)
        ops = list(self._ops(macro))
        total = 0.0
        truncated = False
        op_index = 0
        started = self._state.steps

        while op_index < len(ops):
            if self._state.steps >= self._episode_config.max_env_steps:
                truncated = True
                break
            op = ops[op_index]
            self._maybe_shift()
            step_index = self._state.steps
            total += self._episode_config.step_cost
            self._state.steps += 1
            reward_delta, extra_ops = self._execute(op, step_index, op_index)
            total += reward_delta
            ops.extend(extra_ops)
            op_index += 1

        if self._state.steps >= self._episode_config.max_env_steps:
            truncated = True
        if self._state.steps > started:
            self._history.append(decision_event(
                macro, self._state.last_outcome, total, self._state.steps - started
            ))
        return self._obs(), float(total), False, truncated, self._info()

    def render(self) -> np.ndarray:
        return self._obs()

    def public_view(self) -> Any:
        if self._state is None:
            raise RuntimeError("call reset() before public_view()")
        config = self._episode_config
        return build_public_observation(
            geometry=self._geometry(),
            state=RenderState(
                position=self._agent_position(),
                gate_locked=False,
                deliveries=0,
                steps=self._state.steps,
            ),
            frame=self._obs(),
            public_text=(
                f"step={self._state.steps} side={self._state.side} "
                f"routes=bridge,safe station=(1,4) last={self._state.last_outcome}"
            ),
            history=self._history,
            last_outcome=self._state.last_outcome,
            candidate_rules=tuple(
                CandidateRule(rule_id=rule, description=description)
                for rule, description in REGIME_DESCRIPTIONS.items()
            ),
            public_costs={
                "primitive_step": abs(config.step_cost),
                "probe_detour": float(PROBE_ONE_WAY),
                "delivery_reward": float(config.delivery_reward),
                "safe_steps": float(config.safe_steps),
                "bridge_steps": float(config.bridge_steps),
                "recovery_steps": float(config.recovery_steps),
            },
            available_actions=FAMILY_B_ACTIONS,
        )

    def evaluator_state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("call reset() before evaluator_state()")
        config = self._episode_config
        return {
            "scenario_id": config.scenario_id,
            "initial_mechanic": config.initial_regime,
            "mechanic_id": self._state.regime,
            "active_mechanic": self._state.regime,
            "true_rule": self._state.regime,
            "shift_time": self._shift_time,
            "shift_hazard": config.shift_hazard,
            "transition_kernel": {
                "stable": {"slip": config.slip_stable},
                "slippery": {"slip": config.slip_slippery},
            },
            "slip_probability": (
                config.slip_stable
                if self._state.regime == "stable"
                else config.slip_slippery
            ),
            "c_shifted": self._state.shifted,
            "side": self._state.side,
            "steps_taken": self._state.steps,
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
            return action if action in FAMILY_B_ACTIONS_FROZEN else "WAIT"
        try:
            index = int(action)
        except (TypeError, ValueError):
            return "WAIT"
        return FAMILY_B_ACTIONS[index] if 0 <= index < len(FAMILY_B_ACTIONS) else "WAIT"

    def _ops(self, macro: str) -> list[tuple[str, str | None]]:
        config = self._episode_config
        if macro == "GO_SAFE":
            return [("safe_step", None)] * config.safe_steps
        if macro == "CROSS_BRIDGE":
            return [("cross_step", None)] * config.bridge_steps
        if macro.startswith("PROBE_"):
            regime = macro.removeprefix("PROBE_").lower()
            return (
                [("travel", None)] * PROBE_ONE_WAY
                + [("sense", regime)]
                + [("travel", None)] * PROBE_ONE_WAY
            )
        return [("wait", None)]

    def _maybe_shift(self) -> None:
        assert self._state is not None and self._rng is not None
        config = self._episode_config
        if config.disable_shift or self._state.shifted:
            return
        step = self._state.steps
        if self._rng.draw("SHIFT_TRIGGER", step, 0) < config.shift_hazard:
            self._state.regime = self._rng.choice(
                "SHIFT_TARGET", step, 0, config.shift_targets
            )
            self._state.shifted = True
            self._shift_time = step

    def _execute(
        self, op: tuple[str, str | None], step_index: int, op_index: int
    ) -> tuple[float, list[tuple[str, str | None]]]:
        assert self._state is not None and self._rng is not None
        config = self._episode_config
        kind, payload = op
        reward_delta = 0.0
        extra_ops: list[tuple[str, str | None]] = []

        if kind == "safe_step":
            self._state.last_outcome = "none"
            if op_index == config.safe_steps - 1:
                self._state.side = "goal"
                self._state.last_outcome = "success"
                reward_delta += config.delivery_reward
        elif kind == "cross_step":
            if op_index != config.bridge_steps - 1:
                self._state.last_outcome = "none"
            else:
                slip_probability = (
                    config.slip_stable
                    if self._state.regime == "stable"
                    else config.slip_slippery
                )
                slipped = (
                    self._rng.draw("HAZARD_OUTCOME", step_index, 0) < slip_probability
                )
                if slipped:
                    self._state.last_outcome = "failure"
                    self._state.side = "start"
                    extra_ops = [("recover", None)] * config.recovery_steps
                else:
                    self._state.side = "goal"
                    self._state.last_outcome = "success"
                    reward_delta += config.delivery_reward
        elif kind == "sense":
            assert payload is not None
            correct = payload == self._state.regime
            flipped = (
                self._rng.draw("PROBE_FEEDBACK", step_index, 0) < config.feedback_noise
            )
            confirmed = correct if not flipped else not correct
            self._state.last_outcome = "accept" if confirmed else "reject"
        elif kind == "travel":
            if op_index < PROBE_ONE_WAY:
                self._state.last_outcome = "none"
        elif kind == "wait":
            self._state.last_outcome = "none"
        # Recovery ticks preserve the public crossing failure.

        return reward_delta, extra_ops

    def _geometry(self) -> Any:
        from .macro import Geometry

        return Geometry(
            start=(1, 2),
            gate=(GRID_WIDTH - 2, 2),
            station=(1, 4),
            grid=(GRID_WIDTH, GRID_HEIGHT),
        )

    def _agent_position(self) -> tuple[int, int]:
        geometry = self._geometry()
        return geometry.gate if self._state.side == "goal" else geometry.start

    @staticmethod
    def _render_distractors(image: np.ndarray, config: BridgeConfig) -> None:
        """Deterministic decorative pixels derived ONLY from public map id."""

        seed = int.from_bytes(
            hashlib.blake2b(
                f"{config.map_id}:{config.distractor_seed}".encode("utf-8"), digest_size=4
            ).digest(),
            "big",
        )
        for index in range(12):
            value = (seed >> (2 * index)) & 3
            x = 2 + (index % (GRID_WIDTH - 4))
            y = 1 + (index // (GRID_WIDTH - 4))
            image[y * CELL : (y + 1) * CELL, x * CELL : (x + 1) * CELL] = (
                70 + value * 20,
                70 + value * 20,
                90 + value * 10,
            )

    def _obs(self) -> np.ndarray:
        image = np.full((GRID_HEIGHT * CELL, GRID_WIDTH * CELL, 3), 35, dtype=np.uint8)
        image[1 * CELL : 2 * CELL, 2 * CELL : (GRID_WIDTH - 2) * CELL] = (60, 130, 90)
        image[2 * CELL : 3 * CELL, 2 * CELL : (GRID_WIDTH - 2) * CELL] = (120, 100, 60)
        image[4 * CELL : 5 * CELL, 0 * CELL : 3 * CELL] = (60, 110, 170)
        self._render_distractors(image, self._episode_config)
        x, y = self._agent_position()
        image[y * CELL : (y + 1) * CELL, x * CELL : (x + 1) * CELL] = (230, 220, 80)
        return image

    def _info(self) -> dict[str, Any]:
        public = self.public_view()
        return {
            "public_obs_hash": public.public_obs_hash,
            "public_history_hash": public.public_history_hash,
            "physical_steps_elapsed": self._state.steps if self._state else 0,
            "max_env_steps": self._episode_config.max_env_steps,
            "available_actions": list(FAMILY_B_ACTIONS),
        }
