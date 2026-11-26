"""Exhaustive finite-horizon solver for enumerated microtasks.

Evaluator only. The model keeps the essential Gate/Lock structure but is small
enough to enumerate exactly: hidden `(rule, shifted)` state, public macro
actions with explicit duration distributions, in-macro shift chance events,
and noisy probe feedback. Larger environments must use an approximate
reference and must be labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from ruleshift.envs.macro import Coord, Geometry, MacroExecutor
from ruleshift.evaluator.bayes_filter import RuleShiftFilter
from ruleshift.public_api.schemas import (
    COMMIT_ACTIONS,
    FAMILY_A_ACTIONS,
    PROBE_ACTIONS,
)

Belief = dict[tuple[str, bool], float]
FrozenBelief = tuple[tuple[tuple[str, bool], float], ...]


@dataclass(frozen=True)
class MicrotaskConfig:
    """A fully specified tiny POMDP derived from the environment contract."""

    rules: tuple[str, ...] = ("red", "blue", "green")
    initial_rule: str = "red"
    shift_hazard: float = 0.0
    shift_targets: tuple[str, ...] = ("blue", "green")
    feedback_noise: float = 0.0
    reward_success: float = 10.0
    step_cost: float = -0.05
    lockout_steps: int = 0
    durations: dict[str, dict[int, float]] = field(default_factory=dict)
    observation_steps: dict[str, int] = field(default_factory=dict)
    geometry: Geometry | None = None

    def as_filter(self) -> RuleShiftFilter:
        """The shared generative likelihood/transition model."""

        return RuleShiftFilter(
            rules=self.rules,
            initial_rule=self.initial_rule,
            shift_hazard=self.shift_hazard,
            shift_targets=self.shift_targets,
            feedback_noise=self.feedback_noise,
        )


@dataclass(frozen=True)
class Branch:
    """One observation outcome of one macro action."""

    probability: float
    observation: str
    posterior: Belief
    steps: int
    observation_step: int
    success: bool
    position: Coord | None = None
    lockout_remaining: int = 0


def branch_reward(
    config: MicrotaskConfig, branch: Branch, executed_steps: int
) -> float:
    """Reward for the primitive steps actually executed in a branch."""

    reward = executed_steps * config.step_cost
    if branch.success and branch.observation_step < executed_steps:
        reward += config.reward_success
    return reward


def freeze_belief(belief: Belief) -> FrozenBelief:
    """Hashable canonical belief form used by the DP cache."""

    return tuple(
        sorted(
            ((key, value) for key, value in belief.items() if value > 0.0),
            key=lambda item: (item[0][0], item[0][1]),
        )
    )


@lru_cache(maxsize=512)
def _public_plan(geometry: Geometry, position: Coord, action: str) -> tuple[int, int]:
    ops = MacroExecutor(geometry).plan(action, position).ops
    event = next((i for i, op in enumerate(ops) if op.kind in {"commit", "test"}), 0)
    return len(ops), event


def macro_branches(
    config: MicrotaskConfig, belief: Belief, action: str,
    position: Coord | None = None, lockout_remaining: int = 0,
) -> tuple[Branch, ...]:
    """Observation distribution for one macro, including in-macro shifts."""

    likelihood_model = config.as_filter()
    durations = config.durations.get(action, {1: 1.0})
    event = config.observation_steps.get(action)
    geometry = config.geometry
    if geometry is not None:
        position = position if position is not None else geometry.start
        duration, event = _public_plan(geometry, position, action)
        durations = {duration: 1.0}
    is_commit = action.startswith("COMMIT_")
    if geometry is not None and is_commit and lockout_remaining > event:
        is_commit = False
    is_probe = action.startswith("PROBE_")
    observations = ("accept", "reject") if is_commit or is_probe else ("none",)
    results: list[Branch] = []

    for duration, duration_probability in durations.items():
        observation_step = duration - 1 if event is None else event
        if not 0 <= observation_step < duration:
            raise ValueError(f"observation step outside macro duration for {action}")
        at_event = likelihood_model.advance(belief, observation_step + 1)
        for observation in observations:
            effective_action = action if is_commit or is_probe else "WAIT"
            posterior_mass = {
                key: prior * likelihood_model.likelihood(effective_action, observation, key[0])
                for key, prior in at_event.items()
            }
            joint = sum(posterior_mass.values())
            if joint <= 0.0:
                continue
            posterior = {key: value / joint for key, value in posterior_mass.items()}
            rejected = is_commit and observation == "reject"
            # Abstract microtasks model forced lockout ticks. The grid allows
            # travel and probes during lockout and stays at the gate on rejection.
            extra_steps = config.lockout_steps if geometry is None and rejected else 0
            posterior = likelihood_model.advance(
                posterior, duration - observation_step - 1 + extra_steps
            )
            next_position = position
            remaining = max(0, lockout_remaining - duration)
            if geometry is not None:
                if action.startswith("COMMIT_"):
                    next_position = geometry.start if observation == "accept" else geometry.gate
                    if rejected:
                        remaining = config.lockout_steps
                elif is_probe:
                    next_position = geometry.start
                elif action == "GO_TO_STATION":
                    next_position = geometry.station
            results.append(
                Branch(
                    probability=joint * duration_probability,
                    observation=observation,
                    posterior=posterior,
                    steps=duration + extra_steps,
                    observation_step=observation_step,
                    success=is_commit and observation == "accept",
                    position=next_position,
                    lockout_remaining=remaining,
                )
            )
    return tuple(results)


def family_a_microtask(
    *,
    probe_detour_steps: int,
    commit_travel_steps: int,
    feedback_noise: float,
    lockout_steps: int,
    actions: Sequence[str] = FAMILY_A_ACTIONS,
    shift_hazard: float = 0.0,
    step_cost: float = -0.05,
    reward_success: float = 10.0,
) -> MicrotaskConfig:
    """Microtask whose macro durations match ``envs.macro`` exactly.

    A probe costs ``2 * detour + 1`` primitive steps and observes the test
    result after ``detour`` steps (out, test, back); a commit costs
    ``commit_travel + 1`` steps and observes the gate after ``commit_travel``
    steps. Using these instead of a unit-duration abstraction keeps exact
    labels comparable with what the environment actually charges.
    """

    if probe_detour_steps < 0:
        raise ValueError("probe_detour_steps must be non-negative")
    if commit_travel_steps < 0:
        raise ValueError("commit_travel_steps must be non-negative")

    if not 0 <= shift_hazard <= 1 or not 0 <= feedback_noise <= 1:
        raise ValueError("hazard and feedback noise must be probabilities")
    if lockout_steps < 0:
        raise ValueError("lockout must be non-negative")
    station_x = min(commit_travel_steps, probe_detour_steps)
    station_y = probe_detour_steps - station_x
    geometry = Geometry(
        start=(0, 0),
        gate=(commit_travel_steps, 0),
        station=(station_x, station_y),
        grid=(commit_travel_steps + 1, station_y + 1),
    )
    durations: dict[str, dict[int, float]] = {}
    observation_steps: dict[str, int] = {}
    for action in actions:
        duration, event = _public_plan(geometry, geometry.start, action)
        durations[action] = {duration: 1.0}
        observation_steps[action] = event

    return MicrotaskConfig(
        shift_hazard=shift_hazard,
        feedback_noise=feedback_noise,
        reward_success=reward_success,
        step_cost=step_cost,
        lockout_steps=lockout_steps,
        durations=durations,
        observation_steps=observation_steps,
        geometry=geometry,
    )


class ExactSolver:
    """Backward dynamic programming over beliefs and remaining horizon."""

    def __init__(self, config: MicrotaskConfig, actions: Iterable[str]) -> None:
        self.config = config
        self.actions = tuple(actions)
        self._q_cache: dict[tuple, tuple[tuple[str, float], ...]] = {}
        self._branch_cache: dict[tuple, tuple[Branch, ...]] = {}

    def q_values(
        self, belief: FrozenBelief, horizon: int, position: Coord | None = None,
        lockout_remaining: int = 0,
    ) -> tuple[tuple[str, float], ...]:
        """Return ``Q_h(b, a)`` for every action at this belief."""

        if self.config.geometry is not None and position is None:
            position = self.config.geometry.start
        cache_key = (belief, horizon, position, lockout_remaining)
        if cache_key in self._q_cache:
            return self._q_cache[cache_key]

        if horizon <= 0:
            result = tuple((action, 0.0) for action in self.actions)
        elif (
            self.config.geometry is not None
            and position is not None
            and horizon <= max(lockout_remaining, _public_plan(self.config.geometry, position, "COMMIT_RED")[1])
        ):
            # No delivery can finish in time. Every policy consumes exactly
            # this many primitive steps and earns only the common step cost.
            result = tuple((action, horizon * self.config.step_cost) for action in self.actions)
        else:
            current = dict(belief)
            scored: list[tuple[str, float]] = []
            for action in self.actions:
                total = 0.0
                branch_key = (belief, action, position, lockout_remaining)
                if branch_key not in self._branch_cache:
                    self._branch_cache[branch_key] = macro_branches(
                        self.config, current, action, position, lockout_remaining
                    )
                for branch in self._branch_cache[branch_key]:
                    executed = min(branch.steps, horizon)
                    reward = branch_reward(self.config, branch, executed)
                    continuation = self.value(
                        branch.posterior, horizon - executed, branch.position,
                        branch.lockout_remaining,
                    )
                    total += branch.probability * (reward + continuation)
                scored.append((action, total))
            result = tuple(scored)

        self._q_cache[cache_key] = result
        return result

    def value(
        self, belief: Belief | FrozenBelief, horizon: int,
        position: Coord | None = None, lockout_remaining: int = 0,
    ) -> float:
        """Optimal finite-horizon value ``V_h(b)``."""

        if horizon <= 0:
            return 0.0
        frozen = belief if isinstance(belief, tuple) else freeze_belief(belief)
        return max(value for _, value in self.q_values(frozen, horizon, position, lockout_remaining))

    def delta_probe(
        self, belief: Belief, horizon: int, position: Coord | None = None,
        lockout_remaining: int = 0,
    ) -> float:
        """``max_PROBE Q - max_TASK Q`` at this belief and horizon.

        Only actions this solver was constructed with are considered; callers
        that omit a probe or a task action get an explicit error instead of a
        silently mislabelled delta.
        """

        q = dict(self.q_values(freeze_belief(belief), horizon, position, lockout_remaining))
        probe_values = [
            q[action] for action in PROBE_ACTIONS if action in q
        ]
        task_values = [
            q[action] for action in COMMIT_ACTIONS if action in q
        ]
        if not probe_values or not task_values:
            raise ValueError(
                "delta_probe needs at least one probe and one commit action in "
                f"this solver, got {sorted(q)}"
            )
        delta = max(probe_values) - max(task_values)
        return 0.0 if abs(delta) < 1e-10 else delta
