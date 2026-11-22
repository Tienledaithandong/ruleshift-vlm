"""Exhaustive finite-horizon solver for enumerated microtasks.

Evaluator only. The model keeps the essential Gate/Lock structure but is small
enough to enumerate exactly: hidden `(rule, shifted)` state, public macro
actions with explicit duration distributions, in-macro shift chance events,
and noisy probe feedback. Larger environments must use an approximate
reference and must be labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ruleshift.evaluator.bayes_filter import RuleShiftFilter
from ruleshift.public_api.schemas import COMMIT_ACTIONS, PROBE_ACTIONS

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


def _shift_paths(
    shifted: bool, steps: int, hazard: float, targets: tuple[str, ...]
) -> Iterable[tuple[int | None, str | None, float]]:
    """Enumerate no-shift and every one-shift timing/target path."""

    if shifted:
        yield (None, None, 1.0)
        return
    yield (None, None, (1.0 - hazard) ** steps)
    for step in range(steps):
        for target in targets:
            probability = (1.0 - hazard) ** step * hazard / len(targets)
            yield (step, target, probability)


def macro_branches(
    config: MicrotaskConfig, belief: Belief, action: str
) -> tuple[Branch, ...]:
    """Observation distribution for one macro, including in-macro shifts."""

    likelihood_model = config.as_filter()
    is_commit = action.startswith("COMMIT_")
    is_probe = action.startswith("PROBE_")
    observations = ("accept", "reject") if is_commit or is_probe else ("none",)
    durations = config.durations.get(action, {1: 1.0})
    results: list[Branch] = []

    for duration, duration_probability in durations.items():
        observation_step = config.observation_steps.get(action, duration - 1)
        if observation_step >= duration:
            raise ValueError(f"observation step outside macro duration for {action}")

        for observation in observations:
            joint = 0.0
            posterior_mass: Belief = {}
            for (rule, shifted), prior in belief.items():
                if prior <= 0.0:
                    continue
                for shift_step, target, path_probability in _shift_paths(
                    shifted, duration, config.shift_hazard, config.shift_targets
                ):
                    if path_probability <= 0.0:
                        continue
                    rule_at_observation = (
                        target
                        if shift_step is not None and shift_step <= observation_step
                        else rule
                    )
                    likelihood = likelihood_model.likelihood(
                        action, observation, rule_at_observation
                    )
                    mass = prior * path_probability * likelihood
                    if mass <= 0.0:
                        continue
                    joint += mass
                    final_rule = target if shift_step is not None else rule
                    final_state = (final_rule, bool(shifted or shift_step is not None))
                    posterior_mass[final_state] = posterior_mass.get(final_state, 0.0) + mass

            if joint <= 0.0:
                continue
            posterior = {key: value / joint for key, value in posterior_mass.items()}
            extra_steps = (
                config.lockout_steps if is_commit and observation == "reject" else 0
            )
            results.append(
                Branch(
                    probability=joint * duration_probability,
                    observation=observation,
                    posterior=posterior,
                    steps=duration + extra_steps,
                    observation_step=observation_step,
                    success=is_commit and observation == "accept",
                )
            )
    return tuple(results)


class ExactSolver:
    """Backward dynamic programming over beliefs and remaining horizon."""

    def __init__(self, config: MicrotaskConfig, actions: Iterable[str]) -> None:
        self.config = config
        self.actions = tuple(actions)
        self._q_cache: dict[tuple[FrozenBelief, int], tuple[tuple[str, float], ...]] = {}

    def q_values(
        self, belief: FrozenBelief, horizon: int
    ) -> tuple[tuple[str, float], ...]:
        """Return ``Q_h(b, a)`` for every action at this belief."""

        cache_key = (belief, horizon)
        if cache_key in self._q_cache:
            return self._q_cache[cache_key]

        if horizon <= 0:
            result = tuple((action, 0.0) for action in self.actions)
        else:
            current = dict(belief)
            scored: list[tuple[str, float]] = []
            for action in self.actions:
                total = 0.0
                for branch in macro_branches(self.config, current, action):
                    executed = min(branch.steps, horizon)
                    reward = branch_reward(self.config, branch, executed)
                    continuation = (
                        self.value(branch.posterior, horizon - executed)
                        if executed < horizon
                        else 0.0
                    )
                    total += branch.probability * (reward + continuation)
                scored.append((action, total))
            result = tuple(scored)

        self._q_cache[cache_key] = result
        return result

    def value(self, belief: Belief | FrozenBelief, horizon: int) -> float:
        """Optimal finite-horizon value ``V_h(b)``."""

        if horizon <= 0:
            return 0.0
        frozen = belief if isinstance(belief, tuple) else freeze_belief(belief)
        return max(value for _, value in self.q_values(frozen, horizon))

    def delta_probe(self, belief: Belief, horizon: int) -> float:
        """``max_PROBE Q - max_TASK Q`` at this belief and horizon."""

        q = dict(self.q_values(freeze_belief(belief), horizon))
        probe_best = max(q[action] for action in PROBE_ACTIONS if action in q)
        task_best = max(q[action] for action in COMMIT_ACTIONS if action in q)
        return probe_best - task_best
