"""Truncated Monte-Carlo rollout reference for the full environment.

This is deliberately labelled APPROXIMATE. It uses the evaluator's simulator
copies and a belief-informed greedy default policy, but only a truncated
horizon and a small sample count. It must never be described as Bayes-optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ruleshift.envs.config import EnvConfig
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.public_api.schemas import MACRO_ACTIONS


@dataclass(frozen=True)
class ApproximateQ:
    """Approximate action values plus an explicit provenance label."""

    values: dict[str, float]
    label: str
    horizon: int
    samples: int


class TruncatedRolloutReference:
    """Estimate action returns by short, seeded rollouts on copied envs."""

    LABEL = "APPROXIMATE_TRUNCATED_ROLLOUT"

    def __init__(
        self,
        config: EnvConfig,
        horizon: int = 3,
        samples: int = 4,
        seed: int = 0,
    ) -> None:
        self.config = replace(config, max_env_steps=horizon)
        self.horizon = horizon
        self.samples = samples
        self.seed = seed

    @staticmethod
    def _default_action(belief: dict[tuple[str, bool], float] | None) -> str:
        if not belief:
            return "COMMIT_RED"
        marginal: dict[str, float] = {}
        for (rule, _), probability in belief.items():
            marginal[rule] = marginal.get(rule, 0.0) + probability
        rule = max(marginal, key=marginal.get)
        return f"COMMIT_{rule.upper()}"

    def _rollout(
        self, action: str, sample: int, belief: dict[tuple[str, bool], float] | None
    ) -> float:
        env = RuleShiftEnv(config=self.config)
        env.reset(seed=self.seed + sample)
        _, total, _, truncated, info = env.step(action)
        while not truncated:
            _, reward, _, truncated, info = env.step(self._default_action(belief))
            total += reward
            if info["physical_steps_elapsed"] >= self.horizon:
                break
        return total

    def q_values(
        self, belief: dict[tuple[str, bool], float] | None = None
    ) -> ApproximateQ:
        """Return approximate Q estimates for every public macro id."""

        values = {
            action: sum(
                self._rollout(action, sample, belief)
                for sample in range(self.samples)
            )
            / self.samples
            for action in MACRO_ACTIONS
        }
        return ApproximateQ(
            values=values,
            label=self.LABEL,
            horizon=self.horizon,
            samples=self.samples,
        )
