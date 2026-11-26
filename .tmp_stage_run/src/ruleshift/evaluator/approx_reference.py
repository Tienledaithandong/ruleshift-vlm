"""Truncated Monte-Carlo rollout reference for the full environment.

This is deliberately labelled APPROXIMATE. It uses the evaluator's simulator
copies and a belief-informed greedy default policy, but only a truncated
horizon and a small sample count. It must never be described as Bayes-optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random

from ruleshift.envs.config import EnvConfig
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.evaluator.bayes_filter import Belief, RuleShiftFilter
from ruleshift.public_api.schemas import FAMILY_A_ACTIONS


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
        if horizon < 1 or samples < 1:
            raise ValueError("horizon and samples must be positive")
        self.horizon = min(horizon, config.max_env_steps)
        self.config = replace(config, max_env_steps=self.horizon)
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
        self, action: str, sample: int, belief: Belief, hidden_state: tuple[str, bool]
    ) -> float:
        env = RuleShiftEnv(config=self.config)
        rule, shifted = hidden_state
        env.reset(seed=self.seed + sample, options={
            "initial_mechanic": rule,
            "disable_shift": self.config.disable_shift or shifted,
        })
        bayes = RuleShiftFilter(
            rules=self.config.candidate_rules,
            shift_targets=self.config.shift_targets,
            shift_hazard=0.0 if self.config.disable_shift else self.config.shift_hazard,
            feedback_noise=self.config.feedback_noise,
        )
        total = 0.0
        while True:
            before = env.evaluator_state()
            ops = env.executor.plan(action, before["position"]).ops
            _, reward, _, truncated, info = env.step(action)
            total += reward
            if truncated:
                break
            elapsed = info["physical_steps_elapsed"] - before["steps_taken"]
            outcome = env.public_view().public_text.rsplit("last=", 1)[1]
            if outcome in {"accept", "reject"}:
                event = next(i for i, op in enumerate(ops) if op.kind in {"test", "commit"})
                belief = bayes.step(belief, action, outcome, event + 1)
                belief = bayes.advance(belief, elapsed - event - 1)
            else:
                belief = bayes.advance(belief, elapsed)
            action = self._default_action(belief)
        return total

    def q_values(
        self, belief: dict[tuple[str, bool], float] | None = None
    ) -> ApproximateQ:
        """Return approximate Q estimates for every public macro id."""

        if belief is None:
            belief = {(self.config.initial_mechanic, False): 1.0}
        if (not belief or any(rule not in self.config.candidate_rules for rule, _ in belief)
                or any(p < 0 or not math.isfinite(p) for p in belief.values())
                or not math.isclose(sum(belief.values()), 1.0, abs_tol=1e-8)):
            raise ValueError("belief must be a probability distribution on candidate mechanics")
        # Reuse the same sampled initial states and event streams across actions.
        states = sorted(belief)
        samples = random.Random(self.seed).choices(
            states, weights=[belief[state] for state in states], k=self.samples
        )
        values = {
            action: sum(
                self._rollout(action, sample, belief, hidden_state)
                for sample, hidden_state in enumerate(samples)
            )
            / self.samples
            for action in FAMILY_A_ACTIONS
        }
        return ApproximateQ(
            values=values,
            label=self.LABEL,
            horizon=self.horizon,
            samples=self.samples,
        )
