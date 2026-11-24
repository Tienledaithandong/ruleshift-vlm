"""Evaluator-only, shift-aware Bayesian filtering over (rule, shifted).

This module lives in the privileged evaluator package. It uses the true
transition and observation likelihoods and must never be imported by agent
code. The public macro ids are reused only as observation labels; no action
value or true mechanic is exposed to the agent through this filter.
"""

from __future__ import annotations

from dataclasses import dataclass

Belief = dict[tuple[str, bool], float]


@dataclass(frozen=True)
class RuleShiftFilter:
    """Filter over hidden `(active_rule, shift_already_happened)` state.

    The transition applies the per-primitive-step shift hazard while ``c=0``;
    once ``c=1`` no further shift can occur. Observation likelihoods mirror
    the simulator: commits are deterministic given the rule, probe feedback
    is flipped with probability ``feedback_noise``, and movement/wait produce
    an uninformative ``none`` outcome.
    """

    rules: tuple[str, ...] = ("red", "blue", "green")
    initial_rule: str = "red"
    shift_hazard: float = 0.03
    shift_targets: tuple[str, ...] = ("blue", "green")
    feedback_noise: float = 0.10

    def initial_belief(self) -> Belief:
        """Start certain on the publicly known old default rule."""

        return {
            (rule, shifted): 1.0
            if rule == self.initial_rule and not shifted
            else 0.0
            for rule in self.rules
            for shifted in (False, True)
        }

    def _transition_once(self, belief: Belief) -> Belief:
        updated: Belief = {key: 0.0 for key in belief}
        targets = tuple(self.shift_targets)
        for (rule, shifted), probability in belief.items():
            if probability == 0.0:
                continue
            if shifted:
                updated[(rule, True)] += probability
                continue
            updated[(rule, False)] += probability * (1.0 - self.shift_hazard)
            share = probability * self.shift_hazard / len(targets)
            for target in targets:
                key = (target, True)
                updated[key] = updated.get(key, 0.0) + share
        return updated

    def advance(self, belief: Belief, primitive_steps: int = 1) -> Belief:
        """Propagate the belief through *primitive_steps* with no observation."""

        if primitive_steps < 0:
            raise ValueError("primitive_steps must be non-negative")
        result = dict(belief)
        if self.shift_hazard == 0.0:
            return result
        for _ in range(primitive_steps):
            result = self._transition_once(result)
        return result

    def likelihood(self, action: str, observation: str, rule: str) -> float:
        """Return ``P(observation | rule, action)`` under the true model."""

        if action.startswith("COMMIT_"):
            if observation not in {"accept", "reject"}:
                return 0.0
            color = action.removeprefix("COMMIT_").lower()
            accepted = color == rule
            matched = observation == "accept"
            return 1.0 if matched == accepted else 0.0

        if action.startswith("PROBE_"):
            if observation not in {"accept", "reject"}:
                return 0.0
            color = action.removeprefix("PROBE_").lower()
            probability_accept = (
                1.0 - self.feedback_noise if color == rule else self.feedback_noise
            )
            return (
                probability_accept if observation == "accept" else 1.0 - probability_accept
            )

        return float(observation == "none")

    def observe(self, belief: Belief, action: str, observation: str) -> Belief:
        """Bayes update on one observed action outcome."""

        unnormalised = {
            key: probability * self.likelihood(action, observation, key[0])
            for key, probability in belief.items()
        }
        total = sum(unnormalised.values())
        if total <= 0.0:
            raise ValueError(f"observation {observation!r} has zero likelihood")
        return {key: value / total for key, value in unnormalised.items()}

    def step(
        self,
        belief: Belief,
        action: str,
        observation: str,
        primitive_steps: int = 1,
    ) -> Belief:
        """Advance through *primitive_steps*, then condition on the outcome."""

        return self.observe(
            self.advance(belief, primitive_steps), action, observation
        )
