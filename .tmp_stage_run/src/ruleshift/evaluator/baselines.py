"""Privileged or reference baselines (not agent-valid methods).

These baselines use the evaluator's belief, exact solver or true mechanic.
They are kept in the evaluator package so they cannot be imported by agent
code, and every privileged member is labelled as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ruleshift.evaluator.exact_solver import (
    ExactSolver,
    MicrotaskConfig,
    freeze_belief,
    macro_branches,
)

Belief = dict[tuple[str, bool], float]


def _rule_entropy(belief: Belief) -> float:
    marginal: dict[str, float] = {}
    for (rule, _), probability in belief.items():
        marginal[rule] = marginal.get(rule, 0.0) + probability
    return -sum(p * math.log(p) for p in marginal.values() if p > 0.0)


@dataclass(frozen=True)
class MaxInformationGainPolicy:
    """Choose the macro with the largest expected entropy reduction."""

    config: MicrotaskConfig
    actions: tuple[str, ...]

    def choose(self, belief: Belief) -> str:
        before = _rule_entropy(belief)
        best_action = self.actions[0]
        best_gain = float("-inf")
        for action in self.actions:
            expected_after = sum(
                branch.probability * _rule_entropy(branch.posterior)
                for branch in macro_branches(self.config, belief, action)
            )
            gain = before - expected_after
            if gain > best_gain:
                best_gain = gain
                best_action = action
        return best_action


@dataclass(frozen=True)
class ReturnAwareReference:
    """Exact-solver argmax; a reference label, not a deployable agent."""

    solver: ExactSolver

    def choose(self, belief: Belief, horizon: int) -> str:
        q_values = dict(self.solver.q_values(freeze_belief(belief), horizon))
        return max(q_values.items(), key=lambda item: item[1])[0]


@dataclass(frozen=True)
class TrueMechanicOracle:
    """Reads the true rule: a diagnostic upper bound, never a main result."""

    true_rule: str
    is_privileged: bool = True

    def choose(self) -> str:
        return f"COMMIT_{self.true_rule.upper()}"
