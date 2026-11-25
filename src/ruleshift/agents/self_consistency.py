"""Self-consistency over multiple samples with matched call accounting."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Sequence

from ruleshift.public_api.interface import PublicObservation

from .cache import ResponseCache
from .structured_vlm import run_structured_prediction
from .value_selector import Belief, PublicCostModel


def majority_vote(actions: Sequence[str]) -> str:
    """First-occurrence tie break so aggregation is deterministic."""

    if not actions:
        return "WAIT"
    return Counter(actions).most_common(1)[0][0]


@dataclass(frozen=True)
class SelfConsistencySelector:
    """Sample N structured predictions; each extra sample costs one call."""

    backend: object
    base_selector: object
    n_samples: int
    cache: ResponseCache

    def choose(
        self,
        observation: PublicObservation,
        belief: Belief,
        cost: PublicCostModel,
    ) -> str:
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        choices: list[str] = []
        for sample in range(self.n_samples):
            payload, _, errors = run_structured_prediction(
                observation,
                self.backend,
                self.cache,
                decoding_seed=100 + sample,
                do_sample=True,
            )
            if payload is None or errors:
                continue
            choices.append(self.base_selector.choose(payload, belief, cost))
        return majority_vote(choices)
