"""Public-only, non-VLM agent baselines.

These policies receive only :class:`PublicObservation` objects and public
outcomes. They import nothing from ``envs`` or ``evaluator``. The privileged
references live in the evaluator package instead.
"""

from __future__ import annotations

import random

from ruleshift.public_api.interface import PublicObservation
from ruleshift.public_api.schemas import COMMIT_ACTIONS, MACRO_ACTIONS


class NeverProbePolicy:
    """Cycle task colours on public rejections; never emit a probe macro."""

    def __init__(self) -> None:
        self._index = 0

    def choose(self, observation: PublicObservation) -> str:
        return COMMIT_ACTIONS[self._index % len(COMMIT_ACTIONS)]

    def record(self, action: str, outcome: str) -> None:
        if action.startswith("COMMIT_") and outcome == "reject":
            self._index = (self._index + 1) % len(COMMIT_ACTIONS)


class RandomProbePolicy:
    """Uniform macro choice seeded by the public history digest."""

    def __init__(self, public_seed: str) -> None:
        self._rng = random.Random(public_seed)

    def choose(self, observation: PublicObservation) -> str:
        return self._rng.choice(MACRO_ACTIONS)
