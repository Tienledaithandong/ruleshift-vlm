"""Public-only, non-VLM agent baselines.

These policies receive only :class:`PublicObservation` objects and public
outcomes. They import nothing from ``envs`` or ``evaluator``. The privileged
references live in the evaluator package instead.
"""

from __future__ import annotations

import random

from ruleshift.public_api.interface import PublicObservation
from ruleshift.public_api.schemas import TASK_ACTIONS


class NeverProbePolicy:
    """Cycle task colours on public rejections; never emit a probe macro."""

    def __init__(self) -> None:
        self._index = 0

    def choose(self, observation: PublicObservation) -> str:
        tasks = [a for a in observation.available_actions if a in TASK_ACTIONS]
        return tasks[self._index % len(tasks)] if tasks else "WAIT"

    def record(self, action: str, outcome: str) -> None:
        if action.startswith("COMMIT_") and outcome == "reject":
            self._index += 1


class RandomProbePolicy:
    """Uniform macro choice seeded by the public history digest."""

    def __init__(self, public_seed: str) -> None:
        self._rng = random.Random(public_seed)

    def choose(self, observation: PublicObservation) -> str:
        return self._rng.choice(observation.available_actions)
