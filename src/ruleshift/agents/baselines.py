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

    def __init__(
        self,
        public_seed: str,
        *,
        actions: tuple[str, ...] | None = None,
    ) -> None:
        self._rng = random.Random(public_seed)
        self._actions = tuple(actions) if actions is not None else None

    def choose(self, observation: PublicObservation) -> str:
        # By default respect the advertised affordances. Callers may pass an
        # explicit full menu to stress the environment's cross-family clamp.
        menu = self._actions if self._actions is not None else observation.available_actions
        return self._rng.choice(menu)
