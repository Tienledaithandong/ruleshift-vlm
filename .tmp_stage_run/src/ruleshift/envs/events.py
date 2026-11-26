"""Semantic, consumption-order-independent random streams.

Every stochastic draw is a pure function of
``(master_seed, scenario_id, event_type, primitive_step, attempt_index)``.
Taking extra draws for one event type therefore cannot perturb another event,
which is what makes paired scenario comparisons meaningful.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SemanticRNG:
    """A stateless collection of keyed uniform streams."""

    master_seed: int
    scenario_id: str

    def _key(
        self, event_type: str, primitive_step: int, attempt_index: int
    ) -> str:
        return (
            f"{self.master_seed}|{self.scenario_id}|{event_type}|"
            f"{primitive_step}|{attempt_index}"
        )

    def seed_for(
        self, event_type: str, primitive_step: int, attempt_index: int = 0
    ) -> int:
        """Return a deterministic 64-bit seed for one semantic event."""

        digest = hashlib.blake2b(
            self._key(event_type, primitive_step, attempt_index).encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big")

    def draw(
        self, event_type: str, primitive_step: int, attempt_index: int = 0
    ) -> float:
        """Return a value in ``[0, 1)`` for one semantic event."""

        return self.seed_for(event_type, primitive_step, attempt_index) / 2**64

    def choice(
        self,
        event_type: str,
        primitive_step: int,
        attempt_index: int,
        options: Sequence[T],
    ) -> T:
        """Choose one option using the keyed uniform stream."""

        values = tuple(options)
        if not values:
            raise ValueError("choice() requires at least one option")
        index = min(int(self.draw(event_type, primitive_step, attempt_index) * len(values)), len(values) - 1)
        return values[index]
