"""Semantic draw helpers for the grid simulator.

Each helper is a thin, named wrapper around :class:`SemanticRNG` so that the
event type of every stochastic draw is declared at the call site. The helpers
are stateless: a draw depends only on the RNG, the primitive step and the
attempt index, never on how many other draws happened before.
"""

from __future__ import annotations

from ruleshift.envs.events import SemanticRNG


def draw_move(rng: SemanticRNG, step: int, attempt: int = 0) -> float:
    """Uniform draw selecting a movement outcome at ``step``."""

    return rng.draw("move", step, attempt)


def draw_shift(rng: SemanticRNG, step: int) -> float:
    """Uniform draw for a (stochastic) rule shift at ``step``."""

    return rng.draw("shift", step)


def draw_inspect(rng: SemanticRNG, step: int, attempt: int = 0) -> float:
    """Uniform draw for the INSPECT tool at ``step``."""

    return rng.draw("inspect", step, attempt)


def draw_sign(rng: SemanticRNG, step: int, attempt: int = 0) -> float:
    """Uniform draw for the READ_SIGN tool at ``step``."""

    return rng.draw("sign", step, attempt)


__all__ = ["draw_move", "draw_shift", "draw_inspect", "draw_sign"]
