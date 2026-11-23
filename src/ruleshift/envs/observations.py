"""Rendering, public text view, and the public/evaluator data split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ruleshift.public_api.interface import CandidateRule, PublicObservation
from ruleshift.public_api.schemas import MACRO_ACTIONS

from .macro import Coord, Geometry

CELL_PIXELS = 16

RULE_DESCRIPTIONS = {
    "red": "The gate currently accepts a red card.",
    "blue": "The gate currently accepts a blue card.",
    "green": "The gate currently accepts a green card.",
}


@dataclass(frozen=True)
class RenderState:
    """Only public quantities needed by the renderer and public text view."""

    position: Coord
    gate_locked: bool
    deliveries: int
    steps: int


def digest(payload: Any) -> str:
    """Stable short digest of a JSON-serializable public payload."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:16]


def render_rgb(geometry: Geometry, state: RenderState) -> np.ndarray:
    """Render a local RGB frame whose appearance never encodes the mechanic."""

    width, height = geometry.grid
    image = np.full((height * CELL_PIXELS, width * CELL_PIXELS, 3), 40, dtype=np.uint8)

    def paint(cell: Coord, color: tuple[int, int, int]) -> None:
        x, y = cell
        if 0 <= x < width and 0 <= y < height:
            image[
                y * CELL_PIXELS : (y + 1) * CELL_PIXELS,
                x * CELL_PIXELS : (x + 1) * CELL_PIXELS,
            ] = color

    paint(geometry.gate, (150, 150, 150) if not state.gate_locked else (90, 90, 90))
    paint(geometry.station, (60, 110, 170))
    paint(state.position, (230, 220, 80))
    return image


def build_public_text(
    geometry: Geometry, state: RenderState, last_outcome: str | None
) -> str:
    """Describe exactly the publicly visible content of a frame."""

    return (
        f"step={state.steps} pos=({state.position[0]},{state.position[1]}) "
        f"card=red,blue,green gate={'locked' if state.gate_locked else 'open'} "
        f"deliveries={state.deliveries} station=({geometry.station[0]},"
        f"{geometry.station[1]}) last={last_outcome or 'none'}"
    )


def build_public_observation(
    *,
    geometry: Geometry,
    state: RenderState,
    history: Sequence[tuple[str, str | None, str]],
    last_outcome: str | None,
    candidate_rules: tuple[str | CandidateRule, ...],
    public_costs: dict[str, float],
    available_actions: tuple[str, ...] = MACRO_ACTIONS,
) -> PublicObservation:
    """Assemble the canonical agent-visible observation object."""

    state_payload = {
        "position": list(state.position),
        "gate_locked": state.gate_locked,
        "deliveries": state.deliveries,
        "steps": state.steps,
    }

    history_steps = []
    for index, (action, detail, outcome) in enumerate(history, start=1):
        token = f"{action.upper()}"
        if detail:
            token = f"{token} {detail.upper()}"
        history_steps.append(f"Step {index}: {token} -> {outcome or 'none'}")
    public_history = "; ".join(history_steps) if history_steps else "no prior steps"

    return PublicObservation(
        observation_id=f"obs-{state.steps}",
        public_obs_hash=digest(state_payload),
        public_history_hash=digest(list(history)),
        public_history=public_history,
        public_text=build_public_text(geometry, state, last_outcome),
        available_actions=tuple(available_actions),
        candidate_rules=tuple(
            rule
            if isinstance(rule, CandidateRule)
            else CandidateRule(rule_id=rule, description=RULE_DESCRIPTIONS[rule])
            for rule in candidate_rules
        ),
        public_costs=public_costs,
    )
