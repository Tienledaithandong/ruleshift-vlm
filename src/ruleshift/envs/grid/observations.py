"""Assemble the agent-visible RuleShift-Grid observation.

Everything emitted here is derived from public map geometry, public episode
state and the public hypothesis menu. The active rule, the shift schedule and
every noise draw stay exclusively in ``GridEnv.evaluator_state()``.
"""

from __future__ import annotations

import copy
import math
from numbers import Integral
from typing import TYPE_CHECKING, Any, Sequence

from gymnasium import spaces

from ruleshift.envs.observations import digest
from ruleshift.public_api.grid_rules import RULE_BY_ID, GridRule
from ruleshift.public_api.grid_schemas import GRID_ACTIONS, GridPublicObservation
from ruleshift.public_api.interface import CandidateRule

from .render import image_digest

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .grid_env import GridState


_ASCII_CHARS = frozenset(chr(code) for code in range(32, 127))


class PublicPayloadSpace(spaces.Space):
    """Last-resort space for the two fields gymnasium cannot express cleanly.

    Only ``public_map`` (nested dicts with optional ``None`` cells) and
    ``candidate_rules`` (a list of structured dicts) use this. Both are already
    validated by ``validate_grid_public_observation``; this space keeps them in
    the declared shape. ``sample_value`` gives ``observation_space.sample()``
    a benign, non-``None`` placeholder instead of a dict of ``None``s.
    """

    def __init__(self, sample_value: Any = None) -> None:
        super().__init__()
        self.sample_value = sample_value

    def contains(self, x: Any) -> bool:
        return True

    def sample(self) -> Any:
        return copy.deepcopy(self.sample_value)


class PublicText(spaces.Space):
    """Bounded printable-ASCII string; accepts punctuation and empty text.

    Gymnasium's ``Text`` uses a restricted character set and requires a
    non-empty string, so it rejects the canonical ``public_text`` and the empty
    symbolic ``image_hash``.
    """

    def __init__(self, max_length: int = 4096, min_length: int = 0) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.min_length = int(min_length)

    def contains(self, x: Any) -> bool:
        return (
            isinstance(x, str)
            and self.min_length <= len(x) <= self.max_length
            and all(char in _ASCII_CHARS for char in x)
        )

    def sample(self) -> str:
        return "public"


class PublicLabelList(spaces.Space):
    """Bounded list of short printable-ASCII labels (action lists, inventory).

    Gymnasium's ``Sequence.contains`` only accepts tuples, while the payload
    owns plain lists; this space accepts both and bounds their contents.
    """

    def __init__(self, max_items: int = 64, max_label_length: int = 32) -> None:
        super().__init__()
        self.max_items = int(max_items)
        self.max_label_length = int(max_label_length)

    def contains(self, x: Any) -> bool:
        return (
            isinstance(x, (list, tuple))
            and len(x) <= self.max_items
            and all(
                isinstance(item, str)
                and 0 < len(item) <= self.max_label_length
                and all(char in _ASCII_CHARS for char in item)
                for item in x
            )
        )

    def sample(self) -> list[str]:
        return []


class PublicCoord(spaces.Space):
    """An in-bounds ``(x, y)`` pair held as a plain list or tuple."""

    def __init__(self, max_x: int, max_y: int) -> None:
        super().__init__()
        self.max_x = int(max_x)
        self.max_y = int(max_y)

    def contains(self, x: Any) -> bool:
        return (
            isinstance(x, (list, tuple))
            and len(x) == 2
            and all(isinstance(value, Integral) and not isinstance(value, bool) for value in x)
            and 0 <= int(x[0]) <= self.max_x
            and 0 <= int(x[1]) <= self.max_y
        )

    def sample(self) -> list[int]:
        return [0, 0]


class PublicScalar(spaces.Space):
    """A finite numeric cost held as a plain Python ``int`` or ``float``."""

    def __init__(self, low: float = -1e9, high: float = 1e9) -> None:
        super().__init__()
        self.low = float(low)
        self.high = float(high)

    def contains(self, x: Any) -> bool:
        return (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and math.isfinite(x)
            and self.low <= x <= self.high
        )

    def sample(self) -> float:
        return 0.0


def _as_candidate_rules(rules: Sequence[Any]) -> tuple[CandidateRule, ...]:
    """Normalize the public hypothesis menu into ``CandidateRule`` rows."""

    normalized: list[CandidateRule] = []
    for rule in rules:
        if isinstance(rule, CandidateRule):
            normalized.append(rule)
        elif isinstance(rule, str):
            normalized.append(
                CandidateRule(rule_id=rule, description=RULE_BY_ID[rule].description)
            )
        elif isinstance(rule, GridRule):
            normalized.append(
                CandidateRule(rule_id=rule.rule_id, description=rule.description)
            )
        else:
            raise TypeError(f"unsupported public rule entry: {rule!r}")
    return tuple(normalized)


def _format_history(
    history: Sequence[tuple[str, str | None, str]],
) -> str:
    lines = [
        f"Step {index}: {action.upper()}"
        + (f" {detail.upper()}" if detail else "")
        + f" -> {outcome or 'none'}"
        for index, (action, detail, outcome) in enumerate(history, 1)
    ]
    return "; ".join(lines) if lines else "no prior steps"


def _cell_text(coord: Any) -> str:
    return "none" if coord is None else f"({coord[0]},{coord[1]})"


def build_public_text(
    state: "GridState",
    public_map: Any,
    tool_budget_remaining: int,
    last_outcome: str,
) -> str:
    """Describe exactly the public, rule-independent content of a frame.

    Marker coordinates and the key/door/signpost cells are public geometry.
    The active rule, ``true_rule``, ``shift_time`` and signpost clauses are
    never included: their only agent-visible surface is the hypothesis menu
    and the epistemic tools.
    """

    markers = ", ".join(
        f"{color}=({coord[0]},{coord[1]})"
        for color, coord in sorted(public_map.markers.items())
    )
    return (
        f"step={state.steps} pos=({state.position[0]},{state.position[1]}) "
        f"health={state.health} tool={tool_budget_remaining} "
        f"inventory={','.join(state.inventory) or 'none'} "
        f"door={'open' if state.door_open else 'closed'} "
        f"last={last_outcome} markers=[{markers}] "
        f"key={_cell_text(public_map.key)} "
        f"door_cell={_cell_text(public_map.door)} "
        f"signpost={_cell_text(public_map.signpost)}"
    )


def build_grid_observation(
    public_map: Any,
    state: "GridState",
    history: Sequence[tuple[str, str | None, str]],
    last_outcome: str,
    candidate_rules: Sequence[Any],
    public_costs: Any,
    action_mask: Sequence[str],
    tool_budget_remaining: int,
    image: Any | None = None,
) -> GridPublicObservation:
    """Assemble one complete agent-visible grid observation."""

    public_state = {
        "position": list(state.position),
        "health": state.health,
        "inventory": list(state.inventory),
        "door_open": state.door_open,
        "steps": state.steps,
        "tool_budget": state.tool_budget,
        "last_outcome": last_outcome,
    }
    return GridPublicObservation(
        observation_id=f"grid-obs-{state.steps}",
        public_obs_hash=digest(
            {"state": public_state, "public_map": public_map.to_payload()}
        ),
        public_history_hash=digest(list(history)),
        public_history=_format_history(history),
        public_text=build_public_text(state, public_map, tool_budget_remaining, last_outcome),
        available_actions=tuple(GRID_ACTIONS),
        candidate_rules=_as_candidate_rules(candidate_rules),
        public_costs=dict(public_costs),
        public_map=public_map,
        agent_position=tuple(state.position),
        health=state.health,
        inventory=tuple(state.inventory),
        door_open=state.door_open,
        steps=state.steps,
        action_mask=tuple(action_mask),
        tool_budget_remaining=tool_budget_remaining,
        image=image,
        image_hash=image_digest(image) if image is not None else "",
    )


__all__ = [
    "PublicCoord",
    "PublicLabelList",
    "PublicPayloadSpace",
    "PublicScalar",
    "PublicText",
    "build_grid_observation",
    "build_public_text",
]
