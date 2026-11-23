"""Assemble the agent-visible RuleShift-Grid observation.

Everything emitted here is derived from public map geometry, public episode
state and the public hypothesis menu. The active rule, the shift schedule and
every noise draw stay exclusively in ``GridEnv.evaluator_state()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from gymnasium import spaces

from ruleshift.envs.observations import digest
from ruleshift.public_api.grid_rules import RULE_BY_ID, GridRule
from ruleshift.public_api.grid_schemas import GRID_ACTIONS, GridPublicObservation
from ruleshift.public_api.interface import CandidateRule

from .render import image_digest

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .grid_env import GridState


class PublicPayloadSpace(spaces.Space):
    """Permissive gym space for a public payload field.

    The canonical ``GridPublicObservation.to_payload()`` uses plain Python
    lists (not numpy arrays), optional ``None`` cells and an empty image hash
    in symbolic mode. Gymnasium's ``Sequence`` rejects lists, ``Text`` rejects
    the empty string and ``Box`` emits a cast warning for those values, so a
    value space would report a false mismatch. This space documents that the
    field is part of the public payload shape while accepting every payload
    value the grid observation legitimately produces.
    """

    def __init__(self) -> None:
        super().__init__()

    def contains(self, x: Any) -> bool:
        return True

    def sample(self) -> None:  # pragma: no cover - never used by the harness
        return None


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
    "PublicPayloadSpace",
    "build_grid_observation",
    "build_public_text",
]
