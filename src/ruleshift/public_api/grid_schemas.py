"""Dependency-free schemas and validators for the RuleShift-Grid contracts.

This module is additive: the Family A/B contracts in :mod:`public_api.schemas`
and :mod:`public_api.interface` stay untouched. Grid actions therefore live in
their own constants and the grid has its own validators, so a Family A/B
response can never legally emit a grid action such as ``"UP"``.

As in ``schemas.py``, every validator returns a list of human-readable error
strings; an empty list means the payload is valid. Bad payloads never raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ruleshift.permissions import find_forbidden_fields

from .interface import CandidateRule

#: Grid movement and interaction actions. Disjoint from ``MACRO_ACTIONS``.
GRID_PHYSICAL_ACTIONS: tuple[str, ...] = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "INTERACT",
)

#: Epistemic tool actions. Only ``INSPECT`` takes a grid coordinate.
GRID_TOOL_ACTIONS: tuple[str, ...] = ("INSPECT", "READ_SIGN")

GRID_ACTIONS: tuple[str, ...] = GRID_PHYSICAL_ACTIONS + GRID_TOOL_ACTIONS
GRID_ACTIONS_FROZEN: frozenset[str] = frozenset(GRID_ACTIONS)

#: Public rule hypotheses, in the same order as Task 2's ``GRID_HYPOTHESES``.
#: Task 2 carries the cross-check test that keeps the two in lockstep.
GRID_RULE_IDS: tuple[str, ...] = (
    "base",
    "remap_swap",
    "remap_rotate",
    "slippery",
    "hazard_goal_swap",
    "sign_composition",
)

#: Structure of the grid VLM response. ``predicted_outcomes`` is optional.
GRID_VLM_OUTPUT_FIELDS: frozenset[str] = frozenset(
    {
        "hypothesis_probabilities",
        "action",
        "tool_target",
        "predicted_outcomes",
        "confidence",
    }
)

#: Exact keys returned by ``GridPublicObservation.to_payload()``. The raw
#: ``image`` is deliberately excluded: only its digest may cross the boundary.
GRID_PUBLIC_OBSERVATION_FIELDS: frozenset[str] = frozenset(
    {
        "observation_id",
        "public_obs_hash",
        "public_history_hash",
        "public_history",
        "public_text",
        "available_actions",
        "candidate_rules",
        "public_costs",
        "public_map",
        "agent_position",
        "health",
        "inventory",
        "door_open",
        "steps",
        "action_mask",
        "tool_budget_remaining",
        "image_hash",
    }
)


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        # e.g. a Python int too large to convert to a float (10 ** 400).
        return False


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_coord(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_is_int(part) for part in value)
    )


def _is_grid_action(value: Any) -> bool:
    """Hash-safe membership test: unhashable payload values must not raise."""

    return isinstance(value, str) and value in GRID_ACTIONS_FROZEN


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


@dataclass(frozen=True)
class GridAction:
    """One grid decision: a legal action kind plus an optional tool target."""

    kind: str
    target: tuple[int, int] | None = None


@dataclass(frozen=True)
class GridPublicObservation:
    """Agent-visible grid observation, including the optional image digest.

    ``public_map`` is typed as ``Any`` to avoid a circular import with
    ``grid_rules.PublicGridMap``; it is serialized via its ``to_payload()``
    method. The raw ``image`` never appears in a public payload.
    """

    observation_id: str
    public_obs_hash: str
    public_history_hash: str
    public_history: str
    public_text: str
    available_actions: tuple[str, ...]
    candidate_rules: tuple[CandidateRule, ...]
    public_costs: Mapping[str, float]
    public_map: Any
    agent_position: tuple[int, int]
    health: int
    inventory: tuple[str, ...]
    door_open: bool
    steps: int
    action_mask: tuple[str, ...]
    tool_budget_remaining: int
    image: Any | None = None
    image_hash: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the canonical agent-visible grid payload shape."""

        payload: dict[str, Any] = {
            "observation_id": self.observation_id,
            "public_obs_hash": self.public_obs_hash,
            "public_history_hash": self.public_history_hash,
            "public_history": self.public_history,
            "public_text": self.public_text,
            "available_actions": list(self.available_actions),
            "candidate_rules": [
                {"rule_id": rule.rule_id, "description": rule.description}
                for rule in self.candidate_rules
            ],
            "public_costs": dict(self.public_costs),
            "public_map": self.public_map.to_payload(),
            "agent_position": list(self.agent_position),
            "health": self.health,
            "inventory": list(self.inventory),
            "door_open": self.door_open,
            "steps": self.steps,
            "action_mask": list(self.action_mask),
            "tool_budget_remaining": self.tool_budget_remaining,
            "image_hash": self.image_hash,
        }
        assert set(payload) == set(GRID_PUBLIC_OBSERVATION_FIELDS)
        return payload


@dataclass(frozen=True)
class GridDecisionRecord:
    """One row of the grid public decision log.

    The first twenty fields mirror ``DecisionRecord``; the remaining six are
    grid-specific. Privileged trace columns live in a separate evaluator-only
    adapter, never here.
    """

    run_id: str
    scenario_id: str
    agent_method: str
    episode_id: str
    decision_id: str
    public_obs_hash: str
    public_history_hash: str
    available_actions: tuple[str, ...]
    chosen_action: str
    physical_steps_elapsed: int
    actual_reward: float
    vlm_calls_used: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    json_valid: bool
    invalid_action: bool
    model_id: str
    prompt_hash: str
    run_seed: int
    tool_calls_used: int
    tool_cost: float
    advisor_calls_used: int
    information_gain_bits: float
    env_steps_elapsed: int
    success: bool

    def to_row(self) -> dict[str, Any]:
        """Return a JSON-serializable public grid log row."""

        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "agent_method": self.agent_method,
            "episode_id": self.episode_id,
            "decision_id": self.decision_id,
            "public_obs_hash": self.public_obs_hash,
            "public_history_hash": self.public_history_hash,
            "available_actions": list(self.available_actions),
            "chosen_action": self.chosen_action,
            "physical_steps_elapsed": self.physical_steps_elapsed,
            "actual_reward": self.actual_reward,
            "vlm_calls_used": self.vlm_calls_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "json_valid": self.json_valid,
            "invalid_action": self.invalid_action,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "run_seed": self.run_seed,
            "tool_calls_used": self.tool_calls_used,
            "tool_cost": self.tool_cost,
            "advisor_calls_used": self.advisor_calls_used,
            "information_gain_bits": self.information_gain_bits,
            "env_steps_elapsed": self.env_steps_elapsed,
            "success": self.success,
        }


def validate_grid_action(action: Any, action_mask: Any, grid_size: Any) -> list[str]:
    """Check one grid action against the current mask and grid dimensions."""

    if not isinstance(action, GridAction):
        return ["action must be a GridAction"]

    errors: list[str] = []
    kind = action.kind
    if not _is_grid_action(kind):
        errors.append(f"unknown grid action {kind!r}")
    if not _is_sequence(action_mask) or kind not in action_mask:
        errors.append("action is not in action_mask")

    if kind == "INSPECT":
        target = action.target
        if not _is_coord(target):
            errors.append("INSPECT requires a two-integer target")
        elif not (_is_coord(grid_size) and min(grid_size) > 0):
            errors.append("grid_size must be a positive (width, height) pair")
        elif not (0 <= target[0] < grid_size[0] and 0 <= target[1] < grid_size[1]):
            errors.append("INSPECT target is outside the grid")
    elif action.target is not None:
        errors.append("target is only valid for INSPECT")

    return errors


def validate_grid_vlm_output(payload: Any) -> list[str]:
    """Validate the structured grid VLM output, including the no-leak check."""

    if not isinstance(payload, dict):
        return ["grid VLM output must be a JSON object"]

    errors: list[str] = []
    unknown = set(payload) - GRID_VLM_OUTPUT_FIELDS
    if unknown:
        errors.append(f"unknown grid VLM output fields: {sorted(unknown)}")

    for location, name in find_forbidden_fields(payload):
        errors.append(f"privileged field {name!r} leaked at {location}")

    if "hypothesis_probabilities" not in payload:
        errors.append("hypothesis_probabilities is required")
    else:
        probabilities = payload["hypothesis_probabilities"]
        if not isinstance(probabilities, dict):
            errors.append("hypothesis_probabilities must be an object")
        else:
            unknown = set(probabilities) - set(GRID_RULE_IDS)
            if unknown:
                errors.append(
                    "hypothesis_probabilities has unknown ids: "
                    + ", ".join(sorted(map(str, unknown)))
                )
            numeric: list[float] = []
            for rule_id, value in probabilities.items():
                if not _is_number(value):
                    errors.append(
                        f"hypothesis_probabilities[{rule_id!r}] must be a finite number"
                    )
                    continue
                numeric.append(float(value))
                if not 0.0 <= float(value) <= 1.0:
                    errors.append(
                        f"hypothesis_probabilities[{rule_id!r}] must lie in [0, 1]"
                    )
            if numeric and sum(numeric) <= 0.0:
                errors.append("hypothesis_probabilities needs at least one positive value")
            # The frozen VLM may return a subset of the menu and rounded sums.
            # Accept those as public diagnostics instead of forcing a fallback
            # action; calibration metrics read the reported values as-is.


    action = payload.get("action")
    if not _is_grid_action(action):
        errors.append("action is not a valid grid action")

    tool_target = payload.get("tool_target")
    if tool_target is not None and action != "INSPECT":
        errors.append("tool_target is only valid for INSPECT")
    elif tool_target is not None and not _is_coord(tool_target):
        errors.append("tool_target must be a two-integer pair")
    elif action == "INSPECT" and tool_target is None:
        errors.append("tool_target is required for INSPECT")

    if "confidence" in payload:
        confidence = payload["confidence"]
        if not _is_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
            errors.append("confidence must be a finite number in [0, 1]")

    return errors


def validate_grid_public_observation(payload: Any) -> list[str]:
    """Validate an agent-visible grid observation and its no-leak property."""

    if not isinstance(payload, dict):
        return ["grid public observation must be a JSON object"]

    errors: list[str] = []
    if set(payload) != set(GRID_PUBLIC_OBSERVATION_FIELDS):
        errors.append(
            "grid public observation keys must be exactly "
            f"{sorted(GRID_PUBLIC_OBSERVATION_FIELDS)}"
        )

    for location, name in find_forbidden_fields(payload):
        errors.append(f"privileged field {name!r} leaked at {location}")

    for key in ("observation_id", "public_obs_hash", "public_history_hash"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            errors.append(f"{key} must be a non-empty string")
    for key in ("public_history", "public_text", "image_hash"):
        if not isinstance(payload.get(key), str):
            errors.append(f"{key} must be a string")

    if not isinstance(payload.get("public_map"), dict):
        errors.append("public_map must be a serialized object")

    available = payload.get("available_actions")
    if not _is_sequence(available) or not available:
        errors.append("available_actions must be a non-empty array")
    else:
        if any(not _is_grid_action(action) for action in available):
            errors.append("available_actions contains an unknown grid action")
        # Only string entries can be hashed safely; unhashable entries are
        # already reported above and must not make the duplicate check raise.
        string_actions = [action for action in available if isinstance(action, str)]
        if len(set(string_actions)) != len(string_actions):
            errors.append("available_actions contains duplicates")

    mask = payload.get("action_mask")
    if not _is_sequence(mask):
        errors.append("action_mask must be an array")
    elif any(not _is_grid_action(action) for action in mask):
        errors.append("action_mask contains an unknown grid action")

    rules = payload.get("candidate_rules")
    if not _is_sequence(rules) or not rules:
        errors.append("candidate_rules must be a non-empty array")
    else:
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict) or set(rule) != {"rule_id", "description"}:
                errors.append(f"candidate_rules[{index}] must have rule_id/description")
                continue
            if rule["rule_id"] not in GRID_RULE_IDS:
                errors.append(f"candidate_rules[{index}].rule_id is unknown")
            if not isinstance(rule["description"], str) or not rule["description"]:
                errors.append(f"candidate_rules[{index}].description must be a string")

    costs = payload.get("public_costs")
    if not isinstance(costs, dict):
        errors.append("public_costs must be an object")
    else:
        for key, value in costs.items():
            # Grid step costs are negative by design (config.step_cost = -0.05).
            if not _is_number(value):
                errors.append(f"public_costs[{key!r}] must be a finite number")

    position = payload.get("agent_position")
    if not _is_coord(position) or min(position) < 0:
        errors.append("agent_position must be a pair of non-negative integers")

    inventory = payload.get("inventory")
    if not _is_sequence(inventory) or any(
        not isinstance(item, str) for item in inventory
    ):
        errors.append("inventory must be an array of strings")

    for key in ("health", "steps", "tool_budget_remaining"):
        value = payload.get(key)
        if not _is_int(value) or value < 0:
            errors.append(f"{key} must be a non-negative integer")

    if not isinstance(payload.get("door_open"), bool):
        errors.append("door_open must be a boolean")

    return errors


__all__ = [
    "GRID_ACTIONS",
    "GRID_ACTIONS_FROZEN",
    "GRID_PHYSICAL_ACTIONS",
    "GRID_PUBLIC_OBSERVATION_FIELDS",
    "GRID_RULE_IDS",
    "GRID_TOOL_ACTIONS",
    "GRID_VLM_OUTPUT_FIELDS",
    "GridAction",
    "GridDecisionRecord",
    "GridPublicObservation",
    "validate_grid_action",
    "validate_grid_public_observation",
    "validate_grid_vlm_output",
]
