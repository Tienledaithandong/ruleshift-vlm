"""Dependency-free schemas and validators for the Week 1 contracts.

The validators below are intentionally small and use only the standard
library. They are the executable form of the JSON schema shown in the master
plan (section 5.3); no pydantic/jsonschema dependency is needed to run the Week
1 test suite.

Every validator returns a list of human-readable error strings. An empty list
means the payload is valid. Returning errors instead of raising keeps the
checks easy to log in a CI or evaluation harness.
"""

from __future__ import annotations

import math
from typing import Any

from ruleshift.permissions import (
    PUBLIC_OBSERVATION_FIELDS,
    find_forbidden_fields,
)

RULE_IDS: tuple[str, ...] = ("red", "blue", "green")
HAZARD_REGIME_IDS: tuple[str, ...] = ("stable", "slippery")
KNOWN_RULE_IDS: frozenset[str] = frozenset((*RULE_IDS, *HAZARD_REGIME_IDS))

COMMIT_ACTIONS: tuple[str, ...] = (
    "COMMIT_RED",
    "COMMIT_BLUE",
    "COMMIT_GREEN",
)

PROBE_ACTIONS: tuple[str, ...] = (
    "PROBE_RED",
    "PROBE_BLUE",
    "PROBE_GREEN",
)

FAMILY_A_ACTIONS: tuple[str, ...] = (
    *COMMIT_ACTIONS,
    *PROBE_ACTIONS,
    "WAIT",
    "GO_TO_STATION",
)

FAMILY_B_ACTIONS: tuple[str, ...] = (
    "GO_SAFE",
    "CROSS_BRIDGE",
    "PROBE_STABLE",
    "PROBE_SLIPPERY",
    "WAIT",
)

# Union of all public macro ids. Each environment advertises only the subset
# that is legal for its family through PublicObservation.available_actions.
MACRO_ACTIONS: tuple[str, ...] = FAMILY_A_ACTIONS + tuple(
    action for action in FAMILY_B_ACTIONS if action not in FAMILY_A_ACTIONS
)

COMMIT_ACTIONS_FROZEN: frozenset[str] = frozenset(COMMIT_ACTIONS)
PROBE_ACTIONS_FROZEN: frozenset[str] = frozenset(PROBE_ACTIONS)
MACRO_ACTIONS_FROZEN: frozenset[str] = frozenset(MACRO_ACTIONS)
FAMILY_A_ACTIONS_FROZEN: frozenset[str] = frozenset(FAMILY_A_ACTIONS)
FAMILY_B_ACTIONS_FROZEN: frozenset[str] = frozenset(FAMILY_B_ACTIONS)

#: Feedback vocabulary shared by the gate station and the diagnostic probe.
FEEDBACK_OUTCOMES: frozenset[str] = frozenset(
    {"accept", "reject", "success", "failure", "none"}
)

#: Publicly described cost components (values are primitive-step equivalents
#: or reward units, never hidden transition probabilities).
PUBLIC_COST_KEYS: frozenset[str] = frozenset(
    {
        "primitive_step",
        "probe_test",
        "probe_detour",
        "commit_travel",
        "lockout",
        "delivery_reward",
        "safe_steps",
        "bridge_steps",
        "recovery_steps",
    }
)

VLM_OUTPUT_FIELDS: frozenset[str] = frozenset(
    {"candidate_probabilities", "action_predictions", "selected_action"}
)
PREDICTION_FIELDS: frozenset[str] = frozenset(
    {"action_id", "under_rules", "estimated_future_best_action"}
)
RULE_PREDICTION_FIELDS: frozenset[str] = frozenset({"rule_id", "possible_outcomes"})
OUTCOME_FIELDS: frozenset[str] = frozenset({"observation", "p"})

#: JSON-Schema-style declaration of the target structured VLM output. This is a
#: *schema example*, not a claim about correct predictions for any mechanic.
VLM_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RuleShiftStructuredVLMOutput",
    "type": "object",
    "additionalProperties": False,
    "required": sorted(VLM_OUTPUT_FIELDS),
    "properties": {
        "candidate_probabilities": {
            "type": "object",
            "propertyNames": {"enum": sorted(KNOWN_RULE_IDS)},
            "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "action_predictions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(PREDICTION_FIELDS),
                "properties": {
                    "action_id": {"type": "string", "enum": list(MACRO_ACTIONS)},
                    "under_rules": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": sorted(RULE_PREDICTION_FIELDS),
                            "properties": {
                                "rule_id": {"type": "string", "enum": list(RULE_IDS)},
                                "possible_outcomes": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": sorted(OUTCOME_FIELDS),
                                        "properties": {
                                            "observation": {
                                                "type": "string",
                                                "enum": sorted(FEEDBACK_OUTCOMES),
                                            },
                                            "p": {
                                                "type": "number",
                                                "minimum": 0,
                                                "maximum": 1,
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "estimated_future_best_action": {
                        "type": "string",
                        "enum": list(MACRO_ACTIONS),
                    },
                },
            },
        },
        "selected_action": {"type": "string", "enum": list(MACRO_ACTIONS)},
    },
}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sums_to_one(values: list[float], tolerance: float = 1e-6) -> bool:
    return abs(sum(values) - 1.0) <= tolerance


def validate_candidate_probabilities(mapping: Any) -> list[str]:
    """Validate ``b_hat(m)`` over the closed rule set."""

    errors: list[str] = []
    if not isinstance(mapping, dict):
        return ["candidate_probabilities must be an object"]

    if set(mapping) != set(RULE_IDS):
        errors.append(
            "candidate_probabilities keys must be exactly " + ", ".join(RULE_IDS)
        )

    numeric: list[float] = []
    for key, value in mapping.items():
        if not _is_number(value):
            errors.append(f"candidate_probabilities[{key!r}] must be a finite number")
            continue
        numeric.append(float(value))
        if not 0.0 <= float(value) <= 1.0:
            errors.append(f"candidate_probabilities[{key!r}] must lie in [0, 1]")

    if len(numeric) == len(mapping) and numeric and not _sums_to_one(numeric):
        errors.append("candidate_probabilities must sum to 1 within 1e-6")

    return errors


def validate_action_predictions(predictions: Any) -> list[str]:
    """Validate counterfactual outcome predictions for candidate macros."""

    errors: list[str] = []
    if not isinstance(predictions, list) or not predictions:
        return ["action_predictions must be a non-empty array"]

    seen_actions: set[str] = set()
    for index, prediction in enumerate(predictions):
        prefix = f"action_predictions[{index}]"
        if not isinstance(prediction, dict):
            errors.append(f"{prefix} must be an object")
            continue

        if set(prediction) != set(PREDICTION_FIELDS):
            errors.append(f"{prefix} keys must be exactly {sorted(PREDICTION_FIELDS)}")

        action_id = prediction.get("action_id")
        if action_id not in MACRO_ACTIONS_FROZEN:
            errors.append(f"{prefix}.action_id is not a valid macro action")
        elif action_id in seen_actions:
            errors.append(f"{prefix}.action_id is duplicated")
        else:
            seen_actions.add(action_id)

        future_action = prediction.get("estimated_future_best_action")
        if future_action not in MACRO_ACTIONS_FROZEN:
            errors.append(f"{prefix}.estimated_future_best_action is not a valid macro")

        under_rules = prediction.get("under_rules")
        if not isinstance(under_rules, list) or not under_rules:
            errors.append(f"{prefix}.under_rules must be a non-empty array")
            continue

        seen_rules: set[str] = set()
        for rule_index, rule_prediction in enumerate(under_rules):
            rule_prefix = f"{prefix}.under_rules[{rule_index}]"
            if not isinstance(rule_prediction, dict):
                errors.append(f"{rule_prefix} must be an object")
                continue
            if set(rule_prediction) != set(RULE_PREDICTION_FIELDS):
                errors.append(
                    f"{rule_prefix} keys must be exactly "
                    f"{sorted(RULE_PREDICTION_FIELDS)}"
                )

            rule_id = rule_prediction.get("rule_id")
            if rule_id not in KNOWN_RULE_IDS:
                errors.append(f"{rule_prefix}.rule_id is not a known rule id")
            elif rule_id in seen_rules:
                errors.append(f"{rule_prefix}.rule_id is duplicated")
            else:
                seen_rules.add(rule_id)

            outcomes = rule_prediction.get("possible_outcomes")
            if not isinstance(outcomes, list) or not outcomes:
                errors.append(f"{rule_prefix}.possible_outcomes must be non-empty")
                continue

            probabilities: list[float] = []
            for outcome_index, outcome in enumerate(outcomes):
                out_prefix = f"{rule_prefix}.possible_outcomes[{outcome_index}]"
                if not isinstance(outcome, dict):
                    errors.append(f"{out_prefix} must be an object")
                    continue
                if set(outcome) != set(OUTCOME_FIELDS):
                    errors.append(
                        f"{out_prefix} keys must be exactly {sorted(OUTCOME_FIELDS)}"
                    )
                if outcome.get("observation") not in FEEDBACK_OUTCOMES:
                    errors.append(f"{out_prefix}.observation is not in vocabulary")
                probability = outcome.get("p")
                if not _is_number(probability):
                    errors.append(f"{out_prefix}.p must be a finite number")
                    continue
                probabilities.append(float(probability))
                if not 0.0 <= float(probability) <= 1.0:
                    errors.append(f"{out_prefix}.p must lie in [0, 1]")

            if len(probabilities) == len(outcomes) and not _sums_to_one(probabilities):
                errors.append(f"{rule_prefix}.possible_outcomes p-values must sum to 1")

    return errors


def validate_structured_vlm_output(payload: Any) -> list[str]:
    """Validate the complete structured VLM output, including no-leak checks."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["VLM output must be a JSON object"]

    if set(payload) != set(VLM_OUTPUT_FIELDS):
        errors.append(f"VLM output keys must be exactly {sorted(VLM_OUTPUT_FIELDS)}")

    for location, name in find_forbidden_fields(payload):
        errors.append(f"privileged field {name!r} leaked at {location}")

    if payload.get("selected_action") not in MACRO_ACTIONS_FROZEN:
        errors.append("selected_action is not a valid macro action")

    if "candidate_probabilities" in payload:
        errors.extend(
            validate_candidate_probabilities(payload["candidate_probabilities"])
        )
    if "action_predictions" in payload:
        errors.extend(validate_action_predictions(payload["action_predictions"]))

    return errors


def validate_public_observation(payload: Any) -> list[str]:
    """Validate an agent-visible observation payload and its no-leak property."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["public observation must be a JSON object"]

    if set(payload) != set(PUBLIC_OBSERVATION_FIELDS):
        errors.append(
            "public observation keys must be exactly "
            f"{sorted(PUBLIC_OBSERVATION_FIELDS)}"
        )

    for location, name in find_forbidden_fields(payload):
        errors.append(f"privileged field {name!r} leaked at {location}")

    for key in ("observation_id", "public_obs_hash", "public_history_hash"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            errors.append(f"{key} must be a non-empty string")

    if not isinstance(payload.get("public_text"), str):
        errors.append("public_text must be a string")

    available = payload.get("available_actions")
    if not isinstance(available, list) or not available:
        errors.append("available_actions must be a non-empty array")
    else:
        if any(action not in MACRO_ACTIONS_FROZEN for action in available):
            errors.append("available_actions contains an unknown macro id")
        if len(set(available)) != len(available):
            errors.append("available_actions contains duplicates")

    rules = payload.get("candidate_rules")
    if not isinstance(rules, list) or not rules:
        errors.append("candidate_rules must be a non-empty array")
    else:
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict) or set(rule) != {"rule_id", "description"}:
                errors.append(f"candidate_rules[{index}] must have rule_id/description")
                continue
            if rule["rule_id"] not in KNOWN_RULE_IDS:
                errors.append(f"candidate_rules[{index}].rule_id is unknown")
            if not isinstance(rule["description"], str) or not rule["description"]:
                errors.append(f"candidate_rules[{index}].description must be a string")

    costs = payload.get("public_costs")
    if not isinstance(costs, dict):
        errors.append("public_costs must be an object")
    else:
        if any(key not in PUBLIC_COST_KEYS for key in costs):
            errors.append("public_costs contains an unknown cost key")
        for key, value in costs.items():
            if not _is_number(value) or float(value) < 0.0:
                errors.append(f"public_costs[{key!r}] must be a finite non-negative number")

    return errors
