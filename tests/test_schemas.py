"""Tests for the Week 1 public schemas.

These are contract tests only. They do not require Gymnasium, numpy, a VLM, or
an environment implementation.
"""

from __future__ import annotations

import copy

from ruleshift.permissions import PRIVILEGED_FIELDS
from ruleshift.public_api import schemas
from ruleshift.public_api.interface import CandidateRule, PublicObservation


def valid_vlm_output() -> dict:
    """A schema-valid (but scientifically meaningless) example payload."""

    return {
        "candidate_probabilities": {"red": 0.5, "blue": 0.3, "green": 0.2},
        "action_predictions": [
            {
                "action_id": "PROBE_BLUE",
                "under_rules": [
                    {
                        "rule_id": "red",
                        "possible_outcomes": [
                            {"observation": "reject", "p": 0.95},
                            {"observation": "accept", "p": 0.05},
                        ],
                    },
                    {
                        "rule_id": "blue",
                        "possible_outcomes": [
                            {"observation": "accept", "p": 0.90},
                            {"observation": "reject", "p": 0.10},
                        ],
                    },
                ],
                "estimated_future_best_action": "COMMIT_BLUE",
            }
        ],
        "selected_action": "PROBE_BLUE",
    }


def valid_public_observation() -> dict:
    return {
        "observation_id": "obs_0001",
        "public_obs_hash": "sha256:aaa",
        "public_history_hash": "sha256:bbb",
        "public_text": "Gate ahead; one red card in inventory.",
        "available_actions": ["COMMIT_RED", "PROBE_RED", "WAIT"],
        "candidate_rules": [
            {"rule_id": "red", "description": "Gate accepts a red card."},
            {"rule_id": "blue", "description": "Gate accepts a blue card."},
            {"rule_id": "green", "description": "Gate accepts a green card."},
        ],
        "public_costs": {
            "primitive_step": 0.05,
            "probe_test": 1.0,
            "probe_detour": 2.0,
            "lockout": 3.0,
        },
    }


def test_action_menu_is_stable_and_categories_are_disjoint() -> None:
    assert set(schemas.COMMIT_ACTIONS).isdisjoint(schemas.PROBE_ACTIONS)
    assert set(schemas.FAMILY_A_ACTIONS) == set(schemas.COMMIT_ACTIONS) | set(
        schemas.PROBE_ACTIONS
    ) | {"WAIT", "GO_TO_STATION"}
    assert set(schemas.FAMILY_B_ACTIONS) <= set(schemas.MACRO_ACTIONS)
    assert len(set(schemas.MACRO_ACTIONS)) == len(schemas.MACRO_ACTIONS)


def test_valid_structured_output_passes() -> None:
    assert schemas.validate_structured_vlm_output(valid_vlm_output()) == []


def test_valid_public_observation_passes() -> None:
    assert schemas.validate_public_observation(valid_public_observation()) == []


def test_public_observation_dataclass_serializes_to_valid_payload() -> None:
    observation = PublicObservation(
        observation_id="obs_0002",
        public_obs_hash="sha256:ccc",
        public_history_hash="sha256:ddd",
        public_text="Gate ahead.",
        available_actions=("COMMIT_RED", "PROBE_RED"),
        candidate_rules=(
            CandidateRule("red", "Gate accepts a red card."),
            CandidateRule("blue", "Gate accepts a blue card."),
            CandidateRule("green", "Gate accepts a green card."),
        ),
        public_costs={"primitive_step": 0.05},
    )
    assert schemas.validate_public_observation(observation.to_payload()) == []


def test_probabilities_must_sum_to_one() -> None:
    payload = valid_vlm_output()
    payload["candidate_probabilities"]["green"] = 0.5
    errors = schemas.validate_structured_vlm_output(payload)
    assert any("sum to 1" in error for error in errors)


def test_probabilities_reject_unknown_rule_and_boolean() -> None:
    payload = valid_vlm_output()
    payload["candidate_probabilities"] = {"red": 0.5, "blue": 0.5, "purple": 0.0}
    errors = schemas.validate_structured_vlm_output(payload)
    assert any("keys must be exactly" in error for error in errors)

    payload = valid_vlm_output()
    payload["candidate_probabilities"]["red"] = True
    errors = schemas.validate_structured_vlm_output(payload)
    assert any("finite number" in error for error in errors)


def test_invalid_macro_ids_are_rejected() -> None:
    payload = valid_vlm_output()
    payload["selected_action"] = "TELEPORT"
    errors = schemas.validate_structured_vlm_output(payload)
    assert any("selected_action" in error for error in errors)

    payload = valid_vlm_output()
    payload["action_predictions"][0]["estimated_future_best_action"] = "CALL_ORACLE"
    errors = schemas.validate_structured_vlm_output(payload)
    assert any("estimated_future_best_action" in error for error in errors)


def test_outcome_probabilities_must_sum_to_one() -> None:
    payload = valid_vlm_output()
    outcome = payload["action_predictions"][0]["under_rules"][0]["possible_outcomes"][0]
    outcome["p"] = 0.5
    errors = schemas.validate_structured_vlm_output(payload)
    assert any("sum to 1" in error for error in errors)


def test_unknown_feedback_outcome_is_rejected() -> None:
    payload = valid_vlm_output()
    outcome = payload["action_predictions"][0]["under_rules"][0]["possible_outcomes"][0]
    outcome["observation"] = "maybe"
    errors = schemas.validate_structured_vlm_output(payload)
    assert any("observation" in error for error in errors)


def test_privileged_field_is_rejected_in_vlm_output() -> None:
    payload = valid_vlm_output()
    payload["true_rule"] = "blue"  # leaking the hidden mechanic
    errors = schemas.validate_structured_vlm_output(payload)
    assert any("true_rule" in error for error in errors)


def test_privileged_field_is_rejected_in_nested_string() -> None:
    payload = valid_public_observation()
    payload["public_text"] = "Reminder: check oracle_belief before acting."
    errors = schemas.validate_public_observation(payload)
    assert any("oracle_belief" in error for error in errors)


def test_privileged_field_is_rejected_in_public_observation() -> None:
    payload = valid_public_observation()
    payload["shift_time"] = 17
    errors = schemas.validate_public_observation(payload)
    assert any("shift_time" in error for error in errors)


def test_validator_does_not_mutate_its_input() -> None:
    payload = valid_vlm_output()
    expected = copy.deepcopy(payload)
    schemas.validate_structured_vlm_output(payload)
    assert payload == expected


def test_schema_declaration_contains_no_privileged_field_names() -> None:
    def keys_of(value: object) -> set[str]:
        result: set[str] = set()
        if isinstance(value, dict):
            for key, nested in value.items():
                result.add(str(key))
                result |= keys_of(nested)
        elif isinstance(value, list):
            for nested in value:
                result |= keys_of(nested)
        return result

    assert keys_of(schemas.VLM_OUTPUT_SCHEMA).isdisjoint(PRIVILEGED_FIELDS)
