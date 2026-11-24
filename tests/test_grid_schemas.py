"""Contract tests for the additive RuleShift-Grid public schemas.

The first five tests are the verbatim acceptance tests from Task 1's brief;
the rest pin down the remaining public contracts produced by the task.
"""

from dataclasses import dataclass

from ruleshift.public_api.grid_schemas import (
    GRID_ACTIONS,
    GRID_ACTIONS_FROZEN,
    GRID_PHYSICAL_ACTIONS,
    GRID_PUBLIC_OBSERVATION_FIELDS,
    GRID_RULE_IDS,
    GRID_TOOL_ACTIONS,
    GridAction,
    GridDecisionRecord,
    GridPublicObservation,
    validate_grid_action,
    validate_grid_public_observation,
    validate_grid_vlm_output,
)
from ruleshift.public_api.interface import CandidateRule
from ruleshift.public_api.schemas import MACRO_ACTIONS_FROZEN


# --- Verbatim acceptance tests from the brief -------------------------------


def test_grid_actions_are_disjoint_from_macro_actions():
    assert not (set(GRID_ACTIONS) & MACRO_ACTIONS_FROZEN)


def test_inspect_requires_a_target():
    errors = validate_grid_action(GridAction("INSPECT"), ("INSPECT",), (7, 7))
    assert errors


def test_inspect_accepts_an_in_range_target():
    errors = validate_grid_action(GridAction("INSPECT", (3, 4)), ("INSPECT",), (7, 7))
    assert errors == []


def test_valid_grid_vlm_output():
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "UP",
        "tool_target": None,
        "confidence": 0.5,
    }
    assert validate_grid_vlm_output(payload) == []


def test_grid_vlm_output_rejects_privileged_field():
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "UP",
        "shift_time": 3,
    }
    errors = validate_grid_vlm_output(payload)
    assert any("privileged" in error for error in errors)


# --- Constant and action validation contracts -------------------------------


def test_grid_constants_match_the_spec():
    assert GRID_PHYSICAL_ACTIONS == ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT")
    assert GRID_TOOL_ACTIONS == ("INSPECT", "READ_SIGN")
    assert GRID_ACTIONS == GRID_PHYSICAL_ACTIONS + GRID_TOOL_ACTIONS
    assert GRID_ACTIONS_FROZEN == frozenset(GRID_ACTIONS)
    assert GRID_RULE_IDS == (
        "base",
        "remap_swap",
        "remap_rotate",
        "slippery",
        "hazard_goal_swap",
        "sign_composition",
    )


def test_inspect_rejects_out_of_range_and_non_integer_targets():
    assert validate_grid_action(GridAction("INSPECT", (7, 0)), ("INSPECT",), (7, 7))
    assert validate_grid_action(GridAction("INSPECT", (1.5, 2)), ("INSPECT",), (7, 7))
    assert validate_grid_action(GridAction("INSPECT", (1,)), ("INSPECT",), (7, 7))


def test_grid_action_rejects_unknown_kind_and_kind_outside_mask():
    assert validate_grid_action(GridAction("TELEPORT"), GRID_ACTIONS, (7, 7))
    assert validate_grid_action(GridAction("UP"), ("DOWN",), (7, 7))


def test_physical_action_accepts_no_target():
    assert validate_grid_action(GridAction("UP"), ("UP",), (7, 7)) == []


# --- Observation payload contracts ------------------------------------------


@dataclass(frozen=True)
class _FakeMap:
    def to_payload(self) -> dict:
        return {"width": 7, "height": 7}


def make_observation() -> GridPublicObservation:
    return GridPublicObservation(
        observation_id="obs-1",
        public_obs_hash="hash-obs",
        public_history_hash="hash-history",
        public_history="history",
        public_text="public text",
        available_actions=GRID_ACTIONS,
        candidate_rules=(
            CandidateRule("base", "Movement follows the chosen direction."),
        ),
        public_costs={"step_cost": -0.05},
        public_map=_FakeMap(),
        agent_position=(3, 4),
        health=3,
        inventory=("green_key",),
        door_open=False,
        steps=2,
        action_mask=GRID_ACTIONS,
        tool_budget_remaining=3,
        image=object(),
        image_hash="hash-image",
    )


def test_public_observation_payload_has_image_hash_but_not_raw_image():
    payload = make_observation().to_payload()
    assert set(payload) == set(GRID_PUBLIC_OBSERVATION_FIELDS)
    assert "image" not in payload
    assert payload["image_hash"] == "hash-image"
    assert payload["public_map"] == {"width": 7, "height": 7}
    assert payload["agent_position"] == [3, 4]


def test_public_observation_payload_validates():
    payload = make_observation().to_payload()
    assert validate_grid_public_observation(payload) == []


def test_public_observation_validator_rejects_raw_image_and_missing_keys():
    payload = make_observation().to_payload()
    payload["image"] = "raw-pixels"
    errors = validate_grid_public_observation(payload)
    assert errors
    del payload["image"]
    del payload["observation_id"]
    assert validate_grid_public_observation(payload)


# --- VLM output validator contracts -----------------------------------------


def _uniform_probabilities() -> dict[str, float]:
    return {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS}


def test_grid_vlm_output_rejects_bad_probabilities():
    payload = {
        "hypothesis_probabilities": {rule: 0.0 for rule in GRID_RULE_IDS},
        "action": "UP",
    }
    errors = validate_grid_vlm_output(payload)
    assert any("sum" in error for error in errors)

    payload["hypothesis_probabilities"] = {"base": 1.0}
    assert validate_grid_vlm_output(payload)


def test_grid_vlm_output_requires_target_only_for_inspect():
    payload = {"hypothesis_probabilities": _uniform_probabilities(), "action": "INSPECT"}
    assert validate_grid_vlm_output(payload)
    payload["tool_target"] = [3, 4]
    assert validate_grid_vlm_output(payload) == []


def test_grid_vlm_output_rejects_target_on_non_inspect_actions():
    payload = {
        "hypothesis_probabilities": _uniform_probabilities(),
        "action": "UP",
        "tool_target": [3, 4],
    }
    errors = validate_grid_vlm_output(payload)
    assert any("only valid for INSPECT" in error for error in errors)


def test_grid_vlm_output_rejects_unknown_top_level_key_and_bad_confidence():
    payload = {
        "hypothesis_probabilities": _uniform_probabilities(),
        "action": "UP",
        "confidence": 1.5,
        "bogus": 1,
    }
    errors = validate_grid_vlm_output(payload)
    assert errors


def test_validators_never_raise_for_bad_payload_types():
    assert validate_grid_vlm_output(None)
    assert validate_grid_public_observation("not a payload")
    assert validate_grid_action("not an action", GRID_ACTIONS, (7, 7))


def test_validators_never_raise_for_unhashable_membership_values():
    vlm = {"hypothesis_probabilities": _uniform_probabilities(), "action": ["UP"]}
    assert validate_grid_vlm_output(vlm)
    vlm["action"] = {"kind": "UP"}
    assert validate_grid_vlm_output(vlm)

    observation = make_observation().to_payload()
    nested = dict(observation)
    nested["available_actions"] = [{"x": 1}]
    assert validate_grid_public_observation(nested)
    nested = dict(observation)
    nested["action_mask"] = [["UP"]]
    assert validate_grid_public_observation(nested)

    assert validate_grid_action(GridAction(["UP"]), GRID_ACTIONS, (7, 7))
    assert validate_grid_action(GridAction("UP"), [{"x": 1}], (7, 7))


def test_validators_never_raise_for_oversized_numbers():
    huge = 10**400  # Overflows float() conversion.
    vlm = {
        "hypothesis_probabilities": _uniform_probabilities(),
        "action": "UP",
        "confidence": huge,
    }
    assert validate_grid_vlm_output(vlm)

    probabilities = _uniform_probabilities()
    probabilities["base"] = huge
    vlm = {"hypothesis_probabilities": probabilities, "action": "UP"}
    assert validate_grid_vlm_output(vlm)

    observation = make_observation().to_payload()
    observation["public_costs"] = {"step_cost": huge}
    assert validate_grid_public_observation(observation)


# --- Decision record contract -----------------------------------------------


def test_grid_decision_record_row_contains_base_and_grid_fields():
    record = GridDecisionRecord(
        run_id="run-1",
        scenario_id="scenario-1",
        agent_method="grid_stub",
        episode_id="episode-1",
        decision_id="decision-1",
        public_obs_hash="hash-obs",
        public_history_hash="hash-history",
        available_actions=GRID_ACTIONS,
        chosen_action="INSPECT",
        physical_steps_elapsed=4,
        actual_reward=1.0,
        vlm_calls_used=1,
        input_tokens=10,
        output_tokens=20,
        latency_ms=5.0,
        json_valid=True,
        invalid_action=False,
        model_id="stub",
        prompt_hash="hash-prompt",
        run_seed=0,
        tool_calls_used=2,
        tool_cost=3.0,
        advisor_calls_used=1,
        information_gain_bits=0.25,
        env_steps_elapsed=4,
        success=True,
    )
    row = record.to_row()
    assert row["available_actions"] == list(GRID_ACTIONS)
    assert row["tool_calls_used"] == 2
    assert row["tool_cost"] == 3.0
    assert row["advisor_calls_used"] == 1
    assert row["information_gain_bits"] == 0.25
    assert row["env_steps_elapsed"] == 4
    assert row["success"] is True
    assert set(row) == set(vars(record))
