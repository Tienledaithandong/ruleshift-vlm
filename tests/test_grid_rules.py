"""Contract tests for the additive RuleShift-Grid public rule hypotheses.

The first nine tests are the verbatim acceptance tests from Task 2's brief;
the last one wires the real ``PublicGridMap`` serialization into the Task 1
observation validator (Task 1 could only use a stub map).
"""

from dataclasses import replace

from ruleshift.public_api.grid_rules import (
    BASE_RULE,
    GRID_HYPOTHESES,
    RULE_BY_ID,
    PublicGridMap,
    cell_role,
    effective_map,
    key_opens,
    movement_outcomes,
    rules_consistent_with_all_clauses,
    rules_consistent_with_clause,
    sign_clause,
)
from ruleshift.public_api.grid_schemas import (
    GRID_ACTIONS,
    GRID_RULE_IDS,
    GridPublicObservation,
    validate_grid_public_observation,
)
from ruleshift.public_api.interface import CandidateRule

MAP = PublicGridMap(
    width=5, height=5, walls=frozenset(),
    start=(1, 1),
    markers={"red": (3, 1), "blue": (3, 3), "green": (1, 3)},
    key=(0, 4), door=(4, 4), signpost=(2, 2),
)


def test_hypotheses_cover_every_declared_rule_id():
    assert tuple(rule.rule_id for rule in GRID_HYPOTHESES) == GRID_RULE_IDS


def test_base_rule_identity_movement():
    rule = replace(BASE_RULE, slip_prob=0.0)
    assert movement_outcomes(rule, (1, 1), "UP", MAP) == {(1, 0): 1.0}


def test_remap_swap_reverses_vertical():
    rule = replace(RULE_BY_ID["remap_swap"], slip_prob=0.0)
    assert movement_outcomes(rule, (1, 1), "UP", MAP) == {(1, 2): 1.0}


def test_slippery_distribution_sums_to_one():
    outcomes = movement_outcomes(RULE_BY_ID["slippery"], (2, 2), "RIGHT", MAP)
    assert abs(sum(outcomes.values()) - 1.0) < 1e-9
    assert len(outcomes) >= 2


def test_blocked_move_collapses_to_current_cell():
    blocked = replace(MAP, walls=frozenset({(1, 0)}))
    rule = replace(BASE_RULE, slip_prob=0.0)
    assert movement_outcomes(rule, (1, 1), "UP", blocked) == {(1, 1): 1.0}


def test_effective_map_blocks_signpost_and_closed_door():
    closed = effective_map(MAP, door_open=False)
    assert (2, 2) in closed.walls  # signpost
    assert (4, 4) in closed.walls  # closed door
    assert closed.markers == MAP.markers
    assert closed.key == MAP.key
    assert closed.door == MAP.door
    assert closed.signpost == MAP.signpost

    opened = effective_map(MAP, door_open=True)
    assert (2, 2) in opened.walls
    assert (4, 4) not in opened.walls


def test_effective_map_matches_simulator_movement_around_blocked_cells():
    """The helper must resolve movement exactly as the simulator does."""

    closed = effective_map(MAP, door_open=False)
    rule = RULE_BY_ID["slippery"]
    for position in ((2, 1), (2, 3), (1, 2), (3, 2), (4, 3), (3, 4)):
        for action in ("UP", "DOWN", "LEFT", "RIGHT"):
            expected = movement_outcomes(rule, position, action, closed)
            inline = replace(
                MAP,
                walls=frozenset(set(MAP.walls) | {MAP.signpost, MAP.door}),
            )
            assert movement_outcomes(rule, position, action, inline) == expected


def test_every_sign_side_leaves_at_least_two_hypotheses():
    for side in range(3):
        for rule in GRID_HYPOTHESES:
            clause = sign_clause(rule, side)
            assert len(rules_consistent_with_clause(side, clause)) >= 2


def test_full_sign_set_still_needs_transition_evidence():
    for rule in GRID_HYPOTHESES:
        clauses = tuple(sign_clause(rule, side) for side in range(3))
        assert len(rules_consistent_with_all_clauses(clauses)) >= 2


def test_key_opens_green_door():
    assert key_opens(BASE_RULE, "green", "green")
    assert not key_opens(BASE_RULE, "blue", "green")


def test_cell_roles_are_rule_dependent():
    assert cell_role(BASE_RULE, "red") == "goal"
    assert cell_role(RULE_BY_ID["hazard_goal_swap"], "red") == "hazard"


def test_real_map_serialization_validates_against_observation_validator():
    observation = GridPublicObservation(
        observation_id="obs-grid",
        public_obs_hash="hash-obs",
        public_history_hash="hash-history",
        public_history="history",
        public_text="public text",
        available_actions=GRID_ACTIONS,
        candidate_rules=tuple(
            CandidateRule(rule.rule_id, rule.description) for rule in GRID_HYPOTHESES
        ),
        public_costs={"step_cost": -0.05},
        public_map=MAP,
        agent_position=MAP.start,
        health=3,
        inventory=(),
        door_open=False,
        steps=0,
        action_mask=GRID_ACTIONS,
        tool_budget_remaining=3,
        image_hash="hash-image",
    )
    payload = observation.to_payload()
    assert payload["public_map"] == MAP.to_payload()
    assert validate_grid_public_observation(payload) == []
