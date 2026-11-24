from ruleshift.agents.grid_belief import GridBelief
from ruleshift.agents.grid_selector import (
    InfoGainSelector,
    PublicAdvisor,
    _adjacent_side,
    _Candidate,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.tools import sign_side
from ruleshift.public_api.grid_planner import GridPlannerState, HypothesisDynamics
from ruleshift.public_api.grid_rules import (
    PublicGridMap,
    RULE_BY_ID,
    cell_role,
    effective_map,
    movement_outcomes,
    sign_clause,
)
from ruleshift.public_api.grid_schemas import GridAction

# No signpost: the test isolates INSPECT from the cheaper READ_SIGN probe.
MAP = PublicGridMap(5, 5, frozenset(), (1, 1),
                    {"red": (1, 3), "blue": (3, 3), "green": (3, 1)},
                    None, None, None)
MASK = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "INSPECT", "READ_SIGN")


def test_info_gain_prefers_an_informative_tool():
    belief = GridBelief(hypotheses=("base", "hazard_goal_swap"))
    selector = InfoGainSelector(GridConfig())
    action = selector.choose(
        belief, MAP, position=(1, 1), tool_budget=3, action_mask=MASK, step=0,
    )
    assert action.kind == "INSPECT"
    assert action.target is not None


def test_selector_exposes_the_chosen_actions_expected_information_gain():
    belief = GridBelief(hypotheses=("base", "hazard_goal_swap"))
    selector = InfoGainSelector(GridConfig())
    action = selector.choose(
        belief, MAP, position=(1, 1), tool_budget=3, action_mask=MASK, step=0,
    )
    assert action.kind == "INSPECT"
    assert selector.last_information_gain_bits is not None
    assert selector.last_information_gain_bits > 0.0


def test_info_gain_respects_tool_budget():
    belief = GridBelief()
    selector = InfoGainSelector(GridConfig())
    action = selector.choose(
        belief, MAP, position=(1, 1), tool_budget=0,
        action_mask=("UP", "DOWN", "LEFT", "RIGHT", "INTERACT"), step=0,
    )
    assert action.kind not in {"INSPECT", "READ_SIGN"}


def test_info_gain_returns_a_legal_action():
    belief = GridBelief()
    selector = InfoGainSelector(GridConfig())
    action = selector.choose(
        belief, MAP, position=(1, 1), tool_budget=3, action_mask=MASK, step=0,
    )
    assert action.kind in MASK


def test_public_advisor_returns_a_legal_action():
    belief = GridBelief()
    advisor = PublicAdvisor(GridConfig())
    action = advisor.advise(belief, MAP, position=(1, 1))
    assert action.kind in {"UP", "DOWN", "LEFT", "RIGHT", "INTERACT"}


def test_adjacent_side_matches_public_sign_convention():
    signpost = (2, 2)
    # The public sign has three faces; south and west deliberately share face 0.
    clauses = tuple(sign_clause(RULE_BY_ID["base"], side) for side in range(3))
    assert len(set(clauses)) == 3
    expected = {(2, 1): 2, (3, 2): 1, (2, 3): 0, (1, 2): 0}  # N, E, S, W
    for position, side in expected.items():
        assert _adjacent_side(position, signpost) == side
        assert _adjacent_side(position, signpost) == sign_side(position, signpost)


def test_read_sign_uses_the_south_west_face_convention():
    # A south-facing agent reads side 0, the only face that separates
    # "base" from "hazard_goal_swap"; movement is identical under both, so
    # READ_SIGN is the only informative action in this mask.
    sign_map = PublicGridMap(
        5, 5, frozenset(), (2, 3),
        {"red": (4, 4), "blue": (0, 0), "green": (4, 0)},
        None, None, (2, 2),
    )
    belief = GridBelief(hypotheses=("base", "hazard_goal_swap"))
    selector = InfoGainSelector(GridConfig())
    action = selector.choose(
        belief, sign_map, position=(2, 3), tool_budget=3,
        action_mask=("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "READ_SIGN"),
        step=0,
    )
    assert action.kind == "READ_SIGN"


def _hazard_probability_from_dynamics(public_map, rule_id):
    """Reference hazard probability from the public dynamics model."""

    rule = RULE_BY_ID[rule_id]
    model = HypothesisDynamics(rule, public_map)
    distribution = model.predict_proba(
        GridPlannerState((1, 1), False, False), GridAction("RIGHT")
    )
    color_at = {coord: color for color, coord in public_map.markers.items()}
    return sum(
        probability
        for state, probability in distribution.items()
        if (color := color_at.get(state.position)) is not None
        and cell_role(rule, color) == "hazard"
    )


def test_move_hazard_probability_matches_dynamics_with_signpost_blocking():
    public_map = PublicGridMap(
        5, 5, frozenset(), (1, 1),
        {"red": (4, 4), "blue": (2, 1), "green": (4, 0)},
        None, None, (3, 1),
    )
    belief = GridBelief(hypotheses=("slippery",))
    selector = InfoGainSelector(GridConfig())

    actual = selector._move_hazard_probability(
        belief, public_map, (1, 1), "RIGHT", door_open=False
    )
    expected = _hazard_probability_from_dynamics(public_map, "slippery")
    assert abs(actual - expected) < 1e-9


def test_move_hazard_probability_matches_dynamics_with_closed_door_blocking():
    public_map = PublicGridMap(
        5, 5, frozenset(), (1, 1),
        {"red": (4, 4), "blue": (3, 1), "green": (4, 0)},
        None, (2, 1), None,
    )
    belief = GridBelief(hypotheses=("slippery",))
    selector = InfoGainSelector(GridConfig())

    actual = selector._move_hazard_probability(
        belief, public_map, (1, 1), "RIGHT", door_open=False
    )
    expected = _hazard_probability_from_dynamics(public_map, "slippery")
    assert abs(actual - expected) < 1e-9


def test_expected_ig_for_move_matches_dynamics_with_blocked_cells():
    public_map = PublicGridMap(
        5, 5, frozenset(), (1, 1),
        {"red": (4, 4), "blue": (2, 1), "green": (4, 0)},
        None, None, (3, 1),
    )
    belief = GridBelief(hypotheses=("slippery",))
    selector = InfoGainSelector(GridConfig())
    candidate = _Candidate(GridAction("RIGHT"), 1, 0)

    actual = selector._expected_ig(
        belief, public_map, (1, 1), candidate, 0, (), False
    )

    movement_map = effective_map(public_map, door_open=False)
    before = belief.entropy()
    outcomes: dict[tuple[int, int], float] = {}
    for rule_id, rule_mass in belief.marginal().items():
        for landing, mass in movement_outcomes(
            RULE_BY_ID[rule_id], (1, 1), "RIGHT", movement_map
        ).items():
            outcomes[tuple(landing)] = outcomes.get(tuple(landing), 0.0) + rule_mass * mass
    expected_after = 0.0
    for outcome, probability in outcomes.items():
        branch = belief.copy()
        branch.update_move((1, 1), "RIGHT", outcome, public_map, 0)
        expected_after += probability * branch.entropy()
    expected = max(0.0, before - expected_after)
    assert abs(actual - expected) < 1e-9
