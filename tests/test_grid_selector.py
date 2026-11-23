from ruleshift.agents.grid_belief import GridBelief
from ruleshift.agents.grid_selector import (
    InfoGainSelector,
    PublicAdvisor,
    _adjacent_side,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.tools import sign_side
from ruleshift.public_api.grid_rules import PublicGridMap, RULE_BY_ID, sign_clause

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
