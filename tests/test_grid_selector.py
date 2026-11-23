from ruleshift.agents.grid_belief import GridBelief
from ruleshift.agents.grid_selector import InfoGainSelector, PublicAdvisor
from ruleshift.envs.grid.config import GridConfig
from ruleshift.public_api.grid_rules import PublicGridMap

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
