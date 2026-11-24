from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.envs.grid.tools import inspect_cell, read_sign, sign_side
from ruleshift.public_api.grid_rules import BASE_RULE, PublicGridMap
from ruleshift.public_api.grid_schemas import GridAction


def make_env():
    public = PublicGridMap(
        5, 5, frozenset(), (1, 1),
        {"red": (3, 3), "blue": (1, 3), "green": (3, 1)},
        key=None, door=None, signpost=(2, 1),
    )
    env = GridEnv(GridConfig())
    env.reset(seed=0, options={"map": public})
    return env, public


def test_inspect_costs_two_units_and_does_not_advance_steps():
    env, _ = make_env()
    _, _, _, _, info = env.step(GridAction("INSPECT", (1, 3)))
    assert info["tool_budget_remaining"] == 1
    assert info["env_steps"] == 0


def test_inspect_rejects_out_of_range_target():
    env, _ = make_env()
    _, _, _, _, info = env.step(GridAction("INSPECT", (3, 3)))
    assert info["tool_budget_remaining"] == 3
    assert info["tool_rejected"] == "out_of_range"


def test_inspect_rejects_when_budget_insufficient():
    env, _ = make_env()
    env.step(GridAction("INSPECT", (1, 3)))
    _, _, _, _, info = env.step(GridAction("INSPECT", (2, 2)))
    assert info["tool_budget_remaining"] == 1
    assert info["tool_rejected"] == "insufficient_budget"


def test_inspect_is_noisy_but_truthful_without_error():
    _, public = make_env()
    truthful = inspect_cell(public, BASE_RULE, (1, 3), draw=0.99, wrong_index=0,
                            config=GridConfig())
    assert truthful.value == "hazard"
    noisy = inspect_cell(public, BASE_RULE, (1, 3), draw=0.0, wrong_index=0,
                         config=GridConfig())
    assert noisy.value != "hazard"


def test_read_sign_requires_adjacency():
    env, _ = make_env()
    env.step(GridAction("LEFT"))   # move away from the signpost at (2, 1)
    _, _, _, _, info = env.step(GridAction("READ_SIGN"))
    assert info["tool_budget_remaining"] == 3
    assert info["tool_rejected"] == "not_adjacent"


def test_read_sign_costs_one_unit_and_returns_one_clause():
    env, _ = make_env()
    _, _, _, _, info = env.step(GridAction("READ_SIGN"))
    assert info["tool_budget_remaining"] == 2
    assert info["env_steps"] == 0
    assert "last_tool_value" in info


def test_repeated_reads_cost_again():
    env, _ = make_env()
    env.step(GridAction("READ_SIGN"))
    _, _, _, _, info = env.step(GridAction("READ_SIGN"))
    assert info["tool_budget_remaining"] == 1


def test_tool_budget_never_negative():
    env, _ = make_env()
    env.step(GridAction("INSPECT", (1, 3)))
    env.step(GridAction("INSPECT", (2, 2)))
    env.step(GridAction("READ_SIGN"))
    _, _, _, _, info = env.step(GridAction("READ_SIGN"))
    assert info["tool_budget_remaining"] == 0
    assert info["tool_rejected"] == "insufficient_budget"


def test_sign_side_is_deterministic():
    assert sign_side((2, 1), (2, 2)) == 2   # north of the sign
    assert sign_side((3, 2), (2, 2)) == 1   # east of the sign
    assert sign_side((2, 3), (2, 2)) == 0   # south of the sign
