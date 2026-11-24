import numpy as np
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.envs.grid.render import GRID_LINE_COLOR, render_grid
from ruleshift.public_api.grid_rules import PublicGridMap
from ruleshift.public_api.grid_schemas import GridAction


def make_env(rule="base"):
    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=(5, 5), door=(5, 6), signpost=(4, 3),
    )
    env = GridEnv(GridConfig(), observation_mode="visual")
    env.reset(seed=0, options={"map": public, "initial_rule": rule})
    return env


def test_renderer_is_deterministic():
    env = make_env()
    assert np.array_equal(env.render(), env.render())


def test_renderer_is_rule_independent():
    base = make_env("base").render()
    swapped = make_env("hazard_goal_swap").render()
    assert np.array_equal(base, swapped)


def test_set_of_mark_grid_is_drawn():
    image = make_env().render()
    assert np.any(np.all(image == GRID_LINE_COLOR, axis=2))


def test_hud_changes_with_public_state():
    env = make_env()
    first = env.render()
    env.step(GridAction("RIGHT"))
    second = env.render()
    assert not np.array_equal(first, second)
