from pathlib import Path
from ruleshift.public_api.grid_planner import (
    GridPlannerState, HypothesisDynamics, astar, plan_to_goal,
)
from ruleshift.public_api.grid_rules import BASE_RULE, RULE_BY_ID, PublicGridMap


def test_astar_finds_shortest_path_on_open_grid():
    public = PublicGridMap(5, 5, frozenset(), (0, 0),
                           {"red": (4, 4), "blue": (4, 0), "green": (0, 4)},
                           None, None, None)
    model = HypothesisDynamics(BASE_RULE, public)
    plan = astar(GridPlannerState((0, 0), False, False), (4, 4), model)
    assert len(plan) == 8
    assert plan[0].kind in {"UP", "DOWN", "LEFT", "RIGHT"}


def test_astar_avoids_walls():
    # A wall column with a single gap at the bottom forces a detour; the
    # Manhattan path from (0,0) to (4,0) would be 4 steps.
    walls = frozenset({(2, y) for y in range(4)})
    public = PublicGridMap(5, 5, walls, (0, 0),
                           {"red": (4, 0), "blue": (4, 4), "green": (0, 4)},
                           None, None, None)
    model = HypothesisDynamics(BASE_RULE, public)
    plan = astar(GridPlannerState((0, 0), False, False), (4, 0), model)
    assert plan
    assert len(plan) > 4


def test_astar_uses_remapped_movement():
    public = PublicGridMap(5, 5, frozenset(), (2, 2),
                           {"red": (2, 0), "blue": (4, 4), "green": (0, 4)},
                           None, None, None)
    rule = RULE_BY_ID["remap_swap"]
    model = HypothesisDynamics(rule, public)
    plan = astar(GridPlannerState((2, 2), False, False), (2, 0), model)
    state = GridPlannerState((2, 2), False, False)
    for action in plan:
        state = max(model.predict_proba(state, action).items(), key=lambda kv: kv[1])[0]
    assert state.position == (2, 0)


def test_astar_collects_key_and_opens_door():
    public = PublicGridMap(
        5, 5, frozenset({(1, 0), (3, 0), (1, 2), (3, 2)}), (2, 2),
        {"red": (2, 0), "blue": (4, 4), "green": (0, 4)},
        key=(2, 3), door=(2, 1), signpost=None,
    )
    model = HypothesisDynamics(BASE_RULE, public)
    plan = plan_to_goal(BASE_RULE, public, (2, 2))
    state = GridPlannerState((2, 2), False, False)
    for action in plan:
        state = max(model.predict_proba(state, action).items(), key=lambda kv: kv[1])[0]
    assert state.position == (2, 0)


def test_astar_returns_empty_when_unreachable():
    public = PublicGridMap(5, 5, frozenset({(3, 0), (2, 1), (4, 1), (3, 2)}),
                           (0, 0), {"red": (3, 1), "blue": (4, 4), "green": (0, 4)},
                           None, None, None)
    model = HypothesisDynamics(BASE_RULE, public)
    assert astar(GridPlannerState((0, 0), False, False), (3, 1), model) == []


def test_planner_is_public_only():
    source = Path("src/ruleshift/public_api/grid_planner.py").read_text(encoding="utf-8")
    assert "ruleshift.envs" not in source
    assert "ruleshift.evaluator" not in source
