from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_rules import RULE_BY_ID, PublicGridMap
from ruleshift.public_api.grid_schemas import GridAction, validate_grid_public_observation


def make_env(public_map, *, scenario=None, **config_overrides):
    # Tests assert exact positions, so slip is off unless a test opts in.
    settings = {"baseline_slip": 0.0, "slippery_slip": 0.0}
    settings.update(config_overrides)
    config = GridConfig(**settings)
    env = GridEnv(config)
    scenario = scenario or GridScenario(
        scenario_id="test", scenario_type="identification",
        initial_rule="base", shift_rule=None, shift_time=None, map_seed=0,
    )
    env.reset(seed=0, options={"map": public_map, "scenario": scenario})
    return env


def test_reset_returns_public_observation():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 2), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    view = env.public_view()
    assert view.public_map.start == (1, 1)
    assert find_forbidden_fields(view.to_payload()) == []


def test_physical_step_increments_env_steps():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 3), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    _, _, _, _, info = env.step(GridAction("DOWN"))
    assert info["env_steps"] == 1


def test_wall_blocks_movement():
    public = PublicGridMap(5, 5, frozenset({(1, 0)}), (1, 1),
                           {"red": (1, 3), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    env.step(GridAction("UP"))
    assert env.evaluator_state()["position"] == (1, 1)


def test_soft_hazard_damages_and_recovers_without_termination():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (3, 3), "blue": (1, 2), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    _, reward, terminated, _, _ = env.step(GridAction("DOWN"))
    assert reward <= -2.0
    assert env.evaluator_state()["health"] == 1   # max_health=3, damage=2
    assert terminated is False
    env.step(GridAction("DOWN"))
    assert env.evaluator_state()["position"] == (1, 2)
    assert env.evaluator_state()["recovery"] == 0


def test_goal_terminates_with_reward():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 2), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    _, reward, terminated, _, info = env.step(GridAction("DOWN"))
    assert terminated is True
    assert info["success"] is True
    assert reward >= 9.9


def test_key_and_door():
    public = PublicGridMap(
        5, 5, frozenset(), (1, 1),
        {"red": (1, 4), "blue": (3, 3), "green": (3, 1)},
        key=(1, 2), door=(1, 3), signpost=None,
    )
    env = make_env(public)
    env.step(GridAction("DOWN"))       # onto the key
    env.step(GridAction("INTERACT"))   # pick it up
    env.step(GridAction("INTERACT"))   # open the adjacent door
    assert env.evaluator_state()["door_open"] is True
    env.step(GridAction("DOWN"))       # onto the door cell
    _, _, terminated, _, _ = env.step(GridAction("DOWN"))
    assert terminated is True


def test_shift_changes_rule_at_shift_time():
    public = PublicGridMap(7, 7, frozenset(), (3, 3),
                           {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
                           None, None, None)
    scenario = GridScenario(
        scenario_id="shift", scenario_type="change_detection",
        initial_rule="base", shift_rule="remap_swap", shift_time=2, map_seed=0,
    )
    env = make_env(public, scenario=scenario)
    assert env.evaluator_state()["true_rule"] == "base"
    env.step(GridAction("UP"))
    env.step(GridAction("DOWN"))
    assert env.evaluator_state()["true_rule"] == "remap_swap"


def test_truncates_at_max_env_steps():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (3, 3), "blue": (3, 1), "green": (1, 3)},
                           None, None, None)
    env = make_env(public, max_env_steps=3)
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(GridAction("RIGHT"))
    assert truncated is True
    assert terminated is False


def test_slip_is_deterministic_for_seed():
    public = PublicGridMap(7, 7, frozenset(), (3, 3),
                           {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
                           None, None, None)
    scenario = GridScenario("slip", "identification", "slippery", None, None, 0)
    first = make_env(public, scenario=scenario, slippery_slip=0.8)
    second = make_env(public, scenario=scenario, slippery_slip=0.8)
    for _ in range(10):
        first.step(GridAction("RIGHT"))
        second.step(GridAction("RIGHT"))
    assert first.evaluator_state()["position"] == second.evaluator_state()["position"]


def test_public_view_validates_against_grid_schema_with_optional_cells_absent():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 2), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    assert validate_grid_public_observation(env.public_view().to_payload()) == []
    assert env.render() is None


def test_rule_object_uses_config_slip_not_public_default():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 2), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public, baseline_slip=0.0, slippery_slip=0.9)
    assert env._rule_object("base").slip_prob == 0.0
    assert env._rule_object("slippery").slip_prob == 0.9
    # The public planner's rule objects keep their documented defaults.
    assert RULE_BY_ID["base"].slip_prob == 0.10
