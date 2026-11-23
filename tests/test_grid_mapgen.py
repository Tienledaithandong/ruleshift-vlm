"""Tests for RuleShift-Grid config loading and solvable map generation.

The first four tests are the verbatim acceptance tests from Task 3's brief.
The rest pin the config defaults, YAML loading, determinism of ``attempts`` and
the door-required path.
"""

from pathlib import Path

from ruleshift.envs.grid.config import GridConfig, load_grid_config
from ruleshift.envs.grid.mapgen import (
    all_hypotheses_solvable,
    generate_solvable_map,
    is_solvable,
)
from ruleshift.public_api.grid_rules import GRID_HYPOTHESES, PublicGridMap

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_DEV_YAML = REPO_ROOT / "configs" / "grid_dev.yaml"


def test_generated_map_has_expected_dimensions():
    result = generate_solvable_map(0, GridConfig(), GRID_HYPOTHESES)
    public = result.map.public
    assert (public.width, public.height) == (7, 7)
    assert public.start not in public.walls
    assert set(public.markers) == {"red", "blue", "green"}
    assert result.attempts >= 1


def test_generated_map_is_solvable_under_every_hypothesis():
    for seed in range(20):
        result = generate_solvable_map(seed, GridConfig(), GRID_HYPOTHESES)
        for rule in GRID_HYPOTHESES:
            assert is_solvable(result.map.public, rule)


def test_unsolvable_map_is_rejected():
    public = PublicGridMap(
        width=5,
        height=5,
        walls=frozenset({(3, 0), (2, 1), (4, 1), (3, 2)}),
        start=(0, 0),
        markers={"red": (3, 1), "blue": (1, 3), "green": (0, 4)},
        key=None,
        door=None,
        signpost=None,
    )
    rule = GRID_HYPOTHESES[0]
    assert not is_solvable(public, rule)


def test_generator_is_deterministic():
    first = generate_solvable_map(7, GridConfig(), GRID_HYPOTHESES)
    second = generate_solvable_map(7, GridConfig(), GRID_HYPOTHESES)
    assert first.map.public.to_payload() == second.map.public.to_payload()
    assert first.attempts == second.attempts


def test_grid_config_defaults_match_brief():
    config = GridConfig()
    assert config.width == 7 and config.height == 7
    assert config.max_health == 3
    assert not hasattr(config, "max_damage")
    assert config.baseline_slip == 0.10
    assert config.slippery_slip == 0.80
    assert config.shift_window == (10, 40)
    assert config.wall_density == 0.18
    assert config.max_env_steps == 60
    assert config.tool_budget == 3
    assert config.inspect_cost == 2
    assert config.read_sign_cost == 1
    assert config.inspect_range == 2
    assert config.inspect_error == 0.05
    assert config.shift_probability == 0.35
    assert config.soft_hazard_damage == -2.0
    assert config.soft_hazard_recovery_steps == 1
    assert config.goal_reward == 10.0
    assert config.step_cost == -0.05
    assert config.lethal_trap is False
    assert config.door_required is False
    assert config.advisor_cost == 2
    assert config.max_decisions == 200


def test_load_grid_config_reads_dev_yaml_and_ignores_unknown_keys():
    config = load_grid_config(GRID_DEV_YAML)
    assert config.width == 7 and config.height == 7
    assert config.wall_density == 0.18
    assert config.shift_window == (10, 40)
    assert config.max_health == 3
    assert config.baseline_slip == 0.10
    assert config.slippery_slip == 0.80
    assert not hasattr(config, "render_size")
    assert not hasattr(config, "observation_mode")


def test_door_required_maps_are_solvable_under_every_hypothesis():
    config = GridConfig(door_required=True)
    for seed in range(5):
        result = generate_solvable_map(seed, config, GRID_HYPOTHESES)
        assert result.map.door_required
        assert all_hypotheses_solvable(
            result.map.public, GRID_HYPOTHESES, door_required=True
        )
