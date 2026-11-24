"""Tests for RuleShift-Grid config loading and solvable map generation.

The first four tests are the verbatim acceptance tests from Task 3's brief.
The rest pin the config defaults, YAML loading, determinism of ``attempts`` and
the door-required path.
"""

from collections import deque
from pathlib import Path

from ruleshift.envs.grid.config import GridConfig, load_grid_config
from ruleshift.envs.grid.mapgen import (
    all_hypotheses_solvable,
    generate_solvable_map,
    is_solvable,
)
from ruleshift.public_api.grid_rules import (
    GRID_HYPOTHESES,
    PublicGridMap,
    key_opens,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_DEV_YAML = REPO_ROOT / "configs" / "grid_dev.yaml"

_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _independent_reachable(public, start, blocked):
    """Reference BFS used only by tests; deliberately independent of mapgen."""

    if start in blocked:
        return set()
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _DELTAS:
            nxt = (x + dx, y + dy)
            if nxt in seen or nxt in blocked:
                continue
            if not (0 <= nxt[0] < public.width and 0 <= nxt[1] < public.height):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def _independent_solvable(public, rule):
    """Recompute solvability without calling the generator's gate."""

    blocked = set(public.walls)
    hazard = public.markers.get(rule.hazard_color)
    if hazard is not None:
        blocked.add(hazard)
    if public.signpost is not None:
        blocked.add(public.signpost)
    goal = public.markers[rule.goal_color]

    if public.door is not None:
        if not key_opens(rule, rule.key_color, rule.door_color):
            return False
        door_closed = blocked | {public.door}
        reachable_closed = _independent_reachable(public, public.start, door_closed)
        if public.key not in reachable_closed:
            return False
        dx, dy = public.door
        if not any(
            0 <= dx + ox < public.width
            and 0 <= dy + oy < public.height
            and (dx + ox, dy + oy) in reachable_closed
            for ox, oy in _DELTAS
        ):
            return False
        return goal in _independent_reachable(public, public.start, blocked)
    return goal in _independent_reachable(public, public.start, blocked)


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


def test_solver_blocks_signpost_on_only_path():
    """The signpost blocks movement, so it must block the reference BFS too."""

    public = PublicGridMap(
        width=5,
        height=5,
        walls=frozenset((x, y) for x in range(5) for y in range(1, 5)),
        start=(0, 0),
        markers={"red": (4, 0), "blue": (0, 4), "green": (3, 4)},
        key=None,
        door=None,
        signpost=(2, 0),
    )
    rule = GRID_HYPOTHESES[0]
    assert not _independent_solvable(public, rule)
    assert not is_solvable(public, rule)


def test_generated_maps_are_solvable_with_signpost_blocked():
    """Regression: seed 9 and friends exposed a signpost-only solvability gap."""

    seeds = (*range(20), 21, 22, 23, 24, 35, 48, 49, 50)
    for seed in seeds:
        public = generate_solvable_map(seed, GridConfig()).map.public
        for rule in GRID_HYPOTHESES:
            assert _independent_solvable(public, rule), (
                f"seed {seed} is unsolvable under {rule.rule_id} "
                "when the signpost blocks movement"
            )


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


def test_map_with_key_behind_closed_door_is_rejected():
    """A key that is only reachable through the closed door is unwinnable.

    The simulator always blocks a closed door, so the gate must reject this map
    even though ``door_required`` is left at its false default.
    """

    public = PublicGridMap(
        width=5,
        height=5,
        walls=frozenset((2, y) for y in range(5) if y != 2),
        start=(1, 2),
        markers={"red": (3, 1), "blue": (0, 0), "green": (1, 1)},
        key=(3, 2),
        door=(2, 2),
        signpost=None,
    )
    # With the door closed, start (1,2) cannot reach the key at (3,2).
    for rule in GRID_HYPOTHESES:
        assert not is_solvable(public, rule)
        assert not is_solvable(public, rule, door_required=True)
    # Opening the door would make it winnable, confirming the door is the only
    # obstacle and the rejection is for the right reason.
    from dataclasses import replace

    opened = replace(public, walls=frozenset())
    assert is_solvable(opened, GRID_HYPOTHESES[0])


def test_generated_maps_are_winnable_by_oracle_with_deterministic_slip():
    """Regression: every generated map must actually be winnable.

    Seeds 57 and 149 previously passed ``all_hypotheses_solvable`` while both
    the key and the goal sat behind the closed door, so the oracle could not
    finish them within the step budget.
    """

    from ruleshift.envs.grid.entities import GridScenario
    from ruleshift.envs.grid.grid_env import GridEnv
    from ruleshift.evaluator.grid_runner import OraclePolicy, run_grid_episode

    config = GridConfig(baseline_slip=0.0, slippery_slip=0.0)
    for seed in (0, 1, 2, 3, 5, 7, 57, 149):
        result = generate_solvable_map(seed, config)
        scenario = GridScenario(
            scenario_id=f"oracle-solvability-{seed}",
            scenario_type="identification",
            initial_rule="base",
            shift_rule=None,
            shift_time=None,
            map_seed=seed,
            run_seed=seed,
        )
        env = GridEnv(config)
        observation, info = env.reset(
            seed=seed, options={"map": result.map, "scenario": scenario}
        )
        episode = run_grid_episode(
            env, OraclePolicy(env, config), config, scenario, observation, info
        )
        assert episode.success, f"oracle could not win generated map seed {seed}"
