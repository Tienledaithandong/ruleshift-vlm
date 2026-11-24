"""Contract tests for the grid semantic draw helpers and scenario sampler.

The first two tests are the verbatim acceptance tests from Task 4's brief; the
remaining ones cover the new ``noise`` wrappers and ``sample_scenario``.
"""

from ruleshift.envs.events import SemanticRNG
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.noise import draw_inspect, draw_move, draw_shift, draw_sign
from ruleshift.envs.grid.rules import RULE_OBJECTS, sample_scenario


def test_semantic_rng_is_deterministic():
    a = SemanticRNG(11, "scenario")
    b = SemanticRNG(11, "scenario")
    assert [a.draw("move", step) for step in range(5)] == [
        b.draw("move", step) for step in range(5)
    ]


def test_different_seeds_differ():
    assert SemanticRNG(11, "scenario").draw("move", 0) != SemanticRNG(
        12, "scenario"
    ).draw("move", 0)


def test_noise_helpers_use_distinct_semantic_events():
    rng = SemanticRNG(7, "grid")
    assert draw_move(rng, 2) == rng.draw("move", 2)
    assert draw_shift(rng, 2) == rng.draw("shift", 2)
    assert draw_inspect(rng, 2, 1) == rng.draw("inspect", 2, 1)
    assert draw_sign(rng, 2, 1) == rng.draw("sign", 2, 1)


def test_sample_scenario_is_deterministic_and_seeds_both_streams():
    config = GridConfig()
    first = sample_scenario(42, config, "change_detection")
    assert first == sample_scenario(42, config, "change_detection")
    assert first.map_seed == 42
    assert first.run_seed == 42
    assert first.initial_rule == "base"
    assert first.shift_rule in set(RULE_OBJECTS) - {"base"}
    assert 1 <= first.shift_time <= config.max_env_steps - 1

    identification = sample_scenario(3, config, "identification")
    assert identification.shift_rule is None
    assert identification.shift_time is None
    no_shift = sample_scenario(3, config, "no_shift")
    assert no_shift.initial_rule == "base"
    assert no_shift.shift_rule is None
    assert no_shift.shift_time is None


def test_change_detection_shift_time_is_clamped_to_episode_length():
    config = GridConfig(max_env_steps=3, shift_window=(10, 40))
    scenario = sample_scenario(0, config, "change_detection")
    assert scenario.shift_time == 2
