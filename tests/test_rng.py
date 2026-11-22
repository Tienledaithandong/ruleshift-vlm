"""Tests for semantic, consumption-order-independent randomness."""

from __future__ import annotations

from ruleshift.envs.events import SemanticRNG


def test_same_key_gives_same_draw() -> None:
    first = SemanticRNG(master_seed=42, scenario_id="s1")
    second = SemanticRNG(master_seed=42, scenario_id="s1")
    assert first.draw("SHIFT_TRIGGER", 3, 0) == second.draw("SHIFT_TRIGGER", 3, 0)


def test_different_semantic_keys_give_different_draws() -> None:
    rng = SemanticRNG(master_seed=42, scenario_id="s1")
    assert rng.draw("SHIFT_TRIGGER", 3, 0) != rng.draw("SHIFT_TRIGGER", 4, 0)
    assert rng.draw("SHIFT_TRIGGER", 3, 0) != rng.draw("PROBE_FEEDBACK", 3, 0)


def test_other_draws_do_not_change_a_semantic_stream() -> None:
    rng = SemanticRNG(master_seed=42, scenario_id="s1")
    expected = rng.draw("SHIFT_TRIGGER", 7, 0)
    for step in range(20):
        rng.draw("PROBE_FEEDBACK", step, 0)
        rng.choice("SHIFT_TARGET", step, 0, ("blue", "green"))
    assert rng.draw("SHIFT_TRIGGER", 7, 0) == expected


def test_scenario_id_partitions_streams() -> None:
    a = SemanticRNG(master_seed=42, scenario_id="s1").draw("SHIFT_TRIGGER", 1, 0)
    b = SemanticRNG(master_seed=42, scenario_id="s2").draw("SHIFT_TRIGGER", 1, 0)
    assert a != b


def test_draw_is_in_unit_interval() -> None:
    rng = SemanticRNG(master_seed=1, scenario_id="s")
    for step in range(5):
        value = rng.draw("SHIFT_TRIGGER", step, 0)
        assert 0.0 <= value < 1.0


def test_choice_returns_a_member_of_the_options() -> None:
    rng = SemanticRNG(master_seed=9, scenario_id="s")
    assert rng.choice("SHIFT_TARGET", 0, 0, ("blue", "green")) in {"blue", "green"}
