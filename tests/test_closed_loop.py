"""Closed-loop Family A: identical executor, separated traces, paired CIs."""

from __future__ import annotations

import json

import pytest

from ruleshift.evaluator.closed_loop import (
    build_family_a_scenarios,
    run_closed_loop,
    run_episode,
    write_episode_files,
    write_report,
)
from ruleshift.permissions import PRIVILEGED_FIELDS

METHODS = ("never_probe", "random_probe", "direct_stub", "m1_stub", "diva_stub")


def test_episode_is_deterministic_and_bounded() -> None:
    scenarios = build_family_a_scenarios()
    scenario_id, config = scenarios[0]
    first = run_episode(config, "m1_stub", seed=11, scenario_id=scenario_id)
    second = run_episode(config, "m1_stub", seed=11, scenario_id=scenario_id)
    assert first == second
    assert first.steps <= config.max_env_steps
    assert first.total_return == pytest.approx(
        sum(row["reward"] for row in first.public_rows)
    )


def test_public_decision_rows_never_leak_privileged_fields() -> None:
    scenarios = build_family_a_scenarios()
    scenario_id, config = scenarios[0]
    episode = run_episode(config, "never_probe", seed=3, scenario_id=scenario_id)
    for row in episode.public_rows:
        assert set(row).isdisjoint(PRIVILEGED_FIELDS)
        assert row["chosen_action"] in row["available_actions"]


def test_all_public_methods_emit_valid_actions() -> None:
    scenarios = build_family_a_scenarios()
    scenario_id, config = scenarios[0]
    for method in METHODS:
        episode = run_episode(config, method, seed=5, scenario_id=scenario_id)
        assert episode.invalid_actions == 0
        assert episode.total_return == pytest.approx(
            sum(row["reward"] for row in episode.public_rows)
        )


def test_no_shift_and_shift_scenarios_are_both_present() -> None:
    scenarios = build_family_a_scenarios()
    assert any(config.disable_shift for _, config in scenarios)
    assert any(not config.disable_shift and config.shift_hazard > 0 for _, config in scenarios)


def test_closed_loop_writes_public_and_evaluator_files_separately(tmp_path) -> None:
    scenarios = build_family_a_scenarios()[:2]
    results = run_closed_loop(scenarios, METHODS[:2], seeds=(1, 2))
    public_path, evaluator_path = write_episode_files(results, tmp_path)

    public_text = public_path.read_text(encoding="utf-8")
    for field in PRIVILEGED_FIELDS:
        assert field not in public_text

    public_rows = [json.loads(line) for line in public_text.strip().splitlines()]
    private_rows = [
        json.loads(line) for line in evaluator_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert public_rows and len(private_rows) == len(public_rows)
    assert set(public_rows[0]).isdisjoint(PRIVILEGED_FIELDS)
    assert set(private_rows[0]) & PRIVILEGED_FIELDS


def test_paired_summaries_are_computed_with_scenario_unit(tmp_path) -> None:
    scenarios = build_family_a_scenarios()[:3]
    results = run_closed_loop(scenarios, ("never_probe", "m1_stub"), seeds=(1,))
    summary = results[("m1_stub", "never_probe")]
    assert summary.n_scenarios == len(scenarios)
    assert summary.low <= summary.mean_difference <= summary.high

    report = write_report(results, tmp_path / "w7.md")
    text = report.read_text(encoding="utf-8")
    assert "SYNTHETIC" in text
    assert "paired" in text.lower()
