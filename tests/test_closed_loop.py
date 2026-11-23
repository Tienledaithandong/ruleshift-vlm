"""Closed-loop Family A: identical executor, separated traces, paired CIs."""

from __future__ import annotations

import json

import pytest

from ruleshift.envs.config import EnvConfig
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.evaluator.closed_loop import (
    DEFAULT_VLM_CALL_CAP,
    _StubMethod,
    _last_outcome,
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


def test_all_public_methods_emit_actions_in_their_public_menu() -> None:
    scenarios = build_family_a_scenarios()
    scenario_id, config = scenarios[0]
    for method in METHODS:
        episode = run_episode(config, method, seed=5, scenario_id=scenario_id)
        assert episode.total_return == pytest.approx(
            sum(row["reward"] for row in episode.public_rows)
        )
        # The runner must execute only actions in the environment's menu: no
        # cross-menu action may leak into a public decision row.
        assert all(
            row["chosen_action"] in row["available_actions"]
            for row in episode.public_rows
        )


def test_cross_family_action_is_clamped_to_wait_and_counted_invalid() -> None:
    config = build_family_a_scenarios()[0][1]
    episode = run_episode(config, "random_probe", seed=5, scenario_id="clamp")
    # random_probe proposes across families; those must not execute as written.
    assert episode.invalid_actions > 0
    assert all(row["chosen_action"] in row["available_actions"] for row in episode.public_rows)


def test_episode_runs_to_the_step_budget() -> None:
    config = build_family_a_scenarios()[0][1]
    # never_probe always emits WAIT (one primitive step each). The old
    # 12-decision cap stopped the episode at 12 steps, short of the 60-step
    # budget. It must now run to the full step budget.
    episode = run_episode(config, "never_probe", seed=1, scenario_id="steps")
    assert episode.steps == config.max_env_steps


def test_commit_travel_is_position_aware() -> None:
    # Station is two steps off the start so a GO_TO_STATION move leaves the
    # agent's cell; the committed travel cost must track that cell.
    config = EnvConfig(probe_detour_steps=2, scenario_id="pos-aware")
    env = RuleShiftEnv(config=config)
    env.reset(seed=1)
    start_travel = env.public_view().public_costs["commit_travel"]
    # A non-terminating move puts the agent at a different cell. The commit
    # cost must track that cell, not the start tile.
    env.step("GO_TO_STATION")
    moved_travel = env.public_view().public_costs["commit_travel"]
    assert moved_travel != start_travel


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


# ---------------------------------------------------------------------------
# Budget, log integrity and clustering regressions.
# ---------------------------------------------------------------------------
def _config(**overrides: object) -> EnvConfig:
    base = dict(
        feedback_noise=0.0,
        lockout_steps=3,
        probe_detour_steps=2,
        disable_shift=True,
        initial_mechanic="red",
    )
    base.update(overrides)
    return EnvConfig(**base)  # type: ignore[arg-type]


def test_probe_feedback_reaches_the_method_through_public_text() -> None:
    """A completed probe must hand accept/reject to the policy, not "none"."""

    env = RuleShiftEnv(config=_config())
    env.reset(seed=1)
    env.step("PROBE_RED")
    outcome = _last_outcome(env.public_view().public_text)
    assert outcome == "accept"

    method = _StubMethod("m1")
    before = method.belief["red"]
    method.record("PROBE_RED", outcome)
    assert method.belief["red"] > before


def test_call_cap_is_enforced_for_every_model_method() -> None:
    config = build_family_a_scenarios()[0][1]
    for method in ("direct_stub", "m1_stub", "diva_stub"):
        episode = run_episode(
            config,
            method,
            seed=1,
            scenario_id="cap",
            max_vlm_calls_per_episode=3,
        )
        assert episode.vlm_calls <= 3
        assert episode.call_cap_hits > 0
        assert sum(row["vlm_calls_used"] for row in episode.public_rows) == (
            episode.vlm_calls
        )


def test_default_call_cap_matches_the_preregistered_budget() -> None:
    assert DEFAULT_VLM_CALL_CAP == 6
    config = build_family_a_scenarios()[0][1]
    episode = run_episode(config, "m1_stub", seed=2, scenario_id="cap")
    assert episode.vlm_calls <= DEFAULT_VLM_CALL_CAP


def test_public_rows_are_unique_per_scenario_seed_decision() -> None:
    scenarios = build_family_a_scenarios()[:2]
    result = run_closed_loop(
        scenarios, ("never_probe", "m1_stub"), seeds=(1, 2),
        comparisons=(("m1_stub", "never_probe"),),
    )
    keys = [
        (row["scenario_id"], row["seed"], row["method"], row["decision_id"])
        for episode in result.episodes.values()
        for row in episode.public_rows
    ]
    assert len(keys) == len(set(keys))
    # Seeds must be visible in the flattened log, otherwise they collide.
    assert {row["seed"] for episode in result.episodes.values() for row in episode.public_rows} == {1, 2}


def test_paired_summary_clusters_seeds_within_scenario() -> None:
    scenarios = build_family_a_scenarios()[:3]
    result = run_closed_loop(
        scenarios,
        ("never_probe", "m1_stub"),
        seeds=(1, 2, 3),
        comparisons=(("m1_stub", "never_probe"),),
    )
    summary = result[("m1_stub", "never_probe")]
    assert summary.n_scenarios == len(scenarios)
    assert summary.n_episodes == len(scenarios) * 3


def test_documented_method_ids_without_the_stub_suffix_are_accepted() -> None:
    """docs/RUNBOOK.md uses ``m1``/``direct``; those ids must work."""

    scenarios = build_family_a_scenarios()[:1]
    result = run_closed_loop(
        scenarios,
        ("never_probe", "direct", "m1", "diva"),
        seeds=(1,),
        comparisons=(("m1", "direct"),),
    )
    assert ("m1", "direct") in result.summaries
