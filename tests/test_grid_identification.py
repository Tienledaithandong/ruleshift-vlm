"""Focused tests for the independent identification evaluator."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.evaluator.grid_identification import (
    DeclaredCommit,
    EpisodeRecord,
    EvaluationError,
    PairingError,
    build_episode_record,
    classify_posterior,
    mcnemar_exact,
    paired_summary,
    posterior_brier,
    posterior_nll,
    shift_summary,
    summarize_records,
    validate_posterior,
)
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS


def _posterior(top: str = "base", top_probability: float = 0.95) -> dict[str, float]:
    remainder = 1.0 - top_probability
    rest = [rule for rule in GRID_RULE_IDS if rule != top]
    return {
        rule: (top_probability if rule == top else remainder / len(rest))
        for rule in GRID_RULE_IDS
    }


def _scenario(
    *,
    scenario_id: str = "m0-s0-identification-base-None",
    scenario_type: str = "identification",
    initial_rule: str = "base",
    shift_rule: str | None = None,
    shift_time: int | None = None,
) -> GridScenario:
    return GridScenario(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        initial_rule=initial_rule,
        shift_rule=shift_rule,
        shift_time=shift_time,
        map_seed=0,
        run_seed=0,
    )


def _fake_episode(
    *,
    success: bool,
    final_rule: str,
    posterior: dict[str, float] | None,
    env_steps: int = 3,
    tool_units: int = 0,
    actual_shift_time: int | None = None,
    shift_censored: bool = False,
    belief_shift_event_time: int | None = None,
    belief_shift_false_alarm: bool = False,
    public_rows: tuple[dict, ...] = (),
    evaluator_rows: tuple[dict, ...] | None = None,
):
    true_rule = "base"
    return SimpleNamespace(
        method="test_method",
        success=success,
        final_rule_estimate=final_rule,
        env_steps=env_steps,
        tool_units=tool_units,
        terminated=False,
        truncated=False,
        termination_reason="success" if success else "max_decisions",
        actual_shift_time=actual_shift_time,
        shift_censored=shift_censored,
        belief_shift_event_time=belief_shift_event_time,
        belief_shift_false_alarm=belief_shift_false_alarm,
        terminal_posterior=posterior,
        evaluator_rows=evaluator_rows or ({"true_rule": true_rule},),
        public_rows=public_rows,
    )


def _record(
    *,
    scenario_id: str,
    method: str = "m",
    scenario_type: str = "identification",
    initial_rule: str = "base",
    task_success: bool,
    map_correct: bool,
    commitment_status: str = "unavailable",
    declaration_available: bool = False,
    shift_rule: str | None = None,
    scheduled_shift_time: int | None = None,
    actual_shift_time: int | None = None,
    shift_censored: bool = False,
    belief_shift_event_time: int | None = None,
    belief_shift_false_alarm: bool = False,
    tool_units: int = 0,
    terminal_tie: bool = False,
) -> EpisodeRecord:
    return EpisodeRecord(
        evaluation_version="grid-identification-v1",
        scenario_id=scenario_id,
        method=method,
        scenario_type=scenario_type,
        map_seed=0,
        run_seed=0,
        layout_sha256="0" * 64,
        initial_rule=initial_rule,
        terminal_true_rule=initial_rule,
        shift_rule=shift_rule,
        scheduled_shift_time=scheduled_shift_time,
        actual_shift_time=actual_shift_time,
        shift_censored=shift_censored,
        belief_shift_event_time=belief_shift_event_time,
        belief_shift_false_alarm=belief_shift_false_alarm,
        task_success=task_success,
        env_steps=5,
        tool_units=tool_units,
        budget_cap=3,
        terminated=task_success,
        truncated=False,
        termination_reason="success" if task_success else "max_decisions",
        terminal_posterior=_posterior(initial_rule) if map_correct else _posterior("remap_swap"),
        posterior_valid=True,
        posterior_error=None,
        terminal_map=initial_rule if map_correct else "remap_swap",
        terminal_top_probability=0.95,
        terminal_margin=0.9,
        terminal_tie=terminal_tie,
        map_correct=map_correct,
        task_and_map_correct=bool(task_success and map_correct),
        successful_misdiagnosis=bool(task_success and not map_correct),
        task_failure_map_correct=bool(not task_success and map_correct),
        declaration_available=declaration_available,
        declared_rule=None,
        declared_step=None,
        declared_confidence=None,
        declared_margin=None,
        commitment_status=commitment_status,
        settled_declaration=commitment_status in ("correct_commit", "wrong_commit"),
        commitment_correct=commitment_status == "correct_commit",
        earlier_wrong_commitments=0,
        earlier_wrong_commitment_rules=(),
        brier_score=0.0 if map_correct else 0.1,
        nll=0.0 if map_correct else 1.0,
        tool_events=(),
        code_commit="test",
        config_sha256="config",
        manifest_sha256="manifest",
    )


def test_posterior_validation_requires_full_menu_and_normalized_mass():
    with pytest.raises(EvaluationError):
        validate_posterior({"base": 0.9, "remap_swap": 0.1})
    bad = _posterior()
    bad["base"] = 0.5
    with pytest.raises(EvaluationError):
        validate_posterior(bad)
    with pytest.raises(EvaluationError):
        validate_posterior({**_posterior(), "base": float("nan")})


def test_settlement_requires_top_and_margin_and_ties_are_unresolved():
    assert classify_posterior(_posterior("base", 0.90))["settled"] is True
    assert classify_posterior(_posterior("base", 0.89))["settled"] is False
    almost = _posterior("base", 0.95)
    almost["remap_swap"] = 0.90
    almost["base"] = 0.95
    total = sum(almost.values())
    almost = {key: value / total for key, value in almost.items()}
    summary = classify_posterior(almost)
    assert summary["top_rule"] == "base"
    assert summary["settled"] is False
    tie = {rule: 1.0 / 6.0 for rule in GRID_RULE_IDS}
    tie_summary = classify_posterior(tie)
    assert tie_summary["tie"] is True
    assert tie_summary["settled"] is False


def test_brier_and_nll_are_reported_only_for_valid_posteriors():
    posterior = _posterior("base", 0.95)
    assert posterior_brier(posterior, "base") == pytest.approx(0.003)
    assert posterior_nll(posterior, "base") == pytest.approx(math.log(1.0 / 0.95))
    with pytest.raises(EvaluationError):
        posterior_brier({"base": 1.0}, "base")


def test_terminal_true_rule_uses_last_evaluator_row_not_first():
    scenario = _scenario(
        scenario_id="m0-s0-change_detection-base-slippery",
        scenario_type="change_detection",
        initial_rule="base",
        shift_rule="slippery",
        shift_time=3,
    )
    episode = _fake_episode(
        success=True,
        final_rule="slippery",
        posterior=_posterior("slippery", 0.95),
        actual_shift_time=3,
        evaluator_rows=(
            {"true_rule": "base"},
            {"true_rule": "base"},
            {"true_rule": "slippery"},
        ),
    )
    record = build_episode_record(
        episode, scenario, GridConfig(), layout_sha256_value="x"
    )
    assert record.terminal_true_rule == "slippery"
    assert record.initial_rule == "base"
    assert record.map_correct is True
    assert record.brier_score is not None
    assert record.nll is not None and record.nll < 1.0

    stale = build_episode_record(
        _fake_episode(
            success=False,
            final_rule="base",
            posterior=_posterior("base", 0.95),
            actual_shift_time=3,
            evaluator_rows=(
                {"true_rule": "base"},
                {"true_rule": "base"},
                {"true_rule": "slippery"},
            ),
        ),
        scenario,
        GridConfig(),
        layout_sha256_value="x",
    )
    assert stale.terminal_true_rule == "slippery"
    assert stale.map_correct is False


def test_task_success_does_not_imply_correct_identification():
    scenario = _scenario()
    episode = _fake_episode(
        success=True,
        final_rule="remap_swap",
        posterior=_posterior("remap_swap", 0.95),
    )
    record = build_episode_record(
        episode, scenario, GridConfig(), layout_sha256_value="x"
    )
    assert record.task_success is True
    assert record.map_correct is False
    assert record.task_and_map_correct is False
    assert record.successful_misdiagnosis is True
    assert record.commitment_status == "unavailable"


def test_success_before_declaration_is_task_success_plus_unresolved_identification():
    scenario = _scenario()
    episode = _fake_episode(
        success=True,
        final_rule="base",
        posterior=None,
    )
    record = build_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="x",
        declaration_available=True,
        declarations=(),
    )
    assert record.task_success is True
    assert record.map_correct is True
    assert record.commitment_status == "unresolved"
    assert record.posterior_valid is False


def test_explicit_commitments_use_strict_thresholds_and_no_post_terminal_events():
    scenario = _scenario()
    episode = _fake_episode(success=False, final_rule="base", posterior=_posterior())
    correct = build_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="x",
        declaration_available=True,
        declarations=(DeclaredCommit(1, "base", 0.90, 0.80),),
    )
    assert correct.commitment_status == "correct_commit"
    unresolved = build_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="x",
        declaration_available=True,
        declarations=(DeclaredCommit(1, "base", 0.95, 0.05),),
    )
    assert unresolved.commitment_status == "unresolved"
    wrong = build_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="x",
        declaration_available=True,
        declarations=(DeclaredCommit(1, "remap_swap", 0.99, 0.9),),
    )
    assert wrong.commitment_status == "wrong_commit"
    earlier_then_correct = build_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="x",
        declaration_available=True,
        declarations=(
            DeclaredCommit(1, "remap_swap", 0.99, 0.9),
            DeclaredCommit(2, "base", 0.99, 0.9),
        ),
    )
    assert earlier_then_correct.commitment_status == "correct_commit"
    assert earlier_then_correct.earlier_wrong_commitments == 1
    assert earlier_then_correct.earlier_wrong_commitment_rules == ("remap_swap",)
    with pytest.raises(EvaluationError):
        build_episode_record(
            episode,
            scenario,
            GridConfig(),
            layout_sha256_value="x",
            declaration_available=True,
            declarations=(DeclaredCommit(99, "base", 0.99, 0.9),),
        )


def test_summary_keeps_all_static_episodes_in_denominators_and_does_not_score_absent_commitments():
    records = [
        _record(scenario_id="a", task_success=True, map_correct=True),
        _record(scenario_id="b", task_success=True, map_correct=False),
        _record(scenario_id="c", task_success=False, map_correct=True),
        _record(scenario_id="d", task_success=False, map_correct=False),
    ]
    summary = summarize_records(records)["methods"]["m"]["static_identification"]
    assert summary["episodes"] == 4
    assert summary["task_success_rate"] == 0.5
    assert summary["map_accuracy"] == 0.5
    assert summary["task_and_map_rate"] == 0.25
    assert summary["successful_misdiagnosis"] == 1
    assert summary["commitment_accuracy"] is None
    assert summary["commitment_unavailable"] == 4

    declared = [
        _record(
            scenario_id="a",
            task_success=True,
            map_correct=True,
            declaration_available=True,
            commitment_status="correct_commit",
        ),
        _record(
            scenario_id="b",
            task_success=True,
            map_correct=False,
            declaration_available=True,
            commitment_status="wrong_commit",
        ),
        _record(
            scenario_id="c",
            task_success=False,
            map_correct=True,
            declaration_available=True,
            commitment_status="unresolved",
        ),
        _record(
            scenario_id="d",
            task_success=False,
            map_correct=False,
            declaration_available=True,
            commitment_status="unresolved",
        ),
    ]
    declared_summary = summarize_records(declared)["methods"]["m"]["static_identification"]
    assert declared_summary["commitment_accuracy"] == 0.25
    assert declared_summary["commitment_wrong"] == 1
    assert declared_summary["commitment_unresolved"] == 2
    assert declared_summary["selective_error"] == 0.5


def test_base_preservation_is_static_only_and_100_percent_is_observed_not_assumed():
    records = [
        _record(
            scenario_id="base-ok",
            task_success=True,
            map_correct=True,
            initial_rule="base",
        ),
        _record(
            scenario_id="base-bad",
            task_success=False,
            map_correct=False,
            initial_rule="base",
        ),
        _record(
            scenario_id="shift",
            task_success=True,
            map_correct=True,
            initial_rule="base",
            scenario_type="change_detection",
            shift_rule="slippery",
            scheduled_shift_time=5,
            actual_shift_time=5,
        ),
    ]
    summary = summarize_records(records)["methods"]["m"]
    assert summary["base_identification"]["episodes"] == 2
    assert summary["base_identification"]["task_successes"] == 1
    assert summary["base_identification"]["task_success_rate"] == 0.5
    assert summary["static_identification"]["episodes"] == 2
    assert summary["shift"]["episodes"] == 1


def test_shift_summary_distinguishes_censoring_false_alarm_and_proxy_detection():
    records = [
        _record(
            scenario_id="actual",
            task_success=True,
            map_correct=True,
            shift_rule="slippery",
            scheduled_shift_time=5,
            actual_shift_time=5,
            belief_shift_event_time=7,
        ),
        _record(
            scenario_id="false-alarm",
            task_success=False,
            map_correct=False,
            shift_rule="slippery",
            scheduled_shift_time=5,
            actual_shift_time=8,
            belief_shift_event_time=3,
            belief_shift_false_alarm=True,
        ),
        _record(
            scenario_id="censored",
            task_success=False,
            map_correct=False,
            shift_rule="slippery",
            scheduled_shift_time=5,
            actual_shift_time=None,
            shift_censored=True,
            belief_shift_event_time=2,
            belief_shift_false_alarm=False,
        ),
    ]
    summary = shift_summary(records)
    assert summary["scheduled_shift_episodes"] == 3
    assert summary["actual_shift_episodes"] == 2
    assert summary["censored_before_shift"] == 1
    assert summary["pre_shift_false_alarm_episodes"] == 1
    assert summary["policy_shift_event_episodes"] == 3
    assert summary["latency_steps"] == [2]
    assert summary["detection_basis"] == "policy_belief_shift_probability_proxy"


def test_paired_summary_requires_exact_unique_ids_and_never_silently_intersects():
    left = [
        _record(scenario_id="a", task_success=True, map_correct=True),
        _record(scenario_id="b", task_success=False, map_correct=False),
    ]
    right = [
        _record(scenario_id="a", task_success=False, map_correct=False),
        _record(scenario_id="b", task_success=False, map_correct=False),
    ]
    result = paired_summary(left, right, metric="task_success")
    assert result["episodes"] == 2
    assert result["only_left"] == 1
    assert result["only_right"] == 0
    assert result["both"] == 0
    assert result["neither"] == 1
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5, alternative="two-sided") == pytest.approx(2 / 32)
    with pytest.raises(PairingError):
        paired_summary(left, right + [_record(scenario_id="c", task_success=True, map_correct=True)])
    duplicate = left + [_record(scenario_id="a", task_success=True, map_correct=True)]
    with pytest.raises(PairingError):
        paired_summary(duplicate, right)


def test_manifest_is_explicit_and_geometry_hashed():
    from ruleshift.evaluator.grid_manifest import (
        build_manifest,
        layout_sha256,
        manifest_from_dict,
        manifest_to_dict,
        write_manifest,
    )
    from ruleshift.envs.grid.mapgen import generate_solvable_map

    manifest = build_manifest(GridConfig(), map_ids=[0], run_seeds=[0])
    assert len(manifest.entries) == 11
    assert manifest.sha256
    assert len({entry["layout_sha256"] for entry in manifest.entries}) == 1
    assert layout_sha256(generate_solvable_map(0, GridConfig()).map.public) == manifest.entries[0]["layout_sha256"]
    restored = manifest_from_dict(manifest_to_dict(manifest))
    assert restored.sha256 == manifest.sha256
    assert restored.scenario_ids == manifest.scenario_ids
    with pytest.raises(Exception):
        build_manifest(GridConfig(), map_ids=[0, 0], run_seeds=[0])
