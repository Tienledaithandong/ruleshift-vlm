"""Regression tests for versioned v2 declaration scoring."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ruleshift.agents.grid_v05 import DiagnosticBeliefPolicyV4ReadoutV2
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_identification import (
    EVALUATION_VERSION as V1_VERSION,
)
from ruleshift.evaluator.grid_identification_v2 import (
    EVALUATION_VERSION as V2_VERSION,
)
from ruleshift.evaluator.grid_identification_v2 import (
    EvaluationError,
    build_v2_episode_record,
    paired_transitions_v2,
    summarize_v2,
    validate_epistemic_events,
)
from ruleshift.evaluator.grid_runner import build_grid_scenarios, run_grid_episode
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS


def _scenario(scenario_id: str):
    config = GridConfig()
    for scenario in build_grid_scenarios(10, 5, config):
        if scenario.scenario_id == scenario_id:
            return scenario
    raise AssertionError(f"missing scenario {scenario_id}")


def _episode(scenario_id: str):
    config = GridConfig()
    scenario = _scenario(scenario_id)
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    policy = DiagnosticBeliefPolicyV4ReadoutV2(config)
    episode = run_grid_episode(
        env, policy, config, scenario, observation, info
    )
    return scenario, episode, policy


def _event(**overrides):
    event = {
        "kind": "DECLARE_RULE",
        "rule_id": "base",
        "confidence": 0.95,
        "posterior_margin": 0.80,
        "step": 0,
        "decision_index": 0,
        "source": "pre_action",
        "terminal": False,
        "evidence_refs": [1],
    }
    event.update(overrides)
    return event


def test_v2_is_versioned_separately_from_v1():
    assert V1_VERSION == "grid-identification-v1"
    assert V2_VERSION == "grid-identification-v2"


def test_declaration_gate_refuses_low_confidence_and_empty_refs():
    with pytest.raises(EvaluationError, match="confidence gate"):
        validate_epistemic_events(
            [_event(confidence=0.89)], terminal_step=3
        )
    with pytest.raises(EvaluationError, match="margin gate"):
        validate_epistemic_events(
            [_event(posterior_margin=0.09)], terminal_step=3
        )
    with pytest.raises(EvaluationError, match="evidence_refs"):
        validate_epistemic_events(
            [_event(evidence_refs=[])], terminal_step=3
        )
    with pytest.raises(EvaluationError, match="unknown declared rule"):
        validate_epistemic_events(
            [_event(rule_id="made_up")], terminal_step=3
        )
    with pytest.raises(EvaluationError, match="must be terminal"):
        validate_epistemic_events(
            [_event(source="final_observation")], terminal_step=3
        )


def test_event_order_and_one_declaration_per_decision_are_enforced():
    first = _event(decision_index=2)
    second = _event(decision_index=1)
    with pytest.raises(EvaluationError, match="decision order"):
        validate_epistemic_events([first, second], terminal_step=3)
    with pytest.raises(EvaluationError, match="same decision"):
        validate_epistemic_events(
            [_event(decision_index=1), _event(decision_index=1)],
            terminal_step=3,
        )


def test_declaration_cap_and_closed_vocabularies_are_enforced():
    events = [
        _event(decision_index=index, rule_id=GRID_RULE_IDS[index % 6])
        for index in range(7)
    ]
    with pytest.raises(EvaluationError, match="too many declaration"):
        validate_epistemic_events(events, terminal_step=10)
    with pytest.raises(EvaluationError, match="unknown abstention reason"):
        validate_epistemic_events(
            [
                {
                    "kind": "ABSTAIN",
                    "reason": "vibes",
                    "step": 0,
                    "decision_index": 0,
                }
            ],
            terminal_step=3,
        )
    with pytest.raises(EvaluationError, match="unknown request channel"):
        validate_epistemic_events(
            [
                {
                    "kind": "REQUEST_EVIDENCE",
                    "channel": "INSPECT",
                    "status": "pending",
                    "step": 0,
                    "decision_index": 0,
                }
            ],
            terminal_step=3,
        )


def test_final_terminal_index_is_allowed_one_past_the_bound():
    events = validate_epistemic_events(
        [_event(step=3, decision_index=4, source="final_observation", terminal=True)],
        terminal_step=3,
        max_decisions=3,
    )
    assert events[0]["decision_index"] == 4


def test_terminal_map_alone_is_unresolved_not_a_declaration():
    scenario, episode, _ = _episode(
        "m6-s0-identification-base-None"
    )
    record = build_v2_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="test",
        events=(),
    )
    assert record.map_correct
    assert record.terminal_map == "base"
    assert record.unresolved
    assert record.final_declared_rule is None
    assert record.final_declaration_correct is False
    assert record.final_declaration_wrong is False
    assert record.declaration_coverage is False


def test_final_declaration_is_the_last_event_and_wrong_events_are_kept():
    scenario, episode, _ = _episode(
        "m1-s1-identification-base-None"
    )
    record = build_v2_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="test",
    )
    assert record.final_declared_rule == "base"
    assert record.final_declaration_correct
    assert record.final_declaration_source == "final_observation"
    assert record.wrong_declaration_events == 1
    assert record.wrong_declaration_rules == ("slippery",)
    assert record.any_wrong_declaration
    assert record.revision_count == 1
    assert record.strict_clean_correct is False


def test_summary_readiness_and_paired_transitions():
    scenario, episode, _ = _episode(
        "m3-s0-identification-remap_rotate-None"
    )
    base = build_v2_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="test",
    )
    records = []
    for rule in GRID_RULE_IDS:
        for index in range(50):
            correct = rule in {
                "base",
                "remap_swap",
                "remap_rotate",
                "slippery",
                "hazard_goal_swap",
                "sign_composition",
            }
            tool_units = 3 if rule in {"hazard_goal_swap", "sign_composition"} else 0
            records.append(
                replace(
                    base,
                    scenario_id=f"m{index % 10}-s{index % 5}-{rule}",
                    method="synthetic",
                    terminal_true_rule=rule,
                    terminal_map=rule,
                    map_correct=True,
                    final_declared_rule=rule,
                    final_declaration_correct=correct,
                    final_declaration_wrong=not correct,
                    unresolved=False,
                    declaration_coverage=True,
                    task_success=True,
                    tool_units=tool_units,
                    strict_clean_correct=correct,
                    any_wrong_declaration=not correct,
                    wrong_declaration_events=0 if correct else 1,
                    wrong_declaration_rules=() if correct else (rule,),
                )
            )
    summary = summarize_v2(records)
    assert summary["overall"]["episodes"] == 300
    assert summary["overall"]["unconditional_declaration_accuracy"] == 1.0
    assert summary["readiness"]["all_pass"]
    assert summary["by_rule"]["hazard_goal_swap"]["episodes"] == 50

    left = records[:2]
    right = [replace(record, final_declaration_correct=False) for record in left]
    paired = paired_transitions_v2(left, right)
    assert paired["n"] == 2
    assert paired["declaration_correct"]["left_only"] == 2
    assert paired["declaration_correct"]["right_only"] == 0


def test_summary_weights_repeated_request_statuses():
    scenario, episode, _ = _episode(
        "m5-s0-identification-sign_composition-None"
    )
    record = build_v2_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="test",
    )
    record = replace(
        record,
        request_count=5,
        request_status_counts=(("executed", 3), ("unmet", 2)),
    )
    summary = summarize_v2([record])
    assert summary["overall"]["request_statuses"] == {
        "executed": 3,
        "unmet": 2,
    }
