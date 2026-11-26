"""Regression tests for the v0.5 separated epistemic/control policy."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ruleshift.agents import grid_v05
from ruleshift.agents.grid_v05 import (
    DiagnosticBeliefPolicyV4ReadoutV2,
    DiagnosticBeliefPolicyV5,
    DiagnosticBeliefPolicyV5NoRequest,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_identification_v2 import (
    EvaluationError,
    build_v2_episode_record,
    validate_development_slice,
)
from ruleshift.evaluator.grid_runner import build_grid_scenarios, run_grid_episode
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS


def _scenario(scenario_id: str):
    config = GridConfig()
    for scenario in build_grid_scenarios(10, 5, config):
        if scenario.scenario_id == scenario_id:
            return scenario
    raise AssertionError(f"missing scenario {scenario_id}")


def _run(scenario_id: str, policy_cls, config: GridConfig | None = None):
    config = config or GridConfig()
    scenario = _scenario(scenario_id)
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    policy = policy_cls(config)
    episode = run_grid_episode(
        env, policy, config, scenario, observation, info
    )
    return scenario, episode, policy


def _actions(episode) -> tuple[tuple[str, object], ...]:
    return tuple(
        (str(row.get("chosen_action")), row.get("tool_target"))
        for row in episode.public_rows
    )


def test_candidate_preserves_control_and_declarations_of_matched_readout():
    scenario_id = "m1-s1-identification-base-None"
    _, base_episode, base_policy = _run(
        scenario_id, DiagnosticBeliefPolicyV4ReadoutV2
    )
    _, candidate_episode, candidate_policy = _run(
        scenario_id, DiagnosticBeliefPolicyV5
    )
    assert _actions(base_episode) == _actions(candidate_episode)
    assert base_episode.success == candidate_episode.success
    assert base_episode.env_steps == candidate_episode.env_steps
    assert base_episode.tool_units == candidate_episode.tool_units
    assert base_episode.final_rule_estimate == candidate_episode.final_rule_estimate
    assert base_policy.declaration_events() == candidate_policy.declaration_events()
    assert candidate_policy.request_events()
    assert not base_policy.request_events()


def test_terminal_declaration_is_timestamped_and_not_an_action():
    _, episode, policy = _run(
        "m3-s0-identification-remap_rotate-None",
        DiagnosticBeliefPolicyV4ReadoutV2,
    )
    final = episode.epistemic_events[-1]
    assert final["kind"] == "DECLARE_RULE"
    assert final["rule_id"] == "remap_rotate"
    assert final["source"] == "final_observation"
    assert final["terminal"] is True
    assert final["step"] == episode.env_steps
    assert final["decision_index"] > len(episode.public_rows)
    assert final["confidence"] >= 0.90
    assert final["posterior_margin"] >= 0.10
    assert final["evidence_refs"]
    assert episode.success
    before = len(episode.public_rows)
    policy.finalize(None, {"env_steps": episode.env_steps})
    assert len(episode.public_rows) == before
    assert policy.declaration_events()[-1] == final


def test_abstention_does_not_create_a_declaration_or_action():
    _, episode, policy = _run(
        "m6-s0-identification-base-None", DiagnosticBeliefPolicyV4ReadoutV2
    )
    assert policy.declaration_events() == ()
    final = episode.epistemic_events[-1]
    assert final["kind"] == "ABSTAIN"
    assert final["terminal"] is True
    assert final["step"] == episode.env_steps
    assert final["reason"] == "insufficient_evidence"
    assert episode.success


def test_revision_and_wrong_event_accounting():
    scenario, episode, _ = _run(
        "m1-s1-identification-base-None", DiagnosticBeliefPolicyV4ReadoutV2
    )
    declarations = [
        event
        for event in episode.epistemic_events
        if event["kind"] == "DECLARE_RULE"
    ]
    assert len(declarations) == 2
    assert declarations[0]["rule_id"] == "slippery"
    assert declarations[-1]["rule_id"] == "base"
    assert declarations[-1]["source"] == "final_observation"
    record = build_v2_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="test",
    )
    assert record.final_declaration_correct
    assert record.wrong_declaration_events == 1
    assert record.wrong_declaration_rules == ("slippery",)
    assert record.any_wrong_declaration
    assert record.revision_count == 1
    assert record.revision_episode
    assert not record.strict_clean_correct


def test_terminal_map_is_never_converted_to_a_declaration():
    scenario, episode, _ = _run(
        "m6-s0-identification-base-None", DiagnosticBeliefPolicyV4ReadoutV2
    )
    record = build_v2_episode_record(
        episode,
        scenario,
        GridConfig(),
        layout_sha256_value="test",
        events=(),
    )
    assert record.terminal_map == "base"
    assert record.map_correct
    assert record.final_declared_rule is None
    assert record.unresolved
    assert not record.declaration_coverage
    assert not record.final_declaration_correct


def test_request_provides_no_free_evidence_or_budget():
    scenario = _scenario("m5-s0-identification-sign_composition-None")
    config = GridConfig()
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    policy = DiagnosticBeliefPolicyV5(config)
    policy.reset(observation, info)
    before_budget = int(info["tool_budget_remaining"])
    policy.choose(observation, info)
    assert policy.request_events()
    request = policy.request_events()[-1]
    assert request["channel"] in {"SIGN", "BLUE"}
    assert request["status"] == "pending"
    assert policy.evidence_ledger() == ()
    assert int(info["tool_budget_remaining"]) == before_budget
    assert policy._pending_request is not None


def test_candidate_is_public_and_does_not_import_evaluator_or_environment():
    module_path = Path(grid_v05.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden = (
        "ruleshift.evaluator",
        "ruleshift.envs",
        "ruleshift.envs.grid.grid_env",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not any(module.startswith(item) for item in forbidden)
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(alias.name.startswith(item) for item in forbidden)

    _, episode, policy = _run(
        "m5-s0-identification-sign_composition-None",
        DiagnosticBeliefPolicyV5,
    )
    ledger_ids = {int(item["evidence_id"]) for item in policy.evidence_ledger()}
    assert ledger_ids
    for event in episode.epistemic_events:
        assert "true_rule" not in event
        assert "hidden_role" not in event
        assert "evaluator" not in event
        if event["kind"] == "DECLARE_RULE":
            assert set(event["evidence_refs"]) <= ledger_ids
    for item in policy.evidence_ledger():
        assert set(item["likelihoods"]) == set(GRID_RULE_IDS)
        assert item["kind"] in {"move", "sign", "inspect"}


def test_deterministic_candidate_run():
    scenario_id = "m5-s1-identification-sign_composition-None"
    _, first_episode, first_policy = _run(
        scenario_id, DiagnosticBeliefPolicyV5
    )
    _, second_episode, second_policy = _run(
        scenario_id, DiagnosticBeliefPolicyV5
    )
    assert _actions(first_episode) == _actions(second_episode)
    assert first_episode.epistemic_events == second_episode.epistemic_events
    assert first_episode.success == second_episode.success
    assert first_episode.env_steps == second_episode.env_steps
    assert first_episode.tool_units == second_episode.tool_units
    assert first_episode.final_rule_estimate == second_episode.final_rule_estimate
    assert first_policy.evidence_ledger() == second_policy.evidence_ledger()
    assert first_policy.declaration_events() == second_policy.declaration_events()


def test_exhausted_tool_budget_makes_no_tool_calls():
    config = GridConfig(tool_budget=0)
    _, episode, policy = _run(
        "m1-s4-identification-hazard_goal_swap-None",
        DiagnosticBeliefPolicyV5,
        config=config,
    )
    assert episode.tool_units == 0
    assert all(
        event["chosen_action"] not in {"READ_SIGN", "INSPECT"}
        for event in episode.public_rows
    )
    assert len(episode.public_rows) <= config.max_decisions
    assert episode.epistemic_events
    assert policy.request_events() == () or all(
        event["channel"] in {"SIGN", "BLUE"}
        for event in policy.request_events()
    )


def test_decision_bound_is_finite_and_finalized_once():
    config = GridConfig(max_decisions=2)
    _, episode, policy = _run(
        "m6-s0-identification-base-None",
        DiagnosticBeliefPolicyV5,
        config=config,
    )
    assert len(episode.public_rows) <= 2
    terminal_events = [
        event for event in episode.epistemic_events if event.get("terminal")
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0]["decision_index"] <= config.max_decisions + 1
    before = len(episode.epistemic_events)
    policy.finalize(None, {"env_steps": episode.env_steps})
    assert len(policy.epistemic_events()) == before


def test_development_manifest_rejects_reserved_or_wrong_slices():
    with pytest.raises(EvaluationError, match="development maps"):
        validate_development_slice(
            SimpleNamespace(map_ids=(0, 10), run_seeds=(0,), entries=())
        )
    with pytest.raises(EvaluationError, match="reserved"):
        validate_development_slice(
            SimpleNamespace(map_ids=(0, 210), run_seeds=(0,), entries=()),
            development_maps=(0, 210),
            development_seeds=(0,),
            forbidden_maps=(210,),
        )
    with pytest.raises(EvaluationError, match="300"):
        validate_development_slice(
            SimpleNamespace(
                map_ids=tuple(range(10)),
                run_seeds=tuple(range(5)),
                entries=(),
            )
        )


def test_no_request_ablation_is_the_matched_readout_control():
    scenario_id = "m3-s0-identification-remap_rotate-None"
    _, base_episode, base_policy = _run(
        scenario_id, DiagnosticBeliefPolicyV4ReadoutV2
    )
    _, ablation_episode, ablation_policy = _run(
        scenario_id, DiagnosticBeliefPolicyV5NoRequest
    )
    assert _actions(base_episode) == _actions(ablation_episode)
    assert base_episode.epistemic_events == ablation_episode.epistemic_events
    assert base_policy.request_events() == ablation_policy.request_events() == ()
