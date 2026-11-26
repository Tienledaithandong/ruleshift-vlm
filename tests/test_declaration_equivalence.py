"""Behavioral-equivalence gate for declaration instrumentation.

The fixture was generated from the frozen repaired-v0.4 code before the
declaration hook existed.  It covers every exposed static development scenario
on maps 0-9, seeds 0-4 for v0.3 and v0.4.  The hook may only add read-only
events; it must not change any action, outcome, step count or tool spend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ruleshift.agents.grid_baselines import (
    DiagnosticBeliefPolicy,
    DiagnosticBeliefPolicyV4,
    NoAdaptationPolicy,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_manifest import canonical_json
from ruleshift.evaluator.grid_runner import build_grid_scenarios, run_grid_episode

FIXTURE = Path(__file__).parent / "fixtures" / "v04_equivalence_maps0_9_seeds0_4.json"
METHODS = {
    "diagnostic_belief_v3": DiagnosticBeliefPolicy,
    "diagnostic_belief_v4": DiagnosticBeliefPolicyV4,
}


def _scenarios() -> list:
    return [
        scenario
        for scenario in build_grid_scenarios(10, 5, GridConfig())
        if scenario.scenario_type == "identification"
    ]


@pytest.mark.parametrize("method", sorted(METHODS))
def test_declaration_instrumentation_preserves_every_action(method: str) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expected = fixture["methods"][method]
    scenarios = _scenarios()
    assert {scenario.scenario_id for scenario in scenarios} == set(expected)
    cls = METHODS[method]

    for scenario in sorted(scenarios, key=lambda item: item.scenario_id):
        env = GridEnv(GridConfig())
        observation, info = env.reset(
            seed=scenario.run_seed,
            options={"scenario": scenario},
        )
        policy = cls(GridConfig())
        policy.name = method
        episode = run_grid_episode(
            env, policy, GridConfig(), scenario, observation, info
        )
        trace = [
            {"kind": row["chosen_action"], "target": row.get("tool_target")}
            for row in episode.public_rows
        ]
        digest = hashlib.sha256(canonical_json(trace).encode("utf-8")).hexdigest()
        record = expected[scenario.scenario_id]
        assert digest == record["actions_sha256"], scenario.scenario_id
        assert bool(episode.success) == record["success"], scenario.scenario_id
        assert episode.final_rule_estimate == record["final_rule_estimate"], scenario.scenario_id
        assert int(episode.env_steps) == record["env_steps"], scenario.scenario_id
        assert int(episode.tool_units) == record["tool_units"], scenario.scenario_id

        events = policy.commitment_events()
        assert isinstance(events, tuple)
        for event in events:
            assert set(event) == {
                "step",
                "declared_rule",
                "confidence",
                "posterior_margin",
                "source",
            }
            assert event["source"] == "policy.safe_commit"
            assert event["declared_rule"] in {
                "base",
                "remap_swap",
                "remap_rotate",
                "slippery",
                "hazard_goal_swap",
                "sign_composition",
            }
            assert 0.0 <= event["confidence"] <= 1.0
            assert 0.0 <= event["posterior_margin"] <= 1.0
        assert [event["step"] for event in events] == sorted(
            event["step"] for event in events
        )
        assert all(
            left["declared_rule"] != right["declared_rule"]
            for left, right in zip(events, events[1:])
        )


def test_no_adaptation_has_no_declaration_api() -> None:
    assert not callable(getattr(NoAdaptationPolicy(GridConfig()), "commitment_events", None))
