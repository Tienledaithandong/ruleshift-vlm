"""Regression tests for the v0.5 direct cell-role feedback policy."""

from types import SimpleNamespace

from ruleshift.agents.grid_baselines import (
    DiagnosticBeliefPolicy,
    DiagnosticBeliefPolicyV4,
    DiagnosticBeliefPolicyV5,
)
from ruleshift.agents.grid_belief import GridBelief
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import build_grid_scenarios, run_grid_episode
from ruleshift.public_api.grid_schemas import GridAction

SHARED_FAILURES = {
    "m0-s3-identification-sign_composition-None": "sign_composition",
    "m5-s1-identification-hazard_goal_swap-None": "hazard_goal_swap",
    "m7-s2-change_detection-base-sign_composition": "sign_composition",
    "m9-s1-change_detection-base-sign_composition": "sign_composition",
}


def _run(scenario, policy):
    config = GridConfig()
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    return run_grid_episode(env, policy, config, scenario, observation, info)


def test_v5_recovers_all_four_shared_failures():
    config = GridConfig()
    scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_grid_scenarios(10, 5, config)
        if scenario.scenario_id in SHARED_FAILURES
    }
    assert set(scenarios) == set(SHARED_FAILURES)
    for scenario_id, expected_rule in SHARED_FAILURES.items():
        episode = _run(scenarios[scenario_id], DiagnosticBeliefPolicyV5(config))
        assert episode.success, scenario_id
        assert episode.final_rule_estimate == expected_rule
        assert episode.tool_units <= 3


def test_v5_preserves_the_historical_v3_and_v4_failures_on_the_same_scenarios():
    config = GridConfig()
    scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_grid_scenarios(10, 5, config)
        if scenario.scenario_id in SHARED_FAILURES
    }
    for policy_class in (DiagnosticBeliefPolicy, DiagnosticBeliefPolicyV4):
        for scenario_id, scenario in scenarios.items():
            episode = _run(scenario, policy_class(config))
            assert not episode.success, (policy_class.__name__, scenario_id)


def test_v5_records_a_direct_safe_role_only_on_a_marker():
    policy = DiagnosticBeliefPolicyV5(GridConfig())
    policy.belief = GridBelief(  # type: ignore[assignment]
        prior={"base": 0.5, "sign_composition": 0.5}
    )
    policy.previous_observation = object()  # type: ignore[assignment]
    observation = SimpleNamespace(
        agent_position=(2, 2),
        public_map=SimpleNamespace(markers={"blue": (2, 2)}),
        to_payload=dict,
    )
    action = GridAction("UP")
    policy._record_direct_cell_role(  # type: ignore[arg-type]
        action, observation, {"last_outcome": "move", "env_steps": 3}
    )
    marginal = policy.belief.marginal()
    assert policy.direct_role_observations == 1
    assert marginal["sign_composition"] > 0.99
    assert marginal["base"] < 0.01


def test_v5_ignores_blocked_moves_and_non_marker_cells():
    policy = DiagnosticBeliefPolicyV5(GridConfig())
    policy.belief = GridBelief()  # type: ignore[assignment]
    policy.previous_observation = object()  # type: ignore[assignment]
    marker_observation = SimpleNamespace(
        agent_position=(2, 2),
        public_map=SimpleNamespace(markers={"blue": (2, 2)}),
        to_payload=dict,
    )
    plain_observation = SimpleNamespace(
        agent_position=(1, 1),
        public_map=SimpleNamespace(markers={"blue": (2, 2)}),
        to_payload=dict,
    )
    action = GridAction("UP")
    policy._record_direct_cell_role(  # type: ignore[arg-type]
        action,
        marker_observation,
        {"last_outcome": "blocked", "env_steps": 2},
    )
    policy._record_direct_cell_role(  # type: ignore[arg-type]
        action,
        plain_observation,
        {"last_outcome": "move", "env_steps": 3},
    )
    assert policy.direct_role_observations == 0
