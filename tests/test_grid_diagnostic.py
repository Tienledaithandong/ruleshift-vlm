"""Tests for the Phase 0 sign -> blue-inspect diagnostic policy."""

from ruleshift.agents.grid_baselines import DiagnosticBeliefPolicy, as_observation
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import build_grid_scenarios, run_grid_episode


def _run(rule: str):
    config = GridConfig()
    scenario = next(
        scenario
        for scenario in build_grid_scenarios(1, 1, config)
        if scenario.scenario_type == "identification"
        and scenario.initial_rule == rule
    )
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    policy = DiagnosticBeliefPolicy(config)
    episode = run_grid_episode(
        env, policy, config, scenario, observation, info
    )
    actions = [row["chosen_action"] for row in episode.public_rows]
    return episode, policy, as_observation(observation), actions


def test_hazard_reads_sign_then_inspects_blue():
    episode, policy, observation, actions = _run("hazard_goal_swap")
    assert actions.count("READ_SIGN") == 1
    assert actions.count("INSPECT") == 1
    inspect_rows = [
        row for row in episode.public_rows if row["chosen_action"] == "INSPECT"
    ]
    assert inspect_rows[0]["tool_target"] == list(observation.public_map.markers["blue"])
    assert policy.most_likely_rule() == "hazard_goal_swap"
    assert episode.final_rule_estimate == "hazard_goal_swap"
    assert episode.success
    assert episode.tool_units == 3


def test_base_reads_sign_but_does_not_inspect():
    episode, policy, _observation, actions = _run("base")
    assert actions.count("READ_SIGN") == 1
    assert "INSPECT" not in actions
    assert policy.most_likely_rule() == "base"
    assert episode.success
    assert episode.tool_units == 1


def test_sign_composition_reads_sign_then_inspects_blue():
    episode, policy, _observation, actions = _run("sign_composition")
    assert actions.count("READ_SIGN") == 1
    assert actions.count("INSPECT") == 1
    assert policy.most_likely_rule() == "sign_composition"
    assert episode.final_rule_estimate == "sign_composition"
    assert episode.success
    assert episode.tool_units == 3


def test_movement_identifiable_rule_does_not_spend_tools():
    episode, policy, _observation, actions = _run("remap_swap")
    assert "READ_SIGN" not in actions
    assert "INSPECT" not in actions
    assert policy.most_likely_rule() == "remap_swap"
    assert episode.success
    assert episode.tool_units == 0

def test_blocked_side_zero_uses_informative_face():
    config = GridConfig()
    scenarios = {
        scenario.initial_rule: scenario
        for scenario in build_grid_scenarios(5, 5, config)
        if scenario.scenario_id.startswith("m4-s0-identification-")
    }
    for rule in ("hazard_goal_swap", "sign_composition"):
        scenario = scenarios[rule]
        env = GridEnv(config)
        observation, info = env.reset(
            seed=scenario.run_seed, options={"scenario": scenario}
        )
        policy = DiagnosticBeliefPolicy(config)
        episode = run_grid_episode(
            env, policy, config, scenario, observation, info
        )
        actions = [row["chosen_action"] for row in episode.public_rows]
        assert actions.count("READ_SIGN") == 1
        assert actions.count("INSPECT") == 1
        assert episode.tool_units == 3
        assert episode.final_rule_estimate == rule
        assert episode.success
