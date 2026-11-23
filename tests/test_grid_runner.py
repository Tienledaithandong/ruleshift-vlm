from ruleshift.agents.grid_baselines import InfoGainPolicy, NoAdaptationPolicy
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import OraclePolicy, run_grid_episode
from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_rules import PublicGridMap
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS


def scenario():
    return GridScenario("runner-test", "change_detection", "base",
                        "remap_swap", 3, 0)


def env():
    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=None, door=None, signpost=(4, 3),
    )
    env = GridEnv(GridConfig(baseline_slip=0.0, slippery_slip=0.0))
    observation, info = env.reset(
        seed=0, options={"map": public, "scenario": scenario()}
    )
    return env, observation, info


def run(policy):
    grid_env, observation, info = env()
    return run_grid_episode(
        grid_env, policy, GridConfig(), scenario(), observation, info
    )


def test_no_adaptation_keeps_the_base_rule():
    policy = NoAdaptationPolicy(GridConfig())
    episode = run(policy)
    assert policy.most_likely_rule() == "base"


def test_info_gain_uses_a_tool_when_informative():
    episode = run(InfoGainPolicy(GridConfig()))
    assert episode.tool_units > 0


def test_total_cost_formula():
    config = GridConfig()
    episode = run(InfoGainPolicy(config))
    assert episode.total_cost == episode.env_steps + config.advisor_cost * episode.advisor_calls


def test_budgets_are_matched():
    config = GridConfig()
    episode = run(InfoGainPolicy(config))
    assert episode.env_steps <= config.max_env_steps
    assert episode.tool_units <= config.tool_budget


def test_public_rows_have_no_privileged_fields():
    episode = run(InfoGainPolicy(GridConfig()))
    for row in episode.public_rows:
        assert find_forbidden_fields(row) == []


def test_evaluator_trace_has_the_true_rule():
    episode = run(InfoGainPolicy(GridConfig()))
    assert episode.evaluator_rows[0]["true_rule"] in GRID_RULE_IDS


def test_oracle_is_an_upper_bound_on_a_deterministic_map():
    config = GridConfig(baseline_slip=0.0, slippery_slip=0.0)
    no_adapt = run(NoAdaptationPolicy(config))
    oracle_env, observation, info = env()
    oracle = run_grid_episode(
        oracle_env, OraclePolicy(oracle_env, config), config,
        scenario(), observation, info,
    )
    assert oracle.total_return >= no_adapt.total_return


def test_build_grid_scenarios_is_deterministic():
    from ruleshift.evaluator.grid_runner import build_grid_scenarios

    config = GridConfig()
    first = build_grid_scenarios(2, 2, config)
    second = build_grid_scenarios(2, 2, config)
    assert first == second
    assert len(first) == 2 * 2 * (len(GRID_RULE_IDS) + 4 + 1)
    assert {item.run_seed for item in first} == {0, 1}
    assert {item.scenario_type for item in first} == {
        "identification", "change_detection", "no_shift"
    }


def test_run_grid_benchmark_runs_every_method():
    from ruleshift.evaluator.grid_runner import (
        build_grid_scenarios,
        run_grid_benchmark,
    )

    scenarios = build_grid_scenarios(1, 1, GridConfig())[:2]
    methods = {"no_adaptation": lambda config, env: NoAdaptationPolicy(config)}
    episodes = run_grid_benchmark(methods, scenarios, GridConfig())
    assert len(episodes) == 2
    assert all(episode.method == "no_adaptation" for episode in episodes)
