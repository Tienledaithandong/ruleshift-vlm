from ruleshift.agents.grid_baselines import (
    InfoGainPolicy,
    NoAdaptationPolicy,
    PublicAdvisorPolicy,
    RandomProbePolicy,
)
from ruleshift.agents.grid_belief import GridBelief
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import OraclePolicy, run_grid_episode
from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_rules import PublicGridMap
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS, GridAction


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


# ---------------------------------------------------------------------------
# PublicAdvisorPolicy / RandomProbePolicy coverage
# ---------------------------------------------------------------------------


def tool_map():
    """A map with one in-range marker and an adjacent signpost."""

    return PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (3, 1), "blue": (1, 5), "green": (5, 1)},
        key=None, door=None, signpost=(4, 3),
    )


def reset_with(public, spec, config):
    grid_env = GridEnv(config)
    observation, info = grid_env.reset(
        seed=spec.run_seed, options={"map": public, "scenario": spec}
    )
    return grid_env, observation, info


def force_shift(policy):
    """Put the public change-point belief firmly in the trigger regime."""

    policy.belief = GridBelief(p_shift=1.0, shift_window=(0, 60))
    policy.belief.predict(1)
    assert policy.belief.shift_probability() > 0.5


def test_public_advisor_triggers_increments_and_leaves_belief_untouched():
    config = GridConfig()
    _, observation, info = reset_with(tool_map(), scenario(), config)
    policy = PublicAdvisorPolicy(config)
    policy.reset(observation, info)
    force_shift(policy)

    before = dict(policy.belief.states)
    first = policy.choose(observation, info)
    second = policy.choose(observation, info)

    assert isinstance(first, GridAction)
    assert first.kind in observation["action_mask"]
    assert isinstance(second, GridAction)
    assert policy.advisor_calls == 2
    assert policy.shift_detected() is True
    # advise() is read-only: it must never mutate the policy's belief.
    assert policy.belief.states == before


def test_public_advisor_follows_base_plan_without_a_trigger():
    config = GridConfig()
    _, observation, info = reset_with(tool_map(), scenario(), config)
    policy = PublicAdvisorPolicy(config)
    baseline = NoAdaptationPolicy(config)
    policy.reset(observation, info)
    baseline.reset(observation, info)

    assert policy.belief.shift_probability() == 0.0
    action = policy.choose(observation, info)

    assert isinstance(action, GridAction)
    assert action.kind in observation["action_mask"]
    assert policy.advisor_calls == 0
    assert policy.shift_detected() is False
    assert action == baseline.choose(observation, info)


def test_public_advisor_reports_calls_and_false_alarm_on_no_shift():
    class TriggeredAdvisor(PublicAdvisorPolicy):
        def reset(self, observation, info):
            super().reset(observation, info)
            force_shift(self)

    config = GridConfig(baseline_slip=0.0, slippery_slip=0.0)
    no_shift = GridScenario("no-shift-advisor", "no_shift", "base", None, None, 0, 0)
    grid_env, observation, info = reset_with(tool_map(), no_shift, config)
    episode = run_grid_episode(
        grid_env, TriggeredAdvisor(config), config, no_shift, observation, info
    )

    assert episode.advisor_calls > 0
    assert episode.shift_detected is True
    assert episode.scenario_type == "no_shift"
    assert episode.false_alarm is True
    assert episode.evaluator_rows[0]["false_alarm"] is True
    assert episode.total_cost == (
        episode.env_steps + config.advisor_cost * episode.advisor_calls
    )


def test_random_probe_chooses_only_legal_affordable_tools():
    config = GridConfig()
    _, observation, info = reset_with(tool_map(), scenario(), config)
    policy = RandomProbePolicy(config, seed=0)
    policy.reset(observation, info)
    force_shift(policy)

    kinds = set()
    for _ in range(40):
        action = policy.choose(observation, info)
        assert isinstance(action, GridAction)
        assert action.kind in observation["action_mask"]
        if action.kind == "INSPECT":
            assert observation["tool_budget_remaining"] >= config.inspect_cost
            assert action.target is not None
        elif action.kind == "READ_SIGN":
            assert observation["tool_budget_remaining"] >= config.read_sign_cost
            assert action.target is None
        kinds.add(action.kind)
    assert {"INSPECT", "READ_SIGN"} <= kinds


def test_random_probe_falls_back_to_base_plan_without_affordable_tools():
    config = GridConfig(tool_budget=0, baseline_slip=0.0, slippery_slip=0.0)
    _, observation, info = reset_with(tool_map(), scenario(), config)
    assert "INSPECT" not in observation["action_mask"]
    assert "READ_SIGN" not in observation["action_mask"]

    policy = RandomProbePolicy(config, seed=0)
    baseline = NoAdaptationPolicy(config)
    policy.reset(observation, info)
    baseline.reset(observation, info)
    force_shift(policy)

    action = policy.choose(observation, info)
    assert isinstance(action, GridAction)
    assert action.kind in observation["action_mask"]
    assert action == baseline.choose(observation, info)

    # Without a trigger the same fallback applies on a normal-budget map.
    full = GridConfig()
    _, full_obs, full_info = reset_with(tool_map(), scenario(), full)
    calm = RandomProbePolicy(full, seed=0)
    calm_baseline = NoAdaptationPolicy(full)
    calm.reset(full_obs, full_info)
    calm_baseline.reset(full_obs, full_info)
    assert calm.choose(full_obs, full_info) == calm_baseline.choose(full_obs, full_info)
