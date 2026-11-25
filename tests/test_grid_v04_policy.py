"""Focused tests for the exploratory v0.4 diagnostic policy."""

from types import SimpleNamespace

from ruleshift.agents.grid_baselines import DiagnosticBeliefPolicyV4
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import build_grid_scenarios, run_grid_episode
from ruleshift.public_api.grid_schemas import GridAction


class _Belief:
    def __init__(self, marginal: dict[str, float], top: str = "base") -> None:
        self._marginal = dict(marginal)
        self._top = top

    def marginal(self) -> dict[str, float]:
        return dict(self._marginal)

    def most_likely(self) -> str:
        return self._top


def test_base_plan_fallback_replaces_first_legal_action():
    policy = DiagnosticBeliefPolicyV4(GridConfig())
    policy.belief = _Belief({"slippery": 1.0}, top="slippery")  # type: ignore[assignment]
    observation = SimpleNamespace(
        agent_position=(0, 0),
        action_mask=("LEFT",),
        public_map=None,
        to_payload=dict,
    )
    policy.base_plan = (  # type: ignore[method-assign]
        lambda rule, obs: [] if rule == "slippery" else [GridAction("LEFT")]
    )
    assert policy._commit(observation).kind == "LEFT"
    assert policy.plan_fallback_count == 1


def test_sign_side_gain_selects_face_that_separates_base_from_remap():
    policy = DiagnosticBeliefPolicyV4(GridConfig())
    policy.belief = _Belief(  # type: ignore[assignment]
        {"base": 0.5, "remap_rotate": 0.5}, top="base"
    )
    assert policy._sign_side_gain(0) == 0.0
    assert policy._sign_side_gain(1) == 1.0


def test_stable_commit_requires_two_consecutive_top_rules():
    policy = DiagnosticBeliefPolicyV4(GridConfig())
    policy.belief = _Belief({"base": 0.95, "slippery": 0.05}, top="base")  # type: ignore[assignment]
    policy._observe_top_rule()
    assert not policy._commit_is_safe()
    policy._observe_top_rule()
    assert policy._commit_is_safe()
    policy.belief = _Belief({"remap_rotate": 0.95, "base": 0.05}, top="remap_rotate")  # type: ignore[assignment]
    policy._observe_top_rule()
    assert not policy._commit_is_safe()


def test_v4_fixes_the_traced_empty_plan_failures():
    config = GridConfig()
    scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_grid_scenarios(100, 5, config)
        if scenario.scenario_type == "identification"
        and scenario.initial_rule == "base"
        and scenario.scenario_id in {
            "m2-s2-identification-base-None",
            "m57-s4-identification-base-None",
        }
    }
    assert len(scenarios) == 2
    for scenario in scenarios.values():
        env = GridEnv(config)
        observation, info = env.reset(
            seed=scenario.run_seed, options={"scenario": scenario}
        )
        policy = DiagnosticBeliefPolicyV4(config)
        episode = run_grid_episode(
            env, policy, config, scenario, observation, info
        )
        assert episode.success
        assert episode.final_rule_estimate == "base"


def test_v4_records_sign_side_memory_and_recovers_base():
    config = GridConfig()
    scenario = next(
        scenario
        for scenario in build_grid_scenarios(100, 5, config)
        if scenario.scenario_id == "m78-s1-identification-base-None"
    )
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    policy = DiagnosticBeliefPolicyV4(config)
    episode = run_grid_episode(
        env, policy, config, scenario, observation, info
    )
    assert episode.success
    assert episode.final_rule_estimate == "base"
    assert 0 in policy.sign_sides_observed
