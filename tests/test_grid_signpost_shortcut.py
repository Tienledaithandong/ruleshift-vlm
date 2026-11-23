from ruleshift.agents.grid_baselines import _BasePolicy
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import run_grid_episode
from ruleshift.public_api.grid_rules import (
    GRID_HYPOTHESES, RULE_BY_ID, PublicGridMap,
    rules_consistent_with_all_clauses, sign_clause,
)
from ruleshift.public_api.grid_schemas import GridAction


def test_full_sign_set_is_behaviorally_ambiguous():
    for rule in GRID_HYPOTHESES:
        clauses = tuple(sign_clause(rule, side) for side in range(3))
        consistent = rules_consistent_with_all_clauses(clauses)
        assert len(consistent) >= 2
        signatures = {
            (RULE_BY_ID[rule_id].goal_color,
             tuple(sorted(RULE_BY_ID[rule_id].move_map.items())))
            for rule_id in consistent
        }
        assert len(signatures) >= 2


class SignOnlyPolicy(_BasePolicy):
    """Read all three sides via a fixed walk, then commit to the first rule."""

    ROUTE = ("READ_SIGN", "RIGHT", "DOWN", "READ_SIGN",
             "DOWN", "LEFT", "READ_SIGN")

    def __init__(self, config):
        super().__init__(use_belief=False)
        self._clauses = {}
        self._route_index = 0
        self._committed = None

    def choose(self, observation, info):
        if self._route_index < len(self.ROUTE):
            return GridAction(self.ROUTE[self._route_index])
        if self._committed is None:
            clauses = tuple(self._clauses[side] for side in range(3))
            consistent = rules_consistent_with_all_clauses(clauses)
            self._committed = consistent[0] if consistent else "base"
        plan = self.base_plan(self._committed, observation)
        return plan[0] if plan else GridAction("UP")

    def record(self, action, observation, reward, info):
        if action.kind == "READ_SIGN" and "last_tool_value" in info:
            from ruleshift.public_api.grid_rules import sign_side
            side = sign_side(self.previous_observation.agent_position,
                             observation.public_map.signpost)
            self._clauses[side] = info["last_tool_value"]
        if self._route_index < len(self.ROUTE):
            self._route_index += 1
        super().record(action, observation, reward, info)

    def most_likely_rule(self):
        return self._committed or "base"

    def shift_detected(self):
        return False


def test_scripted_signpost_only_policy_underperforms():
    # start (2,2), signpost (2,3): north side is adjacent at the start.
    public = PublicGridMap(
        7, 7, frozenset(), (2, 2),
        {"red": (1, 1), "blue": (5, 5), "green": (4, 4)},
        key=None, door=None, signpost=(2, 3),
    )
    scenario = GridScenario(
        "sign-shortcut", "identification", "sign_composition",
        None, None, 0,
    )
    config = GridConfig(baseline_slip=0.0, slippery_slip=0.0, max_env_steps=20)
    env = GridEnv(config, observation_mode="symbolic")
    obs, info = env.reset(seed=0, options={"map": public, "scenario": scenario})
    episode = run_grid_episode(env, SignOnlyPolicy(config), config, scenario, obs, info)
    assert episode.tool_units == 3
    assert episode.success is False


def test_unlimited_tool_budget_is_an_ablation_not_a_baseline():
    from ruleshift.agents.grid_baselines import InfoGainPolicy

    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=None, door=None, signpost=(4, 3),
    )
    scenario = GridScenario("ablation", "identification", "base", None, None, 0)

    def run(budget):
        config = GridConfig(tool_budget=budget, baseline_slip=0.0)
        env = GridEnv(config)
        obs, info = env.reset(seed=0, options={"map": public, "scenario": scenario})
        return run_grid_episode(env, InfoGainPolicy(config), config, scenario, obs, info)

    limited = run(3)
    unlimited = run(99)
    assert unlimited.tool_units > limited.tool_units
