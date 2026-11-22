"""The evaluator trace is a separate privileged container, not a solver."""

from __future__ import annotations

from ruleshift.envs.config import EnvConfig
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.evaluator.adapter import trace_from_env


def test_trace_mirrors_privileged_state_without_inventing_solver_values() -> None:
    env = RuleShiftEnv(config=EnvConfig(initial_mechanic="blue", disable_shift=True))
    env.reset(seed=1)
    env.step("WAIT")

    trace = trace_from_env(env)
    state = env.evaluator_state()
    assert trace.true_rule_after == state["mechanic_id"] == "blue"
    assert trace.shift_time == state["shift_time"] is None
    assert trace.bayes_belief is None
    assert trace.q_star is None
    assert trace.delta_probe is None
