"""Baselines: public-only agent policies and clearly labelled privileged ones."""

from __future__ import annotations

import pytest

from ruleshift.agents.baselines import NeverProbePolicy, RandomProbePolicy
from ruleshift.evaluator.baselines import (
    MaxInformationGainPolicy,
    ReturnAwareReference,
    TrueMechanicOracle,
)
from ruleshift.evaluator.exact_solver import ExactSolver, MicrotaskConfig, freeze_belief
from ruleshift.public_api.interface import CandidateRule, PublicObservation


def public_observation() -> PublicObservation:
    return PublicObservation(
        observation_id="obs-1",
        public_obs_hash="sha256:a",
        public_history_hash="sha256:b",
        public_text="step=5 gate=open last=reject",
        available_actions=("COMMIT_RED", "COMMIT_BLUE", "PROBE_RED", "WAIT"),
        candidate_rules=tuple(
            CandidateRule(rule, f"Gate accepts {rule}.") for rule in ("red", "blue", "green")
        ),
        public_costs={"primitive_step": 0.05},
    )


def config() -> MicrotaskConfig:
    actions = ("COMMIT_RED", "COMMIT_BLUE", "PROBE_RED")
    return MicrotaskConfig(
        shift_hazard=0.0,
        feedback_noise=0.0,
        reward_success=10.0,
        step_cost=-0.05,
        durations={action: {1: 1.0} for action in actions},
        observation_steps={action: 0 for action in actions},
    )


def uncertain_belief() -> dict[tuple[str, bool], float]:
    return {("red", False): 0.5, ("blue", True): 0.5}


def test_never_probe_never_emits_probe_and_uses_public_outcomes() -> None:
    policy = NeverProbePolicy()
    observation = public_observation()
    assert policy.choose(observation) == "COMMIT_RED"
    policy.record("COMMIT_RED", "reject")
    assert policy.choose(observation) == "COMMIT_BLUE"
    policy.record("COMMIT_BLUE", "accept")
    assert policy.choose(observation) == "COMMIT_BLUE"


def test_random_probe_is_reproducible_from_public_hash() -> None:
    observation = public_observation()
    first = RandomProbePolicy(observation.public_history_hash)
    second = RandomProbePolicy(observation.public_history_hash)
    assert [first.choose(observation) for _ in range(20)] == [
        second.choose(observation) for _ in range(20)
    ]
    samples = {first.choose(observation) for _ in range(200)}
    assert samples <= set(observation.available_actions)
    assert any(action.startswith("PROBE_") for action in samples)
    assert any(action.startswith("COMMIT_") for action in samples)


def test_information_gain_avoids_uninformative_probe() -> None:
    noisy = MicrotaskConfig(
        feedback_noise=0.5,
        durations={"COMMIT_RED": {1: 1.0}, "PROBE_RED": {1: 1.0}},
        observation_steps={"COMMIT_RED": 0, "PROBE_RED": 0},
    )
    action = MaxInformationGainPolicy(
        noisy, actions=("COMMIT_RED", "PROBE_RED")
    ).choose(uncertain_belief())
    assert action == "COMMIT_RED"


def test_return_aware_reference_uses_solver_argmax() -> None:
    solver = ExactSolver(config(), actions=("COMMIT_RED", "COMMIT_BLUE", "PROBE_RED"))
    q = dict(solver.q_values(freeze_belief(uncertain_belief()), 2))
    best = max(q.items(), key=lambda item: item[1])[0]
    assert ReturnAwareReference(solver).choose(uncertain_belief(), horizon=2) == best


def test_true_rule_oracle_is_explicitly_privileged() -> None:
    oracle = TrueMechanicOracle("blue")
    assert oracle.choose() == "COMMIT_BLUE"
    assert oracle.is_privileged is True
