"""Self-consistency with every extra sample counted toward the call budget."""

from __future__ import annotations

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.self_consistency import SelfConsistencySelector, majority_vote
from ruleshift.agents.value_selector import PublicCostModel, ValueAwareSelector
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.public_api.interface import CandidateRule, PublicObservation


def observation() -> PublicObservation:
    return PublicObservation(
        observation_id="obs-1",
        public_obs_hash="sha256:image",
        public_history_hash="sha256:history",
        public_text="step=5 gate=open last=reject",
        available_actions=("COMMIT_RED", "PROBE_RED", "WAIT"),
        candidate_rules=tuple(
            CandidateRule(rule, f"Gate accepts {rule}.") for rule in ("red", "blue", "green")
        ),
        public_costs={"primitive_step": 0.05},
    )


def test_majority_vote_is_deterministic_on_ties() -> None:
    assert majority_vote(["PROBE_RED", "COMMIT_RED", "PROBE_RED"]) == "PROBE_RED"
    assert majority_vote(["COMMIT_RED", "PROBE_RED"]) == "COMMIT_RED"
    assert majority_vote(["COMMIT_RED", "PROBE_RED", "PROBE_RED", "COMMIT_RED"]) == "COMMIT_RED"
    assert majority_vote([]) == "WAIT"


def test_self_consistency_spends_one_call_per_sample(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache.jsonl")
    selector = SelfConsistencySelector(
        backend=StubVLMBackend(),
        base_selector=ValueAwareSelector(),
        n_samples=3,
        cache=cache,
    )
    action = selector.choose(
        observation(),
        {"red": 0.5, "blue": 0.5, "green": 0.0},
        PublicCostModel(lockout_steps=8, remaining_horizon=3, learning_rate=1.0),
    )
    assert action in ("COMMIT_RED", "PROBE_RED", "WAIT")
    lines = (tmp_path / "cache.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
