"""Static data interfaces for agent-visible objects.

These containers are deliberately inert: they carry no environment reference,
no transition model and no step method. Week 2 will populate equivalent objects
from the Gymnasium environment; the agent-side contract stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ruleshift.permissions import PUBLIC_OBSERVATION_FIELDS

from .schemas import MACRO_ACTIONS, RULE_IDS


@dataclass(frozen=True)
class CandidateRule:
    """A natural-language description of a candidate hidden mechanic.

    Only the rule id and the public description are exposed. No likelihood
    table, transition kernel or true-rule flag is part of this object.
    """

    rule_id: str
    description: str


@dataclass(frozen=True)
class PublicObservation:
    """The complete agent-visible observation at one decision point.

    Hash fields refer to immutable snapshots so that cache keys can include
    the public history without embedding hidden state. The text-input pilot
    receives a readable recent history and the public observation digest.
    """

    observation_id: str
    public_obs_hash: str
    public_history_hash: str
    public_text: str
    available_actions: tuple[str, ...]
    candidate_rules: tuple[CandidateRule, ...]
    public_costs: Mapping[str, float] = field(default_factory=dict)
    public_history: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the canonical agent-visible payload shape."""

        payload: dict[str, Any] = {
            "observation_id": self.observation_id,
            "public_obs_hash": self.public_obs_hash,
            "public_history_hash": self.public_history_hash,
            "public_history": self.public_history,
            "public_text": self.public_text,
            "available_actions": list(self.available_actions),
            "candidate_rules": [
                {"rule_id": rule.rule_id, "description": rule.description}
                for rule in self.candidate_rules
            ],
            "public_costs": dict(self.public_costs),
        }
        assert set(payload) == set(PUBLIC_OBSERVATION_FIELDS)
        return payload


@dataclass(frozen=True)
class DecisionRecord:
    """One row of the PUBLIC decisions.jsonl log (master plan section 11).

    Privileged columns live in a separate evaluator-only adapter, never here.
    """

    run_id: str
    scenario_id: str
    agent_method: str
    episode_id: str
    decision_id: str
    public_obs_hash: str
    public_history_hash: str
    available_actions: tuple[str, ...]
    chosen_action: str
    physical_steps_elapsed: int
    actual_reward: float
    vlm_calls_used: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    json_valid: bool
    invalid_action: bool
    model_id: str
    prompt_hash: str
    run_seed: int

    def to_row(self) -> dict[str, Any]:
        """Return a JSON-serializable public log row."""

        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "agent_method": self.agent_method,
            "episode_id": self.episode_id,
            "decision_id": self.decision_id,
            "public_obs_hash": self.public_obs_hash,
            "public_history_hash": self.public_history_hash,
            "available_actions": list(self.available_actions),
            "chosen_action": self.chosen_action,
            "physical_steps_elapsed": self.physical_steps_elapsed,
            "actual_reward": self.actual_reward,
            "vlm_calls_used": self.vlm_calls_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "json_valid": self.json_valid,
            "invalid_action": self.invalid_action,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "run_seed": self.run_seed,
        }


__all__ = [
    "CandidateRule",
    "DecisionRecord",
    "PublicObservation",
    "MACRO_ACTIONS",
    "RULE_IDS",
]
