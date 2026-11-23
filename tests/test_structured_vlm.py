"""Structured outcome prompting: public input, cached, schema-validated output."""

from __future__ import annotations

from pathlib import Path

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.structured_vlm import render_outcomes_prompt, run_structured_prediction
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.permissions import PRIVILEGED_FIELDS
from ruleshift.public_api.interface import CandidateRule, PublicObservation


def observation() -> PublicObservation:
    return PublicObservation(
        observation_id="obs-1",
        public_obs_hash="sha256:image",
        public_history_hash="sha256:history",
        public_history="Step 1: PROBE_RED -> reject.",
        public_text="step=5 gate=open last=reject",
        available_actions=("COMMIT_RED", "PROBE_RED", "WAIT"),
        candidate_rules=tuple(
            CandidateRule(rule, f"Gate accepts {rule}.") for rule in ("red", "blue", "green")
        ),
        public_costs={"primitive_step": 0.05, "lockout": 8.0},
    )


def test_templates_do_not_supply_copyable_example_values() -> None:
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "ruleshift" / "prompts"
    for name in ("direct_v1.txt", "outcomes_v1.txt"):
        text = (prompts_dir / name).read_text(encoding="utf-8")
        for literal in ("0.34", "0.33", "PROBE_RED", "COMMIT_RED"):
            assert literal not in text, f"{name} leaks copyable example value {literal!r}"


def test_outcomes_prompt_contains_only_public_fields() -> None:
    prompt = render_outcomes_prompt(observation())
    assert "step=5 gate=open" in prompt
    assert "COMMIT_RED" in prompt
    assert "PROBE_RED" in prompt
    for field in PRIVILEGED_FIELDS:
        assert field not in prompt


def test_structured_prediction_is_valid_and_cacheable(tmp_path) -> None:
    obs = observation()
    cache = ResponseCache(tmp_path / "cache.jsonl")
    backend = StubVLMBackend()

    payload, first_response, errors = run_structured_prediction(obs, backend, cache)
    assert errors == []
    assert payload is not None
    assert payload["selected_action"] in obs.available_actions
    assert first_response.backend_id == "STUB_SYNTHETIC"

    second_payload, second_response, second_errors = run_structured_prediction(
        obs, backend, cache
    )
    assert second_payload == payload
    assert second_errors == []
    assert second_response.text == first_response.text
