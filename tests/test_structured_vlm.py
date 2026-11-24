"""Structured outcome prompting: public input, cached, schema-validated output."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.structured_vlm import (
    build_prediction_request,
    render_outcomes_prompt,
    run_structured_prediction,
)
from ruleshift.agents.vlm_backend import StubVLMBackend, TransformersVLMBackend
from ruleshift.permissions import PRIVILEGED_FIELDS
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
        public_costs={"primitive_step": 0.05, "lockout": 8.0},
    )


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


def test_rgb_request_forwards_pixels_and_changes_fingerprint() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    obs = replace(observation(), image=frame)
    backend = TransformersVLMBackend(model_id="dummy", input_mode="rgb")

    rgb_request = build_prediction_request(obs, backend, direct=True)
    assert rgb_request.input_mode == "rgb"
    assert rgb_request.image is frame

    changed = replace(
        obs,
        image=frame.copy(),
        public_obs_hash="sha256:different-image",
    )
    changed.image[0, 0, 0] = 255
    changed_request = build_prediction_request(changed, backend, direct=True)
    assert changed_request.fingerprint() != rgb_request.fingerprint()

    text_backend = TransformersVLMBackend(model_id="dummy")
    text_request = build_prediction_request(obs, text_backend, direct=True)
    assert text_request.input_mode == "text"
    assert text_request.image is None
    assert text_request.fingerprint() != rgb_request.fingerprint()
