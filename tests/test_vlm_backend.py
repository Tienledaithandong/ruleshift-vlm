"""VLM request/response contract, JSON parsing, and public-history cache keys."""

from __future__ import annotations

import json

from ruleshift.agents.vlm_backend import (
    ModelRequest,
    StubVLMBackend,
    _flatten_rendered,
    build_chat_messages,
    parse_structured_output,
    render_prompt,
)
from ruleshift.agents.cache import ResponseCache
from ruleshift.permissions import PRIVILEGED_FIELDS


def make_request(**overrides: object) -> ModelRequest:
    values = dict(
        model_id="stub-v0",
        prompt="public prompt",
        image_hash="sha256:image",
        history_hash="sha256:history",
        available_actions=("COMMIT_RED", "PROBE_RED", "WAIT"),
        candidate_hash="sha256:candidates",
        decoding_seed=7,
    )
    values.update(overrides)
    return ModelRequest(**values)


def test_prompt_template_exposes_only_public_fields() -> None:
    prompt = render_prompt(
        public_text="step=1 pos=(0,0)",
        candidate_rules=("red: accepts red", "blue: accepts blue"),
        available_actions=("COMMIT_RED", "WAIT"),
        public_costs={"primitive_step": 0.05},
        public_history="a0=WAIT r=-0.05",
    )
    assert "step=1 pos=(0,0)" in prompt
    assert "COMMIT_RED" in prompt
    for field in PRIVILEGED_FIELDS:
        assert field not in prompt


def test_stub_output_parses_to_schema_valid_payload() -> None:
    request = make_request()
    response = StubVLMBackend().generate(request)
    payload, errors = parse_structured_output(
        response.text, request.available_actions, direct=True
    )
    assert errors == []
    assert payload is not None
    assert payload["selected_action"] in request.available_actions
    assert response.input_tokens > 0 and response.output_tokens > 0


def test_stub_is_deterministic_for_same_request() -> None:
    request = make_request()
    first = StubVLMBackend().generate(request)
    second = StubVLMBackend().generate(request)
    assert first.text == second.text


def test_malformed_json_gets_deterministic_fallback() -> None:
    backend = StubVLMBackend(malformed_every=1)
    response = backend.generate(make_request())
    payload, errors = parse_structured_output(response.text, ("COMMIT_RED", "WAIT"), direct=True)
    assert payload is None
    assert errors
    assert any("json" in error.lower() for error in errors)


def test_invalid_selected_action_is_rejected() -> None:
    backend = StubVLMBackend(invalid_action_every=1)
    response = backend.generate(make_request())
    payload, errors = parse_structured_output(response.text, ("COMMIT_RED", "WAIT"), direct=True)
    assert payload is None
    assert any("action" in error.lower() for error in errors)


def test_text_fingerprint_ignores_pixels_and_history() -> None:
    first = make_request(input_mode="text")
    second = make_request(
        input_mode="text",
        image_hash="sha256:different-image",
        history_hash="sha256:different-history",
    )
    assert first.fingerprint() == second.fingerprint()


def test_rgb_fingerprint_changes_with_public_history_and_pixels() -> None:
    first = make_request(input_mode="rgb")
    second = make_request(input_mode="rgb", history_hash="sha256:different-history")
    third = make_request(input_mode="rgb", image_hash="sha256:different-image")
    assert first.fingerprint() != second.fingerprint()
    assert first.fingerprint() != third.fingerprint()
    assert first.fingerprint() == make_request(input_mode="rgb").fingerprint()


def test_cache_hits_and_never_stores_privileged_fields(tmp_path) -> None:
    request = make_request()
    cache = ResponseCache(tmp_path / "cache.jsonl")
    assert cache.get(request) is None

    response = StubVLMBackend().generate(request)
    cache.put(request, response)
    cached = cache.get(request)
    assert cached is not None
    assert cached.text == response.text

    raw = (tmp_path / "cache.jsonl").read_text(encoding="utf-8")
    for field in PRIVILEGED_FIELDS:
        assert field not in raw
    record = json.loads(raw.strip().splitlines()[0])
    assert record["history_hash"] == request.history_hash


def test_audit_wires_max_new_tokens_into_every_request(tmp_path) -> None:
    """The config value must reach the request; it was silently dropped before,
    so every call used the 256-token default and the JSON was truncated."""
    from ruleshift.evaluator.offline_audit import (
        build_offline_contexts,
        run_offline_audit,
    )

    seen: list[int] = []

    class Recorder:
        model_id = "rec-v0"

        def generate(self, request: ModelRequest):
            seen.append(request.max_new_tokens)
            return StubVLMBackend(model_id="rec-v0").generate(request)

    cache = ResponseCache(tmp_path / "response_cache.jsonl")
    run_offline_audit(build_offline_contexts(1), Recorder(), cache, max_new_tokens=640)
    assert seen == [640]
