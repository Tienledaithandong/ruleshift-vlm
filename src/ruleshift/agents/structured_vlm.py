"""Structured outcome prompting on the public interface only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ruleshift.public_api.interface import PublicObservation

from .cache import ResponseCache
from .vlm_backend import ModelRequest, ModelResponse, parse_structured_output

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "outcomes_v1.txt"
#: Stamped into the request fingerprint so caches record the template actually
#: used (``ModelRequest`` cannot default to this because it is backend-agnostic).
OUTCOMES_PROMPT_VERSION = "outcomes_v1"


def render_outcomes_prompt(observation: PublicObservation) -> str:
    """Fill the public-only outcome-prediction prompt template."""

    template = PROMPT_PATH.read_text(encoding="utf-8")
    descriptions = [
        f"{rule.rule_id}: {rule.description}" for rule in observation.candidate_rules
    ]
    candidate_actions = [
        action
        for action in observation.available_actions
        if action.startswith(("PROBE_", "COMMIT_"))
    ]
    return template.format(
        public_history=observation.public_text,
        public_text=observation.public_text,
        candidate_rules=", ".join(descriptions),
        available_actions=", ".join(observation.available_actions),
        candidate_actions=", ".join(candidate_actions),
        public_costs=", ".join(
            f"{key}={value}" for key, value in sorted(observation.public_costs.items())
        ),
    )


def run_structured_prediction(
    observation: PublicObservation,
    backend: object,
    cache: ResponseCache,
    decoding_seed: int = 7,
) -> tuple[dict | None, ModelResponse, list[str]]:
    """Request, cache and validate a structured prediction from public input."""

    prompt = render_outcomes_prompt(observation)
    descriptions = [
        f"{rule.rule_id}: {rule.description}" for rule in observation.candidate_rules
    ]
    candidate_hash = hashlib.sha256(
        json.dumps(descriptions, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request = ModelRequest(
        model_id=backend.model_id,  # type: ignore[attr-defined]
        prompt=prompt,
        image_hash=observation.public_obs_hash,
        history_hash=observation.public_history_hash,
        available_actions=tuple(observation.available_actions),
        candidate_hash=candidate_hash,
        decoding_seed=decoding_seed,
        prompt_version=OUTCOMES_PROMPT_VERSION,
    )
    cached = cache.get(request)
    if cached is None:
        response = backend.generate(request)  # type: ignore[attr-defined]
        cache.put(request, response)
    else:
        response = cached

    payload, errors = parse_structured_output(response.text, request.available_actions)
    return payload, response, errors
