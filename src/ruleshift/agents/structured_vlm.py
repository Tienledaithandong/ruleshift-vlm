"""Structured outcome prompting on the public interface only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ruleshift.public_api.interface import PublicObservation
from ruleshift.public_api.schemas import TASK_ACTIONS

from .cache import ResponseCache
from .vlm_backend import PROMPT_VERSION, ModelRequest, ModelResponse, parse_structured_output, render_prompt

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "outcomes_v1.txt"
#: Stamped into the request fingerprint so caches record the template actually
#: used (``ModelRequest`` cannot default to this because it is backend-agnostic).
OUTCOMES_PROMPT_VERSION = "outcomes_v4"


def render_outcomes_prompt(observation: PublicObservation, max_new_tokens: int = 256) -> str:
    """Fill the public-only outcome-prediction prompt template."""

    template = PROMPT_PATH.read_text(encoding="utf-8")
    descriptions = [
        f"{rule.rule_id}: {rule.description}" for rule in observation.candidate_rules
    ]
    candidate_actions = [
        action
        for action in observation.available_actions
        if action.startswith("PROBE_") or action in TASK_ACTIONS
    ]
    return template.format(
        public_history=observation.public_history or "(empty)",
        public_text=observation.public_text,
        candidate_rules=", ".join(descriptions),
        available_actions=", ".join(observation.available_actions),
        candidate_actions=", ".join(candidate_actions),
        public_costs=", ".join(
            f"{key}={value}" for key, value in sorted(observation.public_costs.items())
        ),
        max_new_tokens=max_new_tokens,
    )


def build_prediction_request(
    observation: PublicObservation,
    backend: object,
    *,
    direct: bool = False,
    decoding_seed: int = 7,
    do_sample: bool = False,
) -> ModelRequest:
    """Construct the same public input and cache identity for every runner."""

    descriptions = [
        f"{rule.rule_id}: {rule.description}" for rule in observation.candidate_rules
    ]
    candidate_hash = hashlib.sha256(
        json.dumps(descriptions, sort_keys=True).encode("utf-8")
    ).hexdigest()
    budget = getattr(backend, "max_new_tokens", 256)
    if direct:
        prompt = render_prompt(
            public_history=observation.public_history or "(empty)",
            public_text=observation.public_text,
            candidate_rules=descriptions,
            available_actions=observation.available_actions,
            public_costs=observation.public_costs,
            max_new_tokens=budget,
        )
    else:
        prompt = render_outcomes_prompt(observation, budget)
    input_mode = str(getattr(backend, "input_mode", "text"))
    image = observation.image if input_mode == "rgb" else None
    return ModelRequest(
        model_id=backend.model_id,  # type: ignore[attr-defined]
        prompt=prompt,
        image_hash=observation.public_obs_hash,
        history_hash=observation.public_history_hash,
        image=image,
        input_mode=input_mode,
        available_actions=tuple(observation.available_actions),
        candidate_hash=candidate_hash,
        decoding_seed=decoding_seed,
        prompt_version=PROMPT_VERSION if direct else OUTCOMES_PROMPT_VERSION,
        candidate_rules=tuple(rule.rule_id for rule in observation.candidate_rules),
        max_new_tokens=budget,
        revision=getattr(backend, "revision", None),
        dtype=getattr(backend, "dtype", None),
        device=getattr(backend, "device", None),
        backend_id=type(backend).__qualname__,
        do_sample=do_sample,
    )


def _predict(
    request: ModelRequest, backend: object, cache: ResponseCache, *, direct: bool
) -> tuple[dict | None, ModelResponse, list[str]]:
    cached = cache.get(request)
    if cached is None:
        response = backend.generate(request)  # type: ignore[attr-defined]
        cache.put(request, response)
    else:
        response = cached

    payload, errors = parse_structured_output(response.text, request.available_actions, request.candidate_rules, direct=direct)
    return payload, response, errors


def run_structured_prediction(
    observation: PublicObservation, backend: object, cache: ResponseCache,
    decoding_seed: int = 7, *, do_sample: bool = False,
) -> tuple[dict | None, ModelResponse, list[str]]:
    request = build_prediction_request(observation, backend, decoding_seed=decoding_seed, do_sample=do_sample)
    return _predict(request, backend, cache, direct=False)


def run_direct_prediction(
    observation: PublicObservation, backend: object, cache: ResponseCache, decoding_seed: int = 7,
) -> tuple[dict | None, ModelResponse, list[str]]:
    request = build_prediction_request(observation, backend, direct=True, decoding_seed=decoding_seed)
    return _predict(request, backend, cache, direct=True)
