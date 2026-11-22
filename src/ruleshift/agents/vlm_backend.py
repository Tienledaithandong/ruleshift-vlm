"""VLM request/response contract and backends.

The default backend here is a DETERMINISTIC STUB. It exists so the offline
harness can be tested and demonstrated on CPU without downloading a model.
Any manifest produced with it is a synthetic dry run, never VLM evidence.
A lazy transformers backend is provided for a real frozen model when one is
actually available.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ruleshift.public_api.schemas import validate_structured_vlm_output

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "direct_v1.txt"
PROMPT_VERSION = "direct_v4"


@dataclass(frozen=True)
class ModelRequest:
    """Immutable, public-only model input and its cache identity."""

    model_id: str
    prompt: str
    image_hash: str
    history_hash: str
    available_actions: tuple[str, ...]
    candidate_hash: str
    decoding_seed: int
    max_new_tokens: int = 768
    prompt_version: str = PROMPT_VERSION

    def fingerprint(self) -> str:
        """Hash every public factor that can change the correct response."""

        payload = {
            "model_id": self.model_id,
            "prompt": self.prompt,
            "image_hash": self.image_hash,
            "history_hash": self.history_hash,
            "available_actions": list(self.available_actions),
            "candidate_hash": self.candidate_hash,
            "decoding_seed": self.decoding_seed,
            "max_new_tokens": self.max_new_tokens,
            "prompt_version": self.prompt_version,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ModelResponse:
    """Raw model output plus accounting fields."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    backend_id: str


def render_prompt(
    *,
    public_history: str,
    public_text: str,
    candidate_rules: Sequence[str],
    available_actions: Sequence[str],
    public_costs: Mapping[str, float],
) -> str:
    """Fill the public-only direct prompt template."""

    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        public_history=public_history,
        public_text=public_text,
        candidate_rules=", ".join(candidate_rules),
        available_actions=", ".join(available_actions),
        public_costs=", ".join(f"{k}={v}" for k, v in sorted(public_costs.items())),
    )


class StubVLMBackend:
    """Deterministic stand-in used only for harness tests and dry runs."""

    def __init__(
        self,
        model_id: str = "stub-v0",
        malformed_every: int = 0,
        invalid_action_every: int = 0,
    ) -> None:
        self.model_id = model_id
        self.malformed_every = malformed_every
        self.invalid_action_every = invalid_action_every
        self._calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        seed = int(request.fingerprint().removeprefix("sha256:")[:8], 16)
        rng = random.Random(seed ^ request.decoding_seed)
        available = list(request.available_actions)

        if self.malformed_every and self._calls % self.malformed_every == 0:
            text = "{ this is not valid JSON"
        else:
            selected = rng.choice(available)
            if self.invalid_action_every and self._calls % self.invalid_action_every == 0:
                selected = "NOT_A_MACRO"
            future = next(
                (action for action in available if action.startswith("COMMIT_")),
                available[0],
            )
            # Emit a small candidate menu so selectors have alternatives to
            # compare. This is still synthetic; it only exercises the harness.
            commits = [a for a in available if a.startswith("COMMIT_")]
            probes = [a for a in available if a.startswith("PROBE_")]
            menu = (commits[:2] + probes[:1]) or [selected]

            def outcome(rule: str, color: str) -> list[dict]:
                acceptance = 0.9 if rule == color else 0.1
                return [
                    {"observation": "accept", "p": acceptance},
                    {"observation": "reject", "p": round(1.0 - acceptance, 10)},
                ]

            predictions = []
            for action in menu:
                color = action.split("_", 1)[1].lower()
                predictions.append(
                    {
                        "action_id": action,
                        "under_rules": [
                            {
                                "rule_id": rule,
                                "possible_outcomes": outcome(rule, color),
                            }
                            for rule in ("red", "blue", "green")
                        ],
                        "estimated_future_best_action": future,
                    }
                )
            payload = {
                "candidate_probabilities": {"red": 0.5, "blue": 0.3, "green": 0.2},
                "action_predictions": predictions,
                "selected_action": selected,
            }
            text = json.dumps(payload, sort_keys=True)

        return ModelResponse(
            text=text,
            input_tokens=len(request.prompt.split()) + len(available),
            output_tokens=len(text.split()),
            latency_ms=0.0,
            model_id=self.model_id,
            backend_id="STUB_SYNTHETIC",
        )


class TransformersVLMBackend:
    """Optional real frozen VLM; nothing is loaded or downloaded at import."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        revision: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.revision = revision
        self.dtype = dtype
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers backend needs transformers") from exc

        model_class = None
        for candidate in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForMultimodalLM",
        ):
            model_class = getattr(__import__("transformers"), candidate, None)
            if model_class is not None:
                break
        if model_class is None:  # pragma: no cover
            raise RuntimeError("no compatible transformers auto model class found")

        kwargs = {"revision": self.revision} if self.revision else {}
        self._processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        import torch

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.dtype, torch.bfloat16)
        self._model = model_class.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, **kwargs
        ).to(self.device)
        self._model.eval()

    def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
        self._load()
        import time

        started = time.perf_counter()
        messages = [{"role": "user", "content": request.prompt}]
        rendered = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(rendered, dict):
            rendered = {"input_ids": rendered}
        inputs = {key: value.to(self.device) for key, value in rendered.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        outputs = self._model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        generated = outputs[:, prompt_tokens:]
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return ModelResponse(
            text=text,
            # Prompt and completion are counted separately: reporting the total
            # sequence length for both made every run look like it had used the
            # same number of prompt and output tokens.
            input_tokens=prompt_tokens,
            output_tokens=int(generated.shape[-1]),
            latency_ms=(time.perf_counter() - started) * 1000,
            model_id=self.model_id,
            backend_id="TRANSFORMERS_FROZEN",
        )


def parse_structured_output(
    text: str, available_actions: Sequence[str]
) -> tuple[dict | None, list[str]]:
    """Parse and validate raw output; return errors and a safe fallback."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    try:
        payload = json.loads(cleaned)
    except (ValueError, TypeError) as exc:
        return None, [f"invalid JSON: {exc}"]

    errors = validate_structured_vlm_output(payload)
    if isinstance(payload, dict) and payload.get("selected_action") not in set(available_actions):
        errors.append("selected_action is not in the available action menu")
    if errors:
        return None, errors
    return payload, []
