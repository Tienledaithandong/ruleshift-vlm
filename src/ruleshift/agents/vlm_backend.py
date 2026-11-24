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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruleshift.public_api.schemas import HAZARD_REGIME_IDS, RULE_IDS, TASK_ACTIONS, validate_structured_vlm_output

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
    max_new_tokens: int = 1024   # raised from 768 to fit the full object
    prompt_version: str = PROMPT_VERSION
    candidate_rules: tuple[str, ...] = RULE_IDS
    revision: str | None = None
    dtype: str | None = None
    backend_id: str = ""
    do_sample: bool = False
    device: str | None = None
    #: "text" means the backend never receives the RGB frame; only the prompt
    #: and decoding metadata identify the request. "rgb" includes the pixels.
    input_mode: str = "text"
    #: Local RGB frame (H x W x 3 uint8) for P0. It is not serialized into the
    #: public cache; ``image_hash`` is the stable identity. ``compare=False``
    #: keeps dataclass equality from touching a NumPy array.
    image: Any | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.input_mode not in ("text", "rgb"):
            raise ValueError(f"unsupported input_mode: {self.input_mode}")

    def fingerprint(self) -> str:
        """Hash every public factor that can change the correct response."""

        payload = {
            "model_id": self.model_id,
            "prompt": self.prompt,
            "input_mode": self.input_mode,
            # A text-only backend never sees the frame, so unused hashes must
            # not split the cache or inflate the distinct-request count.
            "image_hash": self.image_hash if self.input_mode == "rgb" else None,
            "history_hash": self.history_hash if self.input_mode == "rgb" else None,
            "available_actions": list(self.available_actions),
            "candidate_hash": self.candidate_hash,
            "decoding_seed": self.decoding_seed,
            "max_new_tokens": self.max_new_tokens,
            "prompt_version": self.prompt_version,
            "candidate_rules": self.candidate_rules,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": self.device,
            "backend_id": self.backend_id,
            "do_sample": self.do_sample,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()


def _flatten_rendered(rendered: Mapping[str, Any]) -> dict[str, Any]:
    """Unwrap processor values that some transformers versions nest once more.

    A single multimodal example can come back with ``input_ids`` holding another
    ``BatchFeature`` instead of a tensor. Flattening keeps the generate call
    identical across processor versions.
    """

    flat: dict[str, Any] = {}
    for key, value in rendered.items():
        if hasattr(value, "items") and not hasattr(value, "shape"):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def build_chat_messages(request: ModelRequest) -> list[dict]:
    """Build one user message, with the image beside the text when present."""

    if request.image is None:
        return [{"role": "user", "content": request.prompt}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": request.prompt},
                {"type": "image", "image": request.image},
            ],
        }
    ]


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
    max_new_tokens: int = 256,
) -> str:
    """Fill the public-only direct prompt template."""

    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        public_history=public_history,
        public_text=public_text,
        candidate_rules=", ".join(candidate_rules),
        available_actions=", ".join(available_actions),
        public_costs=", ".join(f"{k}={v}" for k, v in sorted(public_costs.items())),
        max_new_tokens=max_new_tokens,
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
            if request.prompt_version.startswith("direct_"):
                payload = {"selected_action": rng.choice(available)}
            else:
                payload = self._structured_payload(request, rng)
            if self.invalid_action_every and self._calls % self.invalid_action_every == 0:
                payload["selected_action"] = "NOT_A_MACRO"
            text = json.dumps(payload, sort_keys=True)

        return ModelResponse(
            text=text,
            input_tokens=len(request.prompt.split()) + len(available),
            output_tokens=len(text.split()),
            latency_ms=0.0,
            model_id=self.model_id,
            backend_id="STUB_SYNTHETIC",
        )


    @staticmethod
    def _structured_payload(request: ModelRequest, rng: random.Random) -> dict:
        tasks = [a for a in request.available_actions if a in TASK_ACTIONS]
        probes = [a for a in request.available_actions if a.startswith("PROBE_")]
        menu = [rng.choice(group) for group in (tasks, probes) if group]
        if not menu:
            menu = [request.available_actions[0]]
        predictions = []
        for action in menu:
            under_rules = []
            for rule in request.candidate_rules:
                if action.startswith(("COMMIT_", "PROBE_")):
                    color = action.split("_", 1)[1].lower()
                    positive, negative = "accept", "reject"
                    probability = 0.9 if rule == color else 0.1
                elif action in ("GO_SAFE", "CROSS_BRIDGE"):
                    positive, negative = "success", "failure"
                    probability = 0.9 if rule == "stable" or action == "GO_SAFE" else 0.1
                else:
                    under_rules.append({"rule_id": rule, "possible_outcomes": [{"observation": "none", "p": 1.0}]})
                    continue
                under_rules.append({
                    "rule_id": rule,
                    "possible_outcomes": [
                        {"observation": positive, "p": probability},
                        {"observation": negative, "p": round(1.0 - probability, 10)},
                    ],
                })
            predictions.append({
                "action_id": action,
                "under_rules": under_rules,
                "estimated_future_best_action": tasks[0] if tasks else menu[0],
            })
        return {
            "candidate_probabilities": dict.fromkeys(request.candidate_rules, 1.0 / len(request.candidate_rules)),
            "action_predictions": predictions,
            "selected_action": rng.choice(menu),
        }


class TransformersVLMBackend:
    """Optional real frozen VLM; nothing is loaded or downloaded at import."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        revision: str | None = None,
        dtype: str = "bfloat16",
        local_files_only: bool = True,
        cache_dir: str | Path | None = None,
        max_new_tokens: int = 256,
        input_mode: str = "text",
    ) -> None:
        if input_mode not in ("text", "rgb"):
            raise ValueError(f"unsupported input_mode: {input_mode}")
        self.model_id = model_id
        self.device = device
        self.revision = revision
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self.input_mode = input_mode
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.max_new_tokens = max_new_tokens
        if dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"unsupported dtype: {dtype}")
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

        kwargs = {"revision": self.revision, "local_files_only": self.local_files_only, "cache_dir": self.cache_dir}
        self._processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        import torch

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]
        self._model = model_class.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, **kwargs
        ).to(self.device)
        self._model.eval()

    def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
        self._load()
        import time

        started = time.perf_counter()
        if request.input_mode != self.input_mode:
            raise ValueError(
                f"request input mode {request.input_mode!r} does not match backend "
                f"mode {self.input_mode!r}"
            )
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        if self.input_mode == "rgb":
            if request.image is None:
                raise ValueError("RGB input mode requires request.image")
            from PIL import Image

            content.append({"type": "image", "image": Image.fromarray(request.image)})
        messages = [{"role": "user", "content": content}]
        rendered = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if not isinstance(rendered, Mapping):
            rendered = {"input_ids": rendered}
        rendered = _flatten_rendered(rendered)
        inputs = {key: value.to(self.device) for key, value in rendered.items()}
        import torch

        kwargs = {"max_new_tokens": request.max_new_tokens, "do_sample": request.do_sample}
        if request.do_sample:
            device = torch.device(self.device)
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.default_generator.manual_seed(request.decoding_seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(request.decoding_seed)
                outputs = self._model.generate(**inputs, **kwargs)
        else:
            outputs = self._model.generate(**inputs, **kwargs)
        generated = outputs[:, inputs["input_ids"].shape[-1] :]
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
    text: str, available_actions: Sequence[str], candidate_rules: Sequence[str] | None = None,
    *, direct: bool | None = None,
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

    if direct:
        if not isinstance(payload, dict) or set(payload) != {"selected_action"}:
            return None, ["direct response must contain only selected_action"]
        if payload["selected_action"] not in available_actions:
            return None, ["selected_action is not in the available action menu"]
        return payload, []

    if candidate_rules is None:
        candidate_rules = HAZARD_REGIME_IDS if "CROSS_BRIDGE" in available_actions else RULE_IDS
    errors = validate_structured_vlm_output(payload, tuple(available_actions), tuple(candidate_rules))
    if not errors and direct is not None:
        predictions = [row["action_id"] for row in payload["action_predictions"]]
        if payload["selected_action"] not in predictions:
            errors.append("selected_action must have a prediction")
        tasks = TASK_ACTIONS.intersection(available_actions)
        probes = {a for a in available_actions if a.startswith("PROBE_")}
        if len(predictions) > 2 or any(group and not group.intersection(predictions) for group in (tasks, probes)):
            errors.append("structured response must cover the offered task and probe alternatives")
    if errors:
        return None, errors
    return payload, []
