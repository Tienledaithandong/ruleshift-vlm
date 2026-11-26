"""Frozen model and stub backends for the bounded VLM decision smoke.

Nothing here is loaded at import time.  The real backend accepts both text and
RGB requests in the same run, records vision-tensor provenance, and uses the
locally verified Qwen2.5-VL snapshot with greedy decoding.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ruleshift.agents.vlm_backend import (
    ModelRequest,
    ModelResponse,
    _flatten_rendered,
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def _tensor_sha256(value: Any) -> str:
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.asarray(value)
    return "sha256:" + hashlib.sha256(array.tobytes()).hexdigest()


def _tensor_shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    return [int(item) for item in shape] if shape is not None else None


class FrozenQwenBackend:
    """Exact local Qwen2.5-VL-3B-Instruct; no download, training, or substitution."""

    backend_id = "TRANSFORMERS_FROZEN_QWEN25_VL_3B"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.model_id = MODEL_ID
        self.revision = REVISION
        self._model = None
        self._processor = None
        self.last_metadata: dict[str, Any] = {}
        self.load_seconds: float | None = None
        self.load_metadata: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load_processor_only(self) -> None:
        if self._processor is not None:
            return
        from transformers import AutoProcessor

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"frozen model snapshot not found: {self.model_path}")
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True
        )

    def load(self) -> None:
        if self.is_loaded:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"frozen model snapshot not found: {self.model_path}")
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"the frozen smoke requires CUDA, got device={self.device!r}")
        torch.cuda.empty_cache()
        started = time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                **load_kwargs,
            )
        except TypeError:
            self._model = AutoModelForImageTextToText.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16,
                **load_kwargs,
            )
        self._model.eval()
        self._model.to(self.device)
        if hasattr(self._model, "generation_config"):
            self._model.generation_config.do_sample = False
        self.load_seconds = time.perf_counter() - started
        self.load_metadata = {
            "model_path": str(self.model_path),
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "load_seconds": self.load_seconds,
            "parameters": int(sum(parameter.numel() for parameter in self._model.parameters())),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "cuda_free_bytes": int(torch.cuda.mem_get_info()[0]),
        }

    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        if self._processor is None:
            raise RuntimeError("processor is not loaded")
        import torch
        from PIL import Image

        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        if request.input_mode == "rgb":
            if request.image is None:
                raise ValueError("RGB request has no image")
            content.append({"type": "image", "image": Image.fromarray(request.image)})
        rendered = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if not hasattr(rendered, "items"):
            rendered = {"input_ids": rendered}
        rendered = _flatten_rendered(rendered)
        inputs = {key: value.to(self.device) for key, value in rendered.items()}
        vision = inputs.get("pixel_values")
        return {
            "inputs": inputs,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "vision_tensor_sha256": _tensor_sha256(vision) if vision is not None else None,
            "vision_tensor_shape": _tensor_shape(vision),
            "image_grid_thw": (
                inputs.get("image_grid_thw").detach().cpu().tolist()
                if inputs.get("image_grid_thw") is not None
                else None
            ),
            "input_ids_sha256": _tensor_sha256(inputs["input_ids"]),
        }

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.load()
        import torch

        torch.cuda.empty_cache()
        started = time.perf_counter()
        prepared = self.prepare(request)
        inputs = prepared["inputs"]
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        generation_finished = time.perf_counter()
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated = outputs[:, prompt_tokens:]
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        self.last_metadata = {
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "input_tokens": prompt_tokens,
            "input_ids_sha256": prepared["input_ids_sha256"],
            "vision_tensor_sha256": prepared["vision_tensor_sha256"],
            "vision_tensor_shape": prepared["vision_tensor_shape"],
            "image_grid_thw": prepared["image_grid_thw"],
            "image_seen": prepared["vision_tensor_sha256"] is not None,
            "gpu_time_s": generation_finished - generation_started,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        return ModelResponse(
            text=text,
            input_tokens=prompt_tokens,
            output_tokens=int(generated.shape[-1]),
            latency_ms=float(self.last_metadata["latency_ms"]),
            model_id=self.model_id,
            backend_id=self.backend_id,
        )


class StubSmokeBackend:
    """Deterministic CPU stub; it exercises the same request and image path."""

    backend_id = "STUB_SYNTHETIC"

    def __init__(
        self,
        *,
        malformed_every: int = 0,
        invalid_action_every: int = 0,
    ) -> None:
        self.model_id = "stub-vlm-smoke"
        self.malformed_every = malformed_every
        self.invalid_action_every = invalid_action_every
        self.calls = 0
        self.last_metadata: dict[str, Any] = {}
        self.image_calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if request.input_mode == "rgb":
            if request.image is None:
                raise ValueError("stub RGB request has no image")
            self.image_calls += 1
            image_hash = "sha256:" + hashlib.sha256(
                np.asarray(request.image, dtype=np.uint8).tobytes()
            ).hexdigest()
            vision_sha = image_hash
            vision_shape = list(request.image.shape)
            image_seen = True
        else:
            vision_sha = None
            vision_shape = None
            image_seen = False
        if self.malformed_every and self.calls % self.malformed_every == 0:
            text = "{ this is not valid JSON"
        else:
            actions = list(request.available_actions)
            action = actions[(self.calls - 1) % len(actions)]
            if self.invalid_action_every and self.calls % self.invalid_action_every == 0:
                action = "NOT_A_MACRO"
            text = json.dumps({"selected_action": action}, sort_keys=True)
        self.last_metadata = {
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "input_tokens": len(request.prompt.split()),
            "input_ids_sha256": None,
            "vision_tensor_sha256": vision_sha,
            "vision_tensor_shape": vision_shape,
            "image_grid_thw": None,
            "image_seen": image_seen,
            "gpu_time_s": 0.0,
            "latency_ms": 0.0,
            "cuda_peak_allocated_bytes": 0,
            "cuda_peak_reserved_bytes": 0,
        }
        return ModelResponse(
            text=text,
            input_tokens=len(request.prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=0.0,
            model_id=self.model_id,
            backend_id=self.backend_id,
        )
