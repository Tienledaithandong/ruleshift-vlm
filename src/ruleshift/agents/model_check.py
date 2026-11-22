"""Preflight checks for the optional real frozen-VLM backend.

Nothing here imports or downloads weights implicitly. The stub remains the
default backend for tests and setup verification.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelReadiness:
    model_id: str
    ready: bool
    local_checkpoint: bool
    cuda_available: bool
    messages: tuple[str, ...]


def _local_checkpoint_exists(model_id: str) -> bool:
    """Check the standard Hugging Face hub cache for a local checkpoint."""

    candidate = Path(model_id)
    if candidate.exists():
        return True
    parts = model_id.split("/")
    if len(parts) != 2:
        return False
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    return (hub / f"models--{parts[0]}--{parts[1]}").exists()


def check_model(
    model_id: str,
    local_files_only: bool = True,
    allow_download: bool = False,
) -> ModelReadiness:
    """Return an explicit go/no-go assessment for a real model run."""

    messages: list[str] = []
    has_torch = importlib.util.find_spec("torch") is not None
    has_transformers = importlib.util.find_spec("transformers") is not None
    if not has_torch or not has_transformers:
        messages.append("BLOCK: torch and transformers must be installed")

    local = _local_checkpoint_exists(model_id)
    if local_files_only and not allow_download and not local:
        messages.append(
            f"BLOCK: local checkpoint for {model_id!r} not found; "
            "download it explicitly or pass --allow-download"
        )

    cuda = False
    if has_torch:
        import torch

        cuda = torch.cuda.is_available()
        if not cuda:
            messages.append("INFO: no CUDA device; CPU inference will be very slow")

    if local:
        messages.append("INFO: local checkpoint found")
    if allow_download:
        messages.append("INFO: downloads explicitly enabled")

    blocking = any(message.startswith("BLOCK") for message in messages)
    return ModelReadiness(
        model_id=model_id,
        ready=not blocking,
        local_checkpoint=local,
        cuda_available=cuda,
        messages=tuple(messages),
    )
