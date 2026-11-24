"""Preflight checks for the optional real frozen-VLM backend.

Nothing here imports or downloads weights implicitly. The stub remains the
default backend for tests and setup verification.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelReadiness:
    model_id: str
    ready: bool
    local_checkpoint: bool
    cuda_available: bool | None
    messages: tuple[str, ...]


def _local_checkpoint_exists(
    model_id: str, revision: str | None = None, cache_dir: str | Path | None = None
) -> bool:
    """Check the selected snapshot, including every indexed weight shard."""

    snapshot = Path(model_id)
    if not snapshot.is_dir():
        hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache = Path(cache_dir or os.getenv("HF_HUB_CACHE", hf_home / "hub"))
        repo = cache / ("models--" + model_id.replace("/", "--"))
        revision = revision or "main"
        ref = repo / "refs" / revision
        if ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
        snapshot = repo / "snapshots" / revision
    if not all((snapshot / name).is_file() for name in ("config.json", "preprocessor_config.json")):
        return False
    if not any((snapshot / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        return False
    if any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = snapshot / name
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                if shards and all((snapshot / shard).is_file() for shard in shards):
                    return True
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
    return False


def check_model(
    model_id: str,
    local_files_only: bool = True,
    allow_download: bool = False,
    *, revision: str | None = None, device: str = "cpu", cache_dir: str | Path | None = None,
) -> ModelReadiness:
    """Return an explicit go/no-go assessment for a real model run."""

    messages: list[str] = []
    if not (device == "cpu" or re.fullmatch(r"cuda(?::[0-9]+)?", device)):
        messages.append(f"BLOCK: unsupported device {device!r}, choose cpu or cuda[:index]")
    has_torch = importlib.util.find_spec("torch") is not None
    has_transformers = importlib.util.find_spec("transformers") is not None
    if not has_torch or not has_transformers:
        messages.append("BLOCK: torch and transformers must be installed")

    local = _local_checkpoint_exists(model_id, revision, cache_dir)
    if not local and (local_files_only or not allow_download):
        messages.append(
            f"BLOCK: local checkpoint for {model_id!r} not found; "
            "download it explicitly or pass --allow-download"
        )

    cuda = None
    if has_torch and device.startswith("cuda"):
        import torch

        cuda = torch.cuda.is_available()
        if not cuda:
            messages.append(f"BLOCK: requested device {device!r} needs CUDA")
        elif ":" in device:
            index = device.partition(":")[2]
            if not index.isdecimal() or int(index) >= torch.cuda.device_count():
                messages.append(f"BLOCK: requested CUDA device {device!r} does not exist")

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


def model_settings(models: dict, key: str, device: str | None, allow_download: bool) -> dict:
    """Resolve the same loader options for preflight and inference."""

    model = models["models"][key]
    runtime = models["runtime"]
    download = allow_download or (runtime["allow_download"] and not runtime["local_files_only"])
    return {
        "model_id": model["hf_id"],
        "revision": model["revision"],
        "device": device or model["device"],
        "cache_dir": runtime.get("cache_dir"),
        "allow_download": download,
        "local_files_only": not download,
    }


def readiness_errors(settings: dict, stage: str, test_config: dict) -> list[str]:
    errors = []
    if not re.fullmatch(r"[0-9a-fA-F]{40}", settings["revision"] or ""):
        errors.append("pin the model revision to a full commit hash")
    if stage == "locked" and (
        test_config.get("status") != "READY"
        or not re.fullmatch(r"[0-9a-fA-F]{40}", test_config.get("frozen_commit") or "")
    ):
        errors.append("locked evaluation requires READY and a full frozen_commit")
    if stage == "locked" and test_config.get("status") == "READY":
        run = test_config.get("locked_run") or {}
        if not all(run.get(field) for field in ("model_id", "revision", "scenarios", "methods", "seeds", "implementation_sha256")):
            errors.append("locked_run requires a model, scenarios, methods, seeds and the implementation hash")
        elif run["revision"] != settings["revision"] or (
            settings.get("model_id") and run["model_id"] != settings["model_id"]
        ):
            errors.append("selected model must match the frozen model and revision")
    return errors
