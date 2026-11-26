"""Checkpoint manifest and verification for the frozen local VLM smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
EXPECTED_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_checkpoint_manifest(snapshot: str | Path) -> dict[str, Any]:
    snapshot = Path(snapshot).resolve()
    if snapshot.name != REVISION:
        raise ValueError(f"snapshot basename is {snapshot.name!r}, expected {REVISION!r}")
    files = []
    for name in EXPECTED_FILES:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen checkpoint file missing: {path}")
        files.append(
            {
                "path": name,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    actual_names = sorted(path.name for path in snapshot.iterdir() if path.is_file())
    if actual_names != sorted(EXPECTED_FILES):
        raise ValueError(f"unexpected snapshot file set: {actual_names}")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_files = sorted(set(index.get("weight_map", {}).values()))
    expected_weight_files = [name for name in EXPECTED_FILES if name.endswith(".safetensors")]
    if weight_files != expected_weight_files:
        raise ValueError(f"index weight files {weight_files} != {expected_weight_files}")
    total_bytes = sum(item["size_bytes"] for item in files)
    if total_bytes != 7_520_892_432:
        raise ValueError(f"unexpected total checkpoint bytes: {total_bytes}")
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "config_model_type": config.get("model_type"),
        "config_architectures": config.get("architectures"),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
        "aggregate_sha256": "sha256:" + hashlib.sha256(_canonical(files).encode("utf-8")).hexdigest(),
        "verified_at_prepare": True,
    }


def verify_checkpoint_manifest(
    manifest: dict[str, Any], *, hash_files: bool = True
) -> None:
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION:
        raise ValueError("checkpoint manifest model/revision mismatch")
    snapshot = Path(manifest["snapshot_path"])
    if snapshot.name != REVISION or not snapshot.is_dir():
        raise ValueError(f"checkpoint snapshot unavailable: {snapshot}")
    files = manifest.get("files", [])
    if [item["path"] for item in files] != list(EXPECTED_FILES):
        raise ValueError("checkpoint manifest file order/set mismatch")
    actual_names = sorted(path.name for path in snapshot.iterdir() if path.is_file())
    if actual_names != sorted(EXPECTED_FILES):
        raise ValueError(f"checkpoint snapshot file set changed: {actual_names}")
    total_bytes = 0
    for item in files:
        path = snapshot / item["path"]
        size = int(path.stat().st_size)
        if size != item["size_bytes"]:
            raise ValueError(f"checkpoint size mismatch: {item['path']}")
        total_bytes += size
        if hash_files and _sha256(path) != item["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {item['path']}")
    if total_bytes != manifest["total_bytes"]:
        raise ValueError("checkpoint total bytes mismatch")
    aggregate = "sha256:" + hashlib.sha256(_canonical(files).encode("utf-8")).hexdigest()
    if aggregate != manifest["aggregate_sha256"]:
        raise ValueError("checkpoint aggregate hash mismatch")
