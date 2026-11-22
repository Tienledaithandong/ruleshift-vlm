"""Public-only response cache keyed by the full immutable request fingerprint."""

from __future__ import annotations

import json
from pathlib import Path

from .vlm_backend import ModelRequest, ModelResponse


class ResponseCache:
    """Append-only JSONL cache; never stores hidden state or labels."""

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: dict[str, ModelResponse] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._entries[record["fingerprint"]] = ModelResponse(**record["response"])

    def get(self, request: ModelRequest) -> ModelResponse | None:
        return self._entries.get(request.fingerprint())

    def put(self, request: ModelRequest, response: ModelResponse) -> None:
        fingerprint = request.fingerprint()
        record = {
            "fingerprint": fingerprint,
            "image_hash": request.image_hash,
            "history_hash": request.history_hash,
            "available_actions": list(request.available_actions),
            "candidate_hash": request.candidate_hash,
            "response": {
                "text": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "model_id": response.model_id,
                "backend_id": response.backend_id,
            },
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._entries[fingerprint] = response
