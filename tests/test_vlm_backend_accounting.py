"""CPU-only accounting checks for the optional frozen-VLM backend."""

from __future__ import annotations

import pytest

from ruleshift.agents.vlm_backend import ModelRequest, TransformersVLMBackend


def test_generate_reports_prompt_and_completion_tokens_separately():
    torch = pytest.importorskip("torch")

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["role"] == "user"
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def batch_decode(self, generated, skip_special_tokens=True):
            assert generated.shape == (1, 2)
            return ["ok"]

    class FakeModel:
        def generate(self, **inputs):
            assert inputs["input_ids"].shape == (1, 3)
            return torch.tensor([[1, 2, 3, 4, 5]])

    backend = TransformersVLMBackend(model_id="fake/frozen", device="cpu")
    backend._processor = FakeProcessor()
    backend._model = FakeModel()
    backend._load = lambda: None
    response = backend.generate(
        ModelRequest(
            model_id="fake/frozen",
            prompt="public prompt",
            image_hash="sha256:test",
            history_hash="sha256:history",
            available_actions=("UP",),
            candidate_hash="sha256:candidates",
            decoding_seed=0,
        )
    )
    assert response.text == "ok"
    assert response.input_tokens == 3
    assert response.output_tokens == 2
    assert response.input_tokens != response.output_tokens
    assert response.backend_id == "TRANSFORMERS_FROZEN"
