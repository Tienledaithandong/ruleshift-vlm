"""Prompt templates must describe the JSON shape without a copyable answer."""

from __future__ import annotations

from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "src" / "ruleshift" / "prompts"
TEMPLATES = ("direct_v1.txt", "outcomes_v1.txt")

#: Literal values from the earlier worked example. When these appeared in the
#: template, the frozen model echoed them verbatim for every distinct input.
FORBIDDEN_EXAMPLE_VALUES = ("0.34", "0.33", "PROBE_RED", "COMMIT_RED")


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_has_no_copyable_example_answer(name: str) -> None:
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    for literal in FORBIDDEN_EXAMPLE_VALUES:
        assert literal not in text, f"{name} still supplies the literal {literal!r}"


def test_outcomes_template_asks_for_values_derived_from_the_public_text() -> None:
    text = (PROMPTS_DIR / "outcomes_v1.txt").read_text(encoding="utf-8").lower()
    assert "placeholder" in text
    assert "derive the probabilities from the public evidence" in text
    assert "repeats the same" not in text


def test_direct_prompt_requests_only_an_action() -> None:
    text = (PROMPTS_DIR / "direct_v1.txt").read_text(encoding="utf-8").lower()
    assert 'containing only "selected_action"' in text
    assert "action_predictions" not in text
    assert "candidate_probabilities" not in text


def test_outcomes_prompt_limits_predictions_to_fit_the_output_budget() -> None:
    text = (PROMPTS_DIR / "outcomes_v1.txt").read_text(encoding="utf-8").lower()
    assert "at most two candidate actions" in text
    assert "within the {max_new_tokens}-token generation limit" in text
