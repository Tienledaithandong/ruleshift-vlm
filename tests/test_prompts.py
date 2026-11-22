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


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_asks_for_values_derived_from_the_public_text(name: str) -> None:
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8").lower()
    assert "placeholder" in text
    assert "derive the numbers from the public text" in text
    assert "different inputs" in text


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_keeps_the_answer_inside_the_token_budget(name: str) -> None:
    """An unbounded answer overflowed the budget and never closed its JSON."""
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8").lower()
    assert "compact on one line" in text
    assert "within the token budget" in text


def test_direct_prompt_limits_predictions_to_fit_the_output_budget() -> None:
    text = (PROMPTS_DIR / "direct_v1.txt").read_text(encoding="utf-8").lower()
    assert "at most two entries" in text
    assert "never add a third" in text


def test_outcomes_prompt_limits_predictions_to_fit_the_output_budget() -> None:
    text = (PROMPTS_DIR / "outcomes_v1.txt").read_text(encoding="utf-8").lower()
    assert "at most two candidate actions" in text


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_requires_a_public_cost_tradeoff(name: str) -> None:
    """The model probed unconditionally until the trade-off was spelled out."""
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8").lower()
    assert "decision procedure" in text
    assert "probe_detour" in text
    assert "do not probe by default" in text
