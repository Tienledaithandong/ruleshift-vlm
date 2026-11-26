"""Frozen direct-action parser for the bounded VLM smoke.

The parser accepts exactly one JSON object with exactly one key,
"selected_action", and validates the action against the request's feasible
menu. A parse/schema/infeasible failure becomes a deterministic COMMIT
fallback, but it remains flagged as a fallback and is never counted as a valid
optimal decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ruleshift.agents.vlm_backend import parse_structured_output

PARSER_VERSION = "vlm-smoke-direct-parser-v1"
FALLBACK_ACTION = "COMMIT"


@dataclass(frozen=True)
class ParsedAction:
    raw_text: str
    selected_action: str
    valid: bool
    fallback: bool
    errors: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "selected_action": self.selected_action,
            "valid": self.valid,
            "fallback": self.fallback,
            "errors": list(self.errors),
        }


def parse_action(raw_text: str, available_actions: Sequence[str]) -> ParsedAction:
    payload, errors = parse_structured_output(
        raw_text, tuple(available_actions), direct=True
    )
    if errors or payload is None:
        return ParsedAction(
            raw_text=raw_text,
            selected_action=FALLBACK_ACTION,
            valid=False,
            fallback=True,
            errors=tuple(errors or ["empty direct response"]),
        )
    action = str(payload["selected_action"])
    if action not in available_actions:
        return ParsedAction(
            raw_text=raw_text,
            selected_action=FALLBACK_ACTION,
            valid=False,
            fallback=True,
            errors=("selected_action is not feasible",),
        )
    return ParsedAction(
        raw_text=raw_text,
        selected_action=action,
        valid=True,
        fallback=False,
        errors=(),
    )
