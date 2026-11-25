"""Privileged, evaluator-only trace for RuleShift-Grid episodes.

This module is the *only* place that assembles a row containing the true rule
and the shift schedule. It is imported by the evaluator and tests, never by
agent code, and its rows are written to a separate trace file from public
decision rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GridEvaluatorTrace:
    """One episode's evaluator-only summary, including hidden ground truth."""

    scenario_id: str
    run_seed: int
    method: str
    true_rule: str
    initial_rule: str
    shift_time: int | None
    scenario_type: str
    map_seed: int
    success: bool
    total_return: float
    env_steps: int
    tool_units: int
    advisor_calls: int
    adaptation_steps: int | None
    false_alarm: bool
    final_rule_estimate: str
    #: Scheduled shift time (the existing shift_time field) and the observed
    #: shift time. None means the episode ended before the shift occurred.
    actual_shift_time: int | None = None
    shift_censored: bool = False
    #: Policy's shift-probability threshold crossing, not ground-truth detection.
    belief_shift_event_time: int | None = None
    belief_shift_false_alarm: bool = False
    termination_reason: str = "unknown"

    def to_row(self) -> dict[str, Any]:
        """Return a JSON-serializable privileged row."""

        return asdict(self)


__all__ = ["GridEvaluatorTrace"]
