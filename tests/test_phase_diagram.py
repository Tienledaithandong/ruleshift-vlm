"""The CPU phase diagram must contain both probe-favourable and task-favourable cells."""

from __future__ import annotations

import math

from ruleshift.evaluator.phase_diagram import compute_phase_grid, write_report


def test_phase_grid_has_both_signs_and_finite_values() -> None:
    rows = compute_phase_grid()
    assert len(rows) > 8
    assert any(row.delta_probe > 0 for row in rows)
    assert any(row.delta_probe < 0 for row in rows)
    assert all(math.isfinite(row.delta_probe) for row in rows)


def test_report_writes_csv_and_markdown(tmp_path) -> None:
    rows = compute_phase_grid()
    paths = write_report(rows, tmp_path)
    assert paths["csv"].exists()
    assert paths["markdown"].exists()
    assert paths["png"].exists()
