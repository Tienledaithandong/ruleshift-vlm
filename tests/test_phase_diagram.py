"""The CPU phase diagram must contain both probe-favourable and task-favourable cells."""

from __future__ import annotations

import math

import yaml

from ruleshift.envs.config import load_dev_grid
from ruleshift.evaluator.phase_diagram import compute_phase_grid, write_report


def test_phase_grid_has_both_signs_and_finite_values() -> None:
    rows = compute_phase_grid()
    assert len(rows) > 8
    assert any(row.delta_probe > 0 for row in rows)
    assert any(row.delta_probe < 0 for row in rows)
    assert all(math.isfinite(row.delta_probe) for row in rows)


def test_report_writes_csv_and_markdown(tmp_path) -> None:
    config = load_dev_grid()
    config["grid"]["probe_detour_steps"] = [2]
    config["grid"]["feedback_noise_rate"] = [0.0]
    config["environment"]["lockout_steps_options"] = [4]
    config["environment"]["max_env_steps"] = 1
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    rows = compute_phase_grid(config_path)
    (cell,) = rows
    assert (cell.probe_detour_steps, cell.noise, cell.lockout, cell.horizon) == (2, 0.0, 4, 1)
    assert cell.delta_probe == 0.0 and cell.preferred == "tie"
    paths = write_report(rows, tmp_path)
    assert paths["csv"].exists()
    assert paths["markdown"].exists()
    assert paths["png"].exists()
    assert "Ties: 1." in paths["markdown"].read_text(encoding="utf-8")
