"""Regression tests for the no-rerun P0 audits."""

from __future__ import annotations

from pathlib import Path

from ruleshift.envs.grid.config import GridConfig
from scripts.audit_grid_existing_rows import audit_rows, parse_scenario_id
from scripts.audit_grid_map_exposure import layout_audit, scan_text


def test_legacy_audit_separates_task_and_terminal_map_outcomes():
    rows = [
        {
            "scenario_id": "m0-s0-identification-base-None",
            "method": "m",
            "success": True,
            "final_rule_estimate": "base",
            "tool_units": 0,
        },
        {
            "scenario_id": "m0-s0-identification-remap_swap-None",
            "method": "m",
            "success": True,
            "final_rule_estimate": "base",
            "tool_units": 0,
        },
        {
            "scenario_id": "m1-s0-identification-base-None",
            "method": "m",
            "success": False,
            "final_rule_estimate": "base",
            "tool_units": 1,
        },
        {
            "scenario_id": "m1-s0-change_detection-base-slippery",
            "method": "m",
            "success": True,
            "final_rule_estimate": "base",
            "tool_units": 1,
            "shift_detected": False,
        },
    ]
    report = audit_rows(rows, "synthetic")
    method = report["methods"]["m"]
    assert method["rows"] == 4
    assert method["task_successes"] == 3
    assert method["map_correct"] == 3
    assert method["old_final_rule_matches_initial_requiring_success"] == 2
    assert method["successful_misdiagnosis"] == 1
    assert method["map_correct_but_task_failure"] == 1
    assert method["base_identification_rows"] == 2
    assert method["base_task_successes"] == 1
    assert method["static_identification_rows"] == 3
    assert method["static_task_successes"] == 2
    assert method["static_map_correct"] == 2
    assert method["static_task_and_map_correct"] == 1
    assert method["static_successful_misdiagnosis"] == 1
    assert method["static_map_correct_but_task_failure"] == 1
    assert method["shift_flag_mismatches"] == 0


def test_legacy_audit_filters_maps_and_reports_duplicates_by_method():
    rows = [
        {
            "scenario_id": "m0-s0-identification-base-None",
            "method": "m",
            "success": True,
            "final_rule_estimate": "base",
        },
        {
            "scenario_id": "m0-s0-identification-base-None",
            "method": "m",
            "success": True,
            "final_rule_estimate": "base",
        },
        {
            "scenario_id": "m5-s0-identification-base-None",
            "method": "m",
            "success": True,
            "final_rule_estimate": "base",
        },
        {
            "scenario_id": "m0-s0-identification-base-None",
            "method": "other",
            "success": True,
            "final_rule_estimate": "base",
        },
    ]
    report = audit_rows(rows, "synthetic", map_max=0)
    assert report["rows"] == 3
    assert report["methods"]["m"]["rows"] == 2
    assert report["methods"]["m"]["duplicate_scenario_ids"] == 1
    assert report["methods"]["other"]["rows"] == 1


def test_scenario_id_parser_rejects_unknown_ids():
    parsed = parse_scenario_id("m12-s3-change_detection-base-slippery")
    assert parsed == {
        "map_seed": 12,
        "run_seed": 3,
        "scenario_type": "change_detection",
        "initial_rule": "base",
        "shift_rule": "slippery",
    }
    assert parse_scenario_id("not-a-scenario") is None


def test_map_exposure_scan_and_layout_audit_are_non_running(tmp_path: Path):
    source = tmp_path / "notes.md"
    source.write_text("m0-s0-identification-base-None\nm7-s1-noise\n", encoding="utf-8")
    counts = scan_text(source)
    assert counts[0] == 1
    assert counts[7] == 1
    audit = layout_audit(GridConfig(), dev_max=1, candidate_maps=[2, 3])
    assert audit["dev_maps"] == 2
    assert audit["candidate_unique_layouts"] == 2
    assert audit["candidate_dev_collisions"] == {}
