"""Week 9 held-out evaluation scaffolding and honest result accounting."""

from __future__ import annotations

import csv
import json

from ruleshift.evaluator.heldout import (
    build_heldout_family_a,
    run_heldout_evaluation,
    write_result_files,
)

METHODS = ("direct_stub", "diva_stub", "m1_stub", "immediate_stub")


def dev_parameter_values() -> dict[str, set[float]]:
    from ruleshift.evaluator.closed_loop import build_family_a_scenarios

    scenarios = build_family_a_scenarios()
    return {
        "detour": {float(c.probe_detour_steps) for _, c in scenarios},
        "noise": {float(c.feedback_noise) for _, c in scenarios},
        "lockout": {float(c.lockout_steps) for _, c in scenarios},
    }


def test_heldout_grid_is_disjoint_from_dev_grid() -> None:
    dev = dev_parameter_values()
    heldout = build_heldout_family_a()
    assert len(heldout) >= 8
    assert {float(c.probe_detour_steps) for _, c in heldout}.isdisjoint(dev["detour"])
    assert {float(c.feedback_noise) for _, c in heldout}.isdisjoint(dev["noise"])
    assert {float(c.lockout_steps) for _, c in heldout}.isdisjoint(dev["lockout"])


def test_heldout_evaluation_is_deterministic() -> None:
    scenarios = build_heldout_family_a()[:4]
    first = run_heldout_evaluation(scenarios, METHODS, seeds=(101,))
    second = run_heldout_evaluation(scenarios, METHODS, seeds=(101,))
    assert first.summaries == second.summaries
    assert first.method_stats == second.method_stats


def test_method_stats_include_cost_and_call_accounting() -> None:
    scenarios = build_heldout_family_a()[:4]
    result = run_heldout_evaluation(scenarios, METHODS, seeds=(101,))
    for method in METHODS:
        stats = result.method_stats[method]
        assert stats["episodes"] == len(scenarios)
        assert stats["mean_return"] == stats["total_return"] / stats["episodes"]
        assert stats["vlm_calls"] >= 0
        assert stats["invalid_actions"] == 0
        assert set(("json_valid", "probes", "tokens", "latency_ms")) <= set(stats)


def test_result_files_and_report_are_written(tmp_path) -> None:
    scenarios = build_heldout_family_a()[:4]
    result = run_heldout_evaluation(scenarios, METHODS, seeds=(101,))
    paths = write_result_files(result, tmp_path / "results", tmp_path / "reports")

    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(scenarios) * len(METHODS)
    assert {"scenario_id", "method", "return", "vlm_calls"} <= set(rows[0])

    summary = json.loads(paths["paired_json"].read_text(encoding="utf-8"))
    assert summary
    for entry in summary.values():
        assert {"mean_difference", "ci95_low", "ci95_high"} <= set(entry)

    report = paths["report"].read_text(encoding="utf-8")
    assert "SYNTHETIC" in report
    assert "negative" in report.lower()
    assert "paired" in report.lower()


def test_real_backend_method_ids_produce_the_proposed_comparisons() -> None:
    """A real run uses ``m1``/``direct``; the proposed label must follow."""

    from ruleshift.evaluator.heldout import proposed_label

    assert proposed_label(("direct", "m1", "immediate")) == "m1"
    assert proposed_label(("direct_stub", "m1_stub")) == "m1_stub"
    assert proposed_label(("direct", "immediate")) is None

    result = run_heldout_evaluation(
        build_heldout_family_a()[:2], ("direct", "m1"), seeds=(101,)
    )
    assert result.summaries
    assert all(method == "m1" for method, _ in result.summaries)
    assert result.max_vlm_calls_per_episode == 6
