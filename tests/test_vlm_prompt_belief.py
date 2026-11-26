from __future__ import annotations

import json
from pathlib import Path

from ruleshift.evaluator.vlm_prompt_belief import (
    ANCHORED_FOOTER,
    ARMS,
    NEUTRAL_FOOTER,
    baseline_diagnostics,
    build_plan,
    score_arm,
)
from ruleshift.evaluator.vlm_run_audit import resolve_run_dir, summarise_run


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_design_is_balanced_and_replicates_the_original_bodies() -> None:
    plan = build_plan()
    assert len(plan.requests) == 48
    assert [len(plan.arm_requests(arm)) for arm in ARMS] == [12, 12, 12, 12]
    assert len({item.request_id for item in plan.requests}) == 48
    assert len({item.context_id for item in plan.requests}) == 12
    for arm in ARMS:
        assert {item.context_id for item in plan.arm_requests(arm)} == {item.context_id for item in plan.contexts}
    for arm in ARMS:
        assert sum(plan.requests[i].arm == arm for i in range(0, 48, 4)) == 3
    for block in range(12):
        assert len({item.context_id for item in plan.requests[block * 4 : block * 4 + 4]}) == 1


def test_b1_and_b3_are_exact_original_prompts_and_instructions_are_matched() -> None:
    from ruleshift.evaluator import vlm_smoke

    plan = build_plan()
    smoke = vlm_smoke.build_plan()
    a1 = {item.context_id: item for item in smoke.base_requests if item.arm == "A1"}
    a2 = {item.context_id: item for item in smoke.base_requests if item.arm == "A2"}
    for item in plan.requests:
        if item.arm == "B1":
            assert item.prompt == a1[item.context_id].prompt
            assert item.prompt.endswith(ANCHORED_FOOTER)
        if item.arm == "B3":
            assert item.prompt == a2[item.context_id].prompt
            assert item.prompt.endswith(ANCHORED_FOOTER)
        if item.arm in ("B2", "B4"):
            assert item.prompt.endswith(NEUTRAL_FOOTER)
            assert '{"selected_action":"COMMIT"}' not in item.prompt
            assert '{"selected_action":"PROBE_SIGN"}' not in item.prompt
            assert '{"selected_action":"PROBE_BLUE"}' not in item.prompt
        if item.arm in ("B1", "B2"):
            assert "posterior" not in item.public_payload
        if item.arm in ("B3", "B4"):
            assert "posterior" in item.public_payload
        assert item.input_mode == "text"
        assert item.context.feasible_actions()


def test_perfect_actions_pass_each_preregistered_eligibility_criterion() -> None:
    plan = build_plan()
    records = [
        {
            "request_id": item.request_id,
            "selected_action": item.label.best_action,
            "valid": True,
            "fallback": False,
        }
        for item in plan.requests
    ]
    baselines = baseline_diagnostics(plan)
    summary = score_arm(records, plan, "B1", baselines=baselines)
    assert summary["valid_count"] == 12
    assert summary["optimal_action_accuracy_all"] == 1.0
    assert summary["mean_regret_all"] == 0.0
    assert all(row["correct"] == 4 for row in summary["optimal_class_accuracy"].values())
    assert summary["eligible_for_visual_smoke_consideration"] is True
    assert baselines["exact"]["mean_regret"] == 0.0


def _write_fake_run(root: Path, *, model_calls: int = 2, nested: bool = True) -> Path:
    run = root / "real" / "real" if nested else root / "real"
    run.mkdir(parents=True)
    _write_jsonl(run / "requests.jsonl", [{"request_id": "x1"}, {"request_id": "x2"}])
    _write_jsonl(
        run / "responses.jsonl",
        [
            {"request_id": "x1", "status": "ok"},
            {"request_id": "x2", "status": "ok"},
        ],
    )
    _write_jsonl(run / "parsed.jsonl", [{"request_id": "x1"}, {"request_id": "x2"}])
    _write_json(
        run / "request_accounting.json",
        {"planned": 2, "attempted": 2, "completed": 2, "model_calls": 2},
    )
    _write_json(run / "summary.json", {"model_calls": model_calls})
    return run


def test_artifact_resolver_prefers_the_nested_real_real_layout(tmp_path: Path) -> None:
    root = tmp_path / "out"
    run = _write_fake_run(root)
    assert resolve_run_dir(root) == run
    audit = summarise_run(root)
    assert audit["resolution"] == "nested_real_real"
    assert audit["raw_counts"]["responses"] == 2
    assert audit["consistency"]["ok"] is True


def test_artifact_resolver_accepts_the_flat_real_layout(tmp_path: Path) -> None:
    root = tmp_path / "out"
    run = _write_fake_run(root, nested=False)
    assert resolve_run_dir(root) == run
    assert summarise_run(root)["resolution"] == "real"


def test_artifact_audit_rejects_summary_log_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "out"
    _write_fake_run(root, model_calls=1)
    audit = summarise_run(root)
    assert audit["consistency"]["ok"] is False
    assert audit["consistency"]["checks"]["summary_model_calls_matches"] is False
