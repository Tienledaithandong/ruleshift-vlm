"""CPU-only tests for the frozen 12-context VLM decision smoke."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ruleshift.agents.vlm_smoke_backend import StubSmokeBackend
from ruleshift.evaluator.vlm_decision_visual import visual_digest
from ruleshift.evaluator.vlm_smoke import (
    BASE_IDS,
    COMMIT_IDS,
    DIAGNOSTIC_ROLES,
    FORBIDDEN_LABEL_KEYS,
    MAX_NEW_TOKENS,
    NUISANCE_IDS,
    SIGN_IDS,
    BLUE_IDS,
    analysis_plan,
    assert_prompt_public,
    baseline_summary,
    build_plan,
    freeze_artifacts,
    image_png_bytes,
    make_model_request,
    manifest_record,
    score_records,
    sha256_text,
    validate_selection,
    verify_freeze,
)
from ruleshift.evaluator.vlm_smoke_parser import parse_action


@pytest.fixture(scope="module")
def plan():
    return build_plan()


def test_selection_is_frozen_and_decisive(plan) -> None:
    validate_selection()
    assert len(BASE_IDS) == 12
    assert len(set(BASE_IDS)) == 12
    assert [len(COMMIT_IDS), len(SIGN_IDS), len(BLUE_IDS)] == [4, 4, 4]
    assert len(plan.base_requests) == 36
    for action, ids in (("COMMIT", COMMIT_IDS), ("PROBE_SIGN", SIGN_IDS), ("PROBE_BLUE", BLUE_IDS)):
        for context_id in ids:
            item = next(x for x in plan.base_requests if x.context_id == context_id and x.arm == "A1")
            assert item.label.best_action == action
            assert set(item.label.optimal_actions) == {action}
            assert item.label.decision_margin >= 0.5
            assert len(item.context.feasible_actions()) >= 2


def test_controls_are_paired_and_change_optimal_set(plan) -> None:
    assert len(plan.nuisance_requests) == 6
    assert len(plan.diagnostic_requests) == 6
    base_by_id = {item.context_id: item for item in plan.base_requests if item.arm == "A3"}
    for item in plan.nuisance_requests:
        base = base_by_id[item.base_context_id]
        assert item.context_id == item.base_context_id
        assert item.label == base.label
        assert item.public_payload == base.public_payload
        assert visual_digest(item.image) != visual_digest(base.image)
    for item in plan.diagnostic_requests:
        base = base_by_id[item.base_context_id]
        assert set(item.label.optimal_actions) != set(base.label.optimal_actions)
        assert item.context.context_id != item.base_context_id
        assert item.diagnostic_role == DIAGNOSTIC_ROLES[item.base_context_id]
        assert visual_digest(item.image) != visual_digest(base.image)


def test_prompts_are_public_and_arm_specific(plan) -> None:
    for item in plan.requests:
        assert_prompt_public(item.prompt)
        assert "latent_rule" not in item.prompt
        assert "optimal_actions" not in item.prompt
        if item.arm == "A1":
            assert "posterior" not in item.prompt.lower()
        if item.arm == "A2":
            assert "posterior" in item.prompt.lower()
        if item.arm in ("A1", "A2"):
            assert item.image is None
        else:
            assert item.image is not None
            assert "posterior" not in item.prompt.lower()
        row = manifest_record(item)
        for key in row:
            assert key not in FORBIDDEN_LABEL_KEYS


def test_manifest_is_deterministic_and_complete(plan) -> None:
    rows_a = [manifest_record(item) for item in plan.requests]
    rows_b = [manifest_record(item) for item in build_plan().requests]
    assert rows_a == rows_b
    assert len(rows_a) == 48
    assert [row["arm"] for row in rows_a].count("A1") == 12
    assert [row["arm"] for row in rows_a].count("A2") == 12
    assert [row["arm"] for row in rows_a].count("A3") == 12
    assert [row["arm"] for row in rows_a].count("RGB_NUISANCE") == 6
    assert [row["arm"] for row in rows_a].count("RGB_DIAGNOSTIC") == 6
    assert all(row["max_new_tokens"] == MAX_NEW_TOKENS for row in rows_a)
    assert all(row["do_sample"] is False for row in rows_a)
    a3 = next(item for item in plan.base_requests if item.arm == "A3")
    assert len(image_png_bytes(a3.image)) > 1000


def test_freeze_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    plan = build_plan()
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# dummy protocol\n", encoding="utf-8")
    checkpoint = {"model_id": "fake", "revision": "fake"}
    out = tmp_path / "frozen"
    freeze = freeze_artifacts(
        plan,
        out,
        checkpoint_manifest=checkpoint,
        checkpoint_manifest_path=out / "checkpoint_manifest.json",
        protocol_path=protocol,
        source_paths=[Path(__file__).resolve()],
    )
    verified = verify_freeze(out, source_paths=[Path(__file__).resolve()])
    assert verified["request_manifest_sha256"] == freeze["request_manifest_sha256"]
    manifest_path = out / "request_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[0]["prompt_sha256"] = "sha256:00"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="request_manifest_sha256"):
        verify_freeze(out, source_paths=[Path(__file__).resolve()])


def test_stub_backend_forwards_image_and_falls_back(plan) -> None:
    a3 = next(item for item in plan.base_requests if item.request_id == "A3-01")
    a1 = next(item for item in plan.base_requests if item.request_id == "A1-01")
    backend = StubSmokeBackend()
    response = backend.generate(make_model_request(a3))
    assert backend.last_metadata["image_seen"] is True
    assert backend.last_metadata["vision_tensor_sha256"]
    assert parse_action(response.text, a3.context.feasible_actions()).valid
    response_text = backend.generate(make_model_request(a1))
    assert backend.last_metadata["image_seen"] is False
    assert parse_action(response_text.text, a1.context.feasible_actions()).valid
    malformed = StubSmokeBackend(malformed_every=1)
    parsed = parse_action(malformed.generate(make_model_request(a1)).text, a1.context.feasible_actions())
    assert parsed.valid is False and parsed.fallback is True
    assert parsed.selected_action == "COMMIT"


def test_fallback_is_separate_from_valid_accuracy(plan) -> None:
    records = []
    for item in plan.base_requests:
        action = item.label.optimal_actions[0]
        fallback = item.request_id == "A1-01"
        records.append(
            {
                "request_id": item.request_id,
                "selected_action": "COMMIT" if fallback else action,
                "valid": not fallback,
                "fallback": fallback,
                "errors": ["forced fallback"] if fallback else [],
            }
        )
    summary = score_records(records, plan, kind="base", arm="A1")
    assert summary["n"] == 12
    assert summary["valid_count"] == 11
    assert summary["fallback_count"] == 1
    assert summary["optimal_action_accuracy_valid"] == 1.0
    assert summary["optimal_action_accuracy_all"] < 1.0
    assert summary["fallback_regret_count"] == 1


def test_baselines_on_selected_subset(plan) -> None:
    baselines = baseline_summary(plan)
    assert baselines["exact"]["optimal_action_accuracy"] == 1.0
    assert baselines["exact"]["mean_regret"] == 0.0
    assert baselines["always_commit"]["optimal_action_accuracy"] < 1.0
    assert baselines["information_gain"]["optimal_action_accuracy"] >= 0.5


def test_analysis_plan_is_predeclared() -> None:
    plan = analysis_plan()
    assert plan["request_cap"] == 48
    assert plan["arms"]["A1"]["n"] == 12
    assert plan["arms"]["A2"]["n"] == 12
    assert plan["arms"]["A3"]["n"] == 12
    assert plan["arms"]["RGB_NUISANCE"]["n"] == 6
    assert plan["arms"]["RGB_DIAGNOSTIC"]["n"] == 6
    assert plan["hard_gates"]["valid_each_arm"].startswith("at least 11")
    assert plan["no_tuning"]
