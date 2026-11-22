"""Offline pilot harness: context selection, labels and failure taxonomy."""

from __future__ import annotations

import json

import pytest

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.envs.config import EnvConfig
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.public_api.schemas import FAMILY_A_ACTIONS
from ruleshift.evaluator.offline_audit import (
    build_offline_contexts,
    run_offline_audit,
    write_taxonomy_report,
)
from ruleshift.permissions import PRIVILEGED_FIELDS


def test_context_selection_has_pilot_scale_and_both_delta_signs() -> None:
    contexts = build_offline_contexts()
    assert 50 <= len(contexts) <= 100
    assert any(context.delta_probe > 0 for context in contexts)
    assert any(context.delta_probe < 0 for context in contexts)
    for context in contexts:
        assert validate_public(context.public_observation.to_payload())
        assert "WAIT" in context.q_values


def validate_public(payload: dict) -> bool:
    from ruleshift.public_api.schemas import validate_public_observation

    return validate_public_observation(payload) == []


def test_audit_is_deterministic_and_metrics_are_bounded(tmp_path) -> None:
    contexts = build_offline_contexts()

    def run(index: int) -> dict:
        records, metrics = run_offline_audit(
            contexts,
            backend=StubVLMBackend(),
            cache=ResponseCache(tmp_path / f"cache-{index}.jsonl"),
        )
        return {"records": records, "metrics": metrics}

    first = run(1)
    second = run(2)
    assert first["records"] == second["records"]
    metrics = first["metrics"]
    assert metrics.n_contexts == len(contexts)
    for rate in (
        metrics.json_valid_rate,
        metrics.valid_action_rate,
        metrics.correct_argmax_rate,
        metrics.cache_hit_rate,
    ):
        assert 0.0 <= rate <= 1.0
    # Hidden noise is not part of the public prompt, so contexts that differ
    # only in feedback noise must share a cache entry. The call count equals
    # the number of distinct public inputs.
    distinct_inputs = len({record["prompt_hash"] for record in first["records"]})
    assert metrics.call_count == distinct_inputs
    assert metrics.cache_hit_rate == pytest.approx(1.0 - distinct_inputs / len(contexts))


def test_manifest_is_public_only_and_labels_are_separate(tmp_path) -> None:
    contexts = build_offline_contexts()
    records, _ = run_offline_audit(
        contexts[:4],
        backend=StubVLMBackend(),
        cache=ResponseCache(tmp_path / "cache.jsonl"),
    )
    manifest_path = tmp_path / "manifest.jsonl"
    evaluator_path = tmp_path / "evaluator_only.jsonl"
    write_taxonomy_report(
        _,
        records,
        contexts[:4],
        manifest_path,
        evaluator_path,
        tmp_path / "report.md",
    )

    public_text = manifest_path.read_text(encoding="utf-8")
    for field in PRIVILEGED_FIELDS:
        assert field not in public_text
    public_rows = [json.loads(line) for line in public_text.strip().splitlines()]
    assert len(public_rows) == 4
    assert "delta_probe" not in public_rows[0]

    private_rows = [
        json.loads(line) for line in evaluator_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(private_rows) == 4
    assert "delta_probe" in private_rows[0]


def test_report_explicitly_labels_synthetic_dry_run(tmp_path) -> None:
    contexts = build_offline_contexts()[:4]
    records, metrics = run_offline_audit(
        contexts,
        backend=StubVLMBackend(),
        cache=ResponseCache(tmp_path / "cache.jsonl"),
    )
    report = tmp_path / "report.md"
    write_taxonomy_report(
        metrics,
        records,
        contexts,
        tmp_path / "manifest.jsonl",
        tmp_path / "evaluator_only.jsonl",
        report,
    )
    text = report.read_text(encoding="utf-8")
    assert "SYNTHETIC" in text
    assert "no vlm" in text.lower() or "not vlm" in text.lower()


# ---------------------------------------------------------------------------
# Label integrity: full menu and durations that match the executor.
# ---------------------------------------------------------------------------
def test_labels_cover_the_full_public_action_menu() -> None:
    from ruleshift.evaluator.offline_audit import ACTIONS

    assert set(ACTIONS) == set(FAMILY_A_ACTIONS)
    for context in build_offline_contexts(9):
        assert set(context.q_values) == set(
            context.public_observation.available_actions
        )


def test_labelled_probe_duration_matches_the_executor() -> None:
    for context in build_offline_contexts(9):
        detour = int(context.public_observation.public_costs["probe_detour"])
        env = RuleShiftEnv(
            config=EnvConfig(probe_detour_steps=detour, disable_shift=True)
        )
        env.reset(seed=1)
        actual = len(env.executor.plan("PROBE_RED", env.geometry.start).ops)
        labelled = next(iter(context.config.durations["PROBE_RED"]))
        assert labelled == actual == 2 * detour + 1


def test_probe_diagnostics_are_stratified_by_sign_and_detour(tmp_path) -> None:
    contexts = build_offline_contexts()
    records, metrics = run_offline_audit(
        contexts,
        backend=StubVLMBackend(),
        cache=ResponseCache(tmp_path / "cache.jsonl"),
    )
    by_key = {stratum.key: stratum for stratum in metrics.strata}
    assert metrics.strata[0].key == "all"
    assert by_key["all"].n_contexts == len(contexts)
    assert by_key["all"].n_probes == sum(
        record["chosen_action"].startswith("PROBE_") for record in records
    )

    # The sign split partitions the contexts, so no aggregate number hides
    # which cells drove the probe rate.
    assert (
        by_key["delta_probe>=0"].n_contexts + by_key["delta_probe<0"].n_contexts
        == by_key["all"].n_contexts
    )
    assert by_key["delta_probe<0"].n_contexts == sum(
        context.delta_probe < 0 for context in contexts
    )

    detours = {
        int(context.public_observation.public_costs["probe_detour"])
        for context in contexts
    }
    assert {f"detour={detour}" for detour in detours} <= set(by_key)
    assert sum(by_key[f"detour={detour}"].n_contexts for detour in detours) == len(
        contexts
    )

    for stratum in metrics.strata:
        assert 0.0 <= stratum.probe_rate <= 1.0
        assert 0.0 <= stratum.correct_argmax_rate <= 1.0
        assert stratum.mean_decision_regret is None or stratum.mean_decision_regret >= 0.0
        assert stratum.n_labelled <= stratum.n_contexts


def test_strata_appear_in_the_report(tmp_path) -> None:
    contexts = build_offline_contexts(9)
    manifest, metrics, report = _run_and_write(tmp_path, contexts)
    text = report.read_text(encoding="utf-8")
    for stratum in metrics.strata:
        assert stratum.key in text
    assert "Correct-argmax rate" in text


def _run_and_write(tmp_path, contexts):
    records, metrics = run_offline_audit(
        contexts,
        backend=StubVLMBackend(),
        cache=ResponseCache(tmp_path / "cache.jsonl"),
    )
    report = tmp_path / "report.md"
    write_taxonomy_report(
        metrics,
        records,
        contexts,
        tmp_path / "manifest.jsonl",
        tmp_path / "evaluator_only.jsonl",
        report,
    )
    return tmp_path / "manifest.jsonl", metrics, report


def test_audit_reports_label_coverage(tmp_path) -> None:
    contexts = build_offline_contexts(9)
    _, metrics = run_offline_audit(
        contexts,
        backend=StubVLMBackend(),
        cache=ResponseCache(tmp_path / "cache.jsonl"),
    )
    assert metrics.label_coverage_rate == pytest.approx(1.0)
