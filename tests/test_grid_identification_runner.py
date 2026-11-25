"""Tests for the explicit-manifest P1 runner; no holdout or VLM calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ruleshift.agents.grid_baselines import NoAdaptationPolicy
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_identification import DeclaredCommit
from ruleshift.evaluator.grid_runner import run_grid_episode
from ruleshift.evaluator.grid_manifest import (
    ManifestError,
    build_manifest,
    manifest_from_dict,
    manifest_to_dict,
    read_manifest,
    write_manifest,
)
from scripts.run_grid_identification import (
    _declaration_payload,
    build_export_payload,
    config_sha256,
    main,
    run_manifest,
    validate_manifest,
)


def test_manifest_layout_and_content_hashes_fail_closed(tmp_path):
    manifest = build_manifest(GridConfig(), map_ids=[0], run_seeds=[0])
    validate_manifest(manifest, GridConfig())
    path = write_manifest(manifest, tmp_path / "manifest.json")
    assert read_manifest(path).sha256 == manifest.sha256
    payload = manifest_to_dict(manifest)
    payload["entries"][0]["layout_sha256"] = "0" * 64
    payload.pop("sha256")
    tampered = manifest_from_dict(payload)
    with pytest.raises(ManifestError):
        validate_manifest(tampered, GridConfig())
    bad_hash = manifest_to_dict(manifest)
    bad_hash["entries"][0]["run_seed"] = 9
    with pytest.raises(ManifestError):
        manifest_from_dict({**bad_hash, "sha256": manifest.sha256})


def test_manifest_hash_is_order_independent():
    manifest = build_manifest(GridConfig(), map_ids=[0], run_seeds=[0])
    payload = manifest_to_dict(manifest)
    payload["entries"] = list(reversed(payload["entries"]))
    reordered = manifest_from_dict(payload)
    assert reordered.sha256 == manifest.sha256
    assert reordered.scenario_ids == manifest.scenario_ids


def test_development_runner_refuses_maps_above_199():
    manifest = build_manifest(GridConfig(), map_ids=[200], run_seeds=[0])
    with pytest.raises(ValueError, match="refuses maps above"):
        run_manifest(
            manifest,
            GridConfig(),
            methods=["no_adaptation"],
            verify_layouts=False,
        )


def test_development_runner_exports_independent_records_and_manifest_hash():
    manifest = build_manifest(GridConfig(), map_ids=[0], run_seeds=[0])
    records = run_manifest(
        manifest,
        GridConfig(),
        methods=["no_adaptation"],
        code_commit="test-commit",
    )
    assert len(records) == 11
    assert {record.manifest_sha256 for record in records} == {manifest.sha256}
    static = [record for record in records if record.scenario_type == "identification"]
    assert len(static) == 6
    assert all(record.declaration_available is False for record in records)
    assert all(record.commitment_status == "unavailable" for record in records)
    assert all(record.terminal_posterior is None for record in records)
    shift = [record for record in records if record.shift_rule is not None]
    assert len(shift) == 4
    assert all(record.scheduled_shift_time is not None for record in shift)
    assert all(
        record.shift_censored == (record.actual_shift_time is None)
        for record in shift
    )
    payload = build_export_payload(
        records,
        manifest,
        GridConfig(),
        methods=["no_adaptation"],
        code_commit="test-commit",
    )
    assert payload["manifest_sha256"] == manifest.sha256
    assert payload["config_sha256"] == config_sha256(GridConfig())
    assert payload["summary"]["methods"]["no_adaptation"]["static_identification"]["episodes"] == 6


def test_public_trajectory_has_no_hidden_scenario_fields():
    manifest = build_manifest(GridConfig(), map_ids=[0], run_seeds=[0])
    scenario = manifest.as_scenario("m0-s0-identification-base-None")
    env = GridEnv(GridConfig())
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    episode = run_grid_episode(
        env, NoAdaptationPolicy(GridConfig()), GridConfig(), scenario, observation, info
    )
    hidden = {
        "true_rule",
        "initial_rule",
        "shift_rule",
        "shift_time",
        "actual_shift_time",
        "belief",
        "posterior",
        "evaluator_rows",
    }
    assert episode.public_rows
    for row in episode.public_rows:
        assert hidden.isdisjoint(row)
    assert episode.evaluator_rows[0]["true_rule"] == "base"


def test_declaration_adapter_never_invents_an_api():
    assert _declaration_payload(SimpleNamespace(), 4) == ([], False)
    events = [
        {
            "step": 1,
            "declared_rule": "base",
            "confidence": 0.95,
            "posterior_margin": 0.8,
        }
    ]
    adapted, available = _declaration_payload(
        SimpleNamespace(commitment_events=lambda: events), 4
    )
    assert available is True
    assert adapted == [DeclaredCommit(1, "base", 0.95, 0.8)]


class _ImmediateDetector:
    name = "immediate_detector"

    def reset(self, observation, info):
        self.observation = observation

    def choose(self, observation, info):
        from ruleshift.public_api.grid_schemas import GridAction

        return GridAction("INTERACT")

    def record(self, action, observation, reward, info):
        self.observation = observation

    def most_likely_rule(self):
        return "base"

    def shift_detected(self):
        return True


def _run_fake_shift(scenario: GridScenario):
    config = GridConfig(max_decisions=4)
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    return run_grid_episode(
        env, _ImmediateDetector(), config, scenario, observation, info
    )


def test_shift_false_alarm_requires_an_observed_shift():
    observed = _run_fake_shift(
        GridScenario(
            scenario_id="m0-s0-change_detection-base-slippery",
            scenario_type="change_detection",
            initial_rule="base",
            shift_rule="slippery",
            shift_time=2,
            map_seed=0,
            run_seed=0,
        )
    )
    assert observed.actual_shift_time == 2
    assert observed.belief_shift_event_time == 1
    assert observed.belief_shift_false_alarm is True
    censored = _run_fake_shift(
        GridScenario(
            scenario_id="m0-s0-change_detection-base-slippery",
            scenario_type="change_detection",
            initial_rule="base",
            shift_rule="slippery",
            shift_time=50,
            map_seed=0,
            run_seed=0,
        )
    )
    assert censored.actual_shift_time is None
    assert censored.shift_censored is True
    assert censored.belief_shift_event_time == 1
    assert censored.belief_shift_false_alarm is False


def test_dry_run_does_not_create_an_environment(tmp_path, capsys):
    manifest = build_manifest(GridConfig(), map_ids=[0], run_seeds=[0])
    path = write_manifest(manifest, tmp_path / "manifest.json")
    assert main(["--manifest", str(path), "--mode", "dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["executed"] is False
    assert output["scenarios"] == 11
