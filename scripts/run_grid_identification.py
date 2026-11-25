"""Versioned validation/development runner for Grid identification studies.

Only explicit manifests are accepted. The default dry-run mode validates the
manifest without creating an environment. Development mode is restricted to
maps below 200. Confirmatory holdout execution is intentionally not exposed by
this P1 scaffold; it requires a separately approved protocol and runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ruleshift.agents.grid_baselines import (
    DiagnosticBeliefPolicy,
    DiagnosticBeliefPolicyV4,
    NoAdaptationPolicy,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_identification import (
    EVALUATION_VERSION,
    DeclaredCommit,
    build_episode_record,
    paired_summary,
    summarize_records,
    write_jsonl,
)
from ruleshift.evaluator.grid_manifest import (
    SCENARIO_SUFFIXES,
    ManifestError,
    ScenarioManifest,
    canonical_json,
    layout_sha256,
    read_manifest,
)
from ruleshift.evaluator.grid_runner import GridEpisode, run_grid_episode
from ruleshift.envs.grid.mapgen import generate_solvable_map

DEVELOPMENT_MAP_MAX = 199
METHOD_IDS = (
    "no_adaptation",
    "diagnostic_belief_v3",
    "diagnostic_belief_v4",
)
_FACTORY: dict[str, Callable[[GridConfig, GridEnv], Any]] = {
    "no_adaptation": lambda config, env: NoAdaptationPolicy(config),
    "diagnostic_belief_v3": lambda config, env: DiagnosticBeliefPolicy(config),
    "diagnostic_belief_v4": lambda config, env: DiagnosticBeliefPolicyV4(config),
}


def config_sha256(config: GridConfig) -> str:
    return hashlib.sha256(
        canonical_json(asdict(config)).encode("utf-8")
    ).hexdigest()


def validate_manifest(
    manifest: ScenarioManifest,
    config: GridConfig,
    *,
    verify_layouts: bool = True,
) -> None:
    """Fail closed if the manifest does not describe the canonical scenarios."""

    if not manifest.entries:
        raise ManifestError("manifest has no entries")
    if manifest.sha256 != canonical_manifest_hash(manifest):
        raise ManifestError("manifest content hash is inconsistent")
    layouts: dict[int, str] = {}
    if verify_layouts:
        for map_id in manifest.map_ids:
            layouts[map_id] = layout_sha256(
                generate_solvable_map(map_id, config).map.public
            )
        if len(set(layouts.values())) != len(layouts):
            raise ManifestError("manifest map ids have duplicate public layouts")
    seen: set[str] = set()
    for entry in manifest.entries:
        scenario_id = str(entry["scenario_id"])
        if scenario_id in seen:
            raise ManifestError(f"duplicate scenario in manifest: {scenario_id}")
        seen.add(scenario_id)
        scenario = manifest.as_scenario(scenario_id)
        expected_id = (
            f"m{scenario.map_seed}-s{scenario.run_seed}-"
            f"{scenario.scenario_type}-{scenario.initial_rule}-{scenario.shift_rule}"
        )
        if scenario.scenario_id != expected_id:
            raise ManifestError(
                f"scenario id does not match its fields: {scenario.scenario_id}"
            )
        if verify_layouts:
            expected_layout = layouts[int(entry["map_seed"])]
            if str(entry["layout_sha256"]) != expected_layout:
                raise ManifestError(
                    f"layout hash mismatch for {scenario.scenario_id}"
                )
    expected_ids = {
        f"m{map_id}-s{seed}-{suffix}"
        for map_id in manifest.map_ids
        for seed in manifest.run_seeds
        for suffix in SCENARIO_SUFFIXES
    }
    missing = sorted(expected_ids - seen)
    extra = sorted(seen - expected_ids)
    if missing or extra:
        raise ManifestError(
            f"manifest scenario set differs; missing={missing[:5]}, extra={extra[:5]}"
        )
    expected_count = len(expected_ids)
    if len(manifest.entries) != expected_count:
        raise ManifestError(
            f"manifest has {len(manifest.entries)} entries, expected {expected_count}"
        )


def canonical_manifest_hash(manifest: ScenarioManifest) -> str:
    return hashlib.sha256(
        canonical_json([dict(entry) for entry in manifest.entries]).encode("utf-8")
    ).hexdigest()


def _declaration_payload(policy: Any, terminal_step: int) -> tuple[list[Any], bool]:
    """Read a declaration API only if the policy explicitly exposes one.

    The P0 baselines do not expose this API. The adapter is deliberately
    conservative: no API means no commitment score, never an inferred one.
    """

    provider = getattr(policy, "commitment_events", None)
    if not callable(provider):
        return [], False
    raw_events = provider()
    if raw_events is None:
        return [], True
    if not isinstance(raw_events, (list, tuple)):
        raise ValueError("commitment_events() must return a sequence")
    adapted: list[DeclaredCommit] = []
    for raw in raw_events:
        if isinstance(raw, DeclaredCommit):
            adapted.append(raw)
        elif isinstance(raw, Mapping):
            adapted.append(DeclaredCommit(**dict(raw)))
        else:
            raise ValueError(f"unsupported declaration event: {raw!r}")
    return adapted, True


def run_manifest(
    manifest: ScenarioManifest,
    config: GridConfig,
    *,
    methods: Sequence[str],
    code_commit: str = "uncommitted",
    verify_layouts: bool = True,
) -> list[Any]:
    """Run the explicit development manifest and return independent records."""

    validate_manifest(manifest, config, verify_layouts=verify_layouts)
    unknown = sorted(set(methods) - set(METHOD_IDS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if not methods:
        raise ValueError("at least one method is required")
    map_ids = set(manifest.map_ids)
    if any(map_id > DEVELOPMENT_MAP_MAX for map_id in map_ids):
        raise ValueError(
            "development runner refuses maps above "
            f"{DEVELOPMENT_MAP_MAX}; holdout execution is not authorized"
        )
    config_hash = config_sha256(config)
    records: list[Any] = []
    by_method: dict[str, list[Any]] = {method: [] for method in methods}
    for method in methods:
        factory = _FACTORY[method]
        for entry in manifest.entries:
            scenario = manifest.as_scenario(str(entry["scenario_id"]))
            env = GridEnv(config)
            observation, info = env.reset(
                seed=scenario.run_seed,
                options={"scenario": scenario},
            )
            policy = factory(config, env)
            try:
                policy.name = method  # type: ignore[attr-defined]
            except AttributeError:
                pass
            episode = run_grid_episode(
                env, policy, config, scenario, observation, info
            )
            declarations, declaration_available = _declaration_payload(
                policy, episode.env_steps
            )
            record = build_episode_record(
                episode,
                scenario,
                config,
                layout_sha256_value=str(entry["layout_sha256"]),
                declarations=declarations,
                declaration_available=declaration_available,
                code_commit=code_commit,
                config_sha256=config_hash,
                manifest_sha256=manifest.sha256,
            )
            records.append(record)
            by_method[method].append(record)
    # Every method must have exactly the same unique scenario set.
    for index, method in enumerate(methods[1:], start=1):
        paired_summary(
            by_method[methods[0]],
            by_method[method],
            metric="task_success",
        )
    return records


def build_export_payload(
    records: Sequence[Any],
    manifest: ScenarioManifest,
    config: GridConfig,
    *,
    methods: Sequence[str],
    code_commit: str,
) -> dict[str, Any]:
    summary = summarize_records(records)
    paired: dict[str, Any] = {}
    by_method: dict[str, list[Any]] = {method: [] for method in methods}
    for record in records:
        by_method.setdefault(record.method, []).append(record)
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            paired[f"{left}__vs__{right}"] = {
                "task_success": paired_summary(
                    by_method[left], by_method[right], metric="task_success"
                ),
                "map_correct": paired_summary(
                    by_method[left], by_method[right], metric="map_correct"
                ),
                "task_and_map_correct": paired_summary(
                    by_method[left],
                    by_method[right],
                    metric="task_and_map_correct",
                ),
            }
    summary["paired"] = paired
    return {
        "evaluation_version": EVALUATION_VERSION,
        "manifest_sha256": manifest.sha256,
        "config_sha256": config_sha256(config),
        "code_commit": code_commit,
        "methods": list(methods),
        "maps": list(manifest.map_ids),
        "run_seeds": list(manifest.run_seeds),
        "episodes": len(records),
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("dry-run", "development"), default="dry-run")
    parser.add_argument("--methods", default=",".join(METHOD_IDS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--code-commit", default="uncommitted")
    args = parser.parse_args(argv)

    manifest = read_manifest(args.manifest)
    config = GridConfig()
    validate_manifest(manifest, config, verify_layouts=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHOD_IDS))
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    if args.mode == "dry-run":
        payload = {
            "mode": "dry-run",
            "evaluation_version": EVALUATION_VERSION,
            "manifest_sha256": manifest.sha256,
            "config_sha256": config_sha256(config),
            "maps": list(manifest.map_ids),
            "run_seeds": list(manifest.run_seeds),
            "scenarios": len(manifest.entries),
            "methods": methods,
            "executed": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not args.out:
        parser.error("--out is required in development mode")
    records = run_manifest(
        manifest,
        config,
        methods=methods,
        code_commit=args.code_commit,
        verify_layouts=True,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / "records.jsonl")
    payload = build_export_payload(
        records,
        manifest,
        config,
        methods=methods,
        code_commit=args.code_commit,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(out_dir),
                "episodes": len(records),
                "manifest_sha256": manifest.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
