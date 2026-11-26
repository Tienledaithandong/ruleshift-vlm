"""Development runner for the v2 separated epistemic/control protocol.

Only the exposed 10 x 5 development slice is accepted. The runner never
converts terminal MAP into a declaration and never executes a reserved map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ruleshift.agents.grid_v05 import (
    DiagnosticBeliefPolicyV4ReadoutV2,
    DiagnosticBeliefPolicyV5,
    DiagnosticBeliefPolicyV5NoRequest,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_identification_v2 import (
    EVALUATION_VERSION,
    build_v2_episode_record,
    paired_transitions_v2,
    summarize_records_v2,
    validate_development_slice,
    write_jsonl_v2,
)
from ruleshift.evaluator.grid_manifest import (
    ScenarioManifest,
    canonical_json,
    read_manifest,
)
from ruleshift.evaluator.grid_runner import run_grid_episode

METHOD_IDS = (
    "v4_readout_v2",
    "v5",
    "v5_no_request",
)
_FACTORY: dict[str, Callable[[GridConfig], Any]] = {
    "v4_readout_v2": DiagnosticBeliefPolicyV4ReadoutV2,
    "v5": DiagnosticBeliefPolicyV5,
    "v5_no_request": DiagnosticBeliefPolicyV5NoRequest,
}


def config_sha256(config: GridConfig) -> str:
    return hashlib.sha256(
        canonical_json(asdict(config)).encode("utf-8")
    ).hexdigest()


def run_manifest(
    manifest: ScenarioManifest,
    config: GridConfig,
    *,
    methods: Sequence[str],
    code_commit: str = "uncommitted",
) -> list[Any]:
    validate_development_slice(manifest)
    unknown = sorted(set(methods) - set(METHOD_IDS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if not methods:
        raise ValueError("at least one method is required")
    config_hash = config_sha256(config)
    records: list[Any] = []
    for method in methods:
        factory = _FACTORY[method]
        for entry in manifest.entries:
            scenario = manifest.as_scenario(str(entry["scenario_id"]))
            env = GridEnv(config)
            observation, info = env.reset(
                seed=scenario.run_seed,
                options={"scenario": scenario},
            )
            policy = factory(config)
            policy.name = method  # type: ignore[attr-defined]
            episode = run_grid_episode(
                env, policy, config, scenario, observation, info
            )
            record = build_v2_episode_record(
                episode,
                scenario,
                config,
                layout_sha256_value=str(entry["layout_sha256"]),
                events=episode.epistemic_events,
                declaration_available=callable(
                    getattr(policy, "finalize", None)
                ),
                code_commit=code_commit,
                config_sha256=config_hash,
                manifest_sha256=manifest.sha256,
            )
            records.append(record)
    return records


def build_export_payload(
    records: Sequence[Any],
    manifest: ScenarioManifest,
    config: GridConfig,
    *,
    methods: Sequence[str],
    code_commit: str,
) -> dict[str, Any]:
    by_method: dict[str, list[Any]] = {method: [] for method in methods}
    for record in records:
        by_method.setdefault(record.method, []).append(record)
    paired: dict[str, Any] = {}
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            paired[f"{left}__vs__{right}"] = paired_transitions_v2(
                by_method[left], by_method[right]
            )
    summary = summarize_records_v2(records)
    summary["paired"] = paired
    final_observation = sum(
        record.final_declaration_source == "final_observation"
        for record in records
    )
    declarations_by_source: dict[str, int] = {}
    for record in records:
        for event in record.declaration_events:
            source = str(event["source"])
            declarations_by_source[source] = (
                declarations_by_source.get(source, 0) + 1
            )
    request_statuses: dict[str, int] = {}
    for record in records:
        for status, count in record.request_status_counts:
            request_statuses[status] = request_statuses.get(status, 0) + count
    return {
        "evaluation_version": EVALUATION_VERSION,
        "manifest_sha256": manifest.sha256,
        "scope": manifest.scope,
        "config_sha256": config_sha256(config),
        "code_commit": code_commit,
        "methods": list(methods),
        "maps": list(manifest.map_ids),
        "run_seeds": list(manifest.run_seeds),
        "episodes": len(records),
        "diagnostics": {
            "final_observation_declarations": final_observation,
            "declarations_by_source": dict(sorted(declarations_by_source.items())),
            "request_statuses": dict(sorted(request_statuses.items())),
        },
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--mode", choices=("dry-run", "development"), default="dry-run"
    )
    parser.add_argument("--methods", default=",".join(METHOD_IDS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--code-commit", default="uncommitted")
    args = parser.parse_args(argv)

    manifest = read_manifest(args.manifest)
    config = GridConfig()
    validate_development_slice(manifest)
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
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_v2(records, out_dir / "records.jsonl")
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
