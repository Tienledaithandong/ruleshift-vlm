"""Explicit, hashable scenario manifests for RuleShift-Grid studies.

The manifest is the contract between a preregistration and a runner: it names
the exact map geometries, run seeds, scenario types and shift schedules before
any episode is executed. Layout hashes cover public geometry only, so two map
integer ids with the same geometry cannot accidentally become independent
samples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.mapgen import generate_solvable_map
from ruleshift.evaluator.grid_runner import build_grid_scenarios
from ruleshift.public_api.grid_rules import PublicGridMap

__all__ = [
    "ManifestError",
    "SCENARIO_SUFFIXES",
    "ScenarioManifest",
    "build_manifest",
    "canonical_json",
    "layout_sha256",
    "manifest_from_dict",
    "manifest_to_dict",
    "read_manifest",
    "write_manifest",
]

MANIFEST_VERSION = "grid-manifest-v1"
SCENARIO_SUFFIXES = (
    "identification-base-None",
    "identification-remap_swap-None",
    "identification-remap_rotate-None",
    "identification-slippery-None",
    "identification-hazard_goal_swap-None",
    "identification-sign_composition-None",
    "change_detection-base-remap_swap",
    "change_detection-base-slippery",
    "change_detection-base-hazard_goal_swap",
    "change_detection-base-sign_composition",
    "no_shift-base-None",
)
_EXPECTED_PER_MAP_SEED = len(SCENARIO_SUFFIXES)


class ManifestError(ValueError):
    """A manifest is malformed, duplicated, or inconsistent with the config."""


def canonical_json(value: Any) -> str:
    """Stable JSON used for all hashes in this module."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def layout_sha256(public_map: PublicGridMap) -> str:
    """Hash public geometry, not the integer map id."""

    return hashlib.sha256(
        canonical_json(public_map.to_payload()).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ScenarioManifest:
    """One fixed scenario list with a content hash and geometry hashes."""

    version: str
    entries: tuple[dict[str, Any], ...]
    sha256: str

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(str(entry["scenario_id"]) for entry in self.entries)

    @property
    def map_ids(self) -> tuple[int, ...]:
        return tuple(sorted({int(entry["map_seed"]) for entry in self.entries}))

    @property
    def run_seeds(self) -> tuple[int, ...]:
        return tuple(sorted({int(entry["run_seed"]) for entry in self.entries}))

    def scenario_by_id(self, scenario_id: str) -> dict[str, Any]:
        for entry in self.entries:
            if entry["scenario_id"] == scenario_id:
                return dict(entry)
        raise KeyError(scenario_id)

    def as_scenario(self, scenario_id: str) -> GridScenario:
        entry = self.scenario_by_id(scenario_id)
        return GridScenario(
            scenario_id=str(entry["scenario_id"]),
            scenario_type=str(entry["scenario_type"]),
            initial_rule=str(entry["initial_rule"]),
            shift_rule=entry.get("shift_rule"),
            shift_time=entry.get("shift_time"),
            map_seed=int(entry["map_seed"]),
            run_seed=int(entry["run_seed"]),
        )


def _manifest_hash(entries: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json(list(entries)).encode("utf-8")
    ).hexdigest()


def build_manifest(
    config: GridConfig,
    map_ids: Sequence[int],
    run_seeds: Sequence[int],
) -> ScenarioManifest:
    """Build a manifest from exact integer map ids and run seeds.

    Only the public geometry is generated here; no episode or policy is run.
    """

    maps = tuple(int(value) for value in map_ids)
    seeds = tuple(int(value) for value in run_seeds)
    if not maps or not seeds:
        raise ManifestError("map_ids and run_seeds must be non-empty")
    if len(set(maps)) != len(maps):
        raise ManifestError("map_ids must be unique")
    if len(set(seeds)) != len(seeds):
        raise ManifestError("run_seeds must be unique")
    if any(value < 0 for value in maps + seeds):
        raise ManifestError("map_ids and run_seeds must be non-negative")
    if any(value >= 1_000_000 for value in maps):
        raise ManifestError("map id is unreasonably large")

    max_map = max(maps)
    max_seed = max(seeds)
    # The canonical generator emits all six identification rules, four
    # change-detection families and one no-shift control for each map/seed.
    all_scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_grid_scenarios(max_map + 1, max_seed + 1, config)
    }
    expected = len(maps) * len(seeds) * _EXPECTED_PER_MAP_SEED
    selected: list[GridScenario] = []
    for map_id in maps:
        for seed in seeds:
            for suffix in SCENARIO_SUFFIXES:
                scenario_id = f"m{map_id}-s{seed}-{suffix}"
                try:
                    selected.append(all_scenarios[scenario_id])
                except KeyError as exc:
                    raise ManifestError(
                        f"generator did not emit expected scenario {scenario_id}"
                    ) from exc
    if len(selected) != expected:
        raise ManifestError(
            f"expected {expected} scenarios, generated {len(selected)}"
        )
    if len({scenario.scenario_id for scenario in selected}) != len(selected):
        raise ManifestError("duplicate scenario ids in manifest")

    layouts: dict[int, str] = {}
    for map_id in maps:
        generated = generate_solvable_map(map_id, config)
        layouts[map_id] = layout_sha256(generated.map.public)
    if len(set(layouts.values())) != len(layouts):
        raise ManifestError("distinct map ids have duplicate public layouts")

    entries: list[dict[str, Any]] = []
    for scenario in selected:
        entries.append(
            {
                "scenario_id": scenario.scenario_id,
                "map_seed": int(scenario.map_seed),
                "run_seed": int(scenario.run_seed),
                "scenario_type": scenario.scenario_type,
                "initial_rule": scenario.initial_rule,
                "shift_rule": scenario.shift_rule,
                "shift_time": scenario.shift_time,
                "layout_sha256": layouts[int(scenario.map_seed)],
            }
        )
    entries.sort(key=lambda item: item["scenario_id"])
    digest = _manifest_hash(entries)
    return ScenarioManifest(
        version=MANIFEST_VERSION,
        entries=tuple(entries),
        sha256=digest,
    )


def manifest_to_dict(manifest: ScenarioManifest) -> dict[str, Any]:
    return {
        "version": manifest.version,
        "sha256": manifest.sha256,
        "entries": [dict(entry) for entry in manifest.entries],
    }


def manifest_from_dict(payload: Mapping[str, Any]) -> ScenarioManifest:
    if payload.get("version") != MANIFEST_VERSION:
        raise ManifestError("unsupported manifest version")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest entries must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ManifestError("manifest entry must be an object")
        required = (
            "scenario_id",
            "map_seed",
            "run_seed",
            "scenario_type",
            "initial_rule",
            "shift_rule",
            "shift_time",
            "layout_sha256",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ManifestError(f"manifest entry missing keys: {missing}")
        normalized.append(dict(raw))
    normalized.sort(key=lambda entry: str(entry["scenario_id"]))
    ids = [str(entry["scenario_id"]) for entry in normalized]
    if len(set(ids)) != len(ids):
        raise ManifestError("manifest has duplicate scenario ids")
    digest = _manifest_hash(normalized)
    declared = payload.get("sha256")
    if declared is not None and str(declared) != digest:
        raise ManifestError(
            f"manifest hash mismatch: declared {declared}, computed {digest}"
        )
    return ScenarioManifest(
        version=MANIFEST_VERSION,
        entries=tuple(normalized),
        sha256=digest,
    )


def read_manifest(path: str | Path) -> ScenarioManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ManifestError("manifest root must be an object")
    return manifest_from_dict(payload)


def write_manifest(manifest: ScenarioManifest, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out
