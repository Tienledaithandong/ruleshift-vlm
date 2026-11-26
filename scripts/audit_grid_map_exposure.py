"""Inventory map exposure and hash candidate layouts without running episodes.

This is a data-governance audit, not an evaluation. It scans existing text and
result artifacts for scenario ids, then hashes public layouts for the development
bank and a proposed candidate namespace. No policy is instantiated.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.mapgen import generate_solvable_map
from ruleshift.evaluator.grid_manifest import canonical_json, layout_sha256

SCENARIO = re.compile(r"\bm(\d+)-s\d+-")
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".txt",
    ".ipynb",
    ".csv",
    ".toml",
}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def scan_text(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    try:
        if path.stat().st_size > 3_000_000:
            return counts
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return counts
    for match in SCENARIO.finditer(text):
        counts[int(match.group(1))] += 1
    return counts


def scan_roots(roots: list[Path]) -> dict[str, Any]:
    per_map: Counter[int] = Counter()
    per_file: dict[str, int] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            counts = scan_text(path)
            if counts:
                per_file[str(path)] = sum(counts.values())
                per_map.update(counts)
    return {
        "files_with_scenario_ids": len(per_file),
        "scenario_references": sum(per_map.values()),
        "map_reference_counts": {
            str(map_id): count for map_id, count in sorted(per_map.items())
        },
        "top_files": dict(
            sorted(per_file.items(), key=lambda item: (-item[1], item[0]))[:20]
        ),
    }


def scan_git_refs(
    candidate_start: int, candidate_count: int
) -> dict[str, Any]:
    """Read-only search of committed branch refs for candidate references."""

    refs_result = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes"],
        capture_output=True,
        text=True,
        check=False,
    )
    refs = [line for line in refs_result.stdout.splitlines() if line]
    scenario_counts: Counter[int] = Counter()
    range_mentions: list[str] = []
    candidate_end = candidate_start + candidate_count - 1
    number_pattern = re.compile(
        rf"\b(?:{candidate_start}|{candidate_start + 1}|{candidate_end})\b"
    )
    for ref in refs:
        scenario = subprocess.run(
            ["git", "grep", "-n", "-E", r"m(20[0-9])-s", ref, "--"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in scenario.stdout.splitlines():
            for match in SCENARIO.finditer(line):
                scenario_counts[int(match.group(1))] += 1
        ranges = subprocess.run(
            ["git", "grep", "-n", "-E", r"200.?229|200.?209", ref, "--"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in ranges.stdout.splitlines():
            if number_pattern.search(line) or "200" in line:
                range_mentions.append(line)
    return {
        "refs_scanned": len(refs),
        "scenario_map_counts": {
            str(map_id): count for map_id, count in sorted(scenario_counts.items())
        },
        "candidate_range_mentions": range_mentions[:40],
    }


def layout_audit(config: GridConfig, dev_max: int, candidate_maps: list[int]) -> dict[str, Any]:
    dev: dict[int, str] = {}
    for map_id in range(dev_max + 1):
        dev[map_id] = layout_sha256(generate_solvable_map(map_id, config).map.public)
    candidate: dict[int, str] = {}
    for map_id in candidate_maps:
        candidate[map_id] = layout_sha256(
            generate_solvable_map(map_id, config).map.public
        )
    dev_by_hash: dict[str, list[int]] = defaultdict(list)
    for map_id, digest in dev.items():
        dev_by_hash[digest].append(map_id)
    candidate_by_hash: dict[str, list[int]] = defaultdict(list)
    for map_id, digest in candidate.items():
        candidate_by_hash[digest].append(map_id)
    duplicate_dev = {
        digest: ids for digest, ids in dev_by_hash.items() if len(ids) > 1
    }
    duplicate_candidate = {
        digest: ids for digest, ids in candidate_by_hash.items() if len(ids) > 1
    }
    candidate_dev_collisions = {
        str(map_id): dev_by_hash[digest]
        for map_id, digest in candidate.items()
        if digest in dev_by_hash
    }
    return {
        "dev_maps": dev_max + 1,
        "candidate_maps": candidate_maps,
        "candidate_unique_layouts": len(candidate_by_hash),
        "duplicate_dev_layouts": duplicate_dev,
        "duplicate_candidate_layouts": duplicate_candidate,
        "candidate_dev_collisions": candidate_dev_collisions,
        "candidate_layouts": {
            str(map_id): digest for map_id, digest in candidate.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["results", "reports", "docs", "scripts", "configs", "notebooks"],
    )
    parser.add_argument("--out", default="reports/p0_map_exposure_audit.json")
    parser.add_argument("--candidate-start", type=int, default=210)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--git-refs", action="store_true")
    args = parser.parse_args()
    config = GridConfig()
    roots = [Path(value) for value in args.roots]
    exposure = scan_roots(roots)
    candidate_maps = list(
        range(args.candidate_start, args.candidate_start + args.candidate_count)
    )
    layouts = layout_audit(config, 199, candidate_maps)
    git_exposure = (
        scan_git_refs(args.candidate_start, args.candidate_count)
        if args.git_refs
        else None
    )
    exposed = {int(key) for key in exposure["map_reference_counts"]}
    payload = {
        "note": (
            "Development-only governance audit. No policy or episode was run; "
            "candidate layouts are hashed, not selected as a holdout."
        ),
        "exposure": exposure,
        "exposed_map_ids": sorted(exposed),
        "exposed_map_count": len(exposed),
        "candidate_namespace": {
            "start": args.candidate_start,
            "count": args.candidate_count,
            "already_referenced": sorted(exposed & set(candidate_maps)),
        },
        "layout_audit": layouts,
        "git_refs": git_exposure,
        "config": canonical_json(config.__dict__),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(path),
                "files_with_scenario_ids": exposure["files_with_scenario_ids"],
                "scenario_references": exposure["scenario_references"],
                "exposed_map_count": len(exposed),
                "max_exposed_map": max(exposed) if exposed else None,
                "candidate_already_referenced": sorted(exposed & set(candidate_maps)),
                "candidate_unique_layouts": layouts["candidate_unique_layouts"],
                "candidate_dev_collisions": layouts["candidate_dev_collisions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
