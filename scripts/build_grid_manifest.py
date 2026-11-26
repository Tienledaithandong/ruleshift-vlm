"""Build an explicit, geometry-hashed Grid scenario manifest.

This only generates public layouts and scenario metadata. It never resets an
environment or runs a policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ruleshift.envs.grid.config import GridConfig
from ruleshift.evaluator.grid_manifest import (
    SCENARIO_SCOPE_SUFFIXES,
    build_manifest,
    write_manifest,
)


def _parse_ints(values: list[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                result.extend(range(int(start), int(end) + 1))
            else:
                result.append(int(part))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+", required=True, help="ids or ranges, e.g. 0-9")
    parser.add_argument("--seeds", nargs="+", required=True, help="run seeds or ranges")
    parser.add_argument(
        "--scope",
        choices=sorted(SCENARIO_SCOPE_SUFFIXES),
        default="full",
        help="static_identification is the approved CPU design; full is deferred",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = GridConfig()
    manifest = build_manifest(
        config,
        _parse_ints(args.maps),
        _parse_ints(args.seeds),
        scope=args.scope,
    )
    write_manifest(manifest, args.out)
    print(
        json.dumps(
            {
                "out": str(Path(args.out)),
                "scope": manifest.scope,
                "sha256": manifest.sha256,
                "scenarios": len(manifest.entries),
                "maps": list(manifest.map_ids),
                "run_seeds": list(manifest.run_seeds),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
