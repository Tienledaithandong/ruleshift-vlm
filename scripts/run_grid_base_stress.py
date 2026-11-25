"""Export and analyze the CPU-only base-preservation stress study.

The study uses all 100 v0.3 map layouts and 5 run seeds per map. It is not a
gate evaluation: every episode is a true-base identification episode, and the
output reports failure and misdiagnosis rates with uncertainty intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ruleshift.envs.grid.config import load_grid_config
from ruleshift.evaluator.grid_base_stress import (
    analyze_base_stress,
    build_base_stress_scenarios,
    write_base_stress_report,
)
from scripts.export_grid_baselines import export_grid_baselines

ROOT = _ROOT

METHODS = ("diagnostic_belief", "no_adaptation")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--maps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--out",
        default="results/grid_phase0/base_stress_100x5.jsonl",
        help="raw JSONL output path",
    )
    parser.add_argument(
        "--analysis-dir",
        default="results/grid_phase0/base_stress_100x5_analysis",
        help="directory for analysis.json and summary.md",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_grid_config(ROOT / args.config)
    scenarios = build_base_stress_scenarios(args.maps, args.seeds, config)
    raw_path = ROOT / args.out
    export_grid_baselines(config, METHODS, raw_path, scenarios=scenarios)

    rows = _read_jsonl(raw_path)
    diagnostic = [row for row in rows if row["method"] == "diagnostic_belief"]
    baseline = [row for row in rows if row["method"] == "no_adaptation"]
    payload = analyze_base_stress(
        diagnostic,
        baseline,
        n_maps=args.maps,
        seeds=args.seeds,
        n_boot=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload["raw_path"] = str(raw_path.relative_to(ROOT))
    payload["raw_sha256"] = _sha256(raw_path)
    payload["config"] = args.config
    write_base_stress_report(payload, ROOT / args.analysis_dir)

    diag = payload["methods"]["diagnostic_belief"]
    print(json.dumps({
        "raw": str(raw_path),
        "analysis": str(ROOT / args.analysis_dir),
        "raw_sha256": payload["raw_sha256"],
        "diagnostic_failure": diag["task_failure"],
        "diagnostic_misdiagnosis": diag["misdiagnosis"],
    }, indent=2))


if __name__ == "__main__":
    main()