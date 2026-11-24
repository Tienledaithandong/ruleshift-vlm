"""Generate the Week 3 CPU phase diagram from the exact microtask solver.

Usage: python scripts/phase_diagram.py [--config configs/dev.yaml] [--out reports]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.evaluator.phase_diagram import compute_phase_grid, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    rows = compute_phase_grid(config_path)
    paths = write_report(rows, Path(args.out))
    for name, path in paths.items():
        print(f"{name}: {path}")
    positive = sum(row.delta_probe > 0 for row in rows)
    negative = sum(row.delta_probe < 0 for row in rows)
    ties = len(rows) - positive - negative
    print(f"cells: {len(rows)} (probe-favourable {positive}, task-favourable {negative}, ties {ties})")


if __name__ == "__main__":
    main()
