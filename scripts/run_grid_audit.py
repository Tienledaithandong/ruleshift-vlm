"""Run the RuleShift-Grid structural audit and write a Markdown report.

Usage: python scripts/run_grid_audit.py [--config PATH] [--maps N] [--out DIR]
Exit status is 1 when any audited invariant fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Always prefer this checkout's src over any installed copy: the installed
# site-packages build does not carry the grid modules or prompt data files.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ruleshift.envs.grid.config import load_grid_config
from ruleshift.evaluator.grid_metrics import audit_grid


def _markdown(result: dict, config_path: str, maps: int) -> str:
    status = "PASS" if result["passed"] else "FAIL"
    lines = [
        "# RuleShift-Grid structural audit",
        "",
        f"- Config: `{config_path}`",
        f"- Maps checked: {maps}",
        f"- Status: **{status}**",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(["", "## Failures", ""])
    if result["failures"]:
        lines.extend(f"- {failure}" for failure in result["failures"])
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--maps", type=int, default=10)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    result = audit_grid(load_grid_config(config_path), n_maps=args.maps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = out / "grid_audit.md"
    report.write_text(
        _markdown(result, str(args.config), args.maps), encoding="utf-8"
    )
    print(f"grid audit: {'PASS' if result['passed'] else 'FAIL'}: {report}")
    for failure in result["failures"]:
        print(f"  - {failure}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
