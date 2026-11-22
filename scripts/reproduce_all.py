"""Regenerate every synthetic artifact and verify the test suite on CPU.

Usage: python scripts/reproduce_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = (
    "scripts/phase_diagram.py",
    "scripts/run_offline_audit.py",
    "scripts/run_closed_loop.py",
    "scripts/run_main_eval.py",
    "scripts/run_diagnostics.py",
    "scripts/run_stress.py",
    "scripts/check_reproducibility.py",
)

EXPECTED = (
    "reports/phase_diagram.md",
    "reports/w4_failure_taxonomy.md",
    "reports/w7_family_A.md",
    "reports/w9_main_results.md",
    "reports/w10_error_decomposition.md",
    "reports/w11_stress.md",
    "reports/w11_reproducibility.md",
)


def main() -> None:
    for script in SCRIPTS:
        print(f"==> {script}")
        completed = subprocess.run(
            [sys.executable, script], cwd=ROOT, capture_output=True, text=True
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(f"failed: {script}")
        print(completed.stdout.strip())

    missing = [name for name in EXPECTED if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True
    )
    print(completed.stdout.strip())
    if completed.returncode != 0:
        raise SystemExit("pytest failed")
    print("reproduction complete — all synthetic artifacts regenerated")


if __name__ == "__main__":
    main()
