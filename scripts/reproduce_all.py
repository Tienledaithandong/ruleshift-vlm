"""Regenerate every synthetic artifact and verify the test suite on CPU.

Usage: python scripts/reproduce_all.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = (
    ("scripts/phase_diagram.py",),
    ("scripts/run_offline_audit.py",),
    ("scripts/run_offline_audit.py", "--design", "history",
     "--out", "results/history", "--report", "results/history"),
    ("scripts/run_closed_loop.py",),
    ("scripts/family_b_transfer.py",),
    ("scripts/run_main_eval.py",),
    ("scripts/run_diagnostics.py",),
    ("scripts/run_stress.py",),
    ("scripts/check_reproducibility.py",),
)

EXPECTED = (
    "reports/phase_diagram.md",
    "reports/w4_failure_taxonomy.md",
    "results/history/w4_failure_taxonomy.md",
    "reports/w7_family_A.md",
    "reports/w8_transfer.md",
    "reports/w9_main_results.md",
    "reports/w10_error_decomposition.md",
    "reports/w11_stress.md",
    "reports/w11_reproducibility.md",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="regenerate artifacts but do not run pytest",
    )
    args = parser.parse_args()

    # A synthetic reproduction is a cold run: do not silently turn the
    # published call count into a cache-hit count on a second invocation.
    for cache_name in (ROOT / "data" / "stub_cache.jsonl",
                       ROOT / "results" / "history" / "stub_cache.jsonl"):
        cache_name.unlink(missing_ok=True)

    for command in COMMANDS:
        print(f"==> {' '.join(command)}", flush=True)
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", *command], cwd=ROOT
        )
        if completed.returncode != 0:
            raise SystemExit(f"failed: {' '.join(command)}")

    missing = [name for name in EXPECTED if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")

    if args.skip_tests:
        print("reproduction complete (tests skipped)")
        return

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT
    )
    if completed.returncode != 0:
        raise SystemExit("pytest failed")
    print("reproduction complete (all synthetic artifacts regenerated)")


if __name__ == "__main__":
    main()
