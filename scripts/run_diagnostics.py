"""Generate the Week 10 single-component attribution report (SYNTHETIC)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.evaluator.diagnostics import (
    INTERVENTIONS,
    run_diagnostic,
    write_diagnostic_report,
)
from ruleshift.evaluator.offline_audit import build_offline_contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, default=81)
    parser.add_argument("--out", default="reports/w10_error_decomposition.md")
    args = parser.parse_args()

    contexts = build_offline_contexts(args.contexts)
    backend = StubVLMBackend()
    cache = ResponseCache(None)
    results = {
        intervention: run_diagnostic(contexts, backend, cache, intervention)
        for intervention in INTERVENTIONS
    }
    path = write_diagnostic_report(results, Path(args.out))
    print("SYNTHETIC diagnostics — stub backend, no VLM was run")
    for intervention, result in results.items():
        print(
            f"{intervention}: mean_regret={result.mean_regret:.3f} "
            f"error_rate={result.error_rate:.3f}"
        )
    print(f"report={path}")


if __name__ == "__main__":
    main()
