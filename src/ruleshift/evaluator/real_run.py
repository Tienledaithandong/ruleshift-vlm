"""Artifact writers for a genuine frozen-VLM pilot run.

These writers make no synthetic claim. They are only used by
`scripts/run_real_pilot.py` with a real backend; the setup/test scripts use
the explicitly labelled synthetic writers instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ruleshift.evaluator.offline_audit import AuditMetrics, OfflineContext


def write_real_artifacts(
    records: Sequence[dict],
    contexts: Sequence[OfflineContext],
    metrics: AuditMetrics,
    metadata: Mapping[str, Any],
    out_dir: Path,
    report_path: Path,
) -> dict[str, Path]:
    """Write public manifest, separate evaluator labels and a real-run report."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "decisions.jsonl"
    labels = out_dir / "evaluator_only.jsonl"

    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    label_rows = [
        {
            "context_id": context.context_id,
            "delta_probe": context.delta_probe,
            "q_values": context.q_values,
            "belief": {
                f"{rule}:{shifted}": value
                for (rule, shifted), value in context.belief.items()
            },
        }
        for context in contexts
    ]
    labels.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows),
        encoding="utf-8",
    )

    lines = [
        "# Frozen-VLM offline pilot (empirical run)",
        "",
        f"Model: `{metadata.get('model_id')}`",
        f"Device: `{metadata.get('device')}`",
        f"Revision: `{metadata.get('revision')}`",
        "",
        "This report is produced only when the real backend is used. It still "
        "requires preregistration and a frozen test configuration before any "
        "hypothesis claim is made.",
        "",
        "| Diagnostic | Value |",
        "|---|---:|",
        f"| Contexts | {metrics.n_contexts} |",
        f"| Model calls | {metrics.call_count} |",
        f"| JSON valid rate | {metrics.json_valid_rate:.3f} |",
        f"| Valid action rate | {metrics.valid_action_rate:.3f} |",
        f"| Over-probe rate | {metrics.over_probe_rate:.3f} |",
        f"| Under-probe rate | {metrics.under_probe_rate:.3f} |",
        f"| Mean decision regret | {metrics.mean_decision_regret:.3f} |",
        f"| Mean input tokens per model call | {metrics.mean_input_tokens:.1f} |",
        f"| Mean output tokens per model call | {metrics.mean_output_tokens:.1f} |",
        f"| Mean model-call latency (s) | {metrics.mean_model_latency_ms / 1000:.3f} |",
        f"| Cache hit rate | {metrics.cache_hit_rate:.3f} |",
        "",
        "Public decisions and privileged labels are written to separate files "
        "and must never be joined into model input.",
        "",
    ]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"manifest": manifest, "labels": labels, "report": report_path}
