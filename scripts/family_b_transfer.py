"""Generate the Week 8 Family A->B transfer report from analytical references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.evaluator.family_b_reference import FamilyBCosts, delta_probe_family_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/w8_transfer.md")
    args = parser.parse_args()

    rows = []
    for safe_steps in (2, 8, 12):
        for slip_slippery in (0.5, 0.7, 0.9):
            config = FamilyBCosts(
                safe_steps=safe_steps,
                bridge_steps=1,
                recovery_steps=10,
                probe_steps=5,
                slip_stable=0.0,
                slip_slippery=slip_slippery,
            )
            rows.append((safe_steps, slip_slippery, delta_probe_family_b(config, 0.5)))

    lines = [
        "# Week 8 — Family A to Family B transfer (analytical, CPU)",
        "",
        "Family B changes the **transition law**: the shortcut's slip probability "
        "is regime-dependent, not merely its reward label. The analytical reference "
        "below is exact for the single-decision abstraction with a perfectly "
        "revealing probe and uses evaluator-side "
        "probabilities. It is a transfer diagnostic, not a VLM result.",
        "",
        "| safe route steps | slippery slip probability | delta_probe | preferred |",
        "|---:|---:|---:|---|",
    ]
    for safe_steps, slip, delta in rows:
        lines.append(
            f"| {safe_steps} | {slip:.2f} | {delta:+.4f} | "
            f"{'probe' if delta > 0 else 'task'} |"
        )
    positive = sum(delta > 0 for _, _, delta in rows)
    lines += [
        "",
        f"Cells: {len(rows)}; probe-favourable: {positive}; "
        f"task-favourable: {len(rows) - positive}.",
        "",
        "## What transfers and what does not",
        "",
        "- **Transfers:** the decision is still probe-versus-task under a hidden "
        "variable, probed evidence is noisy and costs primitive steps, and a "
        "probe is worth its cost only when it can change the later action. The "
        "analytical reference evaluates this tradeoff independently of the "
        "Family A agent selector. Family B agent transfer has not been implemented.",
        "- **Does not transfer automatically:** Family A's evidence is a "
        "deterministic accept/reject under a card law; Family B's evidence is a "
        "noisy regime report and the task action itself is stochastic. The "
        "exact `Delta_probe` labels from Week 3 therefore do **not** cover "
        "Family B; those labels are Family A microtasks only.",
        "- **Remaining gap:** an exact Family B solver would need to enumerate "
        "slip outcomes inside macros. Until then these analytical cells are "
        "labelled as an exact *single-decision* reference, and full episodes "
        "must use an approximate reference.",
        "",
        "The current map variants alter decorative pixels while keeping the "
        "geometry, action menu, costs and transition probabilities fixed.",
        "",
    ]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path} ({len(rows)} cells, {positive} probe-favourable)")


if __name__ == "__main__":
    main()
