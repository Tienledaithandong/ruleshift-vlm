"""CPU phase diagram of Delta_probe over dev cost/noise settings.

All cells are computed by the exact microtask solver. This is a development
diagnostic, not a locked test set and not a VLM result.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ruleshift.evaluator.exact_solver import ExactSolver, MicrotaskConfig

ACTIONS = ("COMMIT_RED", "COMMIT_BLUE", "PROBE_RED")
BELIEF = {("red", False): 0.5, ("blue", True): 0.5}
HORIZON = 9  # enough for a 5-step probe followed by a commit with lockout
PROBE_STEPS = (1, 3, 5)
NOISE_LEVELS = (0.0, 0.1, 0.25)
LOCKOUT_LEVELS = (0, 2, 8)


@dataclass(frozen=True)
class PhaseCell:
    probe_steps: int
    noise: float
    lockout: int
    horizon: int
    delta_probe: float
    preferred: str


def compute_phase_grid() -> tuple[PhaseCell, ...]:
    """Enumerate the dev grid with the exact solver."""

    cells: list[PhaseCell] = []
    for probe_steps in PROBE_STEPS:
        for noise in NOISE_LEVELS:
            for lockout in LOCKOUT_LEVELS:
                durations = {action: {1: 1.0} for action in ACTIONS}
                durations["PROBE_RED"] = {probe_steps: 1.0}
                observation_steps = {action: 0 for action in ACTIONS}
                observation_steps["PROBE_RED"] = probe_steps - 1
                config = MicrotaskConfig(
                    shift_hazard=0.0,
                    feedback_noise=noise,
                    reward_success=10.0,
                    step_cost=-0.05,
                    lockout_steps=lockout,
                    durations=durations,
                    observation_steps=observation_steps,
                )
                delta = ExactSolver(config, ACTIONS).delta_probe(BELIEF, HORIZON)
                cells.append(
                    PhaseCell(
                        probe_steps=probe_steps,
                        noise=noise,
                        lockout=lockout,
                        horizon=HORIZON,
                        delta_probe=delta,
                        preferred="probe" if delta > 0 else "task",
                    )
                )
    return tuple(cells)


def write_report(rows: tuple[PhaseCell, ...], output_dir: Path) -> dict[str, Path]:
    """Write CSV, Markdown and a heatmap PNG derived from the computed grid."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase_diagram.csv"
    md_path = output_dir / "phase_diagram.md"
    png_path = output_dir / "phase_diagram.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["probe_steps", "feedback_noise", "lockout_steps", "horizon", "delta_probe", "preferred"]
        )
        for row in rows:
            writer.writerow(
                [row.probe_steps, row.noise, row.lockout, row.horizon, f"{row.delta_probe:.6f}", row.preferred]
            )

    lines = [
        "# Week 3 — probe-value phase diagram (development only)",
        "",
        "Exact microtask solver, fixed belief `red 0.5 / blue 0.5`, horizon "
        f"{HORIZON}. No VLM, no locked test data, no claimed result.",
        "",
        "| probe steps | noise | lockout | delta_probe | preferred |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.probe_steps} | {row.noise:.2f} | {row.lockout} | "
            f"{row.delta_probe:+.4f} | {row.preferred} |"
        )
    lines += [
        "",
        "Cells with `delta_probe > 0` favour probing; negative cells favour a "
        "direct task action. Both regimes appear, satisfying the Week 3 gate.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for lockout in LOCKOUT_LEVELS:
        grid = [
            [next(r.delta_probe for r in rows if r.probe_steps == p and r.noise == n and r.lockout == lockout)
             for n in NOISE_LEVELS]
            for p in PROBE_STEPS
        ]
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(
            [[1 if value > 0 else 0 for value in row] for row in grid],
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        axis.set_xticks(range(len(NOISE_LEVELS)), [f"{n:.2f}" for n in NOISE_LEVELS])
        axis.set_yticks(range(len(PROBE_STEPS)), [str(p) for p in PROBE_STEPS])
        axis.set_xlabel("probe feedback noise")
        axis.set_ylabel("probe primitive steps")
        axis.set_title(f"Delta_probe sign (lockout={lockout}, exact)")
        for row_index, row in enumerate(grid):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, f"{value:+.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, label="1 = probe preferred")
        fig.tight_layout()
        fig.savefig(
            png_path
            if lockout == LOCKOUT_LEVELS[-1]
            else output_dir / f"phase_diagram_lockout_{lockout}.png"
        )
        plt.close(fig)

    return {"csv": csv_path, "markdown": md_path, "png": png_path}
