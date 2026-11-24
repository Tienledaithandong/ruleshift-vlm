"""CPU phase diagram of Delta_probe over dev cost/noise settings.

All cells are computed by the exact microtask solver with the same macro
durations the environment charges (``2 * detour + 1`` for a probe, public gate
travel + 1 for a commit). This is a development diagnostic, not a locked test
set and not a VLM result.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ruleshift.envs.config import load_dev_grid
from ruleshift.evaluator.exact_solver import (
    ExactSolver,
    family_a_microtask,
)
from ruleshift.public_api.schemas import FAMILY_A_ACTIONS

BELIEF = {("red", False): 0.5, ("blue", True): 0.5}
#: Probe, commit and the largest lockout must all fit inside the horizon.
HORIZON = 24
PROBE_DETOURS = (1, 3, 5)
NOISE_LEVELS = (0.0, 0.1, 0.25)
LOCKOUT_LEVELS = (0, 2, 8)
#: Public gate travel on the default 5x5 dev grid (start (0,0), gate (4,0)).
DEV_COMMIT_TRAVEL = 4


@dataclass(frozen=True)
class PhaseCell:
    probe_detour_steps: int
    probe_steps: int
    noise: float
    lockout: int
    horizon: int
    delta_probe: float
    preferred: str


def compute_phase_grid(config_path: Path | None = None) -> tuple[PhaseCell, ...]:
    """Enumerate the dev grid with the exact solver."""

    dev = load_dev_grid(config_path)
    grid = dev["grid"]
    environment = dev["environment"]
    width, height = environment["grid_size"]
    horizon = min(HORIZON, environment["max_env_steps"])
    cells: list[PhaseCell] = []
    for detour in grid["probe_detour_steps"]:
        if not 0 <= detour <= width + height - 2:
            raise ValueError(f"station detour {detour} is outside the configured grid")
        for noise in grid["feedback_noise_rate"]:
            for lockout in environment["lockout_steps_options"]:
                config = family_a_microtask(
                    probe_detour_steps=detour,
                    commit_travel_steps=width - 1,
                    feedback_noise=noise,
                    lockout_steps=lockout,
                    step_cost=environment["step_cost"],
                    reward_success=environment["delivery_reward"],
                    actions=FAMILY_A_ACTIONS,
                )
                delta = ExactSolver(config, FAMILY_A_ACTIONS).delta_probe(
                    BELIEF, horizon
                )
                cells.append(
                    PhaseCell(
                        probe_detour_steps=detour,
                        probe_steps=2 * detour + 1,
                        noise=noise,
                        lockout=lockout,
                        horizon=horizon,
                        delta_probe=delta,
                        preferred="probe" if delta > 0 else "task" if delta < 0 else "tie",
                    )
                )
    return tuple(cells)


def write_report(rows: tuple[PhaseCell, ...], output_dir: Path) -> dict[str, Path]:
    """Write CSV, Markdown and a heatmap PNG derived from the computed grid."""

    output_dir = Path(output_dir)
    if not rows:
        raise ValueError("phase diagram needs at least one development cell")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase_diagram.csv"
    md_path = output_dir / "phase_diagram.md"
    png_path = output_dir / "phase_diagram.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "probe_detour_steps",
                "probe_steps",
                "feedback_noise",
                "lockout_steps",
                "horizon",
                "delta_probe",
                "preferred",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.probe_detour_steps,
                    row.probe_steps,
                    row.noise,
                    row.lockout,
                    row.horizon,
                    f"{row.delta_probe:.6f}",
                    row.preferred,
                ]
            )

    lines = [
        "# Week 3 — probe-value phase diagram (development only)",
        "",
        "Exact microtask solver with the environment's real macro durations "
        f"(probe = 2·detour+1 steps), fixed belief `red 0.5 / blue 0.5`, "
        f"horizon {rows[0].horizon}, full public macro menu. No VLM, no locked test "
        "data, no claimed result.",
        "",
        "| probe detour | probe steps | noise | lockout | delta_probe | preferred |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.probe_detour_steps} | {row.probe_steps} | "
            f"{row.noise:.2f} | {row.lockout} | "
            f"{row.delta_probe:+.4f} | {row.preferred} |"
        )
    lines += [
        "",
        f"Probe-favourable cells: {sum(r.delta_probe > 0 for r in rows)}. "
        f"Task-favourable cells: {sum(r.delta_probe < 0 for r in rows)}. "
        f"Ties: {sum(r.delta_probe == 0 for r in rows)}.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lockouts = sorted({row.lockout for row in rows})
    detours = sorted({row.probe_detour_steps for row in rows})
    noise_levels = sorted({row.noise for row in rows})
    for lockout in lockouts:
        grid = [
            [
                next(
                    r.delta_probe
                    for r in rows
                    if r.probe_detour_steps == detour
                    and r.noise == noise
                    and r.lockout == lockout
                )
                for noise in noise_levels
            ]
            for detour in detours
        ]
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(
            [[1 if value > 0 else -1 if value < 0 else 0 for value in row] for row in grid],
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            aspect="auto",
        )
        axis.set_xticks(range(len(noise_levels)), [f"{n:.2f}" for n in noise_levels])
        axis.set_yticks(
            range(len(detours)),
            [f"{d} ({2 * d + 1} steps)" for d in detours],
        )
        axis.set_xlabel("probe feedback noise")
        axis.set_ylabel("probe detour (probe steps)")
        axis.set_title(f"Delta_probe sign (lockout={lockout}, exact)")
        for row_index, row in enumerate(grid):
            for column_index, value in enumerate(row):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, ticks=[-1, 0, 1], label="task / tie / probe")
        fig.tight_layout()
        fig.savefig(
            png_path
            if lockout == lockouts[-1]
            else output_dir / f"phase_diagram_lockout_{lockout}.png"
        )
        plt.close(fig)

    return {"csv": csv_path, "markdown": md_path, "png": png_path}
