"""Week 9 held-out evaluation scaffolding (SYNTHETIC while using the stub).

The scenario values here are deliberately disjoint from the development grid.
Nothing is frozen: `configs/preregistered_test.yaml` stays DRAFT_NOT_FROZEN
until a real frozen-VLM validation has been run.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from ruleshift.agents.model_check import readiness_errors
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.envs.config import EnvConfig, default_config_path
from ruleshift.evaluator.closed_loop import (
    ClosedLoopOutput,
    run_closed_loop,
)
from ruleshift.evaluator.metrics_ci import PairedSummary
from ruleshift.evaluator.provenance import implementation_digest

#: Method ids for the proposed method, in preference order. The ``_stub``
#: suffix only reflects which backend was injected, so a real-backend run uses
#: ``m1`` and the same code path still finds the proposed method.
PROPOSED_LABELS = ("m1", "m1_stub")
DEFAULT_PREREGISTRATION = default_config_path("preregistered_test.yaml")
HELD_OUT_DETOUR = (4, 6)
HELD_OUT_NOISE = (0.05, 0.15)
HELD_OUT_LOCKOUT = (4, 6)
HELD_OUT_HAZARD = (0.04, 0.07)


def build_heldout_family_a(max_env_steps: int = 60) -> tuple[tuple[str, EnvConfig], ...]:
    """Build held-out scenarios with parameter values unseen in development."""

    scenarios: list[tuple[str, EnvConfig]] = []
    for detour in HELD_OUT_DETOUR:
        for noise in HELD_OUT_NOISE:
            for lockout in HELD_OUT_LOCKOUT:
                for index, hazard in enumerate(HELD_OUT_HAZARD):
                    scenario_id = (
                        f"heldA-d{detour}-n{noise}-k{lockout}-h{index}"
                    )
                    scenarios.append(
                        (
                            scenario_id,
                            EnvConfig(
                                probe_detour_steps=detour,
                                feedback_noise=noise,
                                lockout_steps=lockout,
                                shift_hazard=hazard,
                                disable_shift=False,
                                initial_mechanic="red",
                                max_env_steps=max_env_steps,
                                scenario_id=scenario_id,
                            ),
                        )
                    )
    return tuple(scenarios)


@dataclass
class HeldOutResult:
    closed_loop: ClosedLoopOutput
    method_stats: dict[str, dict[str, float | int]]
    summaries: dict[tuple[str, str], PairedSummary]
    max_vlm_calls_per_episode: int = 6

    @property
    def episodes(self):
        return self.closed_loop.episodes


def proposed_label(methods: tuple[str, ...]) -> str | None:
    """Return the proposed-method id actually present in *methods*."""

    for candidate in PROPOSED_LABELS:
        if candidate in methods:
            return candidate
    return None


def run_heldout_evaluation(
    scenarios: tuple[tuple[str, EnvConfig], ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...] = (101,),
    backend: object | None = None,
    cache: object | None = None,
    max_vlm_calls_per_episode: int = 6,
    preregistration: Path = DEFAULT_PREREGISTRATION,
) -> HeldOutResult:
    """Run paired held-out episodes and accounting for every method."""

    if backend is not None and not isinstance(backend, StubVLMBackend):
        frozen = yaml.safe_load(Path(preregistration).read_text(encoding="utf-8"))
        errors = readiness_errors({"revision": getattr(backend, "revision", None)}, "locked", frozen)
        if errors:
            raise ValueError("; ".join(errors))
        # Compare the submitted plan before issuing any generation request.
        # JSON normalizes tuple-valued EnvConfig fields to YAML-compatible lists.
        requested = {
            "scenarios": [{"id": key, "config": asdict(config)} for key, config in scenarios],
            "methods": list(methods),
            "seeds": list(seeds),
            "model_id": getattr(backend, "model_id", None),
            "revision": backend.revision,
            "implementation_sha256": implementation_digest(),
        }
        budget = frozen.get("call_and_step_budget", {})
        if (frozen.get("locked_run") != json.loads(json.dumps(requested))
                or budget.get("max_vlm_calls_per_episode") != max_vlm_calls_per_episode
                or any(config.max_env_steps != budget.get("max_env_steps") for _, config in scenarios)):
            raise ValueError("scenarios, seeds, methods, model, source, prompts and budgets must match the frozen run")
    proposed = proposed_label(tuple(methods))
    comparisons: tuple[tuple[str, str], ...] = (
        tuple((proposed, method) for method in methods if method != proposed)
        if proposed is not None
        else ()
    )
    result = run_closed_loop(
        scenarios,
        methods,
        seeds,
        comparisons,
        backend=backend,
        cache=cache,
        max_vlm_calls_per_episode=max_vlm_calls_per_episode,
    )

    stats: dict[str, dict[str, float | int]] = {}
    for method in methods:
        episodes = [
            episode
            for (_, _, name), episode in result.episodes.items()
            if name == method
        ]
        decisions = [row for episode in episodes for row in episode.public_rows]
        total_return = sum(episode.total_return for episode in episodes)
        calls = sum(episode.vlm_calls for episode in episodes)
        tokens = sum(
            episode.input_tokens + episode.output_tokens for episode in episodes
        )
        stats[method] = {
            "episodes": len(episodes),
            "total_return": total_return,
            "mean_return": total_return / len(episodes),
            "steps": sum(episode.steps for episode in episodes),
            "probes": sum(episode.probes for episode in episodes),
            "vlm_calls": calls,
            "json_valid": sum(row["json_valid"] is True for row in decisions),
            "invalid_json": sum(row["json_valid"] is False for row in decisions),
            "invalid_actions": sum(episode.invalid_actions for episode in episodes),
            "tokens": tokens,
            "latency_ms": sum(episode.latency_ms for episode in episodes),
            "call_cap_hits": sum(episode.call_cap_hits for episode in episodes),
        }
    return HeldOutResult(
        result, stats, result.summaries, max_vlm_calls_per_episode
    )


def write_result_files(
    result: HeldOutResult, results_dir: Path, report_dir: Path
) -> dict[str, Path]:
    """Write per-episode CSV, paired JSON and the Week 9 report."""

    results_dir = Path(results_dir)
    report_dir = Path(report_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "main_family_A.csv"
    json_path = results_dir / "main_paired.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario_id",
                "seed",
                "method",
                "return",
                "steps",
                "probes",
                "vlm_calls",
                "input_tokens",
                "output_tokens",
                "invalid_actions",
            ]
        )
        for (scenario_id, seed, method), episode in sorted(result.episodes.items()):
            writer.writerow(
                [
                    scenario_id,
                    seed,
                    method,
                    f"{episode.total_return:.6f}",
                    episode.steps,
                    episode.probes,
                    episode.vlm_calls,
                    episode.input_tokens,
                    episode.output_tokens,
                    episode.invalid_actions,
                ]
            )

    summary_rows = {
        f"{method}_minus_{baseline}": {
            "mean_difference": summary.mean_difference,
            "ci95_low": summary.low,
            "ci95_high": summary.high,
            "n_scenarios": summary.n_scenarios,
        }
        for (method, baseline), summary in result.summaries.items()
    }
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    lines = [
        "# Week 9 - Main held-out evaluation" + (" (SYNTHETIC)" if result.closed_loop.synthetic else ""),
        "",
        ("**SYNTHETIC.** This dry run used the deterministic stub and a development "
         "fixture. It provides no empirical VLM result."
         if result.closed_loop.synthetic else
         "This run used an injected model backend and matched the preregistered run specification."),
        "",
        "Seeds are averaged within each scenario. Intervals resample scenarios. "
        f"All model methods had a cap of {result.max_vlm_calls_per_episode} calls per episode.",
        "",
        "## Paired differences (proposed minus baseline)",
        "",
        "| Comparison | Scenarios | Mean difference | 95% paired CI |",
        "|---|---:|---:|---|",
    ]
    for (method, baseline), summary in result.summaries.items():
        lines.append(
            f"| {method} − {baseline} | {summary.n_scenarios} | "
            f"{summary.mean_difference:+.3f} | [{summary.low:+.3f}, {summary.high:+.3f}] |"
        )

    lines += [
        "",
        "## Cost and call accounting",
        "",
        "| Method | Episodes | Mean return | Probes | VLM calls | Tokens "
        "| Invalid actions | Cap fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, stats in result.method_stats.items():
        lines.append(
            f"| {method} | {stats['episodes']} | {stats['mean_return']:+.3f} | "
            f"{stats['probes']} | {stats['vlm_calls']} | {stats['tokens']} | "
            f"{stats['invalid_actions']} | {stats['call_cap_hits']} |"
        )

    negative = [
        key for key, summary in result.summaries.items() if summary.low <= 0.0
    ]
    lines += [
        "",
        "## Negative and inconclusive results",
        "",
        "All comparisons whose paired interval does not support the proposed "
        "method are reported here rather than omitted:",
        "",
    ]
    if negative:
        for method, baseline in negative:
            summary = result.summaries[method, baseline]
            verdict = "entirely at or below zero" if summary.high <= 0 else "includes zero"
            lines.append(f"- {method} minus {baseline}: interval {verdict}.")
    else:
        lines.append(
            "- All reported intervals lie above zero."
        )
    report_path = report_dir / "w9_main_results.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "csv": csv_path,
        "paired_json": json_path,
        "report": report_path,
    }
