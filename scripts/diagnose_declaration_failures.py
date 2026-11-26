"""Reproduce and diagnose the exposed v3/v4 declaration failures.

The script is intentionally CPU-only and restricted to the ten exposed
development maps (0-9) in the frozen development manifest. It re-runs the frozen
policy objects only to capture the public decision trajectory that the original
record format did not store. It does not change policy code, run maps above 9,
or write anything outside the requested reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ruleshift.agents.grid_baselines import (
    DiagnosticBeliefPolicy,
    DiagnosticBeliefPolicyV4,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.declaration_diagnosis import (
    aggregate_categories,
    classify_trace,
    outcome_counts,
    paired_transition,
)
from ruleshift.evaluator.grid_manifest import read_manifest
from ruleshift.evaluator.grid_runner import run_grid_episode
from ruleshift.public_api.grid_rules import RULE_BY_ID, cell_role

ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "e65e1915503246f0f07ed15c4884ac6952925ee8"
DEV_MAP_MAX = 9


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("records file is empty")
    return rows


def validate_records(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 900:
        raise ValueError(f"expected 900 frozen records, found {len(rows)}")
    for row in rows:
        if row.get("method") not in {
            "no_adaptation",
            "diagnostic_belief_v3",
            "diagnostic_belief_v4",
        }:
            raise ValueError(f"unexpected method: {row.get('method')!r}")
        if row.get("scenario_type") != "identification":
            raise ValueError(f"non-static scenario: {row.get('scenario_id')!r}")
        if int(row.get("map_seed", -1)) > DEV_MAP_MAX:
            raise ValueError(f"candidate map refused: {row.get('scenario_id')!r}")
        if int(row.get("run_seed", -1)) not in range(5):
            raise ValueError(f"unexpected seed: {row.get('scenario_id')!r}")
        if not str(row.get("code_commit", "")).startswith("e65e191"):
            raise ValueError(f"unexpected code commit: {row.get('code_commit')!r}")
    for method in {str(r["method"]) for r in rows}:
        count = sum(1 for r in rows if r["method"] == method)
        if count != 300:
            raise ValueError(f"{method} has {count} records, expected 300")


def _marker_color(public_map: Any, target: list[int] | tuple[int, int] | None) -> str | None:
    if target is None:
        return None
    coord = (int(target[0]), int(target[1]))
    for color, position in public_map.markers.items():
        if tuple(position) == coord:
            return str(color)
    return None


def _true_inspect_value(scenario: Any, observation: Any, target: list[int] | tuple[int, int] | None, true_rule: str) -> str | None:
    color = _marker_color(observation.public_map, target)
    if color is None:
        return None
    return str(cell_role(RULE_BY_ID[true_rule], color))


def build_trace(
    scenario: Any,
    method: str,
    *,
    config: GridConfig,
    true_rule: str,
    stored: dict[str, Any],
) -> dict[str, Any]:
    """Run one exposed episode and attach a complete public decision trace."""

    factory = {
        "diagnostic_belief_v3": DiagnosticBeliefPolicy,
        "diagnostic_belief_v4": DiagnosticBeliefPolicyV4,
    }[method]
    env = GridEnv(config)
    observation, info = env.reset(
        seed=int(scenario.run_seed), options={"scenario": scenario}
    )
    policy = factory(config)
    policy.name = method
    is_v4 = method == "diagnostic_belief_v4"
    decisions: list[dict[str, Any]] = []
    original_choose = policy.choose
    original_record = policy.record
    original_safe = policy._commit_is_safe
    current: dict[str, Any] = {}

    def safe_wrapper() -> bool:
        result = bool(original_safe())
        current["actual_safe"] = result
        return result

    def choose(o: Any, i: dict[str, Any]) -> Any:
        current.clear()
        posterior = policy._marginal() if policy.belief is not None else {}
        values = sorted(posterior.values(), reverse=True)
        top = float(values[0]) if values else 0.0
        second = float(values[1]) if len(values) > 1 else 0.0
        decision = {
            "step": int(i.get("env_steps", 0)),
            "posterior": {str(k): float(v) for k, v in posterior.items()},
            "top_rule": (
                str(policy.belief.most_likely()) if policy.belief is not None else None
            ),
            "top": top,
            "second": second,
            "margin": top - second,
            "leader_run": int(getattr(policy, "_top_rule_count", 0)) if is_v4 else None,
            "safe_commit_eligible": False,
            "budget_before": int(o.tool_budget_remaining),
            "position": list(o.agent_position),
            "sign_observed": bool(getattr(policy, "sign_observed", False)),
            "blue_observed": bool(getattr(policy, "blue_observed", False)),
            "tool_attempts": int(getattr(policy, "tool_attempts", 0)),
            "events_before": len(policy.commitment_events()),
        }
        action = original_choose(o, i)
        if is_v4:
            decision["leader_run"] = int(getattr(policy, "_top_rule_count", 0))
        decision["safe_commit_eligible"] = bool(
            current.get(
                "actual_safe",
                decision["top"] >= 0.90
                and decision["margin"] >= 0.10
                and (decision["leader_run"] or 0) >= 2,
            )
        )
        decision["action"] = action.kind
        decision["target"] = list(action.target) if action.target is not None else None
        if action.kind == "INSPECT" and action.target is not None:
            decision["target_color"] = _marker_color(o.public_map, action.target)
            decision["true_tool_value"] = _true_inspect_value(
                scenario, o, action.target, true_rule
            )
        events = policy.commitment_events()
        decision["events_after"] = len(events)
        if len(events) > decision["events_before"]:
            decision["event"] = dict(events[-1])
        decisions.append(decision)
        return action

    def record(
        action: Any, o: Any, reward: float, i: dict[str, Any]
    ) -> None:
        if decisions:
            decisions[-1]["reward"] = float(reward)
            decisions[-1]["next_steps"] = int(i.get("env_steps", 0))
            decisions[-1]["tool_value"] = i.get("last_tool_value")
            decisions[-1]["tool_success"] = i.get("last_tool_success")
        return original_record(action, o, reward, i)

    policy._commit_is_safe = safe_wrapper
    policy.choose = choose
    policy.record = record
    episode = run_grid_episode(env, policy, config, scenario, observation, info)
    events = [dict(event) for event in policy.commitment_events()]
    terminal_posterior = policy._marginal() if policy.belief is not None else {}
    terminal_values = sorted(terminal_posterior.values(), reverse=True)
    terminal_top = float(terminal_values[0]) if terminal_values else 0.0
    terminal_second = float(terminal_values[1]) if len(terminal_values) > 1 else 0.0
    trace = {
        "scenario_id": stored["scenario_id"],
        "scenario_type": str(scenario.scenario_type),
        "map_seed": int(scenario.map_seed),
        "run_seed": int(scenario.run_seed),
        "method": method,
        "status": stored["commitment_status"],
        "true_rule": true_rule,
        "task_success": bool(episode.success),
        "env_steps": int(episode.env_steps),
        "termination_reason": str(episode.termination_reason),
        "terminal_map": str(episode.final_rule_estimate),
        "terminal_posterior": {
            str(k): float(v) for k, v in terminal_posterior.items()
        },
        "terminal": {
            "top": terminal_top,
            "second": terminal_second,
            "margin": terminal_top - terminal_second,
            "run": int(getattr(policy, "_top_rule_count", 0)) if is_v4 else None,
            "tool_units": int(episode.tool_units),
            "termination_reason": str(episode.termination_reason),
        },
        "events": events,
        "decisions": decisions,
    }
    return trace


def assert_replay_matches(trace: dict[str, Any], stored: dict[str, Any]) -> None:
    checks = [
        ("task_success", trace["task_success"], bool(stored["task_success"])),
        ("env_steps", trace["env_steps"], int(stored["env_steps"])),
        ("terminal_map", trace["terminal_map"], str(stored["terminal_map"])),
        (
            "declared_rule",
            str(trace["events"][-1].get("declared_rule")) if trace["events"] else None,
            None if stored["declared_rule"] is None else str(stored["declared_rule"]),
        ),
    ]
    for label, actual, expected in checks:
        if actual != expected:
            raise AssertionError(
                f"replay mismatch for {stored['scenario_id']} {stored['method']} {label}: "
                f"actual={actual!r} expected={expected!r}"
            )
    if trace["events"]:
        if int(trace["events"][-1]["step"]) != int(stored["declared_step"]):
            raise AssertionError(
                f"declaration step mismatch for {stored['scenario_id']}"
            )


def _tool_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    units = [int(r["tool_units"]) for r in rows]
    events: Counter[str] = Counter()
    for row in rows:
        for event in row.get("tool_events") or []:
            events[str(event.get("action"))] += 1
    return {
        "episodes": len(rows),
        "mean_tool_units": (sum(units) / len(units)) if units else 0.0,
        "zero_tool_episodes": sum(1 for value in units if value == 0),
        "tool_units": dict(sorted(Counter(units).items())),
        "tool_events": dict(sorted(events.items())),
    }


def _task_transition_examples(
    rows_v3: list[dict[str, Any]], rows_v4: list[dict[str, Any]]
) -> dict[str, Any]:
    v3 = {r["scenario_id"]: r for r in rows_v3}
    v4 = {r["scenario_id"]: r for r in rows_v4}
    improved = [sid for sid in v3 if not v3[sid]["task_success"] and v4[sid]["task_success"]]
    worsened = [sid for sid in v3 if v3[sid]["task_success"] and not v4[sid]["task_success"]]
    return {
        "improved": improved,
        "worsened": worsened,
        "improved_tools_v3": _tool_stats([v3[sid] for sid in improved]),
        "improved_tools_v4": _tool_stats([v4[sid] for sid in improved]),
    }


def _declaration_semantics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(r["commitment_status"]) for r in rows)
    earlier_wrong = [
        r for r in rows if int(r.get("earlier_wrong_commitments") or 0) > 0
    ]
    final_wrong = [r for r in rows if r["commitment_status"] == "wrong_commit"]
    any_wrong = {
        str(r["scenario_id"]): r
        for r in rows
        if r["commitment_status"] == "wrong_commit"
        or int(r.get("earlier_wrong_commitments") or 0) > 0
    }
    final_correct_clean = [
        r
        for r in rows
        if r["commitment_status"] == "correct_commit"
        and int(r.get("earlier_wrong_commitments") or 0) == 0
    ]
    return {
        "final_statuses": {
            key: int(statuses.get(key, 0))
            for key in ("correct_commit", "wrong_commit", "unresolved")
        },
        "episodes_with_any_wrong_explicit_declaration": len(any_wrong),
        "any_wrong_scenarios": sorted(any_wrong),
        "final_correct_with_earlier_wrong_declaration": len(earlier_wrong),
        "earlier_wrong_scenarios": sorted(
            str(r["scenario_id"]) for r in earlier_wrong
        ),
        "final_correct_clean": len(final_correct_clean),
        "final_wrong": len(final_wrong),
        "terminal_map_correct_but_not_correctly_declared": sum(
            1
            for r in rows
            if bool(r["map_correct"]) and r["commitment_status"] != "correct_commit"
        ),
    }


def _unresolved_terminal_stats(traces: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [t for t in traces if t["status"] == "unresolved"]
    return {
        "episodes": len(rows),
        "terminal_top_at_least_0_90": sum(
            1 for t in rows if float(t["terminal"]["top"]) >= 0.90
        ),
        "terminal_margin_below_0_10": sum(
            1 for t in rows if float(t["terminal"]["margin"]) < 0.10
        ),
        "terminal_ties": sum(
            1 for t in rows if float(t["terminal"]["margin"]) <= 1e-12
        ),
        "zero_tool_episodes": sum(
            1 for t in rows if int(t["terminal"]["tool_units"]) == 0
        ),
        "final_leader_run_1": sum(
            1 for t in rows if int(t["terminal"].get("run") or 0) == 1
        ),
        "decision_counts": dict(
            sorted(Counter(int(t["env_steps"]) for t in rows).items())
        ),
        "tool_unit_counts": dict(
            sorted(Counter(int(t["terminal"]["tool_units"]) for t in rows).items())
        ),
    }


def _v4_cause_by_v3_status(traces: list[dict[str, Any]]) -> dict[str, Any]:
    v4 = {t["scenario_id"]: t for t in traces if t["method"] == "diagnostic_belief_v4"}
    v3 = {t["scenario_id"]: t for t in traces if t["method"] == "diagnostic_belief_v3"}
    counter: Counter[tuple[str, str]] = Counter()
    for sid, trace in v4.items():
        if trace["status"] != "unresolved":
            continue
        counter[(str(trace["diagnosis"]["primary"]), str(v3[sid]["status"]))] += 1
    return {
        f"{cause}|{status}": int(count)
        for (cause, status), count in sorted(counter.items())
    }


def _trace_rows(traces: list[dict[str, Any]], config: GridConfig) -> list[dict[str, Any]]:
    v4 = {t["scenario_id"]: t for t in traces if t["method"] == "diagnostic_belief_v4"}
    v3 = {t["scenario_id"]: t for t in traces if t["method"] == "diagnostic_belief_v3"}
    rows: list[dict[str, Any]] = []
    for trace in sorted(traces, key=lambda t: (t["scenario_id"], t["method"])):
        sid = str(trace["scenario_id"])
        counterpart = v3[sid] if trace["method"] == "diagnostic_belief_v4" else v4[sid]
        events = list(trace.get("events") or [])
        criterion = [
            d for d in trace["decisions"] if d["top"] >= 0.90 and d["margin"] >= 0.10
        ]
        eligible = [d for d in trace["decisions"] if d["safe_commit_eligible"]]
        rows.append(
            {
                "scenario_id": sid,
                "method": trace["method"],
                "status": trace["status"],
                "primary_cause": trace["diagnosis"]["primary"],
                "true_rule": trace["true_rule"],
                "task_success": trace["task_success"],
                "env_steps": trace["env_steps"],
                "termination_reason": trace["termination_reason"],
                "decisions": len(trace["decisions"]),
                "criterion_decisions": len(criterion),
                "eligible_decisions": len(eligible),
                "declaration_events": len(events),
                "declared_rule": events[-1]["declared_rule"] if events else "",
                "declared_step": events[-1]["step"] if events else "",
                "declared_confidence": events[-1]["confidence"] if events else "",
                "terminal_top": trace["terminal"]["top"],
                "terminal_margin": trace["terminal"]["margin"],
                "terminal_run": trace["terminal"].get("run") or "",
                "tool_units": trace["terminal"]["tool_units"],
                "budget_remaining": max(
                    0, int(config.tool_budget) - int(trace["terminal"]["tool_units"])
                ),
                "counterpart_status": counterpart["status"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_payload(rows: list[dict[str, Any]], manifest: Any) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    v3 = sorted(by_method["diagnostic_belief_v3"], key=lambda r: r["scenario_id"])
    v4 = sorted(by_method["diagnostic_belief_v4"], key=lambda r: r["scenario_id"])
    stored4 = {str(r["scenario_id"]): r for r in v4}
    stored3 = {str(r["scenario_id"]): r for r in v3}
    targets = [
        sid
        for sid, row in stored4.items()
        if row["commitment_status"] in {"unresolved", "wrong_commit"}
    ]
    config = GridConfig()
    traces: list[dict[str, Any]] = []
    for sid in sorted(targets):
        scenario = manifest.as_scenario(sid)
        v4_trace = build_trace(
            scenario,
            "diagnostic_belief_v4",
            config=config,
            true_rule=str(stored4[sid]["terminal_true_rule"]),
            stored=stored4[sid],
        )
        assert_replay_matches(v4_trace, stored4[sid])
        traces.append(v4_trace)
        v3_trace = build_trace(
            scenario,
            "diagnostic_belief_v3",
            config=config,
            true_rule=str(stored3[sid]["terminal_true_rule"]),
            stored=stored3[sid],
        )
        assert_replay_matches(v3_trace, stored3[sid])
        traces.append(v3_trace)
    for trace in traces:
        trace["diagnosis"] = classify_trace(trace)
    v4_traces = [t for t in traces if t["method"] == "diagnostic_belief_v4"]
    v3_traces = [t for t in traces if t["method"] == "diagnostic_belief_v3"]
    transitions = paired_transition(
        v3,
        v4,
        left_name="diagnostic_belief_v3",
        right_name="diagnostic_belief_v4",
    )
    episode_index = _trace_rows(traces, config)
    return {
        "metadata": {
            "frozen_commit": FROZEN_COMMIT,
            "manifest_sha256": manifest.sha256,
            "development_maps": list(manifest.map_ids),
            "development_seeds": list(manifest.run_seeds),
            "records": len(rows),
            "candidate_maps_used": 0,
        },
        "outcomes": {method: outcome_counts(values) for method, values in sorted(by_method.items())},
        "transitions": transitions,
        "declaration_semantics": {
            method: _declaration_semantics(values)
            for method, values in sorted(by_method.items())
        },
        "unresolved_stats": _unresolved_terminal_stats(v4_traces),
        "v4_cause_by_v3_status": _v4_cause_by_v3_status(traces),
        "task_transition_examples": _task_transition_examples(v3, v4),
        "trace_categories": {
            "all": aggregate_categories(v4_traces + v3_traces),
            "v4": aggregate_categories(v4_traces),
            "v3_matched": aggregate_categories(v3_traces),
        },
        "tool_stats": {
            method: _tool_stats(values) for method, values in sorted(by_method.items())
        },
        "episode_index": episode_index,
        "traces": traces,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Declaration diagnosis: generated summary")
    lines.append("")
    lines.append("Frozen commit: " + payload["metadata"]["frozen_commit"])
    lines.append("Manifest: " + payload["metadata"]["manifest_sha256"])
    lines.append("")
    lines.append("## Outcomes")
    lines.append("")
    lines.append("| Method | Correct | Wrong | Unresolved | Task | MAP |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method, result in payload["outcomes"].items():
        statuses = result["statuses"]
        lines.append(
            "| " + method + " | " + str(statuses["correct_commit"]) + " | "
            + str(statuses["wrong_commit"]) + " | " + str(statuses["unresolved"])
            + " | " + str(result["task_success"]) + " | " + str(result["map_correct"]) + " |"
        )
    lines.append("")
    lines.append("## v3 to v4 declaration transitions")
    lines.append("")
    for key, value in payload["transitions"]["declaration_status"].items():
        lines.append("- " + key + ": " + str(value))
    lines.append("")
    lines.append("## Trace categories")
    lines.append("")
    for scope, result in payload["trace_categories"].items():
        lines.append("### " + scope)
        for key, value in result["primary"].items():
            lines.append("- " + key + ": " + str(value))
    return chr(10).join(lines) + chr(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "results/p0_dev_static_e65e191/records.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/p0_dev_static_manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/declaration_diagnosis_v1.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=ROOT / "reports/declaration_diagnosis_v1.md",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "reports/declaration_diagnosis_v1.csv",
    )
    parser.add_argument(
        "--trace-jsonl-out",
        type=Path,
        default=None,
        help="optional path for the complete per-decision target traces",
    )
    args = parser.parse_args()
    rows = load_records(args.records)
    validate_records(rows)
    manifest = read_manifest(args.manifest)
    if any(int(map_id) > DEV_MAP_MAX for map_id in manifest.map_ids):
        raise ValueError("development manifest contains a candidate map")
    payload = build_payload(rows, manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_out.write_text(render_markdown(payload), encoding="utf-8")
    write_csv(args.csv_out, payload["episode_index"])
    if args.trace_jsonl_out is not None:
        args.trace_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_jsonl_out.open("w", encoding="utf-8") as handle:
            for trace in payload["traces"]:
                handle.write(json.dumps(trace, sort_keys=True) + chr(10))
    print(
        "wrote " + str(args.out) + ", " + str(args.markdown_out)
        + ", and " + str(args.csv_out)
        + " traces=" + str(len(payload["traces"]))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
