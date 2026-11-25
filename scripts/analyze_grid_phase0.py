"""Analyze the 10-map Phase 0 replication and write reproducible summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ruleshift.agents.grid_baselines import DiagnosticBeliefPolicy, as_observation
from ruleshift.envs.grid.config import GridConfig, load_grid_config
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.envs.grid.mapgen import generate_solvable_map
from ruleshift.evaluator.grid_runner import build_grid_scenarios, run_grid_episode
from ruleshift.public_api.grid_rules import RULE_BY_ID, cell_role

HS_RULES = ("hazard_goal_swap", "sign_composition")


def _rate(successes: int, n: int) -> float:
    return successes / n if n else 0.0


def _metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    steps = sum(int(row["env_steps"]) for row in rows)
    tools = sum(int(row["tool_units"]) for row in rows)
    initial = sum(
        bool(row["success"]) and row["final_rule_estimate"] == row["_initial_rule"]
        for row in rows
    )
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": _rate(successes, n),
        "env_steps_total": steps,
        "env_steps_mean": steps / n if n else 0.0,
        "tool_units_total": tools,
        "tool_units_mean": tools / n if n else 0.0,
        "false_alarms": sum(bool(row["false_alarm"]) for row in rows),
        "final_rule_matches_initial": initial,
    }


def _group_metrics(
    rows: Sequence[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _metrics(group) for name, group in sorted(groups.items())}


def _mcnemar_exact(only_diag: int, only_base: int) -> float:
    n = only_diag + only_base
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_diag, only_base) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def _paired(
    diag: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ids = sorted(set(diag) & set(base))
    both = sum(
        bool(diag[s]["success"]) and bool(base[s]["success"]) for s in ids
    )
    only_diag = sum(
        bool(diag[s]["success"]) and not bool(base[s]["success"]) for s in ids
    )
    only_base = sum(
        not bool(diag[s]["success"]) and bool(base[s]["success"]) for s in ids
    )
    neither = len(ids) - both - only_diag - only_base
    return {
        "episodes": len(ids),
        "both": both,
        "only_diagnostic": only_diag,
        "only_baseline": only_base,
        "neither": neither,
        "diagnostic_success_rate": _rate(both + only_diag, len(ids)),
        "baseline_success_rate": _rate(both + only_base, len(ids)),
        "mcnemar_exact_p": _mcnemar_exact(only_diag, only_base),
    }


def _paired_group(
    diag: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    scenarios: dict[str, Any],
    key: str,
) -> dict[str, dict[str, Any]]:
    ids = sorted(set(diag) & set(base))
    groups: dict[str, list[str]] = defaultdict(list)
    for scenario_id in ids:
        groups[str(getattr(scenarios[scenario_id], key))].append(scenario_id)
    return {
        name: _paired({s: diag[s] for s in group}, {s: base[s] for s in group})
        for name, group in sorted(groups.items())
    }


def _reachable(
    public: Any, start: tuple[int, int], blocked: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    seen = {start}
    queue = [start]
    while queue:
        x, y = queue.pop()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            cell = (nx, ny)
            if not (0 <= nx < public.width and 0 <= ny < public.height):
                continue
            if cell in blocked or cell in seen:
                continue
            seen.add(cell)
            queue.append(cell)
    return seen


def _sign_audit(
    config: GridConfig, maps: int, seeds: int
) -> dict[str, Any]:
    scenarios = build_grid_scenarios(maps, seeds, config)
    by_map: dict[str, Any] = {}
    for scenario in scenarios:
        map_id = scenario.scenario_id.split("-")[0]
        if map_id in by_map:
            continue
        env = GridEnv(config)
        observation, _info = env.reset(
            seed=scenario.run_seed, options={"scenario": scenario}
        )
        observation = as_observation(observation)
        public = observation.public_map
        signpost = public.signpost
        record: dict[str, Any] = {
            "signpost": list(signpost) if signpost is not None else None,
            "side0_reachable": False,
            "side1_reachable": False,
        }
        if signpost is not None:
            signpost = tuple(signpost)
            blocked = (
                set(public.walls)
                | set(public.markers.values())
                | {signpost}
            )
            if public.door is not None:
                blocked.add(public.door)
            reachable = _reachable(public, tuple(observation.agent_position), blocked)
            side0 = DiagnosticBeliefPolicy._sign_cells(
                public, signpost, 0
            )
            side1 = DiagnosticBeliefPolicy._sign_cells(
                public, signpost, 1
            )
            record["side0_cells"] = [list(cell) for cell in side0]
            record["side1_cells"] = [list(cell) for cell in side1]
            record["side0_reachable"] = any(cell in reachable for cell in side0)
            record["side1_reachable"] = any(cell in reachable for cell in side1)
            record["any_informative_face_reachable"] = bool(
                record["side0_reachable"] or record["side1_reachable"]
            )
            record["base_diagnostic_face_reachable"] = bool(
                record["side0_reachable"]
            )
        else:
            record["any_informative_face_reachable"] = False
            record["base_diagnostic_face_reachable"] = False
        by_map[map_id] = record
    return {
        "method": "walls + markers + signpost + closed door",
        "maps": by_map,
        "maps_with_reachable_informative_face": sum(
            bool(record["any_informative_face_reachable"])
            for record in by_map.values()
        ),
        "maps_with_reachable_base_diagnostic_face": sum(
            bool(record["base_diagnostic_face_reachable"])
            for record in by_map.values()
        ),
        "maps_without_signpost": sum(
            record["signpost"] is None for record in by_map.values()
        ),
    }


def _map_identity_audit(config: GridConfig, maps: int) -> dict[str, Any]:
    """Group map indices whose public geometry is byte-for-byte identical."""

    groups: dict[str, list[str]] = defaultdict(list)
    attempts: dict[str, int] = {}
    for map_index in range(maps):
        result = generate_solvable_map(map_index, config)
        public = result.map.public
        identity = json.dumps(
            {
                "width": public.width,
                "height": public.height,
                "walls": sorted(public.walls),
                "markers": sorted(public.markers.items()),
                "signpost": public.signpost,
                "start": public.start,
                "key": public.key,
                "door": public.door,
            },
            sort_keys=True,
        )
        map_id = f"m{map_index}"
        groups[identity].append(map_id)
        attempts[map_id] = int(result.attempts)
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    return {
        "maps_requested": maps,
        "unique_public_maps": len(groups),
        "duplicate_groups": duplicate_groups,
        "duplicate_map_count": sum(len(ids) for ids in duplicate_groups),
        "attempts": attempts,
        "note": (
            "Counts are public geometry identities, not scenario ids. "
            "Duplicate groups are reported explicitly; scenario ids remain "
            "unique."
        ),
    }


def _eligibility(
    diag_rows: Sequence[dict[str, Any]],
    base_rows: Sequence[dict[str, Any]],
    sign_affordance: dict[str, Any],
    scenarios: dict[str, Any],
    methods: Sequence[str],
) -> dict[str, Any]:
    """Metrics restricted to maps with a readable side-0 diagnostic face."""

    unreadable = {
        map_id
        for map_id, record in sign_affordance["maps"].items()
        if not record["base_diagnostic_face_reachable"]
    }

    def map_id(row: dict[str, Any]) -> str:
        return row["scenario_id"].split("-")[0]

    def keep(row: dict[str, Any]) -> bool:
        return map_id(row) not in unreadable

    eligible_diag = [row for row in diag_rows if keep(row)]
    eligible_base = [row for row in base_rows if keep(row)]
    diag_by_id = {row["scenario_id"]: row for row in eligible_diag}
    base_by_id = {row["scenario_id"]: row for row in eligible_base}
    paired = _paired(diag_by_id, base_by_id)
    all_eligible = eligible_diag + eligible_base
    return {
        "unreadable_maps": sorted(unreadable),
        "eligible_maps": sorted(
            set(sign_affordance["maps"]) - unreadable,
            key=lambda value: int(value.removeprefix("m")),
        ),
        "excluded_episode_rows": len(diag_rows) + len(base_rows) - len(all_eligible),
        "aggregate": {
            method: _metrics([row for row in all_eligible if row["method"] == method])
            for method in methods
        },
        "by_type": {
            method: _group_metrics(
                [row for row in all_eligible if row["method"] == method],
                "_scenario_type",
            )
            for method in methods
        },
        "by_initial_rule": {
            method: _group_metrics(
                [
                    row
                    for row in all_eligible
                    if row["method"] == method
                    and row["_scenario_type"] == "identification"
                ],
                "_initial_rule",
            )
            for method in methods
        },
        "paired": paired,
        "gates": _gates(eligible_diag, paired, []),
    }


def _inspect_evidence(
    scenario: Any, config: GridConfig
) -> dict[str, Any] | None:
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    policy = DiagnosticBeliefPolicy(config)
    episode = run_grid_episode(env, policy, config, scenario, observation, info)
    inspect_rows = [
        row for row in episode.public_rows if row["chosen_action"] == "INSPECT"
    ]
    if not inspect_rows:
        return None
    row = inspect_rows[0]
    target = tuple(row["tool_target"])
    public = as_observation(observation).public_map
    by_cell = {tuple(cell): color for color, cell in public.markers.items()}
    color = by_cell.get(target)
    if color is None:
        return {
            "target": list(target),
            "color": None,
            "expected": None,
            "observed": row.get("last_tool_value"),
            "wrong": False,
        }
    expected = cell_role(RULE_BY_ID[scenario.initial_rule], color)
    observed = row.get("last_tool_value")
    return {
        "target": list(target),
        "color": color,
        "expected": expected,
        "observed": observed,
        "wrong": str(observed) != expected,
    }


def _classify_failure(
    row: dict[str, Any], scenario: Any, config: GridConfig
) -> tuple[str, dict[str, Any]]:
    detail: dict[str, Any] = {
        "env_steps": int(row["env_steps"]),
        "tool_units": int(row["tool_units"]),
        "final_rule_estimate": row["final_rule_estimate"],
        "shift_rule": scenario.shift_rule,
        "shift_time": scenario.shift_time,
        "adaptation_steps": row.get("adaptation_steps"),
    }
    if (
        scenario.scenario_type == "identification"
        and scenario.initial_rule in HS_RULES
        and int(row["tool_units"]) == 0
    ):
        return "zero_tool_unidentified", detail
    if scenario.scenario_type == "identification":
        evidence = _inspect_evidence(scenario, config)
        if evidence is not None:
            detail["inspect"] = evidence
            if evidence["wrong"]:
                return "inspect_noise", detail
    if (
        scenario.scenario_type == "change_detection"
        and scenario.shift_time is not None
        and int(row["env_steps"]) >= int(scenario.shift_time)
    ):
        if row.get("adaptation_steps") is None:
            return "post_shift_no_recovery", detail
        return "post_shift_late", detail
    if int(row["env_steps"]) >= int(getattr(config, "max_steps", 60)):
        return "navigation_timeout", detail
    return "other", detail


def _failure_records(
    rows: Sequence[dict[str, Any]],
    scenarios: dict[str, Any],
    config: GridConfig,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        if row["success"]:
            continue
        scenario = scenarios[row["scenario_id"]]
        classification, detail = _classify_failure(row, scenario, config)
        failures.append(
            {
                "scenario_id": row["scenario_id"],
                "scenario_type": scenario.scenario_type,
                "initial_rule": scenario.initial_rule,
                "classification": classification,
                "detail": detail,
            }
        )
    return failures


def _gates(
    diag_rows: Sequence[dict[str, Any]],
    paired_overall: dict[str, Any],
    failures: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ident = [r for r in diag_rows if r["scenario_type"] == "identification"]
    hazard = [r for r in ident if r["_initial_rule"] == "hazard_goal_swap"]
    sign = [r for r in ident if r["_initial_rule"] == "sign_composition"]
    base = [r for r in ident if r["_initial_rule"] == "base"]
    zero_tool = [
        r
        for r in ident
        if r["_initial_rule"] in HS_RULES and int(r["tool_units"]) == 0
    ]
    checks = {
        "identification_at_least_95pct": _metrics(ident)["success_rate"] >= 0.95,
        "hazard_at_least_90pct": _metrics(hazard)["success_rate"] >= 0.90,
        "sign_at_least_90pct": _metrics(sign)["success_rate"] >= 0.90,
        "base_preserved_100pct": _metrics(base)["success_rate"] == 1.0,
        "no_zero_tool_hazard_or_sign": not zero_tool,
        "tool_units_at_most_3": max(
            int(r["tool_units"]) for r in diag_rows
        )
        <= 3,
        "paired_gain_positive": paired_overall["only_diagnostic"]
        > paired_overall["only_baseline"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "counts": {
            "identification": len(ident),
            "hazard": len(hazard),
            "sign": len(sign),
            "base": len(base),
            "zero_tool_hazard_or_sign": len(zero_tool),
            "failures": len(failures),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 0 10-map CPU replication",
        "",
        f"Input: `{payload['input']}`",
        f"Scenarios: {payload['scenario_count']}; episode rows: {payload['row_count']}",
        "",
        "## Overall",
        "",
        "| Method | Success | Rate | Env steps | Tool units |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in payload["aggregate"]["by_method"].items():
        lines.append(
            f"| `{method}` | {metrics['successes']}/{metrics['episodes']} | "
            f"{_fmt(metrics['success_rate'])} | {metrics['env_steps_total']} | "
            f"{metrics['tool_units_total']} |"
        )
    lines.extend(["", "## Identification by initial rule", ""])
    lines.append("| Rule | Diagnostic | Baseline |")
    lines.append("| --- | ---: | ---: |")
    for rule, metrics in payload["aggregate"]["by_initial_rule"].get(
        "diagnostic_belief", {}
    ).items():
        base = payload["aggregate"]["by_initial_rule"].get(
            "no_adaptation", {}
        ).get(rule, {})
        lines.append(
            f"| `{rule}` | {metrics['successes']}/{metrics['episodes']} | "
            f"{base.get('successes', 0)}/{base.get('episodes', 0)} |"
        )
    lines.extend(["", "## Paired comparison", ""])
    paired = payload["paired"]["overall"]
    lines.append(
        f"Both: {paired['both']}; only diagnostic: "
        f"{paired['only_diagnostic']}; only baseline: "
        f"{paired['only_baseline']}; neither: {paired['neither']}; "
        f"exact McNemar p = {paired['mcnemar_exact_p']:.4f}"
    )
    lines.extend(["", "## Acceptance gates", ""])
    for name, passed in payload["gates"]["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(["", "## Failures", ""])
    if not payload["failures"]["diagnostic"]:
        lines.append("None.")
    else:
        lines.append("| Scenario | Classification | Tools | Steps |")
        lines.append("| --- | --- | ---: | ---: |")
        for failure in payload["failures"]["diagnostic"]:
            lines.append(
                f"| `{failure['scenario_id']}` | {failure['classification']} | "
                f"{failure['detail']['tool_units']} | "
                f"{failure['detail']['env_steps']} |"
            )
    lines.extend(["", "## Per-map", ""])
    lines.append("| Map | Diagnostic | Baseline | Tools |")
    lines.append("| --- | ---: | ---: | ---: |")
    for map_id, record in payload["per_map"].items():
        diag = record["diagnostic_belief"]
        base = record["no_adaptation"]
        lines.append(
            f"| `{map_id}` | {diag['successes']}/{diag['episodes']} | "
            f"{base['successes']}/{base['episodes']} | {diag['tool_units_total']} |"
        )
    lines.extend(["", "## Sign affordance", ""])
    audit = payload["sign_affordance"]
    lines.append(
        f"Maps with side 0 reachable without crossing a marker or the closed "
        f"door: {audit['maps_with_reachable_base_diagnostic_face']}/"
        f"{len(audit['maps'])}; maps with any informative face: "
        f"{audit['maps_with_reachable_informative_face']}/{len(audit['maps'])}; "
        f"maps without a signpost: {audit['maps_without_signpost']}."
    )
    lines.extend(["", "## Map independence", ""])
    identity = payload["map_identity"]
    lines.append(
        f"Requested {identity['maps_requested']} maps; "
        f"unique public geometries: {identity['unique_public_maps']}. "
        "Duplicate groups: "
        + (", ".join("=".join(group) for group in identity["duplicate_groups"]) or "none")
        + "."
    )
    lines.append(
        "Duplicate groups are reported explicitly; scenario ids remain unique."
    )
    lines.extend(["", "## Zero-tool hazard/sign", ""])
    zero = payload["zero_tool_hazard_or_sign"]
    lines.append(
        f"{zero['count']} identification episodes used zero tools; "
        f"{zero['successes']} of them succeeded accidentally: "
        + (", ".join(row["scenario_id"] for row in zero["accidental_successes"]) or "none")
        + "."
    )
    lines.extend(["", "## Eligible-map sensitivity", ""])
    eligibility = payload["eligibility"]
    lines.append(
        f"Excluded maps without a reachable side-0 diagnostic face: "
        f"{eligibility['unreadable_maps'] or 'none'}; "
        f"{eligibility['excluded_episode_rows']} method rows excluded."
    )
    lines.append("| Rule | Diagnostic | Baseline |")
    lines.append("| --- | ---: | ---: |")
    eligible_by_rule = eligibility["by_initial_rule"]
    for rule, metrics in eligible_by_rule.get(payload["diagnostic"], {}).items():
        base = eligible_by_rule.get(payload["baseline"], {}).get(rule, {})
        lines.append(
            f"| `{rule}` | {metrics['successes']}/{metrics['episodes']} | "
            f"{base.get('successes', 0)}/{base.get('episodes', 0)} |"
        )
    lines.append("")
    for name, passed in eligibility["gates"]["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}` (eligible maps)")
    lines.extend(["", "## Failure taxonomy", ""])
    for name, count in payload["failures"]["classification_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/grid_phase0/maps10.jsonl")
    parser.add_argument("--out-dir", default="results/grid_phase0/maps10_analysis")
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--maps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--diagnostic", default="diagnostic_belief")
    parser.add_argument("--baseline", default="no_adaptation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_grid_config(_ROOT / args.config)
    scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_grid_scenarios(args.maps, args.seeds, config)
    }
    input_path = _ROOT / args.input
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in rows:
        scenario = scenarios[row["scenario_id"]]
        row["_initial_rule"] = scenario.initial_rule
        row["_scenario_type"] = scenario.scenario_type
    methods = sorted({row["method"] for row in rows})
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    diag_rows = by_method[args.diagnostic]
    base_rows = by_method[args.baseline]
    diag = {row["scenario_id"]: row for row in diag_rows}
    base = {row["scenario_id"]: row for row in base_rows}
    aggregate = {
        "by_method": {m: _metrics(sub) for m, sub in sorted(by_method.items())},
        "by_type": {
            m: _group_metrics(sub, "_scenario_type")
            for m, sub in sorted(by_method.items())
        },
        "by_initial_rule": {
            m: _group_metrics(
                [row for row in sub if row["_scenario_type"] == "identification"],
                "_initial_rule",
            )
            for m, sub in sorted(by_method.items())
        },
    }
    per_map: dict[str, Any] = {}
    map_ids = sorted(
        {row["scenario_id"].split("-")[0] for row in rows},
        key=lambda value: int(value.removeprefix("m")),
    )
    for map_id in map_ids:
        per_map[map_id] = {}
        for method in methods:
            per_map[map_id][method] = _metrics(
                [row for row in by_method[method] if row["scenario_id"].startswith(f"{map_id}-")]
            )
    paired_overall = _paired(diag, base)
    paired = {
        "overall": paired_overall,
        "by_type": _paired_group(diag, base, scenarios, "scenario_type"),
        "by_initial_rule": _paired_group(diag, base, scenarios, "initial_rule"),
    }
    failures = _failure_records(diag_rows, scenarios, config)
    sign_affordance = _sign_audit(config, args.maps, args.seeds)
    gates = _gates(diag_rows, paired_overall, failures)
    map_identity = _map_identity_audit(config, args.maps)
    zero_tool_rows = [
        row
        for row in diag_rows
        if row["_scenario_type"] == "identification"
        and row["_initial_rule"] in HS_RULES
        and int(row["tool_units"]) == 0
    ]
    accidental_successes = [
        {
            "scenario_id": row["scenario_id"],
            "initial_rule": row["_initial_rule"],
            "env_steps": int(row["env_steps"]),
            "final_rule_estimate": row["final_rule_estimate"],
        }
        for row in zero_tool_rows
        if row["success"]
    ]
    eligibility = _eligibility(
        diag_rows, base_rows, sign_affordance, scenarios, methods
    )
    payload = {
        "input": args.input,
        "config": args.config,
        "maps": args.maps,
        "seeds": args.seeds,
        "diagnostic": args.diagnostic,
        "baseline": args.baseline,
        "scenario_count": len(scenarios),
        "row_count": len(rows),
        "aggregate": aggregate,
        "per_map": per_map,
        "paired": paired,
        "failures": {
            "diagnostic": failures,
            "classification_counts": {
                classification: sum(
                    1 for failure in failures if failure["classification"] == classification
                )
                for classification in sorted({f["classification"] for f in failures})
            },
        },
        "gates": gates,
        "sign_affordance": sign_affordance,
        "map_identity": map_identity,
        "zero_tool_hazard_or_sign": {
            "count": len(zero_tool_rows),
            "successes": len(accidental_successes),
            "rows": [row["scenario_id"] for row in zero_tool_rows],
            "accidental_successes": accidental_successes,
        },
        "eligibility": eligibility,
    }
    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "aggregate.json", payload)
    _write_json(out_dir / "per_map.json", per_map)
    _write_json(out_dir / "paired.json", paired)
    _write_json(out_dir / "failures.json", payload["failures"])
    _write_json(out_dir / "sign_affordance.json", sign_affordance)
    _write_json(out_dir / "map_identity.json", map_identity)
    _write_json(out_dir / "eligibility.json", eligibility)
    _write_json(
        out_dir / "zero_tool_hazard_or_sign.json",
        payload["zero_tool_hazard_or_sign"],
    )
    (out_dir / "summary.md").write_text(
        _summary_markdown(payload), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(out_dir), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
