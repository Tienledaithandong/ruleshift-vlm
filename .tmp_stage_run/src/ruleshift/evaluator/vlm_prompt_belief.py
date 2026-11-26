"""Matched 2x2 prompt-anchoring and belief-conditioning diagnostic design.

This module is CPU-only.  It reuses the frozen 12-context development library
and the original direct-action prompt bodies, changing only the output
instruction and/or whether the explicit public posterior is present.  It does
not load a model and does not read model outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ruleshift.agents.vlm_backend import ModelRequest
from ruleshift.evaluator import vlm_smoke
from ruleshift.evaluator.vlm_decision import (
    ACTIONS,
    DecisionContext,
    DecisionLibrary,
    DecisionPrediction,
    ReferenceLabel,
    baseline_predictions,
    build_development_library,
    score_decisions,
)

EXPERIMENT_ID = "vlm-prompt-belief-diagnostic-v1"
SMOKE_VERSION = "vlm-prompt-belief-diagnostic-v1"
PROMPT_VERSION = "vlm-prompt-belief-v1"
PARSER_VERSION = "vlm-smoke-direct-parser-v1"
SCORER_VERSION = "vlm-prompt-belief-scorer-v1"
REFERENCE_VERSION = "vlm-decision-reference-v1"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
MAX_NEW_TOKENS = 32
DECODING_SEED = 20261126
REQUEST_CAP = 48
ARMS = ("B1", "B2", "B3", "B4")
BELIEFS = ("prior_evidence", "explicit_posterior")
OUTPUT_FORMATS = ("anchored", "neutral")
ARM_ORDER = ("B1", "B2", "B3", "B4")

ARM_SPECS: dict[str, dict[str, str]] = {
    "B1": {"belief": "prior_evidence", "output_format": "anchored"},
    "B2": {"belief": "prior_evidence", "output_format": "neutral"},
    "B3": {"belief": "explicit_posterior", "output_format": "anchored"},
    "B4": {"belief": "explicit_posterior", "output_format": "neutral"},
}

ANCHORED_FOOTER = (
    "Return ONLY one compact JSON object with exactly this key: "
    '{"selected_action":"COMMIT"} (replace COMMIT with one feasible action). '
    "Do not include reasoning, markdown, code fences, or extra keys."
)
NEUTRAL_FOOTER = (
    'Return ONLY one compact JSON object with exactly one key, "selected_action". '
    "Its value MUST be one of the feasible actions listed above. "
    "Do not include reasoning, markdown, code fences, or any other key."
)

PAIR_COMPARISONS = (
    ("B2_minus_B1", "B2", "B1", "prompt format without explicit posterior"),
    ("B4_minus_B3", "B4", "B3", "prompt format with explicit posterior"),
    ("B3_minus_B1", "B3", "B1", "explicit posterior under anchored format"),
    ("B4_minus_B2", "B4", "B2", "explicit posterior under neutral format"),
)


@dataclass(frozen=True)
class DiagnosticRequest:
    request_index: int
    request_id: str
    arm: str
    belief: str
    output_format: str
    context_id: str
    context: DecisionContext
    label: ReferenceLabel
    prompt: str
    public_payload: Mapping[str, Any]

    @property
    def input_mode(self) -> str:
        return "text"


@dataclass(frozen=True)
class DiagnosticPlan:
    contexts: tuple[DecisionContext, ...]
    labels: tuple[ReferenceLabel, ...]
    requests: tuple[DiagnosticRequest, ...]

    @property
    def base_requests(self) -> tuple[DiagnosticRequest, ...]:
        return self.requests

    def arm_requests(self, arm: str) -> tuple[DiagnosticRequest, ...]:
        return tuple(item for item in self.requests if item.arm == arm)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_paths(protocol: Path) -> list[Path]:
    root = repo_root()
    return [
        root / "src" / "ruleshift" / "evaluator" / "vlm_decision.py",
        root / "src" / "ruleshift" / "evaluator" / "vlm_smoke.py",
        root / "src" / "ruleshift" / "evaluator" / "vlm_prompt_belief.py",
        root / "src" / "ruleshift" / "evaluator" / "vlm_run_audit.py",
        root / "src" / "ruleshift" / "evaluator" / "vlm_smoke_parser.py",
        root / "src" / "ruleshift" / "evaluator" / "vlm_smoke_checkpoint.py",
        root / "src" / "ruleshift" / "agents" / "vlm_backend.py",
        root / "src" / "ruleshift" / "agents" / "vlm_smoke_backend.py",
        root / "scripts" / "prepare_vlm_prompt_belief.py",
        root / "scripts" / "preflight_vlm_prompt_belief.py",
        root / "scripts" / "run_vlm_prompt_belief.py",
        Path(protocol),
    ]


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(vlm_smoke.canonical_json(value))


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(vlm_smoke.canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(data)


def neutral_prompt(anchored_prompt: str) -> str:
    """Replace only the anchored output instruction with the neutral schema."""

    if not anchored_prompt.endswith(ANCHORED_FOOTER):
        raise ValueError("prompt does not end with the frozen anchored instruction")
    base = anchored_prompt[: -len(ANCHORED_FOOTER)]
    if ANCHORED_FOOTER in base:
        raise ValueError("anchored instruction appears more than once")
    return base + NEUTRAL_FOOTER


def make_model_request(item: DiagnosticRequest) -> ModelRequest:
    return ModelRequest(
        model_id=MODEL_ID,
        prompt=item.prompt,
        image_hash="",
        history_hash="",
        available_actions=item.context.feasible_actions(),
        candidate_hash=vlm_smoke.candidate_hash(item.context),
        decoding_seed=DECODING_SEED,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_version=PROMPT_VERSION,
        candidate_rules=item.context.candidate_rules,
        revision=REVISION,
        dtype="bfloat16",
        backend_id="",
        do_sample=False,
        device="cuda",
        input_mode="text",
        image=None,
    )


def manifest_record(item: DiagnosticRequest) -> dict[str, Any]:
    return {
        "request_index": item.request_index,
        "request_id": item.request_id,
        "arm": item.arm,
        "belief": item.belief,
        "output_format": item.output_format,
        "context_id": item.context_id,
        "input_mode": "text",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(item.prompt),
        "public_payload_sha256": sha256_json(dict(item.public_payload)),
        "feasible_actions": list(item.context.feasible_actions()),
        "candidate_rules": list(item.context.candidate_rules),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding_seed": DECODING_SEED,
        "image_pixel_digest": None,
        "image_png_sha256": None,
        "image_size": None,
    }


def build_plan(library: DecisionLibrary | None = None) -> DiagnosticPlan:
    """Build the deterministic balanced 4-arm x 12-context design."""

    library = library or build_development_library()
    smoke_plan = vlm_smoke.build_plan(library)
    a1 = {item.context_id: item for item in smoke_plan.base_requests if item.arm == "A1"}
    a2 = {item.context_id: item for item in smoke_plan.base_requests if item.arm == "A2"}
    by_id = {context.context_id: (context, label) for context, label in zip(library.contexts, library.labels)}

    requests: list[DiagnosticRequest] = []
    index = 0
    for block, context_id in enumerate(vlm_smoke.BASE_IDS):
        if context_id not in a1 or context_id not in a2 or context_id not in by_id:
            raise ValueError(f"missing frozen context: {context_id}")
        context, label = by_id[context_id]
        source_by_arm = {
            "B1": a1[context_id],
            "B2": a1[context_id],
            "B3": a2[context_id],
            "B4": a2[context_id],
        }
        rotation = block % len(ARM_ORDER)
        ordered_arms = ARM_ORDER[rotation:] + ARM_ORDER[:rotation]
        for arm in ordered_arms:
            source = source_by_arm[arm]
            spec = ARM_SPECS[arm]
            prompt = source.prompt if spec["output_format"] == "anchored" else neutral_prompt(source.prompt)
            index += 1
            request = DiagnosticRequest(
                request_index=index,
                request_id=f"{arm}-{block + 1:02d}",
                arm=arm,
                belief=spec["belief"],
                output_format=spec["output_format"],
                context_id=context_id,
                context=context,
                label=label,
                prompt=prompt,
                public_payload=dict(source.public_payload),
            )
            vlm_smoke.assert_prompt_public(prompt)
            requests.append(request)

    plan = DiagnosticPlan(
        contexts=tuple(by_id[context_id][0] for context_id in vlm_smoke.BASE_IDS),
        labels=tuple(by_id[context_id][1] for context_id in vlm_smoke.BASE_IDS),
        requests=tuple(requests),
    )
    validate_design(plan, smoke_plan)
    return plan


def _action_class_rows(details: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for action in ACTIONS:
        selected = [row for row in details if row.get("best_action") == action]
        correct = sum(bool(row.get("optimal")) for row in selected)
        rows[action] = {
            "n": len(selected),
            "correct": correct,
            "accuracy": (correct / len(selected)) if selected else None,
        }
    return rows


def validate_design(plan: DiagnosticPlan, smoke_plan: Any | None = None) -> None:
    if len(plan.contexts) != 12 or len(set(item.context_id for item in plan.contexts)) != 12:
        raise ValueError("design requires 12 unique contexts")
    if len(plan.requests) != REQUEST_CAP:
        raise ValueError(f"design requires {REQUEST_CAP} requests, got {len(plan.requests)}")
    if len(set(item.request_id for item in plan.requests)) != len(plan.requests):
        raise ValueError("request ids are not unique")
    expected_contexts = set(vlm_smoke.BASE_IDS)
    if {item.context_id for item in plan.requests} != expected_contexts:
        raise ValueError("request contexts differ from the frozen 12-context subset")
    for arm in ARMS:
        selected = plan.arm_requests(arm)
        if len(selected) != 12 or {item.context_id for item in selected} != expected_contexts:
            raise ValueError(f"{arm}: expected each of the 12 contexts once")
    for block, context_id in enumerate(vlm_smoke.BASE_IDS):
        block_rows = plan.requests[block * 4 : block * 4 + 4]
        if len(block_rows) != 4 or {item.context_id for item in block_rows} != {context_id}:
            raise ValueError(f"block {block}: requests are not one context across four arms")
        if len({item.arm for item in block_rows}) != 4:
            raise ValueError(f"block {block}: arms are not unique")
        rotation = block % 4
        expected_order = ARM_ORDER[rotation:] + ARM_ORDER[:rotation]
        if tuple(item.arm for item in block_rows) != expected_order:
            raise ValueError(f"block {block}: arm order is not the frozen rotation")
    for arm in ARMS:
        if sum(plan.requests[i].arm == arm for i in range(0, REQUEST_CAP, 4)) != 3:
            raise ValueError(f"{arm}: first-position balance is not 3/12")
    if smoke_plan is not None:
        a1 = {item.context_id: item for item in smoke_plan.base_requests if item.arm == "A1"}
        a2 = {item.context_id: item for item in smoke_plan.base_requests if item.arm == "A2"}
        for item in plan.requests:
            if item.arm == "B1" and item.prompt != a1[item.context_id].prompt:
                raise ValueError(f"{item.request_id}: B1 does not exactly replicate A1")
            if item.arm == "B3" and item.prompt != a2[item.context_id].prompt:
                raise ValueError(f"{item.request_id}: B3 does not exactly replicate A2")
            if item.arm in ("B2", "B4"):
                source = a1[item.context_id] if item.arm == "B2" else a2[item.context_id]
                if item.prompt != neutral_prompt(source.prompt):
                    raise ValueError(f"{item.request_id}: neutral prompt changed more than the footer")
            if item.belief == "prior_evidence" and "posterior" in item.public_payload:
                raise ValueError(f"{item.request_id}: prior arm contains a posterior payload")
            if item.belief == "explicit_posterior" and "posterior" not in item.public_payload:
                raise ValueError(f"{item.request_id}: posterior arm lacks a posterior payload")
            if item.output_format == "neutral" and '{"selected_action":"COMMIT"}' in item.prompt:
                raise ValueError(f"{item.request_id}: neutral prompt contains an action example")
    if {item.label.best_action for item in plan.requests} != set(ACTIONS):
        raise ValueError("reference labels do not cover all three optimal-action classes")
    for action in ACTIONS:
        count = sum(label.best_action == action for _context, label in zip(plan.contexts, plan.labels))
        if count != 4:
            raise ValueError(f"{action}: expected four reference contexts")


def prompt_manifest(plan: DiagnosticPlan) -> dict[str, Any]:
    return {
        "version": SMOKE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "anchored_footer": ANCHORED_FOOTER,
        "neutral_footer": NEUTRAL_FOOTER,
        "requests": [
            {
                "request_index": item.request_index,
                "request_id": item.request_id,
                "arm": item.arm,
                "belief": item.belief,
                "output_format": item.output_format,
                "context_id": item.context_id,
                "prompt_sha256": sha256_text(item.prompt),
                "prompt": item.prompt,
            }
            for item in plan.requests
        ],
    }


def reference_payload(plan: DiagnosticPlan) -> dict[str, Any]:
    records = []
    for context, label in zip(plan.contexts, plan.labels):
        records.append({"context_id": context.context_id, "reference": asdict(label)})
    return {"version": REFERENCE_VERSION, "records": records}


def analysis_plan_payload() -> dict[str, Any]:
    return {
        "version": SCORER_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "design": {"arms": list(ARMS), "contexts": 12, "request_cap": REQUEST_CAP},
        "eligibility": {
            "valid_min": 11,
            "optimal_min": 8,
            "correct_each_optimal_class": True,
            "non_constant": True,
            "mean_regret_strictly_below_always_commit": True,
        },
        "primary_pairs": [row[0] for row in PAIR_COMPARISONS],
        "paired_unit": "context_id",
        "descriptive_interaction": "(B4 - B3) - (B2 - B1) for accuracy and mean regret",
        "baselines": ["always_commit", "always_probe", "random_valid", "information_gain", "exact"],
        "confirmatory": False,
        "development_subset_previously_exposed": True,
    }


def runtime_contract() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "device": "cuda",
        "local_files_only": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding_seed": DECODING_SEED,
        "training": False,
        "fine_tuning": False,
        "quantization": None,
        "offload": None,
        "model_substitution": False,
    }


def score_arm(records: Sequence[Mapping[str, Any]], plan: DiagnosticPlan, arm: str, *, baselines: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    summary = vlm_smoke.score_records(records, plan, kind="base", arm=arm)
    summary["arm"] = arm
    summary["belief"] = ARM_SPECS[arm]["belief"]
    summary["output_format"] = ARM_SPECS[arm]["output_format"]
    summary["optimal_class_accuracy"] = _action_class_rows(summary["details"])
    if baselines is not None:
        always = baselines["always_commit"]
        criteria = {
            "valid_at_least_11": int(summary["valid_count"]) >= 11,
            "optimal_at_least_8": sum(bool(row["optimal"]) for row in summary["details"]) >= 8,
            "correct_each_optimal_class": all(
                int(row["correct"]) >= 1 for row in summary["optimal_class_accuracy"].values()
            ),
            "non_constant": len(summary["distinct_valid_actions"]) >= 2,
            "mean_regret_below_always_commit": float(summary["mean_regret_all"]) < float(always["mean_regret"]),
        }
        summary["eligibility"] = criteria
        summary["eligible_for_visual_smoke_consideration"] = all(criteria.values())
        summary["comparison_to_always_commit"] = {
            "mean_regret_delta": float(summary["mean_regret_all"]) - float(always["mean_regret"]),
            "accuracy_delta": float(summary["optimal_action_accuracy_all"]) - float(always["optimal_action_accuracy"]),
        }
        if "information_gain" in baselines:
            info = baselines["information_gain"]
            summary["comparison_to_information_gain"] = {
                "mean_regret_delta": float(summary["mean_regret_all"]) - float(info["mean_regret"]),
                "accuracy_delta": float(summary["optimal_action_accuracy_all"]) - float(info["optimal_action_accuracy"]),
            }
    return summary


def baseline_diagnostics(plan: DiagnosticPlan) -> dict[str, Any]:
    library = vlm_smoke.subset_library(plan.contexts, plan.labels)
    result: dict[str, Any] = {}
    for name in ("always_commit", "always_probe", "random_valid", "information_gain", "exact"):
        predictions = baseline_predictions(library, name, seed=DECODING_SEED)
        scores = score_decisions(library, predictions)
        n = len(scores)
        details = [
            {"context_id": score.context_id, "predicted_action": score.predicted_action, "optimal": bool(score.optimal), "regret": float(score.regret), "over_probe": bool(score.over_probe), "under_probe": bool(score.under_probe), "probe_target_correct": score.probe_target_correct}
            for score in scores
        ]
        for row, (context, label) in zip(details, zip(plan.contexts, plan.labels)):
            row["best_action"] = label.best_action
        result[name] = {
            "n": n,
            "optimal_action_accuracy": sum(score.optimal for score in scores) / n,
            "mean_regret": sum(score.regret for score in scores) / n,
            "max_regret": max(score.regret for score in scores),
            "over_probe_rate": sum(score.over_probe for score in scores) / n,
            "under_probe_rate": sum(score.under_probe for score in scores) / n,
            "tool_selection_accuracy": (
                sum(bool(score.probe_target_correct) for score in scores if score.probe_target_correct is not None)
                / sum(1 for score in scores if score.probe_target_correct is not None)
            ) if any(score.probe_target_correct is not None for score in scores) else None,
            "optimal_class_accuracy": _action_class_rows(details),
        }
    return result


def paired_comparisons(plan: DiagnosticPlan, summaries: Mapping[str, Any]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        if arm not in summaries:
            raise ValueError(f"missing summary for {arm}")
        by_arm[arm] = {row["context_id"]: row for row in summaries[arm]["details"]}
    result: dict[str, Any] = {"pairs": {}, "interaction": {}}
    for name, left, right, description in PAIR_COMPARISONS:
        rows = []
        for context_id in vlm_smoke.BASE_IDS:
            a = by_arm[right][context_id]
            b = by_arm[left][context_id]
            rows.append({
                "context_id": context_id,
                "best_action": a["best_action"],
                "right_action": a["selected_action"],
                "left_action": b["selected_action"],
                "right_optimal": bool(a["optimal"]),
                "left_optimal": bool(b["optimal"]),
                "right_regret": float(a["regret"]),
                "left_regret": float(b["regret"]),
                "action_changed": a["selected_action"] != b["selected_action"],
                "accuracy_delta": int(bool(b["optimal"])) - int(bool(a["optimal"])),
                "regret_delta": float(b["regret"]) - float(a["regret"]),
            })
        n = len(rows)
        result["pairs"][name] = {
            "comparison": description,
            "left": left,
            "right": right,
            "n": n,
            "action_change_count": sum(row["action_changed"] for row in rows),
            "right_accuracy": sum(row["right_optimal"] for row in rows) / n,
            "left_accuracy": sum(row["left_optimal"] for row in rows) / n,
            "accuracy_delta_left_minus_right": sum(row["accuracy_delta"] for row in rows) / n,
            "right_mean_regret": sum(row["right_regret"] for row in rows) / n,
            "left_mean_regret": sum(row["left_regret"] for row in rows) / n,
            "mean_regret_delta_left_minus_right": sum(row["regret_delta"] for row in rows) / n,
            "rows": rows,
        }
    for metric in ("accuracy", "mean_regret"):
        if metric == "accuracy":
            values = {arm: float(summaries[arm]["optimal_action_accuracy_all"]) for arm in ARMS}
        else:
            values = {arm: float(summaries[arm]["mean_regret_all"]) for arm in ARMS}
        prompt_without = values["B2"] - values["B1"]
        prompt_with = values["B4"] - values["B3"]
        posterior_anchored = values["B3"] - values["B1"]
        posterior_neutral = values["B4"] - values["B2"]
        result["interaction"][metric] = {
            "values": values,
            "prompt_format_effect_without_posterior": prompt_without,
            "prompt_format_effect_with_posterior": prompt_with,
            "posterior_effect_anchored": posterior_anchored,
            "posterior_effect_neutral": posterior_neutral,
            "interaction_difference": prompt_with - prompt_without,
        }
    return result


def freeze_artifacts(plan: DiagnosticPlan, out_dir: Path, *, checkpoint_manifest: Mapping[str, Any], checkpoint_manifest_path: Path, protocol_path: Path, source_paths: Sequence[Path]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = prompt_manifest(plan)
    manifest = [manifest_record(item) for item in plan.requests]
    write_json(out_dir / "prompt_manifest.json", prompts)
    write_jsonl(out_dir / "prompts.jsonl", [{"request_id": item.request_id, "arm": item.arm, "context_id": item.context_id, "prompt_sha256": sha256_text(item.prompt), "prompt": item.prompt} for item in plan.requests])
    write_jsonl(out_dir / "public_payloads.jsonl", [{"request_id": item.request_id, "public_payload_sha256": sha256_json(dict(item.public_payload)), "public_payload": dict(item.public_payload)} for item in plan.requests])
    write_json(out_dir / "request_manifest.json", manifest)
    write_json(out_dir / "reference_labels.json", reference_payload(plan))
    write_json(out_dir / "analysis_plan.json", analysis_plan_payload())
    write_json(out_dir / "model_runtime.json", runtime_contract())
    write_json(out_dir / "checkpoint_manifest.json", dict(checkpoint_manifest))
    source_hashes = {}
    for path in source_paths:
        path = Path(path).resolve()
        key = path.relative_to(Path.cwd().resolve()).as_posix() if path.is_relative_to(Path.cwd().resolve()) else path.as_posix()
        source_hashes[key] = file_sha256(path)
    freeze = {
        "version": SMOKE_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_hashes": source_hashes,
        "request_manifest_sha256": file_sha256(out_dir / "request_manifest.json"),
        "prompt_manifest_sha256": file_sha256(out_dir / "prompt_manifest.json"),
        "prompts_sha256": file_sha256(out_dir / "prompts.jsonl"),
        "public_payloads_sha256": file_sha256(out_dir / "public_payloads.jsonl"),
        "reference_labels_sha256": file_sha256(out_dir / "reference_labels.json"),
        "analysis_plan_sha256": file_sha256(out_dir / "analysis_plan.json"),
        "model_runtime_sha256": file_sha256(out_dir / "model_runtime.json"),
        "checkpoint_manifest_sha256": file_sha256(out_dir / "checkpoint_manifest.json"),
        "protocol_sha256": file_sha256(protocol_path),
        "context_ids": list(vlm_smoke.BASE_IDS),
        "arms": list(ARMS),
        "request_count": len(plan.requests),
        "request_cap": REQUEST_CAP,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "scorer_version": SCORER_VERSION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding_seed": DECODING_SEED,
        "do_sample": False,
    }
    write_json(out_dir / "freeze.json", freeze)
    (out_dir / "freeze.sha256").write_bytes((file_sha256(out_dir / "freeze.json") + "\n").encode("utf-8"))
    return freeze


def verify_freeze(out_dir: Path, *, source_paths: Sequence[Path] | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    freeze = json.loads((out_dir / "freeze.json").read_text(encoding="utf-8"))
    expected = {
        "request_manifest_sha256": out_dir / "request_manifest.json",
        "prompt_manifest_sha256": out_dir / "prompt_manifest.json",
        "prompts_sha256": out_dir / "prompts.jsonl",
        "public_payloads_sha256": out_dir / "public_payloads.jsonl",
        "reference_labels_sha256": out_dir / "reference_labels.json",
        "analysis_plan_sha256": out_dir / "analysis_plan.json",
        "model_runtime_sha256": out_dir / "model_runtime.json",
        "checkpoint_manifest_sha256": out_dir / "checkpoint_manifest.json",
    }
    for key, path in expected.items():
        actual = file_sha256(path)
        if actual != freeze[key]:
            raise ValueError(f"frozen hash mismatch for {key}: {actual} != {freeze[key]}")
    if source_paths is not None:
        normalized = {str(key).replace(chr(92), "/"): value for key, value in freeze["source_hashes"].items()}
        for path in source_paths:
            path = Path(path).resolve()
            key = path.relative_to(Path.cwd().resolve()).as_posix() if path.is_relative_to(Path.cwd().resolve()) else path.as_posix()
            actual = file_sha256(path)
            if normalized.get(key) != actual:
                raise ValueError(f"source changed after freeze: {key}")
    if (out_dir / "freeze.sha256").read_text(encoding="utf-8").strip() != file_sha256(out_dir / "freeze.json"):
        raise ValueError("freeze.sha256 mismatch")
    return freeze
