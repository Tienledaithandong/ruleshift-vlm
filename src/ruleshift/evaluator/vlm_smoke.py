"""Frozen CPU-side plan for the bounded probe-versus-commit VLM smoke.

This module contains no model loading.  It selects the 12 development contexts
before any model call, renders the three direct prompts, builds the image
controls, and freezes the request/reference/analysis artifacts.  The model
runner and parser live in separate modules so this file can be CPU-tested.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ruleshift.agents.vlm_backend import ModelRequest
from ruleshift.evaluator.vlm_decision import (
    ACTIONS,
    DecisionContext,
    DecisionLibrary,
    DecisionPrediction,
    ReferenceLabel,
    baseline_predictions,
    build_development_library,
    reference_for,
    score_decisions,
)
from ruleshift.evaluator.vlm_decision_visual import (
    blue_role_perturbation,
    render_decision_card,
    visual_digest,
)
from ruleshift.permissions import find_forbidden_fields

SMOKE_VERSION = "vlm-decision-smoke-v1"
PROMPT_VERSION = "vlm-smoke-direct-v1"
PARSER_VERSION = "vlm-smoke-direct-parser-v1"
SCORER_VERSION = "vlm-smoke-scorer-v1"
SELECTION_SALT = "vlm-smoke-selection-2026-11-26"
MAX_NEW_TOKENS = 32
DECODING_SEED = 20261126
MIN_BASE_MARGIN = 0.5
BASE_ACTION_ORDER = ("COMMIT", "PROBE_SIGN", "PROBE_BLUE")

# The four contexts per exact best action were chosen deterministically from the
# 118-context development library before any model output existed.  The first
# three BLUE contexts have large margins; the fourth is the only BLUE context
# with a larger, semantically different blue-role diagnostic.  Its base margin
# is 0.57775, still above the frozen 0.5 floor.
COMMIT_IDS = (
    "vlm-dev-layout-a-base-rotate-50-expensive-0.00-",
    "vlm-dev-layout-b-base-hazard-50-cheap-0.05-sign:The red tile is not the goal.",
    "vlm-dev-layout-c-base-rotate-50-default-0.30-blue:hazard",
    "vlm-dev-layout-a-hazard-confident-cheap-0.30-sign:The red tile is not the goal.",
)
SIGN_IDS = (
    "vlm-dev-layout-a-base-hazard-50-cheap-0.15-blue:safe",
    "vlm-dev-layout-b-all-uniform-cheap-0.30-blue:goal",
    "vlm-dev-layout-b-sign-confident-cheap-0.15-blue:hazard",
    "vlm-dev-layout-c-base-hazard-90-default-0.05-blue:goal",
)
BLUE_IDS = (
    "vlm-dev-layout-c-hazard-sign-50-cheap-0.05-sign:The red tile is not the goal.",
    "vlm-dev-layout-a-base-hazard-sign-uniform-cheap-0.05-sign:The red tile is not the goal.",
    "vlm-dev-layout-c-hazard-sign-50-default-0.00-sign:The red tile is not the goal.",
    "vlm-dev-layout-b-hazard-confident-cheap-0.05-",
)
BASE_IDS = COMMIT_IDS + SIGN_IDS + BLUE_IDS

# Six paired nuisance cards: two per exact best action.  The image changes only
# palette/border; the card is rendered from the same context and label.
NUISANCE_IDS = (
    COMMIT_IDS[0],
    COMMIT_IDS[1],
    SIGN_IDS[0],
    SIGN_IDS[1],
    BLUE_IDS[0],
    BLUE_IDS[3],
)

# Six paired semantic controls.  Each perturbation replaces the observed blue
# role, recomputes the exact posterior, and must change the exact optimal set.
DIAGNOSTIC_ROLES = {
    SIGN_IDS[0]: "goal",
    SIGN_IDS[1]: "hazard",
    SIGN_IDS[2]: "safe",
    SIGN_IDS[3]: "hazard",
    BLUE_IDS[0]: "goal",
    BLUE_IDS[3]: "hazard",
}

# These are reference/evaluator keys, not public decision facts.  They must not
# appear as JSON keys or bare terms in a model prompt.
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "latent_rule",
        "reference",
        "reference_label",
        "q_commit",
        "q_sign",
        "q_blue",
        "best_value",
        "optimal_actions",
        "best_action",
        "decision_class",
        "decision_margin",
        "commit_rule",
        "commit_target",
        "latent_declaration",
        "declaration",
        "declaration_rule",
        "declaration_confidence",
        "declaration_margin",
        "declaration_correct",
        "posterior_entropy",
        "true_rule",
        "q_star",
        "exact_q_values",
    }
)
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(item) for item in sorted(FORBIDDEN_LABEL_KEYS)) + r")\b",
    re.IGNORECASE,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def image_png_bytes(image: np.ndarray) -> bytes:
    from io import BytesIO
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG", optimize=False
    )
    return buffer.getvalue()


def _public_facts(context: DecisionContext, *, include_posterior: bool) -> dict[str, Any]:
    payload = context.to_public_payload(include_posterior=include_posterior)
    payload.pop("context_id", None)
    payload.pop("layout_id", None)
    return payload


def _action_text(context: DecisionContext) -> str:
    actions = ", ".join(context.feasible_actions())
    return (
        "Action semantics (public):\n"
        "- COMMIT: choose the candidate rule with the largest public belief and "
        "move toward that rule's goal color. Ties break in favor of COMMIT.\n"
        "- PROBE_SIGN: inspect the public signpost clause. The observation is "
        "noise-free under the public model.\n"
        "- PROBE_BLUE: inspect the public blue cell role. The observation is "
        "conditioned on the displayed inspection error.\n"
        f"Feasible actions for this decision: {actions}.\n"
    )


_COMMON_HEADER = (
    "You are an agent choosing exactly one action for a partially observed grid task.\n"
    "Choose the action with the highest exact expected control value over the "
    "remaining horizon, using only the public facts supplied here. The objective "
    "accounts for action cost, remaining tool budget, commit step cost, success "
    "reward, hazard penalty, and inspection noise. Ties break in favor of COMMIT.\n"
)
_TEXT_HEADER = _COMMON_HEADER + (
    "The public facts below are complete for this decision. Do not assume hidden "
    "rules, labels, or future observations.\n"
)


def render_prompt(context: DecisionContext, arm: str) -> str:
    """Render one frozen direct-action prompt; no evaluator label is read."""

    if arm not in ("A1", "A2", "A3"):
        raise ValueError(f"unknown arm: {arm}")
    if arm in ("A1", "A2"):
        payload = _public_facts(context, include_posterior=(arm == "A2"))
        body = (
            f"{_TEXT_HEADER}\nPUBLIC FACTS (JSON):\n{canonical_json(payload)}\n\n"
            f"{_action_text(context)}\n"
        )
    else:
        constants = {
            "commit_steps_by_rule": dict(context.commit_steps_by_rule),
            "step_cost": context.step_cost,
            "success_reward": context.success_reward,
            "hazard_penalty": context.hazard_penalty,
            "feasible_actions": list(context.feasible_actions()),
        }
        body = (
            f"{_COMMON_HEADER}"
            "The attached 512x320 decision card is the sole source of the public "
            "map, candidate-rule prior and updated-belief bars, observed evidence, "
            "tool costs, remaining budget, and inspection error. This text supplies only "
            "the payoff constants and the feasible-action menu; it does not repeat "
            "the visual facts.\n\n"
            f"PUBLIC PAYOFF CONSTANTS (JSON):\n{canonical_json(constants)}\n\n"
            f"{_action_text(context)}\n"
        )
    footer = (
        "Return ONLY one compact JSON object with exactly this key: "
        '{"selected_action":"COMMIT"} (replace COMMIT with one feasible action). '
        "Do not include reasoning, markdown, code fences, or extra keys."
    )
    return body + footer


def assert_prompt_public(prompt: str) -> None:
    """Reject a prompt containing evaluator-label names or forbidden fields."""

    hits = find_forbidden_fields(prompt)
    if hits:
        raise ValueError(f"prompt contains privileged fields: {hits}")
    match = _FORBIDDEN_RE.search(prompt)
    if match:
        raise ValueError(f"prompt contains reference-label term: {match.group(0)!r}")


@dataclass(frozen=True)
class PlannedRequest:
    request_id: str
    arm: str
    kind: str
    context_id: str
    context: DecisionContext
    label: ReferenceLabel
    prompt: str
    public_payload: Mapping[str, Any]
    image: np.ndarray | None = field(default=None, compare=False, repr=False)
    image_variant: str = "none"
    base_context_id: str | None = None
    diagnostic_role: str | None = None


@dataclass(frozen=True)
class SmokePlan:
    base_contexts: tuple[DecisionContext, ...]
    base_labels: tuple[ReferenceLabel, ...]
    requests: tuple[PlannedRequest, ...]

    @property
    def base_requests(self) -> tuple[PlannedRequest, ...]:
        return tuple(item for item in self.requests if item.kind == "base")

    @property
    def nuisance_requests(self) -> tuple[PlannedRequest, ...]:
        return tuple(item for item in self.requests if item.kind == "nuisance")

    @property
    def diagnostic_requests(self) -> tuple[PlannedRequest, ...]:
        return tuple(item for item in self.requests if item.kind == "diagnostic")


def _context_map(library: DecisionLibrary) -> dict[str, tuple[DecisionContext, ReferenceLabel]]:
    return {context.context_id: (context, label) for context, label in zip(library.contexts, library.labels)}


def validate_selection(library: DecisionLibrary | None = None) -> None:
    library = library or build_development_library()
    by_id = _context_map(library)
    if len(BASE_IDS) != 12 or len(set(BASE_IDS)) != 12:
        raise ValueError("selection must contain 12 unique contexts")
    expected = {
        "COMMIT": COMMIT_IDS,
        "PROBE_SIGN": SIGN_IDS,
        "PROBE_BLUE": BLUE_IDS,
    }
    for action, ids in expected.items():
        for context_id in ids:
            if context_id not in by_id:
                raise ValueError(f"selected context missing: {context_id}")
            context, label = by_id[context_id]
            if label.best_action != action:
                raise ValueError(f"{context_id}: best action {label.best_action}, expected {action}")
            if set(label.optimal_actions) != {action}:
                raise ValueError(f"{context_id}: selection requires a unique optimum")
            if label.decision_margin < MIN_BASE_MARGIN:
                raise ValueError(f"{context_id}: margin {label.decision_margin} below floor")
            if len(context.feasible_actions()) < 2:
                raise ValueError(f"{context_id}: selection needs a genuine action choice")
            if not context.remaining_budget >= 0:
                raise ValueError(f"{context_id}: invalid remaining budget")
    for context_id, role in DIAGNOSTIC_ROLES.items():
        if context_id not in by_id:
            raise ValueError(f"diagnostic base missing: {context_id}")
        context, label = by_id[context_id]
        if role not in ("goal", "hazard", "safe"):
            raise ValueError(f"bad diagnostic role: {role}")
        perturbed = blue_role_perturbation(context, role)
        fresh = reference_for(perturbed)
        if set(fresh.optimal_actions) == set(label.optimal_actions):
            raise ValueError(f"{context_id}: diagnostic does not change exact optimal set")
    if len(NUISANCE_IDS) != 6 or len(DIAGNOSTIC_ROLES) != 6:
        raise ValueError("controls must each contain six contexts")


def build_plan(library: DecisionLibrary | None = None) -> SmokePlan:
    library = library or build_development_library()
    validate_selection(library)
    by_id = _context_map(library)
    base_contexts = tuple(by_id[context_id][0] for context_id in BASE_IDS)
    base_labels = tuple(by_id[context_id][1] for context_id in BASE_IDS)
    requests: list[PlannedRequest] = []

    for arm in ("A1", "A2", "A3"):
        for index, (context, label) in enumerate(zip(base_contexts, base_labels), start=1):
            prompt = render_prompt(context, arm)
            assert_prompt_public(prompt)
            image = render_decision_card(context, variant="base") if arm == "A3" else None
            requests.append(
                PlannedRequest(
                    request_id=f"{arm}-{index:02d}",
                    arm=arm,
                    kind="base",
                    context_id=context.context_id,
                    context=context,
                    label=label,
                    prompt=prompt,
                    public_payload=(
                        _public_facts(context, include_posterior=False)
                        if arm == "A1"
                        else context.to_public_payload(include_posterior=True)
                    ),
                    image=image,
                    image_variant="base" if image is not None else "none",
                )
            )

    for index, context_id in enumerate(NUISANCE_IDS, start=1):
        context, label = by_id[context_id]
        prompt = render_prompt(context, "A3")
        assert_prompt_public(prompt)
        requests.append(
            PlannedRequest(
                request_id=f"N{index}",
                arm="RGB_NUISANCE",
                kind="nuisance",
                context_id=context.context_id,
                context=context,
                label=label,
                prompt=prompt,
                public_payload=context.to_public_payload(include_posterior=True),
                image=render_decision_card(context, variant="nuisance"),
                image_variant="nuisance",
                base_context_id=context_id,
            )
        )

    for index, (context_id, role) in enumerate(DIAGNOSTIC_ROLES.items(), start=1):
        context, _base_label = by_id[context_id]
        perturbed = blue_role_perturbation(context, role)
        fresh_label = reference_for(perturbed)
        prompt = render_prompt(perturbed, "A3")
        assert_prompt_public(prompt)
        requests.append(
            PlannedRequest(
                request_id=f"D{index}",
                arm="RGB_DIAGNOSTIC",
                kind="diagnostic",
                context_id=perturbed.context_id,
                context=perturbed,
                label=fresh_label,
                prompt=prompt,
                public_payload=perturbed.to_public_payload(include_posterior=True),
                image=render_decision_card(perturbed, variant="base"),
                image_variant="base",
                base_context_id=context_id,
                diagnostic_role=role,
            )
        )

    if len(requests) != 48:
        raise AssertionError(f"planned {len(requests)} requests, expected 48")
    if len({item.request_id for item in requests}) != len(requests):
        raise AssertionError("request ids are not unique")
    return SmokePlan(base_contexts=base_contexts, base_labels=base_labels, requests=tuple(requests))


def candidate_hash(context: DecisionContext) -> str:
    return sha256_json(
        [
            {"rule_id": item["rule_id"], "description": item["description"]}
            for item in context.candidate_descriptions()
        ]
    )


def make_model_request(item: PlannedRequest) -> ModelRequest:
    image_hash = visual_digest(item.image) if item.image is not None else ""
    return ModelRequest(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        prompt=item.prompt,
        image_hash=image_hash,
        history_hash="",
        available_actions=item.context.feasible_actions(),
        candidate_hash=candidate_hash(item.context),
        decoding_seed=DECODING_SEED,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_version=PROMPT_VERSION,
        candidate_rules=item.context.candidate_rules,
        revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
        dtype="bfloat16",
        backend_id="",
        do_sample=False,
        device="cuda",
        input_mode="rgb" if item.image is not None else "text",
        image=item.image,
    )


def manifest_record(item: PlannedRequest) -> dict[str, Any]:
    png = image_png_bytes(item.image) if item.image is not None else None
    return {
        "request_id": item.request_id,
        "arm": item.arm,
        "kind": item.kind,
        "context_id": item.context_id,
        "base_context_id": item.base_context_id,
        "diagnostic_role": item.diagnostic_role,
        "image_variant": item.image_variant,
        "input_mode": "rgb" if item.image is not None else "text",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(item.prompt),
        "public_payload_sha256": sha256_json(dict(item.public_payload)),
        "feasible_actions": list(item.context.feasible_actions()),
        "candidate_rules": list(item.context.candidate_rules),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding_seed": DECODING_SEED,
        "image_pixel_digest": visual_digest(item.image) if item.image is not None else None,
        "image_png_sha256": sha256_bytes(png) if png is not None else None,
        "image_size": [int(item.image.shape[1]), int(item.image.shape[0])] if item.image is not None else None,
    }


def reference_payload(plan: SmokePlan) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for item in plan.requests:
        records.append(
            {
                "request_id": item.request_id,
                "context_id": item.context_id,
                "base_context_id": item.base_context_id,
                "kind": item.kind,
                "reference": asdict(item.label),
            }
        )
    return {"version": SMOKE_VERSION, "reference_version": "vlm-decision-reference-v1", "records": records}


def analysis_plan() -> dict[str, Any]:
    return {
        "version": SMOKE_VERSION,
        "primary_scope": "12-context development smoke; not confirmatory",
        "request_cap": 48,
        "arms": {
            "A1": {"n": 12, "contract": "direct", "information": "prior_plus_evidence_text"},
            "A2": {"n": 12, "contract": "direct", "information": "posterior_text"},
            "A3": {"n": 12, "contract": "direct", "information": "posterior_rgb_card"},
            "RGB_NUISANCE": {"n": 6, "contract": "direct", "information": "same_facts_new_palette"},
            "RGB_DIAGNOSTIC": {"n": 6, "contract": "direct", "information": "fresh_blue_role_posterior"},
        },
        "denominators": {
            "valid_action_rate": "all planned requests in the completed arm",
            "optimal_action_accuracy_valid": "valid non-fallback responses only",
            "optimal_action_accuracy_all": "all completed attempts; fallback counts as failure",
            "mean_regret_valid": "valid non-fallback responses only",
            "mean_regret_all": "all completed attempts, including fallback regret",
            "fallback_regret": "fallback attempts separately",
            "over_under_probe": "valid non-fallback responses only",
            "tool_selection_accuracy": "valid non-fallback PROBE responses only",
        },
        "hard_gates": {
            "leakage": "zero forbidden evaluator-label terms in prompts/public payloads",
            "checkpoint": "frozen snapshot manifest verifies",
            "valid_each_arm": "at least 11 of 12 valid in each completed direct arm",
            "constant_action": "at least two distinct valid actions across the 12 base contexts",
            "correct_commit_and_probe_A2_A3": "at least one correct COMMIT and one correct PROBE in A2 and A3",
            "budget": "at most 48 model requests, no retries",
        },
        "controls": {
            "nuisance_expected": "same exact label and action; palette/border only",
            "diagnostic_expected": "fresh exact label and optimal set; action change is descriptive, not a correctness shortcut",
        },
        "baselines": ["always_commit", "always_probe", "random_valid", "information_gain", "exact"],
        "no_tuning": "no prompt, parser, threshold, or selection changes after the first model request",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(data)


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_artifacts(
    plan: SmokePlan,
    out_dir: Path,
    *,
    checkpoint_manifest: Mapping[str, Any],
    checkpoint_manifest_path: Path,
    protocol_path: Path,
    source_paths: Sequence[Path],
) -> dict[str, Any]:
    """Write the frozen prompts, images, labels, manifest, and analysis plan."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    manifest = [manifest_record(item) for item in plan.requests]
    by_id = {row["request_id"]: row for row in manifest}

    for item in plan.requests:
        record = by_id[item.request_id]
        prompt_rows.append(
            {
                "request_id": item.request_id,
                "arm": item.arm,
                "context_id": item.context_id,
                "prompt_sha256": record["prompt_sha256"],
                "prompt": item.prompt,
            }
        )
        public_rows.append(
            {
                "request_id": item.request_id,
                "public_payload_sha256": record["public_payload_sha256"],
                "public_payload": dict(item.public_payload),
            }
        )
        if item.image is not None:
            png = image_png_bytes(item.image)
            image_path = images_dir / f"{item.request_id}.png"
            image_path.write_bytes(png)
            image_rows.append(
                {
                    "request_id": item.request_id,
                    "path": f"images/{item.request_id}.png",
                    "pixel_digest": record["image_pixel_digest"],
                    "png_sha256": record["image_png_sha256"],
                    "width": record["image_size"][0],
                    "height": record["image_size"][1],
                }
            )

    write_jsonl(out_dir / "prompts.jsonl", prompt_rows)
    write_jsonl(out_dir / "public_payloads.jsonl", public_rows)
    write_jsonl(out_dir / "images_manifest.jsonl", image_rows)
    write_json(out_dir / "request_manifest.json", manifest)
    write_json(out_dir / "reference_labels.json", reference_payload(plan))
    write_json(out_dir / "analysis_plan.json", analysis_plan())
    write_json(out_dir / "checkpoint_manifest.json", dict(checkpoint_manifest))

    source_hashes = {str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else str(path): _file_sha(path) for path in source_paths}
    freeze = {
        "version": SMOKE_VERSION,
        "source_hashes": source_hashes,
        "request_manifest_sha256": _file_sha(out_dir / "request_manifest.json"),
        "prompts_sha256": _file_sha(out_dir / "prompts.jsonl"),
        "public_payloads_sha256": _file_sha(out_dir / "public_payloads.jsonl"),
        "reference_labels_sha256": _file_sha(out_dir / "reference_labels.json"),
        "analysis_plan_sha256": _file_sha(out_dir / "analysis_plan.json"),
        "checkpoint_manifest_sha256": _file_sha(out_dir / "checkpoint_manifest.json"),
        "protocol_sha256": _file_sha(protocol_path),
        "base_ids": list(BASE_IDS),
        "nuisance_ids": list(NUISANCE_IDS),
        "diagnostic_roles": dict(DIAGNOSTIC_ROLES),
        "request_count": len(plan.requests),
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "scorer_version": SCORER_VERSION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding_seed": DECODING_SEED,
        "do_sample": False,
    }
    write_json(out_dir / "freeze.json", freeze)
    freeze_hash = _file_sha(out_dir / "freeze.json")
    (out_dir / "freeze.sha256").write_bytes((freeze_hash + "\n").encode("utf-8"))
    return freeze


def verify_freeze(out_dir: Path, *, source_paths: Sequence[Path] | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    freeze = json.loads((out_dir / "freeze.json").read_text(encoding="utf-8"))
    expected = {
        "request_manifest_sha256": out_dir / "request_manifest.json",
        "prompts_sha256": out_dir / "prompts.jsonl",
        "public_payloads_sha256": out_dir / "public_payloads.jsonl",
        "reference_labels_sha256": out_dir / "reference_labels.json",
        "analysis_plan_sha256": out_dir / "analysis_plan.json",
        "checkpoint_manifest_sha256": out_dir / "checkpoint_manifest.json",
    }
    for key, path in expected.items():
        actual = _file_sha(path)
        if actual != freeze[key]:
            raise ValueError(f"frozen hash mismatch for {key}: {actual} != {freeze[key]}")
    if source_paths is not None:
        for path in source_paths:
            key = str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else str(path)
            actual = _file_sha(path)
            if freeze["source_hashes"].get(key) != actual:
                raise ValueError(f"source changed after freeze: {key}")
    freeze_file_hash = _file_sha(out_dir / "freeze.json")
    if (out_dir / "freeze.sha256").read_text(encoding="utf-8").strip() != freeze_file_hash:
        raise ValueError("freeze.sha256 mismatch")
    return freeze


def subset_library(
    contexts: Sequence[DecisionContext], labels: Sequence[ReferenceLabel]
) -> DecisionLibrary:
    if len(contexts) != len(labels):
        raise ValueError("contexts and labels differ in length")
    return DecisionLibrary(contexts=tuple(contexts), labels=tuple(labels), manifest={"subset": len(contexts)})


def score_records(
    records: Sequence[Mapping[str, Any]],
    plan: SmokePlan,
    *,
    kind: str = "base",
    arm: str | None = None,
) -> dict[str, Any]:
    """Score parsed responses for one arm/control group without hiding fallbacks."""

    if kind == "base":
        specs = tuple(item for item in plan.base_requests if arm is None or item.arm == arm)
    elif kind == "nuisance":
        specs = plan.nuisance_requests
    elif kind == "diagnostic":
        specs = plan.diagnostic_requests
    else:
        raise ValueError(f"unknown kind: {kind}")
    if not specs:
        raise ValueError(f"no specs for kind={kind!r} arm={arm!r}")
    parsed_by_id = {row["request_id"]: row for row in records}
    predictions: list[DecisionPrediction] = []
    contexts: list[DecisionContext] = []
    labels: list[ReferenceLabel] = []
    details: list[dict[str, Any]] = []
    for item in specs:
        parsed = parsed_by_id.get(item.request_id)
        if parsed is None:
            raise ValueError(f"missing parsed response for {item.request_id}")
        valid = bool(parsed.get("valid", False)) and not bool(parsed.get("fallback", False))
        action = str(parsed.get("selected_action", "COMMIT"))
        fallback = bool(parsed.get("fallback", False))
        predictions.append(
            DecisionPrediction(
                context_id=item.context_id,
                action=action,
                fallback=fallback,
            )
        )
        contexts.append(item.context)
        labels.append(item.label)
    library = subset_library(contexts, labels)
    scores = score_decisions(library, predictions)
    score_by_id = {item.context_id: item for item in scores}
    valid_count = 0
    invalid_count = 0
    fallback_count = 0
    optimal_valid = 0
    optimal_all = 0
    valid_scores: list[Any] = []
    fallback_scores: list[Any] = []
    for item, parsed, score in zip(specs, [parsed_by_id[i.request_id] for i in specs], scores):
        valid = bool(parsed.get("valid", False)) and not bool(parsed.get("fallback", False))
        if valid:
            valid_count += 1
            valid_scores.append(score)
        else:
            invalid_count += 1
        if bool(parsed.get("fallback", False)):
            fallback_count += 1
            fallback_scores.append(score)
        if valid and score.optimal:
            optimal_valid += 1
            optimal_all += 1
        details.append(
            {
                "request_id": item.request_id,
                "context_id": item.context_id,
                "best_action": item.label.best_action,
                "optimal_actions": list(item.label.optimal_actions),
                "selected_action": score.predicted_action,
                "valid": valid,
                "fallback": bool(parsed.get("fallback", False)),
                "invalid": bool(score.invalid),
                "optimal": bool(score.optimal),
                "regret": float(score.regret),
                "over_probe": bool(score.over_probe),
                "under_probe": bool(score.under_probe),
                "probe_target_correct": score.probe_target_correct,
                "best_value": float(item.label.best_value),
                "q_commit": float(item.label.q_commit),
                "q_sign": item.label.q_sign,
                "q_blue": item.label.q_blue,
            }
        )

    def _mean(values: Sequence[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    valid_regrets = [float(score.regret) for score in valid_scores]
    all_regrets = [float(score.regret) for score in scores]
    fallback_regrets = [float(score.regret) for score in fallback_scores]
    probe_scores = [score for score in valid_scores if score.probe_target_correct is not None]
    distinct_actions = sorted({score.predicted_action for score in valid_scores})
    correct_commit = sum(
        score.predicted_action == "COMMIT" and score.optimal for score in valid_scores
    )
    correct_probes = sum(
        score.predicted_action.startswith("PROBE") and score.optimal for score in valid_scores
    )
    summary = {
        "n": len(specs),
        "valid_count": valid_count,
        "invalid_or_fallback_count": invalid_count,
        "fallback_count": fallback_count,
        "valid_action_rate": valid_count / len(specs),
        "optimal_action_accuracy_valid": optimal_valid / valid_count if valid_count else None,
        "optimal_action_accuracy_all": optimal_all / len(specs),
        "mean_regret_valid": _mean(valid_regrets),
        "mean_regret_all": _mean(all_regrets),
        "max_regret": max(all_regrets) if all_regrets else None,
        "fallback_regret_mean": _mean(fallback_regrets),
        "fallback_regret_count": len(fallback_regrets),
        "over_probe_rate_valid": (
            sum(score.over_probe for score in valid_scores) / valid_count
            if valid_count
            else None
        ),
        "under_probe_rate_valid": (
            sum(score.under_probe for score in valid_scores) / valid_count
            if valid_count
            else None
        ),
        "tool_selection_accuracy_valid": (
            sum(bool(score.probe_target_correct) for score in probe_scores) / len(probe_scores)
            if probe_scores
            else None
        ),
        "correct_commit_count": correct_commit,
        "correct_probe_count": correct_probes,
        "distinct_valid_actions": distinct_actions,
        "action_counts": {
            action: sum(1 for row in details if row["selected_action"] == action)
            for action in ACTIONS
        },
        "valid_action_counts": {
            action: sum(1 for row in details if row["selected_action"] == action and row["valid"])
            for action in ACTIONS
        },
        "details": details,
    }
    gate = {
        "n_requests": len(specs),
        "applicable": kind == "base",
        "valid_at_least_11": valid_count >= 11,
        "non_constant": len(distinct_actions) >= 2,
    }
    if kind == "base":
        if specs[0].arm in ("A2", "A3"):
            gate["at_least_one_correct_commit"] = correct_commit >= 1
            gate["at_least_one_correct_probe"] = correct_probes >= 1
    summary["gate"] = gate
    summary["gate_pass"] = (
        all(
            bool(value)
            for key, value in gate.items()
            if key not in ("n_requests", "applicable")
        )
        if kind == "base"
        else None
    )
    return summary


def baseline_summary(plan: SmokePlan) -> dict[str, Any]:
    library = subset_library(plan.base_contexts, plan.base_labels)
    result: dict[str, Any] = {}
    for name in ("always_commit", "always_probe", "random_valid", "information_gain", "exact"):
        predictions = baseline_predictions(library, name, seed=DECODING_SEED)
        scores = score_decisions(library, predictions)
        total = len(scores)
        result[name] = {
            "n": total,
            "optimal_action_accuracy": sum(score.optimal for score in scores) / total,
            "mean_regret": sum(score.regret for score in scores) / total,
            "max_regret": max(score.regret for score in scores),
        }
    return result


def control_summary(
    plan: SmokePlan,
    base_summary: Mapping[str, Any],
    nuisance_summary: Mapping[str, Any],
    diagnostic_summary: Mapping[str, Any],
) -> dict[str, Any]:
    base_by_id = {row["request_id"]: row for row in base_summary["details"]}
    nuisance_by_id = {row["request_id"]: row for row in nuisance_summary["details"]}
    diagnostic_by_id = {row["request_id"]: row for row in diagnostic_summary["details"]}
    nuisance_pairs: list[dict[str, Any]] = []
    for item in plan.nuisance_requests:
        # Match by source context, since N and A3 request ids differ.
        base_row = next(
            row for row in base_summary["details"] if row["context_id"] == item.base_context_id
        )
        n_row = nuisance_by_id[item.request_id]
        nuisance_pairs.append(
            {
                "request_id": item.request_id,
                "base_request_id": base_row["request_id"],
                "context_id": item.base_context_id,
                "base_action": base_row["selected_action"],
                "nuisance_action": n_row["selected_action"],
                "changed": base_row["selected_action"] != n_row["selected_action"],
                "base_valid": base_row["valid"],
                "nuisance_valid": n_row["valid"],
                "label_unchanged": True,
                "base_correct": base_row["optimal"],
                "nuisance_correct": n_row["optimal"],
            }
        )
    diagnostic_pairs: list[dict[str, Any]] = []
    for item in plan.diagnostic_requests:
        base_row = next(
            row for row in base_summary["details"] if row["context_id"] == item.base_context_id
        )
        d_row = diagnostic_by_id[item.request_id]
        diagnostic_pairs.append(
            {
                "request_id": item.request_id,
                "base_request_id": base_row["request_id"],
                "base_context_id": item.base_context_id,
                "diagnostic_context_id": item.context_id,
                "diagnostic_role": item.diagnostic_role,
                "base_action": base_row["selected_action"],
                "diagnostic_action": d_row["selected_action"],
                "changed": base_row["selected_action"] != d_row["selected_action"],
                "base_valid": base_row["valid"],
                "diagnostic_valid": d_row["valid"],
                "base_best_action": base_row["best_action"],
                "diagnostic_best_action": d_row["best_action"],
                "optimal_set_changed": set(base_row["optimal_actions"]) != set(d_row["optimal_actions"]),
                "base_correct": base_row["optimal"],
                "diagnostic_correct": d_row["optimal"],
            }
        )
    return {
        "nuisance": {
            "n": len(nuisance_pairs),
            "unchanged_count": sum(not row["changed"] for row in nuisance_pairs),
            "changed_count": sum(row["changed"] for row in nuisance_pairs),
            "pairs": nuisance_pairs,
        },
        "diagnostic": {
            "n": len(diagnostic_pairs),
            "changed_count": sum(row["changed"] for row in diagnostic_pairs),
            "optimal_set_changed_count": sum(row["optimal_set_changed"] for row in diagnostic_pairs),
            "pairs": diagnostic_pairs,
        },
    }
