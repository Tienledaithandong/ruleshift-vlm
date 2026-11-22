"""Offline pilot harness over development contexts (no live environment loop).

Labels come from the exact microtask solver; all model-visible input is public.
The default backend in this repository is a deterministic stub, so generated
artifacts are a SYNTHETIC DRY RUN and must never be reported as VLM evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_backend import (
    ModelRequest,
    ModelResponse,
    parse_structured_output,
    render_prompt,
)
from ruleshift.envs.config import EnvConfig
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.evaluator.exact_solver import (
    ExactSolver,
    MicrotaskConfig,
    family_a_microtask,
    freeze_belief,
)
from ruleshift.public_api.interface import PublicObservation
from ruleshift.public_api.schemas import FAMILY_A_ACTIONS

from .phase_diagram import LOCKOUT_LEVELS, NOISE_LEVELS, PROBE_DETOURS

#: Long enough for a full probe, a commit and the largest lockout to fit, so
#: both ``delta_probe`` signs are reachable with the real macro durations.
HORIZON = 24
#: Labels cover the complete public macro menu, not a hand-picked subset.
ACTIONS = FAMILY_A_ACTIONS
REFERENCE_BELIEF = {("red", False): 0.5, ("blue", True): 0.5}


@dataclass(frozen=True)
class OfflineContext:
    context_id: str
    scenario_id: str
    config: MicrotaskConfig
    belief: dict[tuple[str, bool], float]
    delta_probe: float
    q_values: dict[str, float]
    public_observation: PublicObservation
    horizon: int


@dataclass(frozen=True)
class AuditMetrics:
    n_contexts: int
    json_valid_rate: float
    valid_action_rate: float
    over_probe_rate: float
    under_probe_rate: float
    mean_decision_regret: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_model_latency_ms: float
    call_count: int
    cache_hit_rate: float
    #: Fraction of decisions whose chosen macro has an exact-solver label.
    #: Regret is only averaged over covered decisions.
    label_coverage_rate: float


class _Backend(Protocol):
    model_id: str

    def generate(self, request: ModelRequest) -> ModelResponse: ...


def _microtask_config(
    probe_detour_steps: int, commit_travel_steps: int, noise: float, lockout: int
) -> MicrotaskConfig:
    """Exact microtask whose macro durations match the environment's executor."""

    return family_a_microtask(
        probe_detour_steps=probe_detour_steps,
        commit_travel_steps=commit_travel_steps,
        feedback_noise=noise,
        lockout_steps=lockout,
        actions=ACTIONS,
    )


def build_offline_contexts(count: int = 81) -> tuple[OfflineContext, ...]:
    """Select diverse dev contexts with exact labels and public observations."""

    seeds_per_cell = math.ceil(
        count / (len(PROBE_DETOURS) * len(NOISE_LEVELS) * len(LOCKOUT_LEVELS))
    )
    contexts: list[OfflineContext] = []
    for detour in PROBE_DETOURS:
        for noise in NOISE_LEVELS:
            for lockout in LOCKOUT_LEVELS:
                # Public geometry is identical for the whole cell, so read the
                # commit travel distance from a throwaway environment.
                probe_env = RuleShiftEnv(
                    config=EnvConfig(
                        probe_detour_steps=detour,
                        feedback_noise=noise,
                        lockout_steps=lockout,
                        disable_shift=True,
                    )
                )
                probe_env.reset(seed=1)
                commit_travel = int(
                    probe_env.public_view().public_costs["commit_travel"]
                )
                microtask = _microtask_config(detour, commit_travel, noise, lockout)
                solver = ExactSolver(microtask, ACTIONS)
                q_values = dict(solver.q_values(freeze_belief(REFERENCE_BELIEF), HORIZON))
                delta = solver.delta_probe(REFERENCE_BELIEF, HORIZON)
                for seed in range(1, seeds_per_cell + 1):
                    scenario_id = f"dev-d{detour}-n{noise}-l{lockout}-s{seed}"
                    env_config = EnvConfig(
                        probe_detour_steps=detour,
                        feedback_noise=noise,
                        lockout_steps=lockout,
                        disable_shift=True,
                        initial_mechanic="red",
                        scenario_id=scenario_id,
                    )
                    env = RuleShiftEnv(config=env_config)
                    env.reset(seed=seed)
                    for _ in range(seed - 1):
                        env.step("WAIT")
                    contexts.append(
                        OfflineContext(
                            context_id=scenario_id,
                            scenario_id=scenario_id,
                            config=microtask,
                            belief=dict(REFERENCE_BELIEF),
                            delta_probe=delta,
                            q_values=q_values,
                            public_observation=env.public_view(),
                            horizon=HORIZON,
                        )
                    )
    return tuple(contexts[:count])


def _make_request(
    context: OfflineContext, prompt: str, model_id: str, max_new_tokens: int
) -> ModelRequest:
    observation = context.public_observation
    descriptions = [rule.description for rule in observation.candidate_rules]
    candidate_hash = hashlib.sha256(
        json.dumps(descriptions, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return ModelRequest(
        model_id=model_id,
        prompt=prompt,
        image_hash=observation.public_obs_hash,
        history_hash=observation.public_history_hash,
        available_actions=tuple(observation.available_actions),
        candidate_hash=candidate_hash,
        decoding_seed=7,
        max_new_tokens=max_new_tokens,
    )


def run_offline_audit(
    contexts: tuple[OfflineContext, ...],
    backend: _Backend,
    cache: ResponseCache,
    max_new_tokens: int = 768,
) -> tuple[list[dict], AuditMetrics]:
    """Run the direct prompt over contexts and score probe decisions."""

    records: list[dict] = []
    labels: list[dict] = []
    flags: list[dict] = []
    call_count = 0

    for context in contexts:
        observation = context.public_observation
        description_by_rule = {
            rule.rule_id: rule.description for rule in observation.candidate_rules
        }
        prompt = render_prompt(
            public_history=observation.public_text,
            public_text=observation.public_text,
            candidate_rules=[f"{key}: {value}" for key, value in description_by_rule.items()],
            available_actions=list(observation.available_actions),
            public_costs=dict(observation.public_costs),
        )
        request = _make_request(
            context, prompt, backend.model_id, max_new_tokens
        )
        cached = cache.get(request)
        cache_hit = cached is not None
        if cached is None:
            response = backend.generate(request)
            cache.put(request, response)
            call_count += 1
        else:
            response = cached

        payload, errors = parse_structured_output(response.text, request.available_actions)
        chosen = payload["selected_action"] if payload else "WAIT"
        available = set(request.available_actions)
        valid_action = payload is not None and chosen in available
        over_probe = context.delta_probe < 0 and chosen.startswith("PROBE_")
        under_probe = context.delta_probe > 0 and not chosen.startswith("PROBE_")
        best_q = max(context.q_values.values())
        # An action with no exact label must be reported as uncovered, never
        # scored as if it were worth zero: that silently fabricated regret.
        chosen_q = context.q_values.get(chosen)
        label_covered = chosen_q is not None
        regret = (best_q - chosen_q) if chosen_q is not None else None

        records.append(
            {
                "context_id": context.context_id,
                "public_obs_hash": observation.public_obs_hash,
                "public_history_hash": observation.public_history_hash,
                "available_actions": list(observation.available_actions),
                "chosen_action": chosen,
                "json_valid": payload is not None,
                "invalid_action": not valid_action,
                "parse_error_count": len(errors),
                "model_id": response.model_id,
                "backend_id": response.backend_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "cache_hit": cache_hit,
                "prompt_hash": request.fingerprint(),
                "vlm_calls_used": 0 if cache_hit else 1,
            }
        )
        labels.append(
            {
                "context_id": context.context_id,
                "delta_probe": context.delta_probe,
                "best_q": best_q,
                "chosen_q": chosen_q,
                "decision_regret": regret,
                "label_covered": label_covered,
                "belief": {f"{k[0]}:{k[1]}": v for k, v in context.belief.items()},
            }
        )
        flags.append({"over_probe": over_probe, "under_probe": under_probe})

    n = max(len(records), 1)
    covered_regrets = [
        label["decision_regret"]
        for label in labels
        if label["decision_regret"] is not None
    ]
    covered = sum(label["label_covered"] for label in labels)
    model_calls = [record for record in records if not record["cache_hit"]]
    call_denominator = max(len(model_calls), 1)
    metrics = AuditMetrics(
        n_contexts=len(records),
        json_valid_rate=sum(r["json_valid"] for r in records) / n,
        valid_action_rate=sum(not r["invalid_action"] for r in records) / n,
        over_probe_rate=sum(f["over_probe"] for f in flags) / n,
        under_probe_rate=sum(f["under_probe"] for f in flags) / n,
        mean_decision_regret=(
            sum(covered_regrets) / len(covered_regrets) if covered_regrets else 0.0
        ),
        mean_input_tokens=(
            sum(record["input_tokens"] for record in model_calls) / call_denominator
        ),
        mean_output_tokens=(
            sum(record["output_tokens"] for record in model_calls) / call_denominator
        ),
        mean_model_latency_ms=(
            sum(record["latency_ms"] for record in model_calls) / call_denominator
        ),
        call_count=call_count,
        cache_hit_rate=sum(r["cache_hit"] for r in records) / n,
        label_coverage_rate=covered / n,
    )
    return records, metrics


def write_taxonomy_report(
    metrics: AuditMetrics,
    records: list[dict],
    contexts: tuple[OfflineContext, ...],
    manifest_path: Path,
    evaluator_path: Path,
    report_path: Path,
) -> None:
    """Write public manifest, separate evaluator labels, and a labelled report."""

    for path, rows in ((Path(manifest_path), records),):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    label_path = Path(evaluator_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_rows = [
        {
            "context_id": context.context_id,
            "delta_probe": context.delta_probe,
            "q_values": context.q_values,
        }
        for context in contexts
    ]
    label_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows),
        encoding="utf-8",
    )

    text = f"""# Week 4 — Offline pilot dry run (SYNTHETIC)

**SYNTHETIC DRY RUN. No VLM was run.** The backend used here is a deterministic
stub (`STUB_SYNTHETIC`). These numbers exercise the harness only; they are NOT
evidence about any vision-language model and must not appear in a paper.

Contexts: {metrics.n_contexts}. Model calls: {metrics.call_count}.
Public manifest: `{Path(manifest_path).name}` (public fields only).
Privileged labels: `{Path(evaluator_path).name}` (never joined into model input).

| Diagnostic | Value |
|---|---:|
| JSON valid rate | {metrics.json_valid_rate:.3f} |
| Valid action rate | {metrics.valid_action_rate:.3f} |
| Over-probe rate (probed when exact label says do not) | {metrics.over_probe_rate:.3f} |
| Under-probe rate (did not probe when label says probe) | {metrics.under_probe_rate:.3f} |
| Mean decision regret vs exact Q* (covered decisions only) | {metrics.mean_decision_regret:.3f} |
| Decisions with an exact label (full macro menu) | {metrics.label_coverage_rate:.3f} |
| Mean output tokens | {metrics.mean_output_tokens:.1f} |
| Cache hit rate | {metrics.cache_hit_rate:.3f} |

## Failure taxonomy measured by this harness

1. Invalid JSON / parse failures — counted, deterministic fallback to `WAIT`.
2. Invalid macro ids — counted separately from parse failures.
3. Over-probing — probe chosen in a cell with `delta_probe < 0`.
4. Under-probing — task action chosen in a cell with `delta_probe > 0`.
5. Outcome-prediction error — schema is captured, accuracy scoring is Week 5.
6. Cost accounting — tokens, calls, latency and cache hits per decision.

A real frozen-VLM pilot requires an available checkpoint. When one exists,
swap `StubVLMBackend` for `TransformersVLMBackend`; only then do these rows
become empirical observations, and the synthetic label must be removed.
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(text, encoding="utf-8")
