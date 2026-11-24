"""Offline pilot harness over development contexts (no live environment loop).

Labels come from the exact microtask solver; all model-visible input is public.
The default backend in this repository is a deterministic stub, so generated
artifacts are a SYNTHETIC DRY RUN and must never be reported as VLM evidence.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_backend import (
    ModelRequest,
    ModelResponse,
    parse_structured_output,
)
from ruleshift.agents.structured_vlm import build_prediction_request
from ruleshift.agents.value_selector import (
    PublicCostModel,
    ValueAwareSelector,
    public_macro_durations,
)
from ruleshift.envs.config import EnvConfig, load_dev_grid
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.evaluator.exact_solver import (
    ExactSolver,
    MicrotaskConfig,
    family_a_microtask,
    freeze_belief,
)
from ruleshift.public_api.interface import PublicObservation
from ruleshift.public_api.schemas import FAMILY_A_ACTIONS


#: Long enough for a full probe, a commit and the largest lockout to fit, so
#: both ``delta_probe`` signs are reachable with the real macro durations.
HORIZON = 24
#: Labels cover the complete public macro menu, not a hand-picked subset.
ACTIONS = FAMILY_A_ACTIONS
RULE_PAIRS = (("red", "blue"), ("blue", "green"), ("green", "red"))
HORIZONS = (HORIZON, 16, 20, 12, 28)


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
    #: Public prior for the first pair rule (0.5 = fair draw).
    prior: float = 0.5
    #: Disclosed station error rate for this context (0 = reliable).
    feedback_noise: float = 0.0
    #: Evaluator-only prefix evidence for history contexts.
    prefix: tuple[tuple[str, str], ...] = ()
    #: Evaluator-only provenance (true rule, shift time, ...).
    evaluator_meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditMetrics:
    n_contexts: int
    json_valid_rate: float
    valid_action_rate: float
    over_probe_rate: float
    under_probe_rate: float
    mean_decision_regret: float | None
    mean_input_tokens: float
    mean_output_tokens: float
    mean_model_latency_ms: float
    call_count: int
    cache_hit_rate: float
    #: Fraction of decisions whose chosen macro has an exact-solver label.
    #: Regret is only averaged over covered decisions.
    label_coverage_rate: float
    distinct_requests: int
    invalid_json_count: int
    invalid_action_count: int
    fallback_count: int


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


def build_offline_contexts(count: int = 81, config_path: Path | None = None) -> tuple[OfflineContext, ...]:
    """Controlled, first-decision P1 microtasks with a public initial lottery.

    Unlike live shift episodes, the rule here stays fixed and the station test
    is ideal. The rule draw itself is hidden. A history-free reset supplies
    the public geometry; its concrete latent state never supplies a label.
    Cycle costs, card pairs and remaining budgets before repeating a cell.
    """

    dev = load_dev_grid(config_path)
    if dev["split"] != "dev" or dev["environment"]["family"] != "gate_lock":
        raise ValueError("offline microtask design requires the Family A dev grid")
    environment = dev["environment"]
    detours = dev["grid"]["probe_detour_steps"]
    lockouts = environment["lockout_steps_options"]
    cell_count = len(detours) * len(lockouts)
    capacity = cell_count * len(RULE_PAIRS) * len(HORIZONS)
    if not 1 <= count <= capacity:
        raise ValueError(f"context count must be between 1 and {capacity}")

    contexts: list[OfflineContext] = []
    solvers: dict[tuple[int, int], ExactSolver] = {}
    for index in range(count):
        block, cell = divmod(index, cell_count)
        row, column = divmod(cell, len(detours))
        detour = detours[column]
        lockout = lockouts[(row - column - 1) % len(lockouts)]
        pair = RULE_PAIRS[(cell + block) % len(RULE_PAIRS)]
        horizon = HORIZONS[(row + block // len(RULE_PAIRS)) % len(HORIZONS)]
        scenario_id = f"ideal-d{detour}-l{lockout}-{pair[0]}-{pair[1]}-h{horizon}"
        env = RuleShiftEnv(config=EnvConfig(
            probe_detour_steps=detour,
            feedback_noise=0.0,
            lockout_steps=lockout,
            disable_shift=True,
            max_env_steps=horizon,
            grid_width=environment["grid_size"][0],
            grid_height=environment["grid_size"][1],
            step_cost=environment["step_cost"],
            delivery_reward=environment["delivery_reward"],
        ))
        env.reset(seed=0)
        initial = env.public_view()
        description = (
            f"Controlled card lottery. An unseen fair draw selected either {pair[0]} "
            f"or {pair[1]}. That card stays fixed for this task. The station reliably reports "
            f"whether a card opens the gate. You have {horizon} primitive steps "
            f"to maximize total delivery reward including step costs. {initial.public_text}"
        )
        # The text promises exactly two candidate rules. Keep the menu
        # consistent: leaving GREEN in candidate_rules would contradict it.
        observation = replace(
            initial,
            public_text=description,
            candidate_rules=tuple(
                rule for rule in initial.candidate_rules if rule.rule_id in pair
            ),
        )
        if (detour, lockout) not in solvers:
            config = _microtask_config(
                detour, int(initial.public_costs["commit_travel"]), 0.0, lockout
            )
            config = replace(config, step_cost=environment["step_cost"],
                             reward_success=environment["delivery_reward"])
            solvers[detour, lockout] = ExactSolver(config, ACTIONS)
        solver = solvers[detour, lockout]
        belief = {(color, False): 0.5 for color in pair}
        q_values = dict(solver.q_values(freeze_belief(belief), horizon))
        contexts.append(OfflineContext(
            context_id=scenario_id,
            scenario_id=scenario_id,
            config=solver.config,
            belief=belief,
            delta_probe=solver.delta_probe(belief, horizon),
            q_values=q_values,
            public_observation=observation,
            horizon=horizon,
            prior=0.5,
            feedback_noise=0.0,
        ))
    return tuple(contexts)


def run_offline_audit(
    contexts: tuple[OfflineContext, ...],
    backend: _Backend,
    cache: ResponseCache,
    *,
    method: str = "direct",
) -> tuple[list[dict], AuditMetrics]:
    """Run direct or M1 over contexts and score probe decisions.

    ``method="m1"`` uses the structured outcome prompt and the agent-side
    value selector. It never reads the exact solver or the hidden rule.
    """

    if method not in ("direct", "m1"):
        raise ValueError(f"unknown offline method: {method}")

    if not contexts:
        raise ValueError("offline audit requires at least one decision context")
    records: list[dict] = []
    labels: list[dict] = []
    flags: list[dict] = []
    call_count = 0

    for context in contexts:
        observation = context.public_observation
        direct = method == "direct"
        request = build_prediction_request(observation, backend, direct=direct)
        cached = cache.get(request)
        cache_hit = cached is not None
        if cached is None:
            response = backend.generate(request)
            cache.put(request, response)
            call_count += 1
        else:
            response = cached

        payload, errors = parse_structured_output(
            response.text,
            request.available_actions,
            request.candidate_rules,
            direct=direct,
        )
        if direct:
            chosen = payload["selected_action"] if payload else "WAIT"
            model_selected = chosen
        else:
            model_selected = payload.get("selected_action") if payload else None
            if payload is None:
                chosen = "WAIT"
            else:
                probabilities = payload.get("candidate_probabilities", {})
                belief = {
                    str(rule): float(value)
                    for rule, value in probabilities.items()
                    if float(value) > 0.0
                }
                if not belief:
                    fallback: dict[str, float] = {}
                    for (rule, _shifted), mass in context.belief.items():
                        fallback[rule] = fallback.get(rule, 0.0) + mass
                    belief = fallback
                total = sum(belief.values())
                if total <= 0.0:
                    raise ValueError(f"empty belief for context {context.context_id}")
                belief = {rule: value / total for rule, value in belief.items()}
                probe_duration, commit_duration = public_macro_durations(
                    observation.public_costs
                )
                cost = PublicCostModel.from_public_costs(
                    observation.public_costs,
                    probe_duration=probe_duration,
                    commit_duration=commit_duration,
                    remaining_horizon=context.horizon,
                    learning_rate=0.5,
                )
                chosen = ValueAwareSelector().choose(payload, belief, cost)
        json_valid = not any(error.startswith("invalid JSON") for error in errors)
        invalid_action = any(error.startswith("selected_action is not") for error in errors)
        over_probe = context.delta_probe < -1e-9 and chosen.startswith("PROBE_")
        under_probe = context.delta_probe > 1e-9 and not chosen.startswith("PROBE_")
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
                "method": method,
                "chosen_action": chosen,
                "model_selected_action": model_selected,
                "json_valid": json_valid,
                "schema_valid": payload is not None,
                "invalid_action": invalid_action,
                "fallback": payload is None,
                "parse_error_count": len(errors),
                "model_id": response.model_id,
                "backend_id": response.backend_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "cache_hit": cache_hit,
                "public_prompt": request.prompt,
                "prompt_version": request.prompt_version,
                "decoding_seed": request.decoding_seed,
                "do_sample": request.do_sample,
                "prompt_hash": "sha256:" + hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
                "request_fingerprint": request.fingerprint(),
                "vlm_calls_used": 1,
                "fresh_model_calls": 0 if cache_hit else 1,
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
                "prior": context.prior,
                "feedback_noise": context.feedback_noise,
                "prefix": [list(item) for item in context.prefix],
                "evaluator_meta": context.evaluator_meta,
                "belief": {f"{k[0]}:{k[1]}": v for k, v in context.belief.items()},
            }
        )
        flags.append({"over_probe": over_probe, "under_probe": under_probe})

    n = len(records)
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
        valid_action_rate=sum(r["schema_valid"] for r in records) / n,
        over_probe_rate=sum(f["over_probe"] for f in flags) / n,
        under_probe_rate=sum(f["under_probe"] for f in flags) / n,
        mean_decision_regret=(
            sum(covered_regrets) / len(covered_regrets) if covered_regrets else None
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
        distinct_requests=len({r["prompt_hash"] for r in records}),
        invalid_json_count=sum(not r["json_valid"] for r in records),
        invalid_action_count=sum(r["invalid_action"] for r in records),
        fallback_count=sum(r["fallback"] for r in records),
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

    if len(records) != len(contexts):
        raise ValueError(
            f"manifest has {len(records)} rows but {len(contexts)} contexts were built"
        )
    if metrics.n_contexts != len(records):
        raise ValueError("metrics.n_contexts does not match the manifest row count")

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
            "prior": context.prior,
            "feedback_noise": context.feedback_noise,
            "prefix": [list(item) for item in context.prefix],
            "evaluator_meta": context.evaluator_meta,
        }
        for context in contexts
    ]
    label_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows),
        encoding="utf-8",
    )

    regret_text = f"{metrics.mean_decision_regret:.3f}" if metrics.mean_decision_regret is not None else "unavailable"
    text = f"""# Week 4 — Offline pilot dry run (SYNTHETIC)

**SYNTHETIC DRY RUN. No VLM was run.** The backend used here is a deterministic
stub (`STUB_SYNTHETIC`). These numbers exercise the harness only; they are NOT
evidence about any vision-language model and must not appear in a paper.

Public-text (P1) microtask decisions: {metrics.n_contexts}.
Methods: {', '.join(sorted({str(r.get('method', 'direct')) for r in records}))}.
Distinct public requests: {metrics.distinct_requests}. Model calls: {metrics.call_count}.
Exact reference labels use the specified context design and remaining budget.
Decision regret includes the executed WAIT fallback on invalid responses.
Public manifest: `{Path(manifest_path).name}` (public fields only).
Privileged labels: `{Path(evaluator_path).name}` (never joined into model input).

| Diagnostic | Value |
|---|---:|
| JSON valid rate | {metrics.json_valid_rate:.3f} |
| Schema-valid action rate | {metrics.valid_action_rate:.3f} |
| Invalid JSON / invalid action / WAIT fallbacks | {metrics.invalid_json_count} / {metrics.invalid_action_count} / {metrics.fallback_count} |
| Over-probe rate (probed when exact label says do not) | {metrics.over_probe_rate:.3f} |
| Under-probe rate (did not probe when label says probe) | {metrics.under_probe_rate:.3f} |
| Mean decision regret vs exact Q* (covered decisions only) | {regret_text} |
| Decisions with an exact label (full macro menu) | {metrics.label_coverage_rate:.3f} |
| Mean output tokens | {metrics.mean_output_tokens:.1f} |
| Cache hit rate | {metrics.cache_hit_rate:.3f} |

## Failure taxonomy measured by this harness

1. Invalid JSON / parse failures — counted, deterministic fallback to `WAIT`.
2. Invalid macro ids — counted separately from parse failures.
3. Over-probing — probe chosen with a negative label outside the 1e-9 tie band.
4. Under-probing — non-probe chosen with a positive label outside the tie band.
5. This direct-action pilot records the chosen macro.
6. Cost accounting — tokens, calls, latency and cache hits per decision.

A real frozen-VLM pilot requires an available checkpoint. When one exists,
swap `StubVLMBackend` for `TransformersVLMBackend`; only then do these rows
become empirical observations, and the synthetic label must be removed.
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(text, encoding="utf-8")
