"""Surgery-style offline diagnostics on the agent pipeline.

Each intervention replaces exactly ONE component while holding the others
fixed. With a real VLM this would locate perception/belief/outcome/control
bottlenecks; here the backend is a deterministic stub, so the numbers only
validate the attribution machinery. No generality claim is made.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.value_selector import PublicCostModel, ValueAwareSelector
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.evaluator.offline_audit import OfflineContext
from ruleshift.agents.structured_vlm import run_structured_prediction

INTERVENTIONS: tuple[str, ...] = (
    "baseline",
    "oracle_belief",
    "oracle_outcomes",
    "public_text",
    "oracle_both",
)


@dataclass(frozen=True)
class DiagnosticResult:
    intervention: str
    decisions: tuple[dict, ...]
    mean_regret: float
    error_rate: float


def _rule_marginal(belief: dict[tuple[str, bool], float]) -> dict[str, float]:
    marginal: dict[str, float] = {}
    for (rule, _), probability in belief.items():
        marginal[rule] = marginal.get(rule, 0.0) + probability
    for rule in ("red", "blue", "green"):
        marginal.setdefault(rule, 0.0)
    return marginal


def _true_outcomes(payload: dict) -> dict:
    """Replace VLM outcome predictions with noise-free public rule semantics."""

    transformed = copy.deepcopy(payload)
    for prediction in transformed.get("action_predictions", []):
        action = prediction.get("action_id", "")
        color = action.split("_", 1)[1].lower() if "_" in action else ""
        for rule_prediction in prediction.get("under_rules", []):
            rule_prediction["possible_outcomes"] = [
                {
                    "observation": (
                        "accept" if rule_prediction.get("rule_id") == color else "reject"
                    ),
                    "p": 1.0,
                }
            ]
    return transformed


def _cost(context: OfflineContext) -> PublicCostModel:
    costs = context.public_observation.public_costs
    detour = costs.get("probe_detour", 0.0)
    return PublicCostModel(
        step_cost=costs.get("primitive_step", 0.05),
        success_reward=costs.get("delivery_reward", 10.0),
        probe_duration=int(2 * detour + 1),
        commit_duration=1,
        lockout_steps=int(costs.get("lockout", 0)),
        remaining_horizon=context.horizon,
        learning_rate=0.5,
    )


def run_diagnostic(
    contexts: tuple[OfflineContext, ...],
    backend: StubVLMBackend,
    cache: ResponseCache,
    intervention: str,
) -> DiagnosticResult:
    """Run one intervention across all contexts."""

    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention: {intervention}")
    selector = ValueAwareSelector()
    decisions: list[dict] = []

    for context in contexts:
        payload, _, errors = run_structured_prediction(
            context.public_observation, backend, cache
        )
        if payload is None:
            action = "WAIT"
        else:
            if intervention in ("oracle_outcomes", "oracle_both"):
                payload = _true_outcomes(payload)
            belief = (
                {"red": 1.0, "blue": 0.0, "green": 0.0}
                if intervention in ("oracle_belief", "oracle_both")
                else _rule_marginal(context.belief)
            )
            action = selector.choose(payload, belief, _cost(context))

        best_q = max(context.q_values.values())
        chosen_q = context.q_values.get(action, 0.0)
        regret = best_q - chosen_q
        decisions.append(
            {
                "context_id": context.context_id,
                "intervention": intervention,
                "chosen_action": action,
                "decision_regret": regret,
                "parse_errors": len(errors),
            }
        )

    count = max(len(decisions), 1)
    return DiagnosticResult(
        intervention=intervention,
        decisions=tuple(decisions),
        mean_regret=sum(d["decision_regret"] for d in decisions) / count,
        error_rate=sum(d["decision_regret"] > 1e-6 for d in decisions) / count,
    )


def write_diagnostic_report(
    results: dict[str, DiagnosticResult], path: Path
) -> Path:
    """Write an explicit-component attribution report."""

    baseline = results["baseline"].mean_regret
    reductions = {
        name: baseline - result.mean_regret
        for name, result in results.items()
        if name != "baseline"
    }
    if reductions and max(reductions.values()) > 1e-9:
        largest = max(reductions, key=reductions.get)
    else:
        largest = "none — all interventions are identical under this stub"

    lines = [
        "# Week 10 — Error decomposition by surgical intervention (SYNTHETIC)",
        "",
        "**No VLM was run.** Every row uses a deterministic stub. This shows "
        "that single-component attribution is *measurable*, not where a real "
        "model's errors come from. **No generality claim** is made.",
        "",
        "The dev contexts use a fixed rule prior; the oracle-belief intervention "
        "replaces it with the known scenario start rule (red). The oracle-outcome "
        "intervention replaces VLM outcome predictions with noise-free public "
        "rule semantics. The public-text intervention is currently identical to "
        "baseline because the stub has no image path — a stated limitation, "
        "not a measured zero.",
        "",
        "| Intervention | Mean regret | Error rate | Regret reduction vs baseline |",
        "|---|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result.mean_regret:.3f} | {result.error_rate:.3f} | "
            f"{baseline - result.mean_regret:+.3f} |"
        )
    lines += [
        "",
        f"Largest measured reduction in this synthetic run: **{largest}**. "
        "When all rows are identical the harness is only proving it can run, "
        "not localising any error. With a real VLM this would be the candidate "
        "bottleneck; here it is an "
        "artifact of stub design and must not be reported as a finding about "
        "vision-language agents.",
        "",
        "External MiniGrid port: not attempted. It would require an RGB partial "
        "observation wrapper and a visual encoder that this repository does not "
        "yet have; adding it now would not change any existing result.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
