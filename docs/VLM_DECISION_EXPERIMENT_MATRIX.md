# VLM Decision Experiment Matrix

**Status:** design-only. No arm has been run.
**Purpose:** compare frozen-VLM decision quality under information access and
response contracts, without changing the reference or the development labels.

## 1. Fixed factors

- Development contexts: 118, from vlm-decision-reference-v1.
- Candidate rules: the six public grid rules.
- Control menu: COMMIT, PROBE_SIGN, PROBE_BLUE. The prompt lists only actions
  feasible in that context.
- Reference: exact finite-horizon cost-aware DP, with tie-aware labels.
- Candidate checkpoint for a later pilot: Qwen2.5-VL-3B, frozen revision
  66285546d2b821cf421d4f5eb2576359d3770cd3, local files only. No checkpoint is
  downloaded or loaded in this phase.
- Decoding: deterministic (do_sample false) for the primary pass. Optional
  sampled replicates are a separate robustness check, not a replacement for the
  primary pass.
- Evaluator-side labels never enter a prompt, image, or cache key.

## 2. The 3 x 3 design

The matrix crosses three information conditions with three response contracts.
Every arm sees the same 118 contexts.

| Arm | Information condition | Response contract | What it tests |
| --- | --- | --- | --- |
| A1 T0-D | text, prior and evidence, no posterior | direct action | baseline text decision |
| A2 TP-D | text, prior and posterior | direct action | value of an explicit belief |
| A3 VP-D | RGB card with posterior | direct action | value of a visual belief |
| A4 T0-R | text, prior and evidence, no posterior | action plus short public rationale | does asking for a reason help or hurt |
| A5 TP-R | text, prior and posterior | action plus short public rationale | reasoning with an explicit belief |
| A6 VP-R | RGB card with posterior | action plus short public rationale | visual reasoning |
| A7 T0-C | text, prior and evidence, no posterior | control plus separate declaration/abstain | calibration without a posterior |
| A8 TP-C | text, prior and posterior | control plus separate declaration/abstain | calibration with a posterior |
| A9 VP-C | RGB card with posterior | control plus separate declaration/abstain | calibration with visual belief |

The declaration field is evaluated only for the C arms. It never changes the
control action or the control regret.

## 3. Why these three information conditions

**T0 (prior-only text).** The model receives the map, candidates, prior,
evidence, costs, budget, horizon, and commit steps, but not the posterior. This
tests whether the model can reason from the raw public facts. It is not a
strawman: the prior and evidence are sufficient to compute the posterior, and a
capable reasoner can do so.

**TP (posterior text).** The model receives the same facts plus the public
posterior. This isolates the decision from belief computation. It is the cleanest
test of probe-versus-commit reasoning given an explicit uncertainty state.

**VP (posterior image).** The model receives the RGB decision card, which
contains the same public facts and posterior as a rendered card. This tests
whether the decision survives a visual interface. The card is self-contained;
the prompt does not repeat the evidence values.

The visual card is not assumed to be equivalent to text. The benchmark uses
paired nuisance and diagnostic cards to measure that gap.

## 4. Why these three contracts

**D (direct).** One JSON object with selected_action. It minimizes interface
demand and makes parsing failures visible. It is the primary decision-quality
comparison.

**R (rationale).** One JSON object with selected_action, a short public
cost/benefit rationale, and confidence. The rationale is scored qualitatively
and is not allowed to introduce hidden facts. This arm tests whether an explicit
reasoning request improves or degrades the decision; it does not reward longer
text.

**C (control plus declaration).** One JSON object containing the control action
and a separate declaration field with DECLARE_RULE or ABSTAIN. The declaration
is scored independently. This arm tests whether the model can calibrate
commitment separately from acting. It is not allowed to use declaration as free
evidence or as an alternative to the control action.

## 5. Prompt and payload contract

Direct arm output:

    {"selected_action": "COMMIT" | "PROBE_SIGN" | "PROBE_BLUE"}

Rationale arm output:

    {"selected_action": "...", "rationale": "short public reason", "confidence": 0.0-1.0}

Control-plus-declaration output:

    {"selected_action": "...",
     "declaration": "DECLARE_RULE" | "ABSTAIN",
     "declared_rule": "<candidate id or null>",
     "confidence": 0.0-1.0}

Rules:

- selected_action must be in the context's feasible_actions.
- declared_rule must be one of the six candidate ids, or null for ABSTAIN.
- confidence is the model's stated confidence, not the evaluator posterior.
- a parser failure is recorded as fallback and does not silently become a
  success.
- no chain-of-thought is requested or stored. The rationale field is a short
  public justification, not hidden reasoning.

The text payload contains only public fields. The image card contains only
public facts; its bottom evidence panel and right candidate panel are generated
from the same context.

## 6. Primary outcomes

Control:

- tie-aware optimal-action accuracy;
- mean and maximum regret;
- over-probe and under-probe rates;
- tool-selection accuracy on probe predictions;
- invalid and fallback rates;
- action distribution, including whether an arm wins by always committing.

Declaration (C arms only):

- coverage;
- accuracy given declared;
- abstention rate;
- 10-bin ECE;
- invalid declaration rate.

Secondary:

- parse validity;
- latency and token counts, recorded only if a later run is authorized;
- nuisance and diagnostic stability;
- per-stratum results for budget exhaustion, high noise, zero noise, and near
  ties.

No composite score is used for the primary decision. A single weighted score
would hide the over-probe/under-probe trade-off and the declaration/control
separation.

## 7. Planned paired controls

- **Base vs nuisance card** for every visual arm on every context. Nuisance
  changes palette and border, not facts. The reference label must be identical;
  any action change is a nuisance sensitivity, not a semantic difference.
- **Base vs diagnostic card** on a prespecified subset of contexts with a blue
  evidence value. Diagnostic changes the observed role, conditions the posterior
  exactly, and produces a new reference label. It is a paired semantic context.
- **Text posterior vs text prior-only** for the same contexts: isolates the value
  of explicit belief.
- **Text posterior vs visual posterior** for the same contexts: measures the
  interface gap without changing the label.
- **Direct vs rationale** for the same contexts: tests whether asking for a
  reason changes the action.
- **Direct vs control-plus-declaration**: tests whether adding a declaration
  field changes control behavior.

## 8. Run accounting (design estimate only)

If a later GO authorizes one deterministic pass:

| Block | Requests |
| --- | ---: |
| Six direct/rationale arms x 118 contexts | 708 |
| Three C arms x 118 contexts (one JSON object each) | 354 |
| Visual nuisance pass for A3, A6, A9 | 354 |
| Diagnostic subset (e.g., 40 contexts x 3 visual arms) | 120 |
| Total design estimate | 1,536 |

These are design estimates, not a compute claim. No request has been issued.
A sampled-seed replication would multiply the request count and is not part of
the primary pass.

## 9. Analysis plan

1. Parse every response with the same public schema and record invalid/fallback.
2. Score control and declaration separately.
3. Report overall and per-stratum results with counts, not only percentages.
4. Compare arms by paired context; do not pool contexts as if independent.
5. Treat 118 development contexts as a development set. Do not compute
   confirmatory p-values or call any difference a confirmed effect.
6. Report exact reference behavior and baseline behavior alongside model arms.
7. Inspect errors by decision class: over-probe on commit-preferred, under-probe
   on probe-preferred, wrong tool, confident wrong declaration, parse failure.
8. Do not select a winner by one aggregate metric. The report must show the
   trade-off between probing and committing and the declaration/control split.

## 10. Abort and interpretation rules for a later run

- If invalid or fallback exceeds 20% for an arm, do not interpret its accuracy
  as decision quality; report the interface failure.
- If an arm's action distribution is at least 95% one action, label it as a
  degenerate policy rather than a decision-maker.
- If the visual nuisance control changes actions materially more often than the
  text control changes with a cosmetic prompt change, report visual
  instability; do not average it away.
- If a diagnostic semantic change does not change the reference label, the
  diagnostic is invalid and must not be counted as a semantic control.
- If a model beats always-commit only by probing more, check whether it also
  reduces under-probe without increasing over-probe. A raw accuracy gain is not
  enough.
- No result from this matrix is confirmatory, regardless of effect size.

## 11. What is deliberately absent

- No model inference in this phase.
- No prompt tuning against development labels.
- No reserved maps 210-239.
- No automatic merge.
- No v0.6 symbolic arm.
- No use of declaration accuracy as a control reward.
- No hidden chain-of-thought collection.

A separate human GO is required before any request in this matrix is sent to a
model.
