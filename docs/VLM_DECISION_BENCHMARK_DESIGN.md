# VLM Decision Benchmark Design

**Status:** CPU infrastructure and development design only.
**Reference version:** vlm-decision-reference-v1.
**Inference status:** no VLM was loaded, called, or evaluated in this phase.

## 1. Research question

Can a frozen vision-language model make effective uncertainty-aware
**probe-versus-commit** decisions under a constrained information budget?

The target is decision quality, not symbolic solver accuracy. A good arm should
spend a probe when the expected value of the resulting information exceeds its
cost, commit when it does not, and avoid both reflexive probing and premature
commitment. Rule declaration is a separate action family and is scored
separately.

## 2. Why this is a new benchmark

The closed symbolic phase optimized identification over a fixed public map with
a large fixed tool budget. In v0.5, the active-request arm produced no observed
behavioral effect, and the remaining failures included confident wrong terminal
MAPs and unresolved declarations. A larger symbolic candidate is not the right
next experiment.

This benchmark isolates a smaller decision problem:

- a public map, six candidate rules, and a public posterior;
- a small set of evidence tools with explicit costs;
- a total tool-unit budget and a probe horizon;
- one commit decision with a public MAP-implied goal target;
- an optional, separately scored declaration decision.

## 3. Public context

A context exposes only public facts:

- the public grid map, start position, and marker locations;
- the six candidate rule ids and descriptions;
- the public prior and, when the arm is posterior-conditioned, the public
  posterior;
- observed sign and blue-cell evidence, including probe noise;
- tool costs, total budget, spent units, remaining budget, and horizon;
- feasible high-level actions;
- commit step counts and the reward/cost constants.

The evaluator-side label file is not part of the context. It contains the exact
reference values, the MAP-implied commit target, a sampled latent rule for
declaration scoring, and the reference declaration gate. It must never appear in
a prompt, image, cache key, or public JSONL row.

## 4. Actions

Control actions:

- **COMMIT**: navigate to the red goal of the posterior-MAP rule and execute
  that rule's goal target. The commit target is public; the true rule is not.
- **PROBE_SIGN**: read the signpost face adjacent to the start. On the current
  layouts this is noise-free and reads the clause for side 0. A sign can be
  uninformative when all surviving candidates emit the same clause.
- **PROBE_BLUE**: inspect the blue cell. The observation is the cell role
  (goal, hazard, or safe) with inspection error epsilon. The likelihood is
  one minus epsilon for the true role and epsilon/2 for each other role.

Declaration actions are scored separately:

- **DECLARE_RULE**: name one of the six candidates with a confidence.
- **ABSTAIN**: decline to name a rule.

Declarations do not enter the control value, do not generate free evidence, and
do not change the posterior. They are a measurement of calibrated commitment,
not a substitute for a control action.

## 5. Exact reference

The reference is a finite-horizon cost-aware expected-value dynamic program over
the public posterior, used tool units, remaining budget, probe horizon, and
inspection noise. It is exact for the stated macro model.

At a state with posterior p, remaining budget b, used-tool flags, and k probes
left:

- The commit value is the expected payoff of the MAP-implied target minus the
  step cost.
- The sign probe value is minus its cost plus the expectation over its possible
  clauses of the value after conditioning.
- The blue probe value is minus its cost plus the expectation over its possible
  observed roles of the value after conditioning.
- The state value is the maximum of commit and the feasible probes.
- A probe is feasible only when it is unused and its cost is at most b.
- The root comparison uses the same recurrence with k = max_probes.

The detailed recurrence, likelihoods, tie rules, and numerical conventions are
in `docs/VLM_DECISION_REFERENCE.md`.

## 6. Development layouts and context census

Three public 7x7/6x6 layouts are used. All have a public signpost, a blue marker,
and red/green markers. All three expose sign side 0 from the start position.
The maps are development fixtures, not reserved evaluation maps.

Eleven prior templates are crossed with four evidence ledgers:

- no evidence;
- one of the distinct side-0 sign clauses;
- one of the distinct blue roles;
- sign plus blue.

Four cost/budget regimes are used:

| Regime | Sign cost | Blue cost | Budget | Purpose |
| --- | ---: | ---: | ---: | --- |
| default | 1 | 2 | 3 | both probes often possible |
| cheap | 1 | 1 | 2 | both probes possible at low cost |
| expensive | 2 | 3 | 4 | one probe, or a constrained second probe |
| exhausted | 1 | 2 | 0 | commit is the only feasible action |

Four inspection-noise levels are used: 0.00, 0.05, 0.15, and 0.30. The sign
clause itself is noise-free by design.

The full cross-product contains 3,399 unique contexts after exact-likelihood
filtering and duplicate removal. A deterministic hash-ranked selector chooses
118 development contexts with required coverage of budget exhaustion,
task-ambiguity, commit preference, sign preference, blue preference, near ties,
high noise, and zero noise.

Current library manifest:

| Field | Value |
| --- | --- |
| Contexts | 118 |
| Decision classes | commit_preferred 76, near_tie 21, probe_preferred 20, tie 1 |
| Strata | budget_exhausted 8, ambiguous_task_solvable 8, commit_preferred 44, probe_sign_preferred 14, probe_blue_preferred 14, near_tie 14, high_noise 8, zero_noise 8 |
| Cost/budget groups | 1/1/2: 59, 1/2/0: 3, 1/2/3: 39, 2/3/4: 17 |
| Noise groups | 0.00: 28, 0.05: 27, 0.15: 38, 0.30: 25 |
| Layouts | layout-a 49, layout-b 36, layout-c 33 |
| Exact best actions | COMMIT 81, PROBE_SIGN 23, PROBE_BLUE 14 |

Decision-class and best-action counts differ because a near tie can still have a
single deterministic best action. The reference chooses COMMIT when values are
tied within 1e-9, then PROBE_SIGN, then PROBE_BLUE. The library uses a 0.25
near-tie margin to keep ambiguous cases visible.

## 7. Reference-label coverage

Every selected context has:

- q_commit and, when feasible, q_sign and q_blue;
- the exact best value and the full tie-aware optimal-action set;
- a deterministic best action and a decision class;
- a MAP-implied commit rule, target, and step count;
- a sampled latent rule for declaration scoring;
- a reference declaration gate (threshold 0.90, margin 0.10);
- posterior entropy and a declaration-correctness flag when declared.

The sampled latent rule is evaluator-side. It is drawn deterministically from
the public posterior for fixture purposes. It is not a real environment truth
and it is not evidence that a model recovered the true symbolic rule.

## 8. CPU baselines (development set, n = 118)

| Baseline | Optimal-action accuracy | Mean regret | Notes |
| --- | ---: | ---: | --- |
| always_commit | 0.6864 | 0.4863 | under-probes on 31.4% |
| always_probe | 0.6949 | 0.2698 | over-probes on 28.0% |
| random_valid | 0.6949 | 0.3362 | random feasible action per context |
| information_gain | 0.9068 | 0.0429 | public entropy-per-cost heuristic |
| exact reference | 1.0000 | 0.0000 | sanity check, not a model baseline |

The reference-declaration gate has coverage 0.3729, 44 declared, 42 correct,
accuracy 0.9545, and 10-bin ECE 0.0252. These are evaluator fixture properties,
not model results. The gate's accuracy is not 1.0 because its latent rule is a
posterior draw, not a deterministic oracle.

## 9. Visual condition

The RGB card is 512x320: the map occupies the left 224x224, an evidence/budget
panel sits below it, and a candidate-rule panel occupies the right side. The
card carries the public map, all six rule descriptions, prior/posterior bars,
observed evidence, remaining budget, costs, error rate, and horizon. It is
self-contained and deterministic.

Two controls are separate:

- **Nuisance**: the same public facts in a different palette and panel border.
  It must not change the reference label or evidence ledger; only pixels change.
- **Diagnostic**: a semantic paired context with a different observed blue role,
  a freshly conditioned posterior, and a fresh reference label. It is a new
  context, not a cosmetic edit.

The text condition receives the same facts as a text payload. The RGB condition
receives an image. The benchmark does not claim that the two are pixel-perfect
equivalents; the pairing is at the public-fact level and the visual controls make
that assumption testable.

## 10. Leakage boundary

The following are forbidden in prompts, images, cache keys, and public JSONL:

- q_commit, q_sign, q_blue, best_value;
- optimal_actions, best_action, decision_class, decision_margin;
- latent_rule, declaration_truth, declaration_correct;
- stratum and any selector identity;
- evaluator internals, simulator transitions, or reserved-map identifiers.

Public contexts pass a field-level scanner before writing. The cache accepts
only public request material. Tests must fail if a privileged key or value is
serialized into a public row.

## 11. What this design does not establish

- It does not establish that any VLM can make these decisions.
- It does not establish that text and RGB conditions are equivalent.
- It does not establish a confirmatory effect or a population estimate.
- It does not use reserved maps 210-239.
- It does not support a v0.6 symbolic candidate.
- It does not justify automatic merge or promotion.

The next step is a feasibility decision packet, not a model run.
