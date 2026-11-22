# Benchmark Protocol and Preregistration Draft (v0.1)

**Status:** specified. Test scenarios are **not frozen** in Week 1. Nothing in
this document is a result. Numeric targets are budget-planning hypotheses that
must be revised after the Week 2–4 pilot measurements.

---

## 1. Hypotheses

| ID | Testable hypothesis | Independent variable | Primary response |
|---|---|---|---|
| **H1** (diagnostic) | Frozen VLMs make systematic probe-decision errors as probe cost, feedback noise and task-action diagnosticity vary. | cost × noise × diagnosticity | offline action-decision regret; probe-selection confusion matrix |
| **H2** (method) | Structured counterfactual outcome predictions + value-aware selection improve decisions over direct VLM and a DIVA-style adapted baseline under the *same* input and call cap. | inference-time method | held-out mean closed-loop episodic return |
| **H3** (explanation) | Errors localise to perception, belief inference, outcome prediction or action choice. | input mode + diagnostic intervention | conditional errors by stage |
| H4 (optional robustness) | Positive results persist under held-out shift schedules, layouts and a second mechanic family. | held-out scenario/layout/family | paired return difference with interval estimates |

H1 is descriptive and alone implies no novel algorithm. H2 is the main method
claim and requires closed-loop evidence with matched budgets.

---

## 2. Endpoints and accounting

- **Primary endpoint:** mean **total undiscounted episodic return** on a locked
  test distribution under a fixed maximum number of primitive environment
  steps. Higher is better. Reward includes every actual time and probe cost
  once (see `problem_spec.md` §7).
- **Secondary:** offline decision regret against **exact** Q* on microtasks
  only; `Delta_probe` sign classification accuracy; post-shift return (shift
  time evaluator-only); false-probe rate in no-shift episodes; time to first
  post-shift success; probes/episode; primitive steps spent on probes; VLM
  calls, tokens, latency, invalid JSON and invalid actions; outcome-prediction
  accuracy; belief Brier/NLL when labels are available to the evaluator.
- **Cost/call accounting:** every compared method receives the same public
  action affordances, physical step budget, reward definition, observation and
  per-episode VLM-call cap (proposed 4–6). A 3-sample method spends 3 calls.
  Cache hits still record the original generation cost in the manifest.
- **Approximate references** (MCTS, truncated lookahead) must be labelled
  approximate; only enumerated microtasks get exact Q*. "Bayes-optimal regret"
  is used only where the reference is genuinely exact.

---

## 3. Offline microtask regret

For an enumerated microtask with public belief `b` and remaining horizon `h`:

```
regret(h, a) = Q*_h(b, a*) - Q*_h(b, a),   a* = argmax_a Q*_h(b, a)
Delta_probe(b, h) = max_{a in PROBE} Q*_h(b,a) - max_{a in TASK} Q*_h(b,a)
```

Report probe-decision error probability in bins of exact `Delta_probe`, including
a tie/noise band. An agent decision is scored on the same public `b` used by the
reference; if the agent does not emit a belief, score the action only and say
so. Exactness acceptance conditions are in `problem_spec.md` §11.

---

## 4. Splits by scenario (never by frame)

Define a `scenario_id` as the tuple of: family, map/layout, mechanic set,
probe-cost/noise/diagnosticity config, hazard-process seed, shift-outcome seed
and observation-noise seed.

| Split | Purpose | Who may inspect results |
|---|---|---|
| `train` | environment design only; no VLM prompt tuning | designers |
| `dev` | debugging, cost/noise grid, smoke tests | designers |
| `validation` | prompt and method selection, ablations | designers |
| `test_locked` | one-time main evaluation | frozen until run |

Frames from one scenario never cross splits. Exact-solver labels may guide
diagnostic analysis but **must not** be used to optimise test prompts. The
locked test config (`configs/preregistered_test.yaml`) and its Git commit hash
are frozen before any held-out result is inspected. Save an
`experiment_manifest.json` with package versions, model id/revision,
environment revision, prompt hash, generation parameters, seeds, hardware and
compute environment.

---

## 5. Development grid (not the test grid)

`configs/dev.yaml` holds a deterministic development-only grid: probe detour
`[0, 2, 5]` primitive steps, feedback noise `[0, 0.00, 0.10, 0.25]`, task-action
diagnosticity `low/medium/high`, shift condition `no_shift/low_hazard/high_hazard`,
one family in Week 1–2. It includes at least one configuration where direct
task action is predicted preferable and one where probing is predicted
preferable. **These are hypotheses to validate once the environment exists,
not established results.**

Diagnosticity must be quantified *before* looking at VLM results, via a written
expected-information-gain formula under a preregistered reference belief
(currently an open question, see `problem_spec.md` §15).

---

## 6. Sample plan (budget targets, not power guarantees)

- **CPU/symbolic:** 500–2,000 seeded episodes per inexpensive baseline;
  exact microtask enumeration as affordable.
- **Offline VLM pilot:** 50–100 *distinct* decision contexts spanning positive,
  negative and marginal `Delta_probe`, noise and hazard.
- **First closed-loop VLM test:** 25–50 paired scenarios × 2–3 methods, cap
  4–6 calls/episode.
- **Final subset (if compute allows):** ~100–200 held-out paired scenarios;
  e.g. 120 × 3 methods × 4 calls = 1,440 model calls before ablations. Set the
  actual cap after measuring inference speed.

**Statistics:** independent unit is the scenario/layout or seed cluster, never
an individual frame or decision. Report 95% **paired bootstrap** intervals on
`method_return − baseline_return`, resampling at the scenario level. Also
report effect sizes, invalid-output counts and cost per episode. Do not treat a
single positive difference in ~10 episodes as evidence. Before finalising the
sample size, run a variance pilot and predefine a minimal practically
meaningful effect appropriate to the reward scale (e.g. a fixed absolute return
threshold, or a relative threshold only where the baseline is safely away from
zero).

---

## 7. Comparisons to run

- **B0 Direct** — one call, returns action id.
- **B1 Structured belief** — candidate probabilities + action.
- **B2 Predict outcomes** — outcome estimates + action.
- **B3 DIVA-style VoI** — adaptation of DIVA to physical probes; report as
  *adapted*, not an exact reproduction, and include physical time costs.
- **B4 Self-consistency** — multiple samples, all counted in the call budget.
- **B5 Oracle belief** — diagnostic upper bound only, segmented in tables.
- **B6 Non-VLM** — never-probe, random-probe, max-information-gain (evaluator
  reference where labelled), return-aware Bayes reference, true-rule oracle
  (distinct privileged label).
- **M1 Proposed** — VLM belief + counterfactual outcomes + predicted
  continuation value + public costs + deterministic value-aware selector.
- **Ablations:** M1 minus belief / outcome / cost / consistency / selector.

Input modes P0 RGB-public (primary), P1 text-public, P2 oracle-belief and P3
perfect-component interventions are diagnostic only; P2/P3 are never main
agent-valid methods.

---

## 8. Preregistration requirements

Before touching `test_locked`:

1. Freeze `configs/preregistered_test.yaml`, prompt versions, method code,
   call caps and this protocol; record Git commit hashes.
2. Write the primary endpoint, the single H2 comparison of interest, the
   paired-CI procedure, exclusion rules for invalid outputs, and the
   go/no-go pivot criteria (master plan §9) **before** inspecting results.
3. Log all tested prompt variants; tune only on `validation`.
4. On a negative H2 result, preserve the failure taxonomy and pivot to a
   rigorous diagnostic/benchmark paper rather than moving the goalposts.

---

## 9. Planned output figures (no plots exist in Week 1)

1. Mean episodic return by method with 95% paired CI.
2. Probe error probability vs exact `Delta_probe` bins on microtasks.
3. Probe-cost × noise phase diagram, split by family/diagnosticity.
4. Over-/under-probe confusion matrix by method.
5. Closed-loop return difference by family and shift/no-shift.
6. Actual model-call cost vs return (Pareto).
7. Error attribution from RGB / public-text / oracle interventions.

---

## 10. Scientific honesty constraints

No invented results, no fabricated plots, no claim that a paper is publishable
or that a novel method has been demonstrated. Week 1 establishes that the
protocol is unambiguous and testable; evidence comes later.
