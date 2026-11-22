# RuleShift / Agentic VLM — Integrated Execution Plan for a Coding Agent

**Status:** Implementation directive  
**Date:** 2026-11-22  
**Primary research track:** Value-Aware Probing under Hidden Dynamics Shifts  
**Secondary reference only:** DISS-Switch / Disentangled Transition-Surprise Gating  
**Planning horizon:** 12 weeks, milestone-based rather than deadline-guaranteed  
**Resources:** CPU-first; optional limited free Kaggle/Colab GPU; optional one 24 GB GPU. No assumption of guaranteed cloud GPU quotas.

> **Read this entire document before changing the repository.** This directive integrates *selected robustness and compute-discipline ideas* from the DISS-Switch plan into the existing RuleShift research project. **It does not authorize replacing the project with DISS-Switch, implementing the 44-week roadmap, or changing the primary research question.**

## 0. Source-of-truth hierarchy and working rules

If these files exist in the repository, read them in order:

1. `RuleShift_VLM_Research_12_Week_Master_Plan.md` — authoritative scientific design and 12-week scope.
2. `AGENT_TASK_WEEK1_RuleShift.md` — authoritative first-week task and acceptance tests.
3. **This file** — integration instructions for selected DISS-Switch ideas; it clarifies scope but must not override the primary RQ, data permissions, evaluation integrity, or first-week acceptance criteria.
4. Original DISS-Switch 44-week document — **literature/backlog reference only**. Store as `research/backlog/DISS_Switch_44Week_Reference.md` if provided and permitted; preserve its original content, clearly labeled *not executable*.

If the original source files are not present, this document is sufficient to establish project intent. Do not invent a missing document or claim to have read it. Do not silently overwrite a user's existing experiments or configuration. Record proposed changes in an implementation log.

**Agent operating protocol:** implement only the tasks for the current approved milestone, run the listed tests, and report evidence. Preserve independent experiments in feature branches or small commits. Never mark an objective complete solely because code compiles. Do not fabricate paper findings, benchmark gains, published deadlines, GPU profiles, or citations.

---

## 1. The scientific question — FROZEN

**Working title**

*Do Visual Agents Know When an Experiment Is Worth Its Cost? Task-Aware Probing under Hidden Dynamics Shifts*

**Primary research question**

> When hidden environment dynamics change without warning, do frozen vision-language agents choose probes whose control value justifies their cost, and can structured action-outcome reasoning — **without access to the environment's generative model** — reduce probe-decision regret and improve closed-loop return?

**Terminology note:** prospective action-outcome prediction (before acting) and retrospective *individual counterfactual inference* (after an observed action, holding latent disturbances fixed) are different. In code, logs, prompts, and manuscripts, do not use `counterfactual` interchangeably with ordinary action-conditioned prediction. The working title and scope remain unchanged; this wording clarifies rather than changes the RQ.

### Testable hypotheses

- **H1 — Decision behavior:** Frozen VLMs make measurable over-probing, under-probing, or wrong-probe decisions as probe cost, feedback noise, and diagnosticity of ordinary task actions change.
- **H2 — Method:** Structured belief + action-outcome prediction + cost-aware selection improve **held-out closed-loop episodic return** over direct VLM and a strong decision-aware/VoI prompting baseline under matched resources.
- **H3 — Diagnosis:** Perception, belief inference, outcome prediction, and action selection can be evaluated separately; failures should not be attributed to the wrong stage.
- **H4 — Robustness:** Any H2 improvement remains meaningful when visually irrelevant changes, transient stochastic noise, persistent mechanic shifts, and no-shift episodes are separated and tested.

**What this paper is NOT about:** optimizing *when to call a slow VLM using learned transition surprise*; training a new world model; making a PPO agent quickly adapt; open-set discovery of arbitrary unknown rules. These are potential later studies.

### Non-negotiable constraints

1. **No privileged generative model on the agent side.** Agent sees only public observation, public action menu, actual interaction history, candidate-mechanic descriptions, and public costs.
2. **The Bayesian reference belongs exclusively to the evaluator.** Exact optimality can be claimed only on microtasks actually solved exactly. Short-horizon search/MCTS on larger tasks is an approximate reference.
3. **Closed-loop return is the main endpoint.** Offline action-value diagnostics help explain results but cannot substitute for closed-loop gains.
4. **All physical consequences count.** Probe moves, travel, retries, lockouts, waits, and consequences of failed actions consume primitive environment steps and enter one consistent reward ledger. Do not double count costs.
5. **Fair baselines:** same admissible observations and action affordances; same primitive-step horizon, test scenarios, and VLM-call cap; disclose *actual* calls/tokens/latency.
6. **No training a large VLM** in the first paper. Frozen VLM inference only; no long PPO runs, JEPA training, or online neural distillation.
7. **No preselected success.** Freeze a locked test distribution before method comparison; calibrate on development/validation only. Report negative controls and negative results.

---

## 2. What to take from DISS-Switch — explicit integration matrix

| DISS-Switch component | Decision | Implementation location / condition | Why |
|---|---|---|---|
| Four controlled regimes (stationary, visual-only shift, transient/noisy, persistent mechanic shift) | **ADOPT** | Environment tests and evaluation protocol, once basic family A works | Separates causal rule changes from appearance and observation noise |
| Strict per-episode VLM-call budgets; tokens, wall time, peak VRAM monitoring | **ADOPT** | Inference runner, starting VLM pilot | Necessary for a reproducible limited-compute study |
| Deterministic full-context VLM response caching | **ADOPT** | Inference runner | Reduces repeated calls for unchanged prompt/context/model settings |
| Expected-vs-observed transition summaries and rule-consistency checks | **ADOPT AS ABLATION** | Structured reasoning experiments after baseline VLM pilot | Tests reliability without granting oracle simulator access |
| Code-grounded / symbol-neutral prompts | **OPTIONAL ABLATION** | Only if a pilot demonstrates semantic-prior bias | May reduce reliance on inappropriate prior game semantics |
| Transition residual + CUSUM gate | **DEFER TO EXTENSION B** | Only after primary H2 closed-loop result | Tests a *different* hypothesis: timing of VLM calls |
| Learned perception-invariant encoder, heteroscedastic world-model ensemble | **DEFER** | Separate later proposal | Requires representation-learning validation and training |
| PPO, constant online model adaptation, replay quarantine, residual dynamics adapter | **OUT OF SCOPE** | No code for paper 1 | Introduces confounded adaptation mechanisms and heavy compute |
| Dyna rollouts / few-shot distillation | **OUT OF SCOPE** | No code for paper 1 | Reliant on learned simulator and gradient updates |
| Craftax + AutumnBench + BALROG + GameWorld all in one plan | **DO NOT INTEGRATE** | At most one carefully chosen external check after core result | Excessive engineering and experimental cost |
| Fixed 44-week GPU-hours and predicted submission deadlines | **REFERENCE ONLY** | No booking or promises | Hardware quotas and conference CFPs require independent verification |

**Priority allocation:** dedicate the overwhelming majority of near-term engineering to the primary RQ; the DISS-derived robustness controls should support rather than replace that research.

---

## 3. System architecture and trust boundary

```text
                      PUBLIC / AGENT-SIDE

 Public RGB observation ─┐
 Legal macro-actions ────┤
 Public mechanic text ───┼──> Frozen VLM / baseline advisor
 Action–outcome history ─┘          │
                                  ├─ optional belief estimate b_hat(m)
                                  ├─ predicted action outcomes
                                  └─ predicted continuation values
                                          │
                           cost-aware probe-vs-act selector
                                          │
                                legal MACRO-ACTION
                                          │
                                public macro executor
                                          │
                                actual env transitions
                                          │
                                     next history

              ═════════════ STRICT ACCESS BOUNDARY ═════════════

                       EVALUATOR / OFFLINE ONLY
 True mechanic / shift time / RNG disturbances / transition kernel
 Exact microtask Bayes solver / optional approximate planner
 Ground-truth outcome labels / regret / metric computation
```

**Special rule for macro-actions:** scripts may use *public static geometry* and publicly observed state for navigation, but may not inspect hidden mechanics, secretly condition on shift time, or take a shortcut directly to an outcome. Every primitive `step` made by the macro-executor must be recorded, consumed from the same horizon, and scored under the same reward function.

**Recommended public observation contract:** image/frame or legitimate nonprivileged symbolic view, valid macro-action menu, last observed reward, done/truncated status, current *observable* position if accessible to all agents, and public candidate descriptions. Do not expose `mechanic_id`, `shifted`, `shift_time`, transition probability arrays, posterior computed using hidden variables, or optimal Q-values.

**Structured method (hypothesis to evaluate, not a solved algorithm):**

1. VLM estimates candidate rule belief `b_hat(m)` from **only** public history.
2. VLM predicts action-conditioned outcomes for a small common action menu and, where feasible, future task implications.
3. A deterministic **history-consistency** check flags predictions/rules contradictory to actually observed transitions; it does not query the simulator to test hypothetical actions.
4. A cost-aware selector ranks probe-vs-act choices using predicted near-term reward, predicted continuation value after possible outcomes, and real macro cost. **Do not use just immediate reward or information gain.**
5. Execute only the selected legal action, observe reality, update history, and repeat.

The evaluator's Bayes posterior or exact `Q*` must never leak into steps 1–4. An `oracle-belief` input mode is allowed solely as a separately labeled diagnostic upper bound, never as the headline result.

---

## 4. Environment specification and four regimes

### 4.1 MVP environment families

**Family A — Gate/Lock (mandatory first):** repeated delivery with three candidate accepted tokens or cards. Unannounced within-episode mechanic shift, including no-shift cases. Directly trying the gate can both make progress and reveal the rule; visiting a separate probe station can yield information while incurring detour cost. Wrong gate use may cause a lockout. Include settings where probing is economically optimal *and* settings where direct attempts dominate.

**Family B — Hazard/Bridge (after Family A):** a hidden stable/slippery or safe/hazardous transition regime; a diagnostic interaction can reduce costly route mistakes. Ensure the mechanism truly changes transition dynamics, not merely visual color or reward labels. Limit to two or three candidate regimes initially.

**Microtask:** a very small, exactly enumerable subproblem for computing reference action values and `Delta_probe`; the larger closed-loop gridworld does **not** automatically inherit the term Bayes-optimal.

### 4.2 Robustness regimes — adopted from DISS-Switch

Use the *same reward and action accounting* across all regimes. Distinguish perturbation type from the probe-cost/noise/diagnosticity parameter grid:

| Regime | What changes | Hidden mechanics? | Key test |
|---|---|---|---|
| R0 — Stationary / no shift | Nothing | Unchanged | Avoid unnecessary probes or advisor calls |
| R1 — Visual-only perturbation | Palette/background/irrelevant sprite appearance | Unchanged | No false inference that rules changed |
| R2 — Transient noise / perception error | Occlusion, one-off failed gate feedback, slip, frame corruption | Unchanged or transiently disturbed | Distinguish single anomaly from persistent rule change |
| R3 — Persistent unannounced mechanic shift | Gate binding, route/interaction mechanics | Changes without visible announcement | Recover behavior and return after new evidence |

Do not assume that an isolated noise event must never rationally motivate probing; whether probing is worth its cost depends on belief, horizon, uncertainty, and cost. Test at least one instance where probing is *not* beneficial.

### 4.3 Independent experimental axes

- Probe costs / detours: tune examples such as `0`, `2`, `5` primitive steps on development only.
- Observation feedback noise: examples `0%`, `10%`, `25%`.
- Task-action diagnosticity: low, medium, high (measured operationally, not a decorative scenario label).
- Shift status/schedule: no-shift vs unannounced shift; no shift-time feature in public observation.
- Layout and mechanics combinations: separate development, validation, and locked test scenarios.

The four regimes are **additional controls**, not a substitute for the scientific phase diagram. Make matched comparisons on the same scenario IDs; report results by regime and as predeclared aggregates.

### 4.4 Stochastic test integrity

- Use semantic-event-keyed random streams or predetermined exogenous event traces. A global mutable RNG can make two policies experience incomparable noise simply because they took different actions.
- Never call the evaluator's hidden `shift_time` to select public prompts or agent interventions.
- A deterministic shift test must verify that visual-only changes do not modify transitions, and mechanic shifts do not create unintended visual announcement before affected interaction.
- An episode that terminates before the shift is **not** a post-shift recovery sample; predeclare handling rules.

---

## 5. Bayesian references and fair evaluation

**Exact microtask solver:** enumerate the full finite-horizon belief-state decision process only for tractable instances; verify against independent brute-force enumeration on very small cases. The agent never calls it.

A diagnostic *probe advantage* can be defined as:

```text
Delta_probe(b, s) = max_a∈A_probe Q*(b,s,a) - max_a∈A_task Q*(b,s,a)
```

Here `Q*` must include *immediate task reward, actual action/probe cost, and expected future value after the resulting observation*. Avoid double-counting cost when it is already embedded in environment reward. For a partially observed physical state, use the appropriate joint belief rather than treating `b(m)` as sufficient by assumption.

**Large gridworld:** use a clearly labeled *approximate Bayes reference* or bounded solver if helpful. Separately report privileged-mechanic oracle as an **upper-bound comparator** if implemented. Never conflate these references.

**Main metric:** mean total undiscounted closed-loop episodic return, fixed primitive-step horizon, locked test distribution, with paired uncertainty intervals over *independent episode/scenario units*.

**Supporting diagnostics:** exact microtask action-choice regret (where defined), sign of `Delta_probe`, over-/under-probing, post-shift return using shift time only in evaluator, time to recovery, steps consumed by probing, calls/tokens/latency/VRAM, belief quality, outcome prediction quality, and invalid action/JSON rate.

**Offline and closed-loop are distinct:** offline decision audit uses identical histories; closed-loop evaluation must allow methods to generate different trajectories, while keeping scenario-level seeds/exogenous conditions comparable.

**Baselines, in build order:**

1. Passive / never explicitly probe (still updates beliefs from ordinary task outcomes).
2. Random and fixed probing with the same physical costs.
3. Maximum expected information gain (does not directly optimize return).
4. Exact Bayes on microtasks / approximate Bayes where appropriate, evaluator-side only.
5. Direct frozen VLM with the same legal actions, observations, call budget.
6. Strong decision-aware / DIVA-style prompting adaptation (label as **inspired baseline** unless independently reproduced faithfully).
7. Structured-belief prompt; outcome-prediction prompt; cost-aware selector; controlled component ablations.
8. Optional self-consistency at **measured equal call/token cost**, not an unfair extra-compute treatment.

Do not compare a method that gets a probe result with a baseline prevented from updating its beliefs on task outcomes.

---

## 6. Inference and compute discipline — adopted from DISS-Switch

### 6.1 Cache correctness

Cache a frozen-VLM response *only when the full semantically relevant input and inference settings are identical*. Recommended key components:

```text
(model_id, exact_model_revision, quantization, decoding_params,
 prompt_template_version, candidate_mechanic_text, ordered_public_history,
 exact_preprocessed_image_bytes_or_hashes, legal_action_menu)
```

Use a canonical serialization and SHA-256 hash, with SQLite or content-addressed files. Store cache schema version, model revision, output JSON, token usage if available, measured latency, and run ID. Cached records are immutable; invalidate on prompt/observation preprocessing/model change. **Never include evaluator-only secrets in public prompts or model-visible metadata.** Cache may reduce repeated inference for offline replays but cannot automatically replay a response for a diverged closed-loop history.

### 6.2 VLM scheduling and macro-action scope

- Call the frozen VLM at *decision points* using the same macro-action abstraction for compared methods; low-level moves run in the environment and cost primitive steps.
- Define one common per-episode cap (pilot may try 4–6 calls, but profile actual expense), plus input/output token limits; record rejected/invalid responses as actual calls.
- Baselines need the same cap and opportunity to use their own permitted actions. Do not force them to exhaust a call quota.
- Profile GPU memory and latency empirically before choosing a model; do not assume Kaggle has contiguous 32 GB across two T4s.
- A model or hardware setup that does not fit must trigger fallback to smaller/quantized inference or smaller controlled pilot, not abandonment of the CPU baselines.

### 6.3 Input-format and reliability ablations

Compare only after the direct-VLM pilot:

- Natural-language rule descriptions vs neutral/object-ID wording (possible semantic-prior bias; do not assert it is always helpful).
- Plain action request vs explicit **predicted outcomes under each candidate rule**.
- Recent `expected → observed` transition summary vs no summary.
- History-consistency filter vs no filter; the filter must use **observed history only**.
- Frozen VLM with agent-estimated belief vs *diagnostic-only* evaluator-provided exact belief. Clearly label privilege.

No learned prediction-error model is required to produce `expected → observed` comparisons: the prediction may come from the agent's own earlier output and the observation from the real environment.

---

## 7. Mandatory acceptance tests

Implement tests as appropriate to each milestone. Examples:

```text
[PERMISSIONS]
- Public observation and model prompt contain no true_rule, mechanic_id, shift_time, hidden RNG, oracle Q*.
- Agent package cannot import evaluator or privileged transition implementation.
- Diagnostic oracle-belief mode cannot appear in primary benchmark runs.

[ENVIRONMENT]
- R0 and R1 have identical hidden transition mechanics under matched exogenous events.
- R1 changes appearance only, not game dynamics or reward.
- R2 transient perturbation does not silently become a permanent mechanic change.
- R3 hidden shift is not announced in the initial observation or prompt.
- No-shift episodes are genuinely included in test distribution.
- Macro navigation, probe use, lockouts and waits consume primitive steps and reward consistently.
- Both probe-favored and probe-disfavored dev settings exist (verify with reference, do not assume).

[REFERENCE]
- Exact microtask solver agrees with exhaustive enumeration on tiny instances.
- Delta_probe accounts for continuation value and cost exactly once.
- Large-environment approximate solver is not mislabeled Bayes-optimal.

[EVALUATION]
- Paired scenario IDs are fixed across methods; exogenous events are reproducible.
- Offline and closed-loop metrics are computed separately.
- Test split is immutable during method tuning; validation controls design.
- Every VLM response is parse-validated and cost-accounted, including failures.

[CACHE]
- Same full public context + same config => cache hit.
- Any change to history, images, prompt revision, model revision, or sampling config => cache miss.
- Cache never changes the response an equivalent uncached call would be requested to produce
  (subject to stochastic sampling; seeded sampling or stored samples must be handled explicitly).
```

No fabricated positive curves to satisfy acceptance tests: tests verify contracts; experiments determine whether hypotheses hold.

---

## 8. Integrated 12-week schedule (one milestone at a time)

| Week | Core tasks | DISS-derived integration | Evidence to deliver / exit condition |
|---|---|---|---|
| **1** | Literature matrix; `problem_spec.md`; agent/evaluator permissions; benchmark protocol; CPU repo tests | Add 4-regime spec and compute/caching requirements **as design only** | Docs and schema tests pass; novelty threats explicitly logged |
| **2** | Family A CPU environment, macro executor, shift/no-shift, reward and RNG tests | Implement R0–R3 support incrementally after basic R0/R3 | `pytest -q`, env checker, full trajectory ledger |
| **3** | Bayesian filtering, exact microtask solver, `Delta_probe`, passive/random/info-gain baselines, phase diagram | Evaluate R1/R2 negative-control integrity | Exhaustive-reference cross-check; both pro-/anti-probe dev cases |
| **4** | Frozen VLM 50–100-case offline pilot; RGB, public history, direct action decisions | Full-context cache, bounded tokens/calls, latency/VRAM logs | Structured output logs; over-/under-probing audit |
| **5** | Structured belief and action-conditioned outcome prediction; consistency checks; cost-aware selector | Optional expected-vs-observed / symbol-neutral prompt ablation | Method code and clean component-level validation |
| **6** | DIVA-style and other strong baselines; ablations; preregister tests | 4-regime evaluation grid and immutable test manifests | Locked test config with commit hash; no leakage |
| **7** | Closed-loop Family A, paired scenario evaluation | Record R0–R3 outcomes, calls and cost | Main endpoint and uncertainty estimates on Family A |
| **8** | Family B Hazard/Bridge; tests; cross-family generalization | Apply R0–R3 if mechanics support them meaningfully | Second-family held-out results, no arbitrary shortcut |
| **9** | Full closed-loop held-out runs; compare method vs baselines | False-positive probes under R1/R2, recovery under R3 | Paired CIs, honest negative results, cost/return trade-off |
| **10** | Failure taxonomy: perception vs belief vs prediction vs selector | Focused reliability-ablation checks; optional one external env | Error decomposition supported by logged traces |
| **11** | Repeat independent seeds; missing ablations; reproducibility from clean checkout | Confirm cache correctness and hardware claims | Repro scripts, test suite, config snapshots |
| **12** | Draft paper and limitations; package code/data/figures | DISS-Switch backlog memo, not an implemented second paper | Full draft, reproducible artifact, updated go/no-go decision |

The schedule is an aspiration. If a scientific gate fails, stop feature expansion, diagnose, and document a controlled pivot. Do not invent a conference submission date.

---

## 9. Go / no-go decisions

### Gate A — Before any VLM work (end of Week 3)

- Family A can generate distinguishable mechanics without privileged information leakage.
- Exact small reference passes independent checks.
- Development test cases include **both** probe-optimal and probe-dominated regions under fixed reward accounting.
- Passive and return-aware reference behave logically.

**If not:** repair environment and the decision problem. Do not compensate by training PPO or adding a VLM.

### Gate B — After frozen VLM pilot (end of Week 4)

- Model inputs, outputs, cache entries, and resource consumption are reproducible.
- Offline errors are measurable; if model seems near-optimal on all cases, inspect task difficulty and baseline strength before drawing conclusions.
- The failure taxonomy identifies whether subsequent method work is worth attempting.

**If not:** reduce pilot size or fix input semantics; avoid expensive scaling.

### Gate C — Before locked test (end of Week 6)

- Proposed method can be compared fairly with direct VLM and DIVA-style prompting.
- Any promising improvement is measured on validation **without** evaluator belief or simulator calls.
- Test scenario list and primary endpoint are frozen with a recorded manifest hash.

**If not:** run targeted ablations or pivot to a diagnostic paper. Do not tune on the locked test.

### Gate D — After closed-loop (end of Week 9)

- Report whether gains hold, vanish, or reverse in closed loop.
- Reconcile physical costs, VLM calls/tokens, random seeds, and uncertainty intervals.
- Check whether improvements persist in R0/R1/R2/R3 rather than hiding bad controls in an aggregate.

**If not:** publish/record a transparent failure study if scientifically meaningful; do not fabricate an adaptation claim.

---

## 10. DISS-Switch backlog — NOT authorized in main implementation

Store as research ideas / GitHub issues labeled `future-research`, **not** work in the sprint backlog:

**Extension B — Selective slow-model invocation:** after H2 has a credible positive closed-loop signal, hold the advisor and action-outcome method fixed; independently compare random/fixed, policy confidence/entropy (if a probabilistic policy exists), and transition-residual/CUSUM timing under matched VLM calls. Avoid comparing a new gate+advisor combination against a weaker baseline with both changed.

**Future paper — Learned robustness and adaptation:** train an appearance-robust latent encoder, estimate dynamics residuals with suitable noise controls, examine false alarms under visual shifts, and test optional online adapters or distillation. First establish causal benefit and cost of probes; do not pre-commit to gradient updates or a specific architecture.

**Future external benchmarks:** evaluate one appropriate outside environment after scientific value is clear. Craftax, AutumnBench, BALROG, and GameWorld are possible candidates, **not** promised deliverables; verify each environment's licensing, observation/action interface, mechanics-shift ability, and resource budget before integrating.

---

## 11. Repository deliverables / file layout (suggested)

```text
README.md
pyproject.toml
RuleShift_VLM_Research_12_Week_Master_Plan.md
AGENT_TASK_WEEK1_RuleShift.md
RuleShift_Agent_Integrated_Execution_Plan.md
research/
  literature_matrix.csv
  backlog/
    DISS_Switch_44Week_Reference.md          # optional, non-executable
  decisions.md                              # scientific and scope decisions
  implementation_log.md
  experiment_registry.csv
  preregistration.md
  results/                                   # generated, never fabricated

docs/
  problem_spec.md
  agent_evaluator_permissions.md
  benchmark_protocol.md
  week2_issues.md

src/ruleshift/
  envs/                                     # private hidden state implemented here
  public_api/                               # sanitized observations, legal actions
  agents/                                   # no env/evaluator imports
  macro_actions/                            # public, paid primitive-step executor
  reference/                                # evaluator-only exact/approx planners
  eval/                                     # privileged metrics and paired scenarios
  inference/                                # frozen VLM, parse validation, cache

configs/
  dev.yaml
  validation.yaml
  locked_test.yaml                          # created/frozen only at planned stage

scripts/
  smoke_env.py
  run_microtask_reference.py
  run_phase_diagram.py
  pilot_vlm.py
  run_closed_loop.py
  summarize_results.py

tests/
  test_permissions.py
  test_env_regimes.py
  test_macro_costs.py
  test_reference_exactness.py
  test_scenario_pairing.py
  test_vlm_cache.py
```

File names indicate target structure; **do not create empty fake artifacts just to satisfy a tree listing**. Build incrementally and mark unimplemented targets accurately.

---

## 12. Start now — current coding-agent assignment

**If you are starting at Week 1, work ONLY on the following:**

1. Check repository state; read the master 12-week plan and Week 1 task if available.
2. Update `docs/problem_spec.md` and `docs/benchmark_protocol.md` to include **R0–R3 controlled regimes** as clearly separated robustness conditions, plus cost/noise/diagnosticity axes. Preserve the original RQ and main metric.
3. Write `docs/agent_evaluator_permissions.md`: list all public fields vs forbidden hidden fields, including macro-executor and VLM-cache handling.
4. Add a small `docs/compute_and_cache_policy.md`: full-context cache keys, call/token caps, measured not assumed VRAM, model fallback strategy.
5. Create `research/backlog/README.md` explaining why DISS-Switch is deferred; optionally copy supplied original reference into this folder unchanged.
6. Add/update Week 1 schema/permissions tests that run on CPU; do not start PPO, CUSUM, JEPA, world-model ensembles, or VLM inference yet.
7. Run `pytest -q` (or document the exact setup blocker). Provide a truthful implementation report with files modified, tests run and results, unresolved choices, and the next milestone.

### Definition of done for this assignment

- [ ] Primary RQ, ownership of evaluator reference, and scientific scope preserved.
- [ ] Four regimes clearly specified and tested at **schema/contract level** (implementation follows in Week 2).
- [ ] Agent/evaluator information leakage constraints documented and checked.
- [ ] Evaluation distinguishes cost accounting, VLM-call budgeting, and action-selection quality.
- [ ] DISS-Switch is labeled future/backlog; no forbidden modules were introduced.
- [ ] Documentation and tests are internally consistent; commands and observed results are included in the handoff report.

**Response format after implementation:**

```text
Milestone:
Files created/modified:
Tests run and exact outcomes:
What works end-to-end:
What is only specified (not implemented):
Information-leakage checks:
Compute/cost assumptions still unverified:
Scientific issues / potential pivots:
Next authorized step:
```

**Final instruction:** Do not optimize a slow-model trigger until the primary experiments demonstrate that probing changes the agent's behavior and can improve real closed-loop task return under honest resource and interaction costs.
