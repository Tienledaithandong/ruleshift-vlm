# Declaration failure diagnosis

**Status:** Phase 0.5 diagnostic result. This document does **not** authorize a confirmatory holdout, candidate-map rollouts, policy tuning, or a merge.

## 1. Scope and evidence

- Frozen candidate: repaired v0.4 at commit
  `e65e1915503246f0f07ed15c4884ac6952925ee8`.
- Freeze head: `a3e614b093dfd8237898cb19ac83cab92b8bad50`.
- Development data: 300 static identification episodes on maps 0-9, seeds 0-4,
  from `results/p0_dev_static_e65e191/records.jsonl` (900 rows: v3, v4, and
  no-adaptation). Only v3 and v4 are analyzed here.
- Frozen manifest content SHA-256:
  `95c4e13e533d1161af9cf6971df905c3222d94fbf091d3bf71e5ff3b37cabe80`.
  The development record manifest hash is
  `cfc6504ec4ad4a27494947e0200581ec86d4e781cb278719da149dc6e5b38b00`.
- Maps 210-239 were not touched. No candidate-map episode, holdout episode,
  GPU/VLM call, policy edit, or protocol edit was made.
- The stored record format has no per-decision posterior. The 37 v4 failures
  (33 unresolved, 4 incorrect) and their matched v3 episodes were replayed with
  a public tracing wrapper: 74 episodes in total. Replay assertions matched the
  stored task result, environment steps, terminal MAP, declaration rule, and
  declaration step for every traced episode.
- Complete per-decision posterior trajectories and tool evidence are in
  `docs/declaration_failure_traces_v1.jsonl` (74 records: 37 v4 + 37 matched
  v3). The reproducible driver is
  `scripts/diagnose_declaration_failures.py`; the pure classifiers are in
  `src/ruleshift/evaluator/declaration_diagnosis.py`.

The diagnosis is an **observational replay of the frozen policy**, not a
counterfactual experiment. Statements about what a different protocol would
have done are identified as interpretations, not measured outcomes.

## 2. Bottom line

1. **The safe-commit hook did not miss a legitimate control commit.** All 33
   v4 unresolved episodes had **zero** decisions on which the frozen
   `_commit_is_safe()` branch was eligible. All four incorrect declarations
   did fire the hook. The 33 unresolved are therefore not caused by an
   instrumented branch being skipped.
2. **The 33 unresolved decompose into three mutually exclusive primary causes:**
   14 **early-success stability gate**, 11 **terminal-only evidence**, and 8
   **early-success insufficient evidence**. None ended at the decision limit.
3. **The stability gate is the single largest new loss.** In all 14 cases the
   confidence/margin rule was met at a decision, but v4 required a second
   consecutive decision with the same leader; the episode succeeded before that
   decision existed. v3 declared correctly in all 14.
4. **Early success, not insufficient tools, explains most of the rest.** Five
   terminal-only and four insufficient-evidence episodes were v3 correct
   declarations that v4 lost because its control path reached the goal earlier.
   The remaining ten unresolved episodes were already unresolved in v3.
5. **The four final incorrect declarations are unchanged from v3.** Two are
   rational Bayesian updates after a 5% noisy `INSPECT`; two are premature
   commitments on ambiguous movement/sign evidence. The stability gate did not
   prevent any of the four.
6. **The v4 task and declaration advantages are orthogonal.** v4 task success
   is 298/300 versus 287/300; all 11 improvements are control repairs of the
   slippery/base plan, with zero task regressions. Final declaration coverage is
   263/300 versus 284/300; terminal MAP remains 297/300 in both. The candidate
   did not improve identification; it improved control while losing declaration
   coverage.

**Decision implication:** the repaired v4 candidate should not proceed to
confirmatory evaluation under the current protocol. The failure is a
protocol/control-coupling failure, not a missing hook and not a reason to relax
the confidence gate.

## 3. Mutually exclusive root-cause classification

Each target has exactly one primary label. Secondary factors are reported
separately and may co-occur.

| Primary label | Count | Definition | v3 counterpart |
|---|---:|---|---|
| `early_success_stability_gate` | 14 | A decision met top >= 0.90 and margin >= 0.10, but v4's two-consecutive-leader condition made safe commit false; episode succeeded before a second decision. | 14/14 correct declarations |
| `early_success_terminal_only_evidence` | 11 | No decision met the confidence rule; the terminal posterior met it only after the last action. The episode succeeded. | 5 correct, 6 unresolved |
| `early_success_insufficient_evidence` | 8 | No decision and no terminal posterior met the confidence rule; the episode succeeded. | 4 correct, 4 unresolved |
| `instrumentation_gap` | 0 | Safe branch eligible but no event recorded. | none |
| `max_decisions_insufficient_evidence` | 0 | No qualified posterior by the decision limit. | none |
| `inspect_noise_confident_error` | 2 | Incorrect declaration after an `INSPECT` result that contradicted the true role, with confidence above threshold. | 2/2 same final error |
| `premature_commit_ambiguous_evidence` | 2 | Incorrect declaration without a contradictory inspect; posterior crossed threshold on ambiguous movement/sign evidence; v4 terminal MAP later recovered in both cases. | 2/2 same final error; v3 recovered one terminal MAP |

The 14 stability cases have criterion-metting decisions (1 each) but
**zero safe-commit-eligible decisions**. The 11 terminal-only and 8
insufficient cases have **zero** criterion-metting decisions. Thus every
unresolved label is forced by the trace, not inferred from the aggregate.

### 3.1 What the 33 unresolved episodes are not

- **Not max-decision failures.** Every one terminated by `success`. Decision
  counts were 2:13, 3:9, 4:4, 5:3, 6:2, 7:1, 8:1.
- **Not missed safe commits.** No unresolved trace contains an eligible
  `_commit_is_safe()` decision.
- **Not all evidence-starved.** 25/33 had terminal top >= 0.90; 5 stability
  cases had the full 3-unit tool budget and a decisive sign-composition inspect.
- **Not all genuine ambiguity.** Only 10/33 were also unresolved in v3. The
  other 23 were v3 correct declarations; 14 are directly attributable to the
  stability gate and 9 to v4's earlier success.

### 3.2 Stability-gate mechanism

v4's `_commit_is_safe()` adds `_top_rule_count >= 2` to the unchanged
v3 thresholds (`grid_baselines.py:860`). `_top_rule_count` counts consecutive
`choose()` calls with the same MAP leader, not evidence observations. A
decisive tool result or movement observation often appears on the last
`choose()` before the goal; the next decision never occurs.

| Episode | v4 trajectory (public posterior) | v4 result | v3 result |
|---|---|---|---|
| `m1-s0-identification-remap_swap-None` | step 0 base .167; step 1 remap_swap .947 (margin .895), leader run 1; action RIGHT reaches goal. Terminal .997. | unresolved, 0 eligible | declared remap_swap at step 1 |
| `m0-s3-identification-slippery-None` | steps 0-4 alternate base/remap/slippery; step 5 slippery .99999995, run 1; RIGHT reaches goal. | unresolved, 0 eligible | declared slippery at step 5 |
| `m5-s0-identification-sign_composition-None` | 3 tool units. Step 3 after `INSPECT [1,1] = safe`: sign_composition .974, margin .949, run 1; DOWN reaches goal. | unresolved, 0 eligible | declared sign_composition at step 3 |

The third example is the strongest: the policy had already paid for the full
3-unit budget, the posterior was correct and far above threshold, and there was
one decision left. The stability filter, not evidence insufficiency, blocked
the declaration.

### 3.3 Terminal-only mechanism

A terminal-only case reaches the qualified posterior only after its final
action. There is no time-stamped decision at which the policy could have
declared; converting the terminal estimate would be retroactive and is
prohibited.

| Episode | v4 trajectory | v3 result |
|---|---|---|
| `m3-s1-identification-remap_rotate-None` | step 0 base .167; step 1 remap_rotate .692, run 1; DOWN reaches goal; terminal .9419. No criterion at a decision. | v3 took UP, had step 2 at .9419 and declared; 4 steps |
| `m1-s0-identification-slippery-None` | steps 0-2: base .167, slippery .500, slippery .471; LEFT reaches goal; terminal 1.0. No criterion at a decision. | also unresolved |

Five terminal-only episodes were v3 correct because v3's path produced one
additional decision; six were unresolved in both methods. This is an
**early-success/decision-opportunity** failure, not an instrumentation error.

### 3.4 Insufficient-evidence mechanism

The eight insufficient cases end below the confidence rule. Four were v3
correct because v3 had a longer path; four were unresolved in both. Some had
full tool budgets but remained genuinely ambiguous (for example the hazard
cases ending at top .5000 with a tie).

`m1-s1-identification-remap_rotate-None` is representative: v4 took 3 steps,
ended at top .733; v3 took 10 steps and declared remap_rotate at step 3. The
v4 path terminated before the evidence arrived. This is a control-policy
consequence of early success, not a missing declaration hook.

### 3.5 The four incorrect declarations

| Episode | True | Declared (step, confidence) | Mechanism | Task | v3 |
|---|---|---|---:|---|---|
| `m0-s3-identification-sign_composition-None` | sign_composition | hazard_goal_swap (8, .9744) | `INSPECT` blue returned "goal" where the true role is "safe"; the 5% error is modeled and the posterior rationally moves to .974 | fail, max decisions | same |
| `m5-s1-identification-hazard_goal_swap-None` | hazard_goal_swap | sign_composition (3, .9744) | `INSPECT` blue returned "safe" where the true role is "goal"; same noisy-inspect mechanism | fail, max decisions | same, declared at step 2 |
| `m1-s1-identification-base-None` | base | slippery (3, .9143) | movement/sign evidence made slippery lead; no contradictory tool; terminal MAP later recovered to base | success | same |
| `m2-s2-identification-base-None` | base | slippery (5, .9343) | movement-only evidence; no contradictory tool; v4 terminal MAP later recovered to base and the slippery fallback still reached the goal | success | same wrong declaration, but task failed at the decision limit and terminal MAP stayed slippery |

The two noisy-inspect episodes exhausted the 3-unit budget (sign + inspect) and
had no redundancy left to cross-check the contradictory sample. Their posterior
is Bayesian, not a bug: the policy's inspect likelihood includes the public 5%
error rate. The two premature cases crossed the unchanged .90/.10 threshold on
ambiguous evidence; both satisfied the two-leader gate, so that gate did not
protect against them. In `m2-s2`, the declaration is wrong but the task
succeeds because v4's slippery fallback uses the base plan; declaration failure
and task failure are not the same event.

## 4. Paired v3/v4 trade-off

### 4.1 Outcome counts

| Method | Correct final declaration | Incorrect final declaration | Unresolved | Task success | Terminal MAP correct |
|---|---:|---:|---:|---:|---:|
| v3 | 284 | 4 | 12 | 287 | 297 |
| v4 | 263 | 4 | 33 | 298 | 297 |

Final declaration precision among episodes with a final declaration is nearly
identical: v3 284/288 = 98.6%; v4 263/267 = 98.5%. The failure is **coverage**,
not precision.

### 4.2 Paired declaration transitions (v3 -> v4)

| Transition | Count |
|---|---:|
| correct -> correct | 261 |
| correct -> unresolved | 23 |
| unresolved -> correct | 2 |
| unresolved -> unresolved | 10 |
| wrong -> wrong | 4 |

The 23 correct-to-unresolved losses split as 14 stability, 5 terminal-only,
and 4 insufficient. The two gains are different-path successes (one slippery,
one hazard); they do not offset the stability loss. Net final coverage is
`+2 - 23 = -21`.

### 4.3 Paired task transitions

| Transition | Count |
|---|---:|
| fail -> success | 11 |
| success -> success | 287 |
| fail -> fail | 2 |

No task regressions. The 11 improvements are 10 slippery episodes and 1 base
episode (`m2-s2`); v3 had no tools in all 11 and failed by max decisions. The
v4 repair is the slippery base-plan fallback, not a declaration change. The two remaining v4 task failures are exactly the
two noisy-inspect wrong declarations.

### 4.4 Why task went up while declarations went down

The candidate changed control and declaration eligibility at the same time:

- **Control repair:** v4's `_commit()` falls back to a base plan for slippery
  (`grid_baselines.py:999`), eliminating ten slippery task loops. The same
  fallback also rescues `m2-s2`, where a wrong slippery declaration would
  otherwise loop; this yields +11 task successes with no observed task
  regression.
- **Declaration gate:** v4 requires a second consecutive leader. This removes
  14 declarations that v3 made correctly. The same control repair makes v4
  reach the goal earlier in several episodes, removing another 9 decisions at
  which v3 could declare.
- **Noise:** the four final wrong declarations are unchanged; two are caused
  by the public 5% inspect error, which the gate does not address.

There is no evidence that the declaration loss bought a task gain or a
correctness gain. The two effects are separable and should be evaluated
separately.

## 5. Instrumentation and coverage audit

- Hook location: `DiagnosticBeliefPolicy._record_commitment`
  (`grid_baselines.py:673`), called only from the safe-commit branch in
  `choose()`.
- v4 adds the leader-run condition before that branch. It does not add a new
  declaration action.
- The hook de-duplicates only a **consecutive same-rule** event. If the rule
  changes, a later event is appended; the frozen record builder uses the
  **last** valid event as the final declaration
  (`grid_identification.py:343-370`).
- No unresolved episode had an eligible safe branch. No incorrect declaration
  lacked an event. Thus the hook has complete coverage of the branch it is
  attached to.
- That branch is not the same as "all legitimate explicit declarations." A
  policy can have a high-confidence posterior at a decision and still be
  ineligible because it has not committed control; conversely, a terminal
  posterior can be qualified after the last decision. The hook cannot represent
  either case separately. This is an instrumentation **semantics** limitation,
  not a missing call.

## 6. What is supported and what is not

Supported by the traces:

- The 14 stability losses are directly attributable to the two-consecutive
  leader condition; all 14 were v3 correct and all had a criterion-metting
  decision with leader run 1.
- The 9 additional new losses are early-success cases: 5 terminal-only and 4
  insufficient; v3 had a later declaration opportunity.
- The 10 shared unresolved cases are genuine v4 unresolved episodes as well;
  they are not evidence of a v4-specific regression.
- The four final wrong declarations are the same four episodes in v3 and v4;
  two are explained by contradictory noisy inspect, two by ambiguous evidence.
- The +11 task effect is control repair, not declaration instrumentation.

Not supported:

- "The hook missed declarations." No eligible branch was skipped.
- "Terminal MAP should count as a declaration." That would retroactively
  convert the 34 v4 episodes whose terminal MAP is correct but whose final
  declaration is not, and inflate coverage to 297/300 without a time-stamped
  assertion.
- "Removing the stability gate alone fixes the candidate." It would address 14
  cases but not the 9 early-success losses, and it would not address the two
  noisy-inspect errors.
- "The candidate is a pure regression." It improves task success by 11 and
  keeps final wrong declarations at 4, but loses 21 final declarations and 16
  clean declarations. Under the current identification gate, that is a
  feasibility failure.

## 7. Decision

**NO-GO for the confirmatory holdout.** Do not promote repaired v0.4. Close
the current candidate as a development-stage feasibility failure under the
current protocol, while preserving the frozen candidate, manifest,
preregistration, archive, and draft PR #22 as historical records.

The diagnosis does support **one** protocol-level intervention to investigate,
not a same-protocol policy tweak: separate epistemic declaration from control
commitment and make abstention/evidence requests explicit. The design-only
proposal is in `docs/NEXT_CANDIDATE_PROPOSAL.md`; it is not authorized for
implementation or tuning by this diagnosis.

## 8. Reproduction

```powershell
$env:PYTHONPATH='.;src'
python scripts/diagnose_declaration_failures.py `
  --trace-jsonl-out docs/declaration_failure_traces_v1.jsonl
python scripts/render_declaration_episode_index.py `
  --input reports/declaration_diagnosis_v1.json `
  --out docs/DECLARATION_FAILURE_EPISODES.md
python -m pytest -q tests/test_declaration_diagnosis.py
python -m pytest -q
```

Outputs:

- `reports/declaration_diagnosis_v1.json`: full summaries and 74 traces.
- `reports/declaration_diagnosis_v1.csv`: one row per traced episode.
- `docs/declaration_failure_traces_v1.jsonl`: complete public trajectories
  for the appendix.

The script refuses non-development maps, non-static records, more or fewer than
900 records, and any code commit other than the frozen `e65e191` candidate.
