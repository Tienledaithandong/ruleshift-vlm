# Base-failure diagnosis (post-hoc replay)

**Status: post-hoc root-cause diagnostic. Not confirmatory, not a new rate,
and not a gate change.**

## Question

The 100-map base-stress study found three task failures and eight final-rule
misdiagnoses in 500 true-base episodes. This replay asks a narrower question:
what decision sequence produced those eight outcomes?

## Method

- Input: the eight scenario ids named by
  `results/grid_phase0/base_stress_100x5_analysis/analysis.json`.
- The same `diagnostic_belief` policy, environment, maps, and seeds are used.
- A tracing subclass records, before each decision: the public observation,
  the policy posterior, commit safety, the MAP rule and its plan length, and
  the public sign face. It records the true rule only at episode level. The
  true rule is never passed back to the policy.
- The trace is JSON-serializable and contains no hidden transition data beyond
  the episode-level true rule.

```bash
python scripts/diagnose_grid_base_failures.py
```

Raw trace: `results/grid_phase0/base_failure_traces.json`

SHA-256: `b978fe79582829c39c7bf568b3aeb37eba85d748d2634bc19d749784026f36a6`

This is a selected sample: the eight episodes were chosen because they failed
or were misdiagnosed. Nothing here estimates a population rate.

## What the traces show

| Scenario | Outcome | Steps | Tools | Final estimate | Final top mass | Sign faces read | Decisions with empty MAP plan | Unique positions |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| `m2-s2` | task failure | 60 | 0 | `slippery` | 0.905 | none | 58/60 | 3 |
| `m57-s4` | task failure | 60 | 1 | `slippery` | 0.727 | side 0 | 59/61 | 2 |
| `m78-s1` | task failure | 60 | 1 | `remap_rotate` | 0.835 | side 0 | 0/61 | 6 |
| `m10-s4` | success, misdiagnosed | 5 | 1 | `slippery` | 0.612 | side 0 | 0/5 | 5 |
| `m60-s3` | success, misdiagnosed | 2 | 0 | `remap_rotate` | 0.205 | none | 0/2 | 2 |
| `m63-s4` | success, misdiagnosed | 3 | 0 | `remap_rotate` | 0.508 | none | 0/3 | 2 |
| `m79-s0` | success, misdiagnosed | 4 | 1 | `slippery` | 0.900 | side 0 | 0/4 | 2 |
| `m81-s2` | success, misdiagnosed | 7 | 0 | `slippery` | 0.319 | none | 0/7 | 4 |

### The three task failures

**`m2-s2`: early false posterior, then an empty-plan fallback loop.** The
posterior first favoured a non-base rule at decision 1. By decision 5 it was
`slippery` at 0.934, so `_commit_is_safe()` became true and the policy stopped
probing. Under the wrong MAP rule, `base_plan()` returned an empty route on
58 of the 60 decisions. The fallback then chose the first legal physical
action; the trace is 55 `UP` actions and only three unique positions. The
episode never reaches the goal. The trigger is a base-rule slip that makes a
slippery explanation more likely; the 60-step loss is the empty-plan fallback.

**`m57-s4`: no useful diagnostic action, then the same fallback.** The policy
read sign side 0 once. The clause was "The red tile is not the hazard", which
is consistent with both `base` and `slippery`; there is no sign face that
separates those two rules. The posterior was `slippery` at 0.727 but the
commit test stayed false, so the policy asked for a diagnostic action. None
was available: the sign had already been read and the blue inspect was not
worth its cost because the hazard/sign mass was zero. `choose()` therefore
called `_commit()` anyway. The MAP plan was empty on 59 of 61 decisions and
the trace is 60 `UP` moves (plus one sign read) in two positions.

**`m78-s1`: side 0 cannot separate `base` from `remap_rotate`, and the MAP
rule oscillates.** The policy read side 0 once; it never read side 1, because
`sign_observed` is a single boolean and the probe order always tries side 0
first. Side 0 ("red is not the hazard") is compatible with both `base` and
`remap_rotate`; side 1 ("movement may be remapped") is the face that separates
them. The posterior therefore oscillated between the two rules, with
repeated 0.99+ excursions, and the commit test oscillated with it. The action trace cycles
through a small set of cells (`DOWN`, `RIGHT`, `UP`) and ends at
`remap_rotate` 0.835 after 60 steps. This is not an empty-plan failure; it is
a static-probe and belief-oscillation failure.

### The five successes that were still misdiagnosed

These episodes reached the goal in 2-7 steps. The final belief is measured at
termination, before the posterior has settled, so a successful path can end
with a wrong final label. Two details matter:

- `m60-s3` ends in an almost exact tie: `base`, `hazard_goal_swap`,
  `remap_rotate`, and `sign_composition` all have mass 0.204545..., with
  `remap_rotate` larger only by floating-point noise. The label is an
  arbitrary tie-break, not a confident false belief.
- `m81-s2` is also weak (`slippery` 0.319 versus 0.227 for three other rules),
  while `m79-s0` is strong (0.900) and `m10-s4`/`m63-s4` are intermediate.
  The eight "misdiagnoses" are therefore not a homogeneous set of confident
  wrong beliefs.

## Mechanisms

1. **Empty MAP plan -> first legal action.** When the MAP rule is wrong, its
   plan can be empty. The current fallback emits the first legal physical
   action instead of a safe plan or an exploratory action. In `m2-s2` and
   `m57-s4` this converts a wrong belief into a 60-step loop.
2. **Static side-0 probing.** `_go_read_sign()` always tries side 0 first and
   `sign_observed` is one bit. Side 1 is never read after side 0, even though
   side 1 is the only sign face that separates remaps from the other rules.
   This explains the `m78-s1` failure mode.
3. **No sign can separate `base` from `slippery`.** Both rules have the same
   three clauses. A sign probe cannot correct the `m2-s2`/`m57-s4` false
   posterior; only movement evidence or a different prior/stopping rule can.
4. **Terminal belief is not a decision.** A 2-step success has almost no
   evidence, and a floating-point tie can be counted as a misdiagnosis. The
   identification endpoint is measuring the belief left at termination, not
   whether the policy had enough evidence to identify the rule.

## Counterfactual check (not an efficacy result)

A temporary subclass changed only the empty-plan fallback: if the MAP plan was
empty, it planned to the goal under `base`. On the same seeds, `m2-s2` then
succeeded in 12 steps and `m57-s4` in 11 steps, both ending at `base`;
`m78-s1` still failed. This isolates the fallback as the immediate cause of
the first two task failures and shows that it is separate from the sign-probe
failure. It is a post-hoc, two-episode observation: it is not a benchmark
result, not a validated fix, and not evidence that the base gate would pass.

## What this does not establish

- It does not estimate a new failure rate; the eight episodes are selected.
- It does not show that a fallback or a side-1 probe would improve the
  population; neither change was evaluated on fresh maps.
- It does not change the 100% base gate, the preregistration, or the Phase 1
  stop rule.
- It does not use a VLM, GPU, or hidden true rule in the policy.

## Decision and next step

The diagnosis narrows the next protocol decision:

- **Keep the literal 100% gate.** Then the policy needs a new preregistered
  version with (a) a safe plan-failure fallback and (b) per-side sign evidence
  with posterior-conditioned face selection. `base` versus `slippery` remains
  sign-ambiguous, so movement evidence or an explicit tolerance is still
  required. Evaluate on fresh maps, not these eight.
- **Adopt a powered base endpoint.** Then pre-specify the tolerance and how
  ties/early terminations are scored before collecting a new sample. Do not
  choose the threshold from this 1.6% estimate.

Either way, the next run must be a new benchmark version with a frozen
protocol. This replay is a hypothesis generator, not a license to tune the
current maps or start Phase 1.

### Follow-up: v0.4 candidate

A v0.4 candidate now implements the plan fallback, per-side sign probing, and
a stable-commit condition. On the fresh-map exploratory holdout (maps 100-199,
500 true-base episodes), both v0.3 and v0.4 have 0 task failures, so the
holdout cannot demonstrate a task-level improvement; v0.4 has 4/500
misdiagnoses versus 5/500 for v0.3. On the original maps, v0.4 removes the
three observed task failures but is post-hoc. The literal gate is still failed
and Phase 1 remains blocked; see
[`GRID_V04_HOLDOUT.md`](GRID_V04_HOLDOUT.md).