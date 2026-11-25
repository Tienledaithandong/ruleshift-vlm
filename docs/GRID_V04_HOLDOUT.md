# v0.4 base-policy candidate: exploratory holdout result

**Status: exploratory candidate evaluation. Not a gate result, not a frozen
benchmark, and not a Phase 1 authorization.**

Protocol: [`GRID_V04_CANDIDATE.md`](GRID_V04_CANDIDATE.md), committed before
the candidate was run.

## Candidate

`DiagnosticBeliefPolicyV4` changes three things relative to v0.3:

1. an empty MAP plan falls back to a `base` plan before the first-legal-action
   fallback;
2. sign reads are stored per face, and the face with the highest expected
   information gain is chosen; side 1 is therefore available when it separates
   `base` from the remap rules;
3. a rule must lead on two consecutive decisions before the existing commit
   thresholds can fire.

v0.3 remains `DiagnosticBeliefPolicy` and is unchanged.

## Fresh-map holdout

- Maps: generator indices **100-199**, unused in the v0.3 10-map study, the
  100-map base-stress study, and the 100-map audit.
- Seeds: 0-4; 500 true-base identification episodes per policy.
- Raw: `results/grid_phase0/v04_holdout_100x5.jsonl`; SHA-256
  `02315d1474bfdfb535cfe6406e748bd039a7878a9e0cbd7d92647a00b5315779`.

| Policy | Task failures | Misdiagnoses |
| --- | ---: | ---: |
| v0.3 diagnostic | 0/500 (0.0%) | 5/500 (1.0%) |
| v0.4 candidate | 0/500 (0.0%) | 4/500 (0.8%) |
| `no_adaptation` | 0/500 (0.0%) | 0/500 (0.0%) |

Task-failure Wilson 95% for both diagnostic policies: `[0.0%, 0.76%]`. The
map-cluster interval is degenerate at `[0.0%, 0.0%]` because every map has
the same zero event rate; that is not evidence of certainty.

v0.4 misdiagnosis: 0.8%, Wilson 95% `[0.31%, 2.04%]`, map-cluster 95%
`[0.0%, 1.8%]`. v0.3 misdiagnosis: 1.0%, Wilson 95% `[0.43%, 2.32%]`,
map-cluster `[0.2%, 2.0%]`.

### Paired comparison

- Task success: **500 both, 0 only v0.4, 0 only v0.3, 0 neither**; exact
  McNemar `p = 1`. The holdout has a task-success ceiling: v0.3 also has no
  failures, so it cannot show that v0.4 fixes task failures.
- Misdiagnosis: 4 both, **1 only v0.3** (`m166-s0`), 0 only v0.4; exact
  McNemar `p = 1`. The difference is one episode, not a reliable improvement.

All four v0.4 misdiagnoses still succeeded, in 4-9 steps, with a final
`slippery` belief of 0.542, 0.894, 0.849, and 0.714. They are early
terminations, not task failures. The `base` versus `slippery` sign ambiguity
identified in the v0.3 diagnosis remains.

### Mechanism use

On the 500 holdout episodes the candidate used the base-plan fallback 4 times
in 2 episodes, and read side 1 in 13 episodes. The per-side set was `{0}` in
479 episodes, `{0, 1}` in 13, and empty in 8. The fallback and side-1 path
are therefore exercised, but the holdout did not contain a v0.3 task failure
for them to repair.

## Post-hoc same-map check

The same comparison on maps 0-99 is reported only as a development check;
those maps produced the diagnosis and are not a holdout.

| Policy | Task failures | Misdiagnoses |
| --- | ---: | ---: |
| v0.3 diagnostic | 3/500 (0.6%) | 8/500 (1.6%) |
| v0.4 candidate | 0/500 (0.0%) | 3/500 (0.6%) |

Paired task success: 497 both, **3 only v0.4**, 0 only v0.3, 0 neither; exact
McNemar `p = 0.25`. Paired misdiagnosis: 3 both, **5 only v0.3**, 0 only
v0.4; exact McNemar `p = 0.0625`. The five v0.3-only misdiagnoses include
`m2-s2`, `m57-s4`, and `m78-s1`; v0.4 keeps the three early-success
misdiagnoses `m10-s4`, `m63-s4`, and `m81-s2`.

This shows that the candidate addresses the observed mechanisms on the maps
that motivated it. It is not an efficacy estimate and does not override the
fresh-map ceiling.

## Interpretation

- The v0.3 base gate remains failed: **49/50** on the official 10-map run.
  Neither the holdout nor the same-map check can pass or replace it.
- v0.4 does not regress on the fresh task endpoint, but the fresh holdout
  cannot demonstrate a task-failure improvement because v0.3 already has zero
  failures there.
- The one-episode misdiagnosis difference is not statistically meaningful.
- The residual misdiagnoses are early successful episodes; the terminal
  belief metric still counts unsettled or tied beliefs as wrong. This is a
  measurement issue to decide before the next endpoint, not a reason to
  relabel these episodes.
- The literal 100% gate is still the binding blocker. v0.4 is not frozen, not
  promoted, and not used by the benchmark.

## Decision

Do not start Phase 1. The next step remains a human decision about the base
endpoint and a fully preregistered candidate evaluation. A useful next
holdout would need maps on which v0.3 actually fails, but selecting those
maps after seeing outcomes would invalidate the comparison; the alternative
is a larger pre-registered sample or an explicitly powered tolerance.

## Reproduce

```bash
python scripts/run_grid_v04_holdout.py
python scripts/run_grid_v04_holdout.py --offset 0 --maps 100 \
  --out results/grid_phase0/v04_samesample_100x5.jsonl \
  --analysis-dir results/grid_phase0/v04_samesample_analysis
```

Generated artifacts stay under `results/` and are not committed.