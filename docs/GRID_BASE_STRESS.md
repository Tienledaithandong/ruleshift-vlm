# Base-preservation stress study (100 maps)

**Status: descriptive CPU study, not a gate evaluation and not a preregistered
confirmatory result.**

## Why this study exists

The repaired v0.3 10-map run passed six of seven Phase 0 gates. It failed the
`base preserved 100%` gate on one episode,
`m2-s2-identification-base-None` (true base, zero tools, 60 steps, final
estimate `slippery`). One failure in 50 episodes cannot distinguish a genuine
low-rate defect from an unlucky sample, so this study measures the rate on the
same 100-map generator rather than changing the gate or the policy.

## Protocol

- **Maps:** all 100 v0.3 map layouts.
- **Seeds:** 5 per map, paired across methods.
- **Episodes:** 500 true-base identification episodes per method; no shifts.
- **Methods:** `diagnostic_belief` and `no_adaptation`.
- **Hardware:** CPU only. No frozen VLM was loaded and no GPU job was started.
- **Primary outcomes:**
  - `task_failure`: the episode did not reach the goal (`success == false`),
    the endpoint used by the original base gate;
  - `misdiagnosis`: the final rule estimate was not `base`, regardless of
    whether the goal was reached.
- **Uncertainty:** Wilson 95% intervals over episodes, plus a percentile
  bootstrap that resamples whole maps. The map-cluster interval is marked
  degenerate when every map has the same event rate.

Command:

```bash
python scripts/run_grid_base_stress.py
```

Raw output: `results/grid_phase0/base_stress_100x5.jsonl`

SHA-256: `95d1ccc040a0b9315b47a223823ac86702a727a26b617e82cb0af453c8fd7e8b`

## Results

| Method | Endpoint | Events | Rate | Wilson 95% | Map-cluster 95% |
| --- | --- | ---: | ---: | ---: | ---: |
| `diagnostic_belief` | task failure | 3/500 | 0.6% | [0.2%, 1.7%] | [0.0%, 1.4%] |
| `diagnostic_belief` | misdiagnosis | 8/500 | 1.6% | [0.8%, 3.1%] | [0.6%, 2.8%] |
| `no_adaptation` | task failure | 0/500 | 0.0% | [0.0%, 0.8%] | not estimable |
| `no_adaptation` | misdiagnosis | 0/500 | 0.0% | [0.0%, 0.8%] | not estimable |

The three diagnostic task failures are:

| Scenario | Estimate | Steps | Tool units |
| --- | --- | ---: | ---: |
| `m2-s2-identification-base-None` | `slippery` | 60 | 0 |
| `m57-s4-identification-base-None` | `slippery` | 60 | 1 |
| `m78-s1-identification-base-None` | `remap_rotate` | 60 | 1 |

Five additional episodes were misdiagnosed but still reached the goal:
`m10-s4` (`slippery`), `m60-s3` (`remap_rotate`), `m63-s4`
(`remap_rotate`), `m79-s0` (`slippery`), and `m81-s2` (`slippery`).

Paired task outcomes: 497 both succeed, 0 only diagnostic, 3 only baseline,
0 neither; exact McNemar `p = 0.25`. This is descriptive only and is not a
claim that the diagnostic policy is better or worse than the baseline.

A post-hoc trace of the eight non-base outcomes is in
[`GRID_BASE_FAILURE_DIAGNOSIS.md`](GRID_BASE_FAILURE_DIAGNOSIS.md). It is a
selected-sample diagnosis, not another rate estimate.

## Interpretation

1. **The original failure is reproducible, not a one-off.** `m2-s2` fails in
   the larger base-only sample as well. Two further task failures appear on
   maps 57 and 78. The post-hoc trace shows a repeatable mechanism, not a
   single random walk: a false posterior plus an empty-plan fallback in
   `m2-s2` and `m57-s4`, and an unresolved `base`/`remap_rotate` ambiguity in
   `m78-s1`.
2. **The base task-failure rate is low but not zero.** The point estimate is
   0.6%; the 95% upper bound is 1.7% (Wilson) or 1.4% (map-cluster). A 100%
   gate therefore still fails, and this sample cannot establish a zero rate.
3. **Success understates the identification problem.** Eight episodes ended
   with a non-base estimate, but five still reached the goal. A success-only
   base gate can miss a false belief; the original 49/50 result was already
   an optimistic view of base preservation.
4. **This does not authorize Phase 1.** It characterises a known failure; it
   does not make the hard gate pass, change the benchmark, or provide a
   preregistered tolerance. Those are separate decisions for a human.

## What would be a defensible next step

A new protocol should explicitly decide whether the base requirement is:

- a literal zero-failure gate, in which case the current policy fails and
  must be improved before Phase 1; or
- a powered false-alarm-rate endpoint with a prespecified tolerance, in which
  case the interval above is the relevant evidence and a new confirmatory
  sample is still required.

Changing the threshold after observing these results without a new
preregistration would be post-hoc gate relaxation and is not supported by this
study.