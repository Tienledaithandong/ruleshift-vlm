# Preregistration: RuleShift-Grid identification v1

Status: **FROZEN PROPOSAL — READY FOR HUMAN GO/NO-GO. NOT EXECUTED.**

Freeze date: 2026-11-25. Branch: `research/identification-protocol-v1`.
This document becomes confirmatory only after the explicit human GO in section
12. Until then it is a protocol proposal, not a result and not authorization for
a candidate-map rollout.

## 1. Question and claim boundary

Active rule identification is the primary research question. The candidate is
asked to make an explicit, settled declaration of the active rule from public
observations, and to do so accurately on a frozen map bank.

The paired comparison with no_adaptation uses terminal-MAP accuracy because
no_adaptation exposes no declaration API. That paired comparison is a benchmark
of terminal behaviour, **not** evidence that either policy made an explicit
identification. Terminal-MAP correctness is never described as confirmed active
identification. A candidate MAP-correct episode without a settled declaration
is not a declaration success.

General post-shift adaptation, shift detection latency, VLM inference and GPU
experiments are out of scope. The old full 11-scenario design (1,650 episodes per
method) is deferred and is not part of this experiment.

## 2. Frozen candidate and comparators

| Role | Method id | Identity |
|---|---|---|
| Candidate | `diagnostic_belief_v4` | repaired v0.4 policy at candidate policy commit `e65e1915503246f0f07ed15c4884ac6952925ee8` |
| Historical reference | `diagnostic_belief_v3` | unchanged v0.3 policy at the same commit |
| Non-adaptive baseline | `no_adaptation` | no belief, no tools, no declarations |

The candidate's only change after the behavioural baseline is declaration
instrumentation at its existing safe-commit branch. It does not alter navigation,
probe selection, posterior updates or action execution. The pre-instrumentation
behaviour is recorded from `f4b43a4`; the equivalence report is
`docs/DECLARATION_INSTRUMENTATION.md`.

The candidate is frozen. No tuning, retraining, prompt change, threshold change,
map-specific branch or seed-specific branch is permitted. If a code change is
needed, this preregistration is void and must be re-frozen before any candidate
map is touched.

## 3. Design and sample

| Factor | Frozen value |
|---|---:|
| Maps | 210–239 (30 maps) |
| Run seeds | 5, 6, 7, 8, 9 (5 seeds) |
| Static initial rules per map-seed | 6 |
| Static episodes per method | 900 |
| Methods | 3 |
| Confirmatory CPU episodes | 2,700 |
| Base episodes per method | 150 |
| hazard_goal_swap episodes per method | 150 |
| sign_composition episodes per method | 150 |
| Shift/no-shift episodes | 0 (deferred) |

Every map-seed contributes exactly six static identification scenarios, one per
initial rule. There are no exclusions, replacements or optional stopping.

Manifest: `artifacts/manifests/identification_v1_static.json`
- manifest content SHA-256: `95c4e13e533d1161af9cf6971df905c3222d94fbf091d3bf71e5ff3b37cabe80`
- local Windows working-file SHA-256 (auxiliary, line-ending dependent):
  `646f89632319a443943fc0e1add5cffb9579764e18cef07cdc7b325620d9fef5`; this
  is not the scientific identity
- scope: `static_identification`
- entries: 900; 30 unique public layouts; no layout collision with maps 0–199.

## 4. Endpoints

All endpoints use the 900 static episodes per method. Unresolved, unavailable and
wrong cases are never silently dropped.

| ID | Endpoint | Denominator | Role |
|---|---|---|---:|
| E1 | Correct explicit settled declarations / all static episodes | 900 | primary identification estimate and gate |
| E2 | Independent terminal-MAP accuracy | 900 | primary paired endpoint for H1/H2 |
| E3 | Task success | 900 | descriptive |
| E4 | Base task success | 150 | engineering gate |
| E5 | hazard_goal_swap explicit declarations | 150 | engineering gate |
| E6 | sign_composition explicit declarations | 150 | engineering gate |
| E7 | Zero-tool task success on hazard/sign | 150 + 150 | fail-closed hazard gate |
| E8 | Tool units per episode | 900 | engineering gate |
| E9 | Brier / NLL | valid posteriors | descriptive calibration |
| E10 | Candidate minus v3 terminal MAP | 900 | descriptive only |

A declaration is an event emitted by the policy's declaration API at decision
time. It contains step, rule, confidence and top-second margin. It is **settled**
only when confidence ≥ 0.90 and margin ≥ 0.10. The final declaration before
termination is scored. A terminal MAP, a fallback plan or an inferred preference
is not a declaration.

Commitment status is exactly one of `correct_commit`, `wrong_commit`,
`unresolved` or `unavailable`. Unresolved includes no declaration, an unsettled
declaration and a missing posterior. Unavailable means the policy exposes no
declaration API. Both are failures in E1's denominator. Ties are unresolved.
Earlier wrong declarations are reported separately even when a later declaration
is correct.

Coverage fields are distinct:
- event coverage: episodes with at least one explicit declaration;
- settled coverage: episodes with a settled declaration;
- API availability: episodes whose policy exposes a declaration API;
- unconditional accuracy: `correct_commit / 900`.

The no_adaptation baseline has 0 declaration events and is unavailable for E1;
its terminal MAP is descriptive only. It cannot pass or fail E1 by MAP
substitution.

## 5. Hypotheses and exact rejection rules

The confirmatory p-valued family has exactly two hypotheses. Both use the
equal-map mean paired difference in terminal-MAP accuracy. The map, not the
episode, is the independent unit.

| ID | Comparison | Direction | Metric |
|---|---|---|---|
| H1 | `diagnostic_belief_v4 - no_adaptation` | greater | terminal MAP |
| H2 | `diagnostic_belief_v3 - no_adaptation` | greater | terminal MAP |

For each hypothesis, let `d` be the equal-map mean difference, `L` the lower
bound of its 95% map-cluster percentile bootstrap interval, and `p_holm` the
Holm-adjusted one-sided sign-flip randomization p-value over H1 and H2.

Reject a hypothesis only if **all three** hold:

1. `p_holm < 0.05`;
2. `d >= 0.05` (the prespecified minimum meaningful difference, 5 percentage
   points);
3. `L > 0`.

A non-rejection is reported as non-rejection, never as equivalence. H2 is a
historical reference in the same family; it is not a candidate-superiority test.
The candidate-versus-v3 comparison is descriptive, two-sided, has no minimum
effect and is not in the rejection family. No other p-values are confirmatory.

## 6. Absolute engineering gates

These are deterministic decision rules, not members of the Holm family. They are
reported with exact counts and 95% Wilson intervals. No gate is relaxed after
seeing a candidate-map result.

| Gate | Requirement | Failure handling |
|---|---|---|
| G1 active identification | v4 E1 point estimate ≥ 95.0% | no active-identification claim |
| G2 base preservation | v4 150/150 base task successes | no confirmatory claim; do not relax |
| G3 hazard identification | v4 hazard explicit declarations ≥ 90.0% (150) | gate failure |
| G4 sign identification | v4 sign explicit declarations ≥ 90.0% (150) | gate failure |
| G5 zero-tool hazard/sign | zero raw candidate zero-tool task successes on the 300 hazard/sign episodes | fail closed; no exception |
| G6 tool budget | every candidate episode ≤ 3 tool units | gate failure |

G1 is the primary identification gate. H1 is the primary paired benchmark
hypothesis. A terminal-MAP advantage alone does not satisfy G1. A G1 pass alone
does not establish an advantage over no_adaptation.

For G2, 100% observed is an empirical gate result for the frozen 150-episode
endpoint, not a population guarantee. The one-sided 95% Clopper–Pearson upper
bound with zero failures in 150 trials is about 2.0%; it is reported alongside
the observed 150/150.

For G5, no evidence-coding exception is defined in this preregistration. Any raw
zero-tool hazard or sign success blocks the gate. Raw and evidence-adjusted
counts are both reported; with no approved evidence rule the adjusted count is
not used to rescue a raw failure. Independent verification of tool-event logs is
required if a future exception is proposed; none is approved here.

## 7. Statistical analysis

### 7.1 Map-clustered paired analysis

For each method and map, compute the mean of the binary static outcome over the
30 episodes. For a pair, compute the 30 within-map differences and their equal-
map mean. Required implementation and defaults:

- percentile bootstrap over maps: 20,000 resamples, seed `20261125`, 95%
  interval, linear interpolation between order statistics;
- one-sided sign-flip randomization test over maps: 20,000 permutations,
  `p = (b + 1) / (B + 1)`, where `b` counts permuted means at least the observed
  mean;
- exact sign test over map differences reported as a sensitivity check;
- Holm step-down over H1 and H2 only;
- strict scenario-id pairing: missing, extra or duplicate ids abort analysis;
- no intersection, no episode-level pseudo-replication, no dropping of maps.

The episode-level exact McNemar test is supplementary only. It must not be used
to claim confirmatory significance. The cluster interval is the uncertainty
statement for the paired effect.

### 7.2 Absolute endpoints

E1, E5 and E6 use the unconditional point estimate and a two-sided 95% Wilson
interval. E4 uses the observed count and a one-sided 95% Clopper–Pearson upper
bound. E9 uses only valid full-menu posteriors; Brier is the conventional
multiclass sum of squared class errors and NLL is the negative log probability
assigned to the terminal true rule. Calibration is descriptive.

### 7.3 Multiplicity

Holm correction is applied to exactly H1 and H2. Gates G1–G6 and the descriptive
E3/E9/E10 analyses are not p-valued and are not included in Holm. Reporting many
descriptive intervals does not convert them into confirmatory tests.

## 8. Sample-size justification

The design is 30 maps, each contributing 30 static episodes. Development evidence
from exposed maps 0–9, seeds 0–4 (not confirmatory) gives:

| Contrast | Equal-map mean | Map-level SD |
|---|---:|---:|
| v4 − no_adaptation | 0.8233 | 0.0161 |
| v3 − no_adaptation | 0.8233 | 0.0161 |
| v4 − v3 | 0.0000 | 0.0157 |

Those observed SDs are small because the exposed bank is easy. Planning uses a
deliberately conservative map-level SD of **0.10**, close to the maximum
variability of a bounded difference. With n = 30 maps, a one-sided alpha of
0.025 (the conservative Holm component) and 80% power:

    MDE = (z_0.975 + z_0.80) * SD / sqrt(30)
        = (1.960 + 0.842) * 0.10 / sqrt(30)
        ≈ 0.0512  (5.1 percentage points)

At a one-sided 0.05 level the same calculation gives about 4.5 percentage
points. The frozen design uses the conservative 5.1-point figure as its sample-
size justification. The 95% map-cluster interval half-width under the same
planning SD is about 3.6 percentage points.

For E1, 900 episodes give a Wilson half-width of about 1.4 percentage points
near 95%. That precision does not turn the 95% point-estimate gate into a
population proof. G3/G4 have 150 episodes each and correspondingly wider
intervals. The study is powered for the prespecified 5-point paired effect, not
for proving that an absolute rate is exactly 95% or 90%.

The development observations are reported as development evidence only. They are
not used to tune the candidate and are not confirmatory estimates.

## 9. Execution and freeze controls

Before any candidate-map episode:

1. verify candidate policy commit `e65e191` and analysis commit recorded in the
   run record;
2. verify manifest content SHA-256 `95c4e1…be80`;
3. verify the config SHA-256 `67f7be9716947a9b30f6d30fdcdd72f0abb95277ea7f2f5e8eb09cde03cb8421`;
4. verify all 30 layout hashes and the 900-entry exact scenario-id set;
5. verify no candidate scenario id appears in any development result or log
   outside the frozen manifest;
6. record Python/platform versions, command line, working tree and output path;
7. confirm no VLM, GPU, remote inference or network service is used.

The current branch intentionally does **not arm** a confirmatory entrypoint; the
development runner refuses maps above 199. After a separate explicit GO, a
reviewed authorization commit may add a manifest-hash-pinned confirmatory entry
point. It must not change policy code, the analysis family or this protocol.

Execution is one fixed batch of 2,700 episodes. No peeking, optional stopping,
selective retry, map replacement or episode deletion is allowed. Abort before
scoring on a hash mismatch, layout mismatch, duplicate scenario id, hidden-truth
leak, unknown method or contract error. Environment failures are recorded and
the batch is not silently resumed with a different manifest.

## 10. Failure classification

Every non-success is classified as one or more of:

- task failure with correct terminal MAP;
- task success with wrong terminal MAP (successful misdiagnosis);
- unresolved declaration;
- wrong settled declaration;
- earlier wrong declaration followed by correction;
- unavailable declaration API;
- tool-budget violation;
- zero-tool hazard/sign success;
- environment or contract failure.

Classification is descriptive and never changes a denominator.

## 11. Known pre-freeze development findings and risks

These are exposed-map findings, not confirmatory results. They are recorded now
so the freeze is not mistaken for a prediction of success.

On the 300 exposed static episodes per method (maps 0–9, seeds 0–4):

| Method | Task | Terminal MAP | Correct declarations | Wrong | Unresolved | Unconditional declaration accuracy |
|---|---:|---:|---:|---:|---:|---:|
| v3 | 287 | 297 | 284 | 4 | 12 | 94.67% |
| v4 | 298 | 297 | 263 | 4 | 33 | **87.67%** |
| no_adaptation | 128 | 50 | 0 | 0 | 0 (unavailable 300) | unavailable |

The candidate's development declaration accuracy is below the 95% G1 gate, and
its development sign-composition declaration accuracy is 44/50 = 88%, below the
90% G4 gate. Its hazard declaration accuracy is 47/50 = 94%, and base task
success is 50/50. The v3 reference is 94.67% overall, with a historical base task
result of 49/50. These exposed results mean the candidate may fail G1 and/or G4
on the frozen bank. That is a reason to make a GO decision with the risk visible,
not a reason to tune the candidate after seeing the data.

Other risks:

- no_adaptation has no declaration API, so H1/H2 cannot test declaration
  superiority; they test terminal-MAP behaviour only;
- 30 maps is a finite sample; a significant paired effect does not establish
  performance on all maps;
- the 95% absolute gate is a point-estimate rule with finite-sample uncertainty;
- shift scenarios are deferred, so no post-shift claim is possible;
- the candidate's safe-commit declaration policy may leave unresolved episodes;
  these remain failures by design;
- the historical v3 base 49/50 is preserved and is not repaired.

## 12. Human GO / NO-GO

The study may start only after a human records all of the following:

- [ ] accept candidate policy commit `e65e191` and the behavioral-equivalence
      report;
- [ ] accept manifest content SHA-256 `95c4e1…be80`;
- [ ] accept maps 210–239 and seeds 5–9;
- [ ] accept the 900-per-method static design and 2,700 total CPU episodes;
- [ ] accept H1/H2 as the only confirmatory family, with Holm and the 5-point
      minimum effect;
- [ ] accept G1–G6, including the fail-closed zero-tool gate and the fact that
      development evidence already shows possible G1/G4 failure;
- [ ] accept that v4−v3 is descriptive only;
- [ ] accept that no VLM/GPU arm is in scope;
- [ ] accept that no post-freeze candidate tuning or post-holdout amendment is
      permitted.

No box is checked by this document. The correct next action after review is a
recorded GO or NO-GO; the default is NO-GO.
