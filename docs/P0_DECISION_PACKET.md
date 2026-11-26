# P0 decision packet and freeze-readiness review

Date: 2026-11-25. Branch: `research/identification-protocol-v1`.
Status: **freeze-ready for human GO/NO-GO; no confirmatory episode has been run
on maps 210–239, no candidate tuning was performed, and nothing was merged.**

This packet supersedes the earlier draft's proposal to freeze an unselected
candidate and to run 1,650 full scenarios per method. It records the nine human
decisions, the exact frozen candidate and manifest, the development evidence
that bears on the GO decision, and the remaining risks.

## 1. Decisions recorded

| # | Decision | Recorded state |
|---|---|---|
| 1 | Candidate | repaired v0.4 (`diagnostic_belief_v4`) is the primary candidate. No v0.5, no policy tuning. Declaration instrumentation is allowed only at the existing safe-commit point and must be behaviourally inert. |
| 2 | Holdout | maps **210–239**, seeds **5–9** provisionally approved, subject to identity, hash, exposure and completeness checks. No candidate rollout before final freeze and explicit GO. |
| 3 | Scope | static-only primary experiment: 30 maps × 5 seeds × 6 rules = **900 static episodes per method**; three methods; **2,700 CPU episodes**. The 1,650-per-method full design is deferred. |
| 4 | Statistics | Holm approved in principle; exact confirmatory family, endpoint, direction, effect and rejection rule are enumerated below. Map-level paired analysis and map-cluster intervals are primary; episode-level McNemar is supplementary. Development-only power rationale is recorded. |
| 5 | Base gate | retained at **150 episodes, 100% observed task success**, as engineering readiness. Not a population-reliability claim. Report intervals, base MAP and declaration errors separately. Do not relax after seeing candidate maps. |
| 6 | Comparators | repaired v0.4 candidate; unchanged v0.3 historical diagnostic reference; no_adaptation non-adaptive baseline. No claim of superiority over stronger active-probing policies from no_adaptation results. |
| 7 | VLM | all VLM/GPU arms excluded. Phase 1 CPU only. |
| 8 | Evidence | fail-closed zero-tool hazard/sign gate retained. No evidence-coding exception is approved. Raw and evidence-adjusted counts must both be reported. |
| 9 | Endpoints | terminal MAP is the paired comparison endpoint for all three methods. Explicit declaration accuracy, event coverage, incorrect rate, unresolved rate, confidence and calibration are reported separately. Unresolved episodes remain in the unconditional declaration denominator. Terminal-MAP correctness is not called confirmed active identification. |

"Approved in principle" is not the explicit GO in section 10. The default after
review remains NO-GO.

## 2. Candidate and code freeze

| Item | Frozen value |
|---|---|
| Candidate policy commit | `e65e1915503246f0f07ed15c4884ac6952925ee8` |
| Behavioural baseline (pre-instrumentation) | `f4b43a4ede7f76ab9d5b1f45178fe648ae419042` |
| Coverage/reporting correction | `a46ed21` (evaluator only; does not change policy behaviour) |
| Manifest commit | `21db1e2` |
| Base revision | `origin/main` = `f4b43a4ede7f76ab9d5b1f45178fe648ae419042` |
| Protected prior branch | `research/belief-state` = `3fbebf096d43fc07050fd88ffc821d568f122b4a` |
| Protected prior candidate branch | `research/v05-protocol-and-candidate` = `0478851453dc50d9624b8aa7ce0becabd35a0247` |
| Draft PR #21 | `https://github.com/Tienledaithandong/ruleshift-vlm/pull/21` (untouched; not merged) |

Policy/instrumentation review of `d9e3e83` found and repaired the following
issues: terminal MAP and Brier/NLL had to use the terminal true rule, not the
initial rule; commitments had to be scored only on static episodes; wrong
declarations and earlier wrong declarations had to be separate fields; map
membership had to be an exact id set with duplicate-layout rejection; the runner
must not allow skipping layout verification; and evidence-adjusted zero-tool
counts must not rescue raw failures.

The candidate's declaration hook is attached only where the existing policy
already takes a safe commit. It records step, declared rule, confidence and
margin; it does not choose actions, select probes, update the posterior or alter
the terminal plan. Event ordering is validated and adjacent duplicate rules are
suppressed. Fallback commitments are not relabelled as declarations, and terminal
MAP is never used to synthesize one.

## 3. Behavioural equivalence report

The fixture `tests/fixtures/v04_equivalence_maps0_9_seeds0_4.json` was generated
from the pre-instrumentation policy on the exposed 300 static scenarios. The test
runs all 300 scenarios per method and compares:

- action-trace SHA-256;
- task success;
- final rule estimate;
- environment steps;
- tool units.

Result: **300/300 v3 and 300/300 v4 scenarios identical; 0 behavioural
differences; 3 focused tests pass.** The declaration event count differs only
because the hook is new; actions do not. Declaration event contract and ordering
tests also pass. The no_adaptation policy exposes no declaration API.

Instrumentation is therefore treated as observationally non-invasive. The
development declaration accuracy below is **not** evidence of equivalence; it is
a separate property of the candidate's declaration policy.

## 4. Frozen static design

| Item | Value |
|---|---:|
| Maps | 210–239 (30) |
| Seeds | 5–9 (5) |
| Initial rules | base, remap_swap, remap_rotate, slippery, hazard_goal_swap, sign_composition (6) |
| Static episodes per method | 900 |
| Methods | no_adaptation, v3, v4 |
| Total CPU episodes | 2,700 |
| Base / hazard / sign per method | 150 / 150 / 150 |
| Shift or no-shift episodes | 0 (deferred) |

No 1,650-per-method full design is run. No policy rollout on maps 210–239 has
occurred. The only work on those maps was public layout generation, hashing and
manifest construction; no environment reset, policy instantiation or episode
record exists for them.

## 5. Manifest freeze

Tracked path: `artifacts/manifests/identification_v1_static.json`

| Hash | Value |
|---|---|
| Manifest content SHA-256 (authoritative, canonical entries) | `95c4e13e533d1161af9cf6971df905c3222d94fbf091d3bf71e5ff3b37cabe80` |
| Local Windows working-file SHA-256 | `646f89632319a443943fc0e1add5cffb9579764e18cef07cdc7b325620d9fef5` |
| Config SHA-256 | `67f7be9716947a9b30f6d30fdcdd72f0abb95277ea7f2f5e8eb09cde03cb8421` |
| Entries | 900 |
| Scope | `static_identification` |
| Unique public layouts | 30 |
| Collisions with maps 0–199 | 0 |

The manifest content hash is the identity to verify. The working-file hash is a
convenience record for this Windows checkout and is line-ending dependent; it is
not the scientific identity. The committed blob is tracked with Git-normalized
line endings.

Validation performed without episode execution:

- exact map set 210–239 and seed set 5–9;
- exactly 900 unique scenario ids, six per map-seed combination;
- exactly six initial rules per map-seed, including the two hazard/sign rules;
- all layout hashes recomputed through the public map generator;
- 30 unique candidate layouts and zero collisions with maps 0–199;
- manifest scope and content hash consistent; missing/extra/duplicate ids fail
  closed;
- no candidate scenario id appears anywhere under `results/`; the only
  `m21x-s…` references are the 900 entries inside the frozen manifest itself.

The prior audit found 56 files and 14,938 scenario-id references, all maps 0–199,
none 210–239. A committed-ref scan found no candidate episode ids; six prose lines
on the prior branch name maps 200–209, not candidate scenarios. The candidate
bank was moved to 210–239 to avoid even that prose namespace.

## 6. Exact hypotheses and gates

Only H1 and H2 are p-valued and Holm-corrected. Both are one-sided positive,
map-level, equal-map mean paired differences in terminal-MAP accuracy.

| ID | Comparison | Reject only if |
|---|---|---|
| H1 (primary paired) | v4 − no_adaptation | Holm p < 0.05, mean ≥ 5 pp, cluster CI lower > 0 |
| H2 (historical reference) | v3 − no_adaptation | same rule |
| D1 (descriptive, not in family) | v4 − v3 | two-sided map-cluster interval; no superiority or non-inferiority claim |

Absolute gates:

| Gate | Requirement |
|---|---|
| G1 | v4 unconditional explicit declaration accuracy ≥ 95.0% (900) |
| G2 | v4 base task success 150/150 (do not relax) |
| G3 | v4 hazard_goal_swap explicit declarations ≥ 90.0% (150) |
| G4 | v4 sign_composition explicit declarations ≥ 90.0% (150) |
| G5 | zero raw candidate zero-tool task successes on hazard/sign (300) |
| G6 | every candidate episode ≤ 3 tool units |

G1 is an absolute identification gate; H1 is the paired terminal-behaviour
benchmark. Passing H1 without G1 supports only a terminal-MAP advantage, not an
active-identification claim. Passing G1 without H1 supports only absolute
identification, not an advantage over no_adaptation.

The exact 95% / 90% point-estimate rules and the 5-point minimum effect are
frozen. Wilson and Clopper–Pearson intervals are reported but do not replace the
point-estimate gate. No gate is a population guarantee.

## 7. Development evidence (exposed; not confirmatory)

Maps 0–9, seeds 0–4, 300 static episodes per method, candidate commit `e65e191`:

| Method | Task | Terminal MAP | Correct decl. | Wrong | Unresolved | Event coverage | Unconditional decl. accuracy | Base task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 | 287 | 297 | 284 | 4 | 12 | 96.0% | 94.67% | 49/50 |
| v4 | 298 | 297 | 263 | 4 | 33 | 89.0% | **87.67%** | **50/50** |
| no_adaptation | 128 | 50 | 0 | 0 | 0 (unavailable 300) | 0.0% | unavailable | 50/50 |

Additional v4 details: 1 successful misdiagnosis; 0 MAP-correct/task-failed; 1
episode with an earlier wrong declaration; Brier 0.025507; NLL 0.056736; mean
tool units 1.2767; 0 raw zero-tool hazard/sign successes; hazard declarations
47/50 = 94%; sign declarations 44/50 = 88%.

The candidate therefore **fails G1 and G4 on the exposed development bank** and
passes G2/G3/G5/G6 on that bank. The 150-episode confirmatory endpoints have not
been run. This is the central GO risk: the frozen candidate may fail its primary
identification gate on the candidate bank. The protocol does not tune around it.

Map-level development contrasts (exposed): v4−no_adaptation mean 0.8233, 95%
interval 0.8133–0.8333; v3−no_adaptation identical; v4−v3 mean 0.0, interval
−0.01–0.01. These are development diagnostics only.

## 8. Sample-size justification

Observed development map-level SDs are 0.0161 (v4−baseline), 0.0161
(v3−baseline) and 0.0157 (v4−v3). Planning deliberately uses SD = 0.10, not the
small observed value. With 30 maps, one-sided alpha 0.025 and 80% power:

    MDE = (1.960 + 0.842) × 0.10 / sqrt(30) ≈ 0.0512 (5.1 pp).

At one-sided 0.05 the MDE is about 4.5 pp. The cluster-interval half-width under
the planning SD is about 3.6 pp. The E1 900-episode Wilson half-width near 95% is
about 1.4 pp; the 150-episode base gate has a zero-failure one-sided 95% upper
bound of about 2.0%. The design is powered for the 5-point paired effect, not for
proving an absolute 95% population rate. The absolute gate remains a
finite-sample decision rule.

## 9. Engineering status

- Full CPU regression suite: **496 passed** (exit 0).
- Focused declaration, manifest and map-analysis tests: **30 passed**.
- Behavioural declaration equivalence: **3 passed**, 600/600 scenarios identical.
- `scripts/reproduce_all.py`: **exit 0**, all synthetic artifacts regenerated;
  `determinism=PASS`.
- CLI `--help` exit 0 for the manifest builder, identification runner,
  reproducibility driver and exposure audit.
- Raw development artifacts archived under
  `results/archive/p0_2026-11-25/` (ignored by Git), 85 files / 8,135,664 bytes
  (69 historical `grid_phase0` files, 18 top-level plus 51 nested);
  84 listed artifacts verified against `SHA256SUMS.txt`; checksum-manifest
  SHA-256 `4735b8fc393359f3b23d9ce4c01d6fc049f0fab8f1ae035740ae5b762516df59`.
- The archive is development evidence, not a confirmatory result set. Nothing
  from it is committed as a result.

The current branch intentionally has no confirmatory entry point: the development
runner refuses maps above 199. After a GO, a separate authorization commit may
add a hash-pinned confirmatory entry point without changing candidate policy or
analysis code. That commit must be reviewed before execution.

## 10. Human GO / NO-GO

The confirmatory CPU experiment requires a separate explicit human GO. Before
giving it, the reviewer must accept:

1. candidate policy `e65e191` and behavioural equivalence;
2. manifest `95c4e1…be80`, maps 210–239, seeds 5–9;
3. 900 static episodes per method and 2,700 total CPU episodes;
4. H1/H2 only, Holm, 5-point minimum effect, map-level intervals;
5. G1–G6, including the known development G1/G4 failure risk;
6. v4−v3 descriptive only; no baseline declaration API;
7. no VLM/GPU, no shift, no tuning, no post-holdout amendment.

**Stop condition:** freeze-readiness packet plus draft PR only. Do not execute
the confirmatory holdout, inspect candidate-map outcomes, tune the candidate,
merge, or consume GPU. If the reviewer cannot accept the known gate risk, the
correct action is NO-GO or a separately pre-registered new candidate, not a
silent protocol change.
