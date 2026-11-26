# v0.5 development results

Status: completed CPU-only development run of v0.5-separated-epistemic. This is
not a confirmatory result. No reserved map, holdout, GPU, VLM, or PR #22 work
was performed.

## 1. Identity and artifacts

- Candidate: v0.5-separated-epistemic, method name v5.
- Candidate commit: 751a3414ad8815087f64d85ad5a1523af11077f4.
- Base diagnosis commit: 0146684dc420b56fb8191ee68e265896ec0a5c5f.
- Protocol: docs/EVALUATION_PROTOCOL_V2.md at protocol commit ae1c112.
- Design: docs/V05_DESIGN.md.
- Frozen repaired v0.4: e65e1915503246f0f07ed15c4884ac6952925ee8.
- Development manifest: results/p0_dev_static_manifest.json, SHA-256
  cfc6504ec4ad4a27494947e0200581ec86d4e781cb278719da149dc6e5b38b00.
- GridConfig SHA-256:
  67f7be9716947a9b30f6d30fdcdd72f0abb95277ea7f2f5e8eb09cde03cb8421.
- Evaluation version: grid-identification-v2.
- Data: 300 static identification episodes, exposed maps 0-9 and seeds 0-4.
  Three methods (B, C, D) were run, 900 episode records total.

Official run directory: results/grid_v05_official_final.

- records.jsonl SHA-256:
  00B6F22A5C39AE2BE14C1C422ADC57C992C77F73B72761A6532BBEE471879EBD
- summary.json SHA-256:
  B2D8D18611EFA9AC15FE7DBEF369760986DC81A15666389FE64254A62805C922

A complete rerun in a temporary directory produced byte-identical
records.jsonl and summary.json (same two SHA-256 digests).

## 2. Headline result

B, C, and D produced identical task outcomes, terminal MAP, declarations,
tool events, and action traces on all 300 episodes. C differs from B only in
445 REQUEST_EVIDENCE events and 157 fewer recorded abstention events; no
request changed a control action or a declaration. D duplicates B by
construction.

The candidate therefore provides no behavioural evidence that the active
request channel helps. The observed gain over repaired v0.4 comes from the
separated declaration/readout and the one final-observation declaration
opportunity (B versus v4), not from C's requests.

Against repaired v0.4, B/C/D:

- held task success at 298/300 and terminal MAP at 297/300;
- increased final correct declarations from 263 to 289 (+26);
- reduced final wrong declarations from 4 to 3 (-1);
- reduced unresolved from 33 to 8 (-25);
- increased coverage from 267 to 292 (+25);
- increased strict clean accuracy from 262 to 282 (+20);
- increased any-wrong episodes from 5 to 10;
- introduced one new final wrong declaration, m3-s2-slippery, and no task or
  terminal-MAP regression.

The six recorded point-estimate readiness criteria all pass, but the 95%
Wilson lower bounds for overall, hazard, and sign accuracy fall below the
recorded targets. The strict clean rate is 94.0%. The result is therefore a
nominal development pass, not a robust demonstration of the targets.

## 3. Method-level results

| metric | B | C | D |
|---|---:|---:|---:|
| task successes | 298 | 298 | 298 |
| terminal MAP correct | 297 | 297 | 297 |
| final declarations correct | 289 | 289 | 289 |
| final declarations wrong | 3 | 3 | 3 |
| unresolved | 8 | 8 | 8 |
| coverage | 292 | 292 | 292 |
| unconditional accuracy | 96.33% | 96.33% | 96.33% |
| strict clean correct | 282 | 282 | 282 |
| strict clean accuracy | 94.00% | 94.00% | 94.00% |
| any-wrong episodes | 10 | 10 | 10 |
| wrong declaration events | 10 | 10 | 10 |
| revision episodes | 7 | 7 | 7 |
| first correct, mean steps | 3.6609 | 3.6609 | 3.6609 |
| first correct, median steps | 3 | 3 | 3 |
| tool units, total | 383 | 383 | 383 |
| tool units, mean | 1.2767 | 1.2767 | 1.2767 |
| tool units, maximum | 3 | 3 | 3 |
| zero-tool task successes | 121 | 121 | 121 |
| zero-tool hazard/sign successes | 0 | 0 | 0 |
| decisions, mean | 9.1467 | 9.1467 | 9.1467 |
| decisions, maximum | 62 | 62 | 62 |
| return, mean | 9.2095 | 9.2095 | 9.2095 |
| cost, mean | 8.2100 | 8.2100 | 8.2100 |
| abstention events | 522 | 365 | 522 |
| request events | 0 | 445 | 0 |
| request statuses | none | executed 281, expired 84, unmet 80 | none |

B has 299 declaration events: 286 pre-action and 13 at the final observation.
Of its 292 final declarations, 279 came from a pre-action decision and 13 from
the final observation. Across B/C/D, 897 declaration events comprise 858
pre-action and 39 final-observation events. Every method's 13 final-observation
declarations are the same 13 episodes: 12 correct and 1 wrong.

There are 7 revision episodes and 7 revisions; every revision ends correct.
Five revisions occur pre-action and two at the final observation. The 7 early
wrong declarations account for 7 of the 10 any-wrong episodes. The other 3
any-wrong episodes are the three final wrong declarations.

Strict clean decomposition: 300 - 8 unresolved - 3 final wrong - 7 early-wrong
revision episodes = 282.

Final-declaration precision is 289/292 = 98.97% (v4 was 263/267 = 98.50%).
Unconditional accuracy is 289/300 = 96.33%.

## 4. Per-rule results (B/C/D identical)

| rule | n | task | final correct | final wrong | unresolved | coverage | strict | any-wrong | mean tools |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 50 | 50 | 45 | 0 | 5 | 45 | 43 | 2 | 1.04 |
| remap_swap | 50 | 50 | 50 | 0 | 0 | 50 | 50 | 0 | 0.00 |
| remap_rotate | 50 | 50 | 49 | 0 | 1 | 49 | 45 | 4 | 0.34 |
| slippery | 50 | 50 | 49 | 1 | 0 | 50 | 49 | 1 | 0.28 |
| hazard_goal_swap | 50 | 49 | 47 | 1 | 2 | 48 | 46 | 2 | 3.00 |
| sign_composition | 50 | 49 | 49 | 1 | 0 | 50 | 49 | 1 | 3.00 |

The two task failures are m0-s3 (sign_composition, declared
hazard_goal_swap) and m5-s1 (hazard_goal_swap, declared sign_composition).
Both spend 3 tool units, reach the 60-step cap, and are wrong from a pre-action
declaration. In both cases a 5% noisy INSPECT of the blue face reports the
opposite role, and that observation is assigned a very high likelihood to the
wrong rule. There is no post-declaration correction.

The 8 unresolved episodes are m1-s1-remap_rotate, m1-s4-hazard,
m6-s0 through m6-s4 base (5), and m8-s4-hazard. All other final declarations
are correct except the three wrong episodes m0-s3, m3-s2, and m5-s1.

The final wrong declarations are:

| scenario | true rule | declared rule | source | confidence | tools | steps |
|---|---|---|---|---:|---:|---:|
| m0-s3-identification-sign_composition | sign_composition | hazard_goal_swap | pre_action | 0.9744 | 3 | 60 |
| m3-s2-identification-slippery | slippery | remap_rotate | final_observation | 0.9419 | 0 | 2 |
| m5-s1-identification-hazard_goal_swap | hazard_goal_swap | sign_composition | pre_action | 0.9744 | 3 | 60 |

## 5. Per-map results (B/C/D identical)

| map | n | task | final correct | final wrong | unresolved | strict | any-wrong | mean tools |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m0 | 30 | 29 | 29 | 1 | 0 | 29 | 1 | 1.17 |
| m1 | 30 | 30 | 28 | 0 | 2 | 26 | 2 | 1.23 |
| m2 | 30 | 30 | 30 | 0 | 0 | 27 | 3 | 1.17 |
| m3 | 30 | 30 | 29 | 1 | 0 | 29 | 1 | 1.20 |
| m4 | 30 | 30 | 30 | 0 | 0 | 30 | 0 | 1.17 |
| m5 | 30 | 29 | 29 | 1 | 0 | 27 | 3 | 1.30 |
| m6 | 30 | 30 | 25 | 0 | 5 | 25 | 0 | 1.63 |
| m7 | 30 | 30 | 30 | 0 | 0 | 30 | 0 | 1.33 |
| m8 | 30 | 30 | 29 | 0 | 1 | 29 | 0 | 1.40 |
| m9 | 30 | 30 | 30 | 0 | 0 | 30 | 0 | 1.17 |

The map-level unevenness is concentrated in the five unresolved base episodes
on m6 and the two timeouts caused by noisy blue inspections on m0 and m5.

## 6. Paired comparison with historical baselines

Historical v3 and repaired v4 were scored under protocol v1. They are control
A and are compared descriptively only; they are not controlled protocol-v2
arms.

| metric | v3 (A) | v4 (A) | B/C/D (v2) |
|---|---:|---:|---:|
| task successes | 287 | 298 | 298 |
| terminal MAP correct | 297 | 297 | 297 |
| final correct | 284 | 263 | 289 |
| final wrong | 4 | 4 | 3 |
| unresolved | 12 | 33 | 8 |
| coverage | 288 | 267 | 292 |
| unconditional accuracy | 94.67% | 87.67% | 96.33% |
| strict clean | 278 | 262 | 282 |
| any-wrong episodes | 10 | 5 | 10 |
| mean tool units | 1.2033 | 1.2767 | 1.2767 |

The historical v4 confirmed its recorded failure: 263/300 unconditional and
44/50 sign-composition, so it fails the overall 95% and sign 90% gates. It
remains closed as a confirmatory candidate.

### 6.1 v4 to B/C/D transitions

| v4 v1 status | v2 status | episodes |
|---|---|---:|
| correct | correct | 263 |
| unresolved | correct | 24 |
| unresolved | wrong | 1 |
| unresolved | unresolved | 8 |
| wrong | correct | 2 |
| wrong | wrong | 2 |

No v4-correct episode became wrong or unresolved. Task success and terminal MAP
are exactly invariant: 298/298 and 297/297 matched pairs. The v4 unresolved to
wrong transition is m3-s2-slippery, discussed below. A naive scenario-level
two-sided exact sign test for correct versus not-correct gives p about 3.0e-8
(26 gains, 0 correct losses); this ignores map clustering and is descriptive,
not confirmatory.

### 6.2 v3 to B/C/D transitions

| v3 v1 status | v2 status | episodes |
|---|---|---:|
| correct | correct | 280 |
| unresolved | correct | 7 |
| wrong | correct | 2 |
| correct | unresolved | 4 |
| unresolved | unresolved | 4 |
| unresolved | wrong | 1 |
| wrong | wrong | 2 |

B/C/D gain 9 correct episodes and lose 4, a net +5 over v3, but the paired
change is not significant (naive two-sided exact sign test p about 0.27).
v3's task success is 287 versus 298; that difference is not a controlled
comparison.

## 7. The new final wrong declaration: m3-s2

m3-s2 has true rule slippery and no shift. The agent starts at (4,2); the red
goal is (2,2), the blue marker is (4,4), and green is (1,1). It takes two
DOWN actions and reaches the red goal: (4,2) -> (3,2) -> (2,2). The episode
succeeds in 2 physical steps with 0 tool units.

Public evidence is two movement transitions. Under the public likelihood
model, each observed transition has likelihood 0.90 under remap_rotate and
0.20 under slippery (the other rules are 0.05). After two transitions the
posterior is remap_rotate 0.9418604651, slippery 0.1538461538; the top margin
is 0.8953488372. The declaration gate is satisfied with two compatible
references. B's final observation declares remap_rotate at decision 3, step 2,
after the task has closed. C also recorded an unmet SIGN request at decision 1;
B abstained there. C's request did not change either action.

This is a genuine successful misdiagnosis and an identifiability collision, not
a timing bug. Two nominal slippery moves are possible (0.20 each) but are much
less likely than deterministic rotation; a short successful episode provides no
tool evidence to separate them. Repaired v4 reached the same wrong terminal
MAP and left the episode unresolved. v0.5's final readout converts that
unresolved episode into a confident wrong declaration. The task success does
not validate the rule. No threshold was changed after seeing this case.

## 8. Original 37 problematic episodes, separately

These are the 37 episodes that were unresolved or wrong under historical v4.
They are reported as a fixed historical set, not selected from v0.5 outcomes.

Summary under v2: 26 correct, 8 unresolved, 3 wrong; 5 episodes have any
wrong declaration. The 5 any-wrong episodes are m0-s3, m1-s1-base, m2-s2-base,
m3-s2-slippery, and m5-s1. Comparison: v4 had 0 correct, 33 unresolved, 4
wrong on this set; v2 therefore adds 26 correct, removes 25 unresolved, and
reduces final wrong by one, while the new m3-s2 wrong and the two corrected
early-wrong base episodes raise the any-wrong count to 5.

| scenario | true rule | v4 v1 | v2 final | declared | source | any-wrong | tools | steps |
|---|---|---|---|---|---|---|---:|---:|
| m0-s3-identification-sign_composition | sign_composition | wrong | wrong | hazard_goal_swap | pre_action | yes | 3 | 60 |
| m0-s3-identification-slippery | slippery | unresolved | correct | slippery | pre_action | no | 0 | 6 |
| m1-s0-identification-remap_swap | remap_swap | unresolved | correct | remap_swap | pre_action | no | 0 | 2 |
| m1-s0-identification-slippery | slippery | unresolved | correct | slippery | final_observation | no | 0 | 3 |
| m1-s1-identification-base | base | wrong | correct | base | final_observation | yes | 1 | 7 |
| m1-s1-identification-remap_rotate | remap_rotate | unresolved | unresolved | - | - | no | 0 | 3 |
| m1-s1-identification-remap_swap | remap_swap | unresolved | correct | remap_swap | pre_action | no | 0 | 2 |
| m1-s1-identification-slippery | slippery | unresolved | correct | slippery | pre_action | no | 1 | 3 |
| m1-s2-identification-remap_swap | remap_swap | unresolved | correct | remap_swap | pre_action | no | 0 | 2 |
| m1-s3-identification-remap_swap | remap_swap | unresolved | correct | remap_swap | pre_action | no | 0 | 2 |
| m1-s4-identification-hazard_goal_swap | hazard_goal_swap | unresolved | unresolved | - | - | no | 3 | 2 |
| m1-s4-identification-remap_swap | remap_swap | unresolved | correct | remap_swap | pre_action | no | 0 | 2 |
| m2-s2-identification-base | base | wrong | correct | base | final_observation | yes | 1 | 12 |
| m3-s0-identification-remap_rotate | remap_rotate | unresolved | correct | remap_rotate | final_observation | no | 0 | 2 |
| m3-s1-identification-base | base | unresolved | correct | base | final_observation | no | 1 | 5 |
| m3-s1-identification-remap_rotate | remap_rotate | unresolved | correct | remap_rotate | final_observation | no | 0 | 2 |
| m3-s1-identification-slippery | slippery | unresolved | correct | slippery | final_observation | no | 1 | 7 |
| m3-s2-identification-base | base | unresolved | correct | base | final_observation | no | 1 | 5 |
| m3-s2-identification-remap_rotate | remap_rotate | unresolved | correct | remap_rotate | final_observation | no | 0 | 2 |
| m3-s2-identification-slippery | slippery | unresolved | wrong | remap_rotate | final_observation | yes | 0 | 2 |
| m3-s3-identification-base | base | unresolved | correct | base | final_observation | no | 1 | 5 |
| m3-s3-identification-remap_rotate | remap_rotate | unresolved | correct | remap_rotate | final_observation | no | 0 | 2 |
| m3-s4-identification-remap_rotate | remap_rotate | unresolved | correct | remap_rotate | final_observation | no | 0 | 3 |
| m5-s0-identification-sign_composition | sign_composition | unresolved | correct | sign_composition | pre_action | no | 3 | 4 |
| m5-s1-identification-hazard_goal_swap | hazard_goal_swap | wrong | wrong | sign_composition | pre_action | yes | 3 | 60 |
| m5-s1-identification-sign_composition | sign_composition | unresolved | correct | sign_composition | pre_action | no | 3 | 3 |
| m5-s2-identification-sign_composition | sign_composition | unresolved | correct | sign_composition | pre_action | no | 3 | 3 |
| m5-s3-identification-sign_composition | sign_composition | unresolved | correct | sign_composition | pre_action | no | 3 | 3 |
| m5-s4-identification-sign_composition | sign_composition | unresolved | correct | sign_composition | pre_action | no | 3 | 3 |
| m6-s0-identification-base | base | unresolved | unresolved | - | - | no | 1 | 4 |
| m6-s0-identification-slippery | slippery | unresolved | correct | slippery | pre_action | no | 1 | 3 |
| m6-s1-identification-base | base | unresolved | unresolved | - | - | no | 1 | 2 |
| m6-s2-identification-base | base | unresolved | unresolved | - | - | no | 1 | 2 |
| m6-s3-identification-base | base | unresolved | unresolved | - | - | no | 3 | 6 |
| m6-s4-identification-base | base | unresolved | unresolved | - | - | no | 1 | 4 |
| m8-s1-identification-slippery | slippery | unresolved | correct | slippery | pre_action | no | 0 | 4 |
| m8-s4-identification-hazard_goal_swap | hazard_goal_swap | unresolved | unresolved | - | - | no | 3 | 8 |

## 9. Costs, budgets, and readiness

Tool spend: 383 units over 300 episodes, mean 1.2767, maximum 3. Distribution:
0 units on 121 episodes, 1 unit on 77, 2 units on 0, 3 units on 102. No episode
exceeds the 3-unit budget. Every hazard or sign task success uses tools; the
zero-tool hazard/sign successes count is 0. The information-acquisition cost
is the 383 tool units, mean 1.2767 per episode; these are budget units, not a
separate monetary measure. Physical steps average 8.21; decisions average
9.1467 with maximum 62, below max_decisions=200.

| recorded criterion | result | point | 95% Wilson interval | interpretation |
|---|---|---:|---|---|
| overall unconditional accuracy >= 95% | 289/300 | 96.33% | [93.55%, 97.94%] | point pass; lower bound below 95% |
| hazard declarations >= 90% | 47/50 | 94.00% | [83.78%, 97.94%] | point pass; lower bound below 90% |
| sign declarations >= 90% | 49/50 | 98.00% | [89.50%, 99.65%] | point pass; lower bound below 90% |
| base task success = 100% | 50/50 | 100% | [92.87%, 100%] | observed pass |
| zero raw zero-tool hazard/sign successes | 0 of 98 | 0 | - | pass |
| tool spend <= 3 units/episode | max 3 | 3 | - | pass |

All six recorded point criteria pass. The 95% lower bounds for the three
proportion criteria do not. The strict clean rate is 282/300 = 94.00%, below
the 95% overall target, even though the recorded unconditional criterion is
met. Any-wrong is 10/300 = 3.33%, up from v4's 5/300 = 1.67%.

These readiness criteria were development targets, not confirmatory gates.
Passing them does not establish confirmatory validity, and the wide intervals
must not be presented as a 95% accuracy claim.

## 10. Answer to the coverage question

Coverage rises by 25 episodes versus v4. The composition is:

- 24 v4-unresolved episodes become correct;
- 2 v4-wrong episodes become correct;
- 1 v4-unresolved episode becomes a new final wrong (m3-s2).

Final wrong falls by one (4 to 3) because two old wrong episodes are corrected
and one new wrong appears. Task success is unchanged at 298/300, and terminal
MAP is unchanged at 297/300. There are no task regressions and no terminal-MAP
regressions. The cost is not more final wrong declarations in aggregate, but
one newly asserted wrong rule and five more any-wrong episodes because seven
early wrong declarations are retained in the audit even when later revised.

## 11. Limitations

- Exposed development maps and seeds only; no reserved maps, holdout, or
  confirmatory evaluation.
- 300 episodes with map clustering; the Wilson intervals are scenario-level and
  optimistic about effective sample size.
- The point-target pass is not robust to the recorded uncertainty. A reviewer
  requiring 95% lower confidence would not accept the overall, hazard, or sign
  targets as demonstrated.
- The 5% inspect noise creates the two timeout failures and the confident wrong
  pre-action declarations; the public likelihood model treats the noisy result
  as strong evidence.
- The final-observation readout can turn a wrong terminal MAP into a confident
  wrong assertion (m3-s2). It adds no evidence and cannot be corrected after
  termination.
- Active requests are behaviourally inert. C is identical to B on actions,
  outcomes, tools, and declarations, so this experiment does not establish a
  benefit for REQUEST_EVIDENCE. D duplicates B and is not an independent
  ablation.
- Declaration confidence is the posterior top probability, not a calibrated
  error probability. Evidence references record compatibility, not a minimal
  sufficient explanation.
- Static identification only; no shift scenarios. No GPU or VLM evidence.
- Historical v3/v4 comparisons use protocol v1 and are descriptive only.

## 12. Test and reproducibility status

The full CPU test suite passes: 528 tests, including 21 new v0.5 policy and v2
scorer tests. The new tests cover declaration independence, terminal timing, abstention,
revision and wrong-event accounting, no terminal-MAP conversion, insufficient
evidence, exhausted budget, public/evaluator isolation, determinism, finite
decision bounds, manifest protection, and the D-equals-B construction.

The official records and summary are byte-identical across two complete runs.
This is determinism evidence, not external validity.
