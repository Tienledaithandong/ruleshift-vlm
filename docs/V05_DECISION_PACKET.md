# v0.5 decision packet

Human decision requested: whether to freeze v0.5-separated-epistemic as a
confirmatory candidate and whether to authorize a later independent
confirmation. No confirmatory evaluation, reserved-map rollout, GPU/VLM work,
or PR #22 modification has been performed.

## 1. Recommendation

Result status: nominal development pass. All six readiness criteria recorded
before the official run pass on their point estimates. The candidate is
eligible for a conditional freeze-readiness review.

Scientific status: not confirmatory and not robustly demonstrated. The 95%
Wilson lower bounds for overall, hazard, and sign accuracy are below the
recorded targets, strict clean accuracy is 94.0%, and one new final wrong
declaration was introduced. The active request channel is behaviourally inert.

Recommendation: prepare a conditional freeze-readiness package for a later,
separately authorized independent confirmation. Do not make a confirmatory
claim now and do not execute reserved-map work under this packet. If the
decision rule requires the 95% lower confidence bound to clear the target, the
correct disposition is negative: do not freeze; close or redesign instead.

## 2. Candidate and evidence identity

- Candidate: v0.5-separated-epistemic, method v5.
- Candidate commit: 751a3414ad8815087f64d85ad5a1523af11077f4.
- Protocol: docs/EVALUATION_PROTOCOL_V2.md at ae1c112.
- Design: docs/V05_DESIGN.md.
- Development results: docs/V05_DEVELOPMENT_RESULTS.md.
- Ablation: docs/V05_ABLATION_REPORT.md.
- Frozen repaired v0.4: e65e1915503246f0f07ed15c4884ac6952925ee8.
- Exposed data only: maps 0-9, seeds 0-4, 300 static identification episodes.
- Manifest SHA-256:
  cfc6504ec4ad4a27494947e0200581ec86d4e781cb278719da149dc6e5b38b00.
- Config SHA-256:
  67f7be9716947a9b30f6d30fdcdd72f0abb95277ea7f2f5e8eb09cde03cb8421.
- Official records SHA-256:
  00B6F22A5C39AE2BE14C1C422ADC57C992C77F73B72761A6532BBEE471879EBD.
- Official summary SHA-256:
  B2D8D18611EFA9AC15FE7DBEF369760986DC81A15666389FE64254A62805C922.
- A complete rerun is byte-identical to the official records and summary.

## 3. Result summary

### 3.1 Main outcomes

| metric | repaired v4 (v1) | B/C/D (v2) |
|---|---:|---:|
| task success | 298/300 | 298/300 |
| terminal MAP | 297/300 | 297/300 |
| final correct | 263/300 | 289/300 |
| final wrong | 4/300 | 3/300 |
| unresolved | 33/300 | 8/300 |
| coverage | 267/300 | 292/300 |
| unconditional accuracy | 87.67% | 96.33% |
| strict clean | 262/300 | 282/300 |
| any-wrong episodes | 5/300 | 10/300 |
| mean tool units | 1.2767 | 1.2767 |

B, C, and D are identical on task, terminal MAP, declarations, tools, and every
action trace. C's 445 requests and 157 fewer abstention events changed no
control action and no declaration. D duplicates B; it is not an independent
ablation.

The gain over v4 is therefore attributable to the separated declaration
readout and final-observation opportunity, not to active REQUEST_EVIDENCE.
Task and terminal MAP are exactly preserved. Final wrong falls by one in
aggregate, but one new wrong assertion appears (m3-s2) and any-wrong doubles
because seven early wrong declarations are retained in the audit.

### 3.2 Recorded readiness criteria

| criterion | result | point | 95% Wilson interval | point status | robust status |
|---|---|---:|---|---|---|
| overall accuracy >= 95% | 289/300 | 96.33% | [93.55%, 97.94%] | pass | lower bound below 95% |
| hazard >= 90% | 47/50 | 94.00% | [83.78%, 97.94%] | pass | lower bound below 90% |
| sign >= 90% | 49/50 | 98.00% | [89.50%, 99.65%] | pass | lower bound below 90% |
| base task = 100% | 50/50 | 100% | [92.87%, 100%] | observed pass | observed only |
| zero zero-tool hazard/sign successes | 0/98 | 0 | - | pass | pass |
| tool spend <= 3/episode | max 3 | 3 | - | pass | pass |

Additionally, strict clean accuracy is 282/300 = 94.00%, below the 95% overall
target even though the recorded unconditional criterion is met. Final
declaration precision is 289/292 = 98.97%.

## 4. Positive evidence

- Task performance is preserved exactly against repaired v4: 298/300 and no
  task regressions.
- Terminal MAP is preserved exactly: 297/300, identical map for every episode.
- Final identification improves from 263 to 289 correct; unresolved falls from
  33 to 8; coverage rises from 267 to 292.
- No v4-correct episode becomes wrong or unresolved.
- The base-rule task criterion is 50/50.
- No zero-tool hazard/sign success; tool spend never exceeds 3.
- The result is deterministic across a full rerun.
- Seven revisions all end correct, showing that the revision interface works.
- The original 37 v4 failure episodes improve from 0 correct to 26 correct, with
  eight unresolved and three wrong; no task regression among them.

## 5. Adverse evidence and risks

1. The overall 95% target is met only on the point estimate. Its 95% Wilson
   lower bound is 93.55%. Hazard (83.78% lower) and sign (89.50% lower) also
   have lower bounds below their 90% targets. With 300 clustered development
   episodes, the target is not demonstrated at the recorded confidence.
2. Strict clean accuracy is 94.0%, below the 95% headline. Any-wrong is 10/300,
   double repaired v4's 5/300, because revisions retain their early wrong
   declarations in the audit.
3. One new final wrong declaration appears: m3-s2-slippery is declared
   remap_rotate with confidence 0.9419 after two movement observations, 0
   tools, and 2 steps. It is a successful task episode and a successful
   misdiagnosis. Repaired v4 reached the same wrong terminal MAP but left the
   episode unresolved. This is the exact cost of converting terminal
   posterior into an explicit assertion.
4. Two task failures remain: m0-s3 and m5-s1. In both, a 5% noisy INSPECT of
   the blue face reports the opposite role, producing a confident wrong
   pre-action declaration and a 60-step timeout.
5. Active REQUEST_EVIDENCE has no behavioural effect. C is B on every action,
   outcome, tool unit, and declaration. The request component of the
   hypothesis is unsupported.
6. D duplicates B, so the intended no-request ablation is not independent.
7. The v4 comparison is cross-protocol and descriptive. It is not a controlled
   causal estimate.
8. The final observation is a readout, not new evidence. It cannot correct a
   wrong posterior after termination and can increase confident wrong
   assertions, as m3-s2 shows.
9. The intervals ignore map clustering and therefore understate uncertainty.
   No confirmatory or reserved-map evidence exists.

## 6. Conditional freeze-readiness proposal

This is a proposal only. It does not authorize execution.

If the human accepts the nominal point pass and wants a confirmatory test:

1. Freeze the candidate at commit
   751a3414ad8815087f64d85ad5a1523af11077f4, with the protocol, config,
   manifest, and evaluator identity recorded above. No code, threshold, probe
   order, evidence model, or data-slice changes after freeze.
2. Keep the exposed 300-episode result labeled development. Do not reuse it as
   confirmatory evidence.
3. Prespecify a separate confirmatory run on reserved maps 210-239, seeds 0-4,
   all six rules: 900 episodes (30 maps x 5 seeds x 6 rules). This is the
   natural powered sample for the observed effect; it must be authorized
   explicitly and must not be run under this packet.
4. Preserve the six criteria and the strict/any-wrong/coverage reporting. Add a
   prespecified interval rule: a criterion passes only if its 95% lower bound
   clears the target. Do not lower a threshold after seeing confirmation data.
5. Report task, terminal MAP, unconditional correct, coverage, wrong,
   unresolved, strict clean, any-wrong, revisions, first-correct time, tools,
   steps, decisions, return, per-rule and per-map outcomes, request channels and
   statuses, and the original 37 episodes separately.
6. Stop rules: any task regression, any new wrong rule on a previously correct
   episode, a lower-bound failure, or a tool-budget violation ends the
   confirmation as negative. No v0.6, no policy patching, no threshold
   relaxation, no post-hoc subgroup rescue.
7. If confirmation fails, close the candidate and redesign the benchmark or the
   evidence model. If it passes, the result may support a freeze decision, but
   still not a claim beyond the evaluated grid.

The alternative disposition is equally valid: if the human requires robust
95% evidence from the development run itself, this packet is negative. In that
case do not freeze; close v0.5 and redesign rather than run more development
iterations.

## 7. Explicit non-actions

- No confirmatory evaluation.
- No reserved-map or holdout rollout.
- No GPU or VLM inference.
- No threshold change or policy patching after outcomes.
- No v0.6.
- No merge.
- No change to PR #22, which remains untouched at its prior draft head.
- Historical v1 results and frozen identities are not rewritten.

## 8. Human decision

Choose one:

- Option A: accept the nominal point pass, approve the conditional
  freeze-readiness package, and separately decide whether to authorize the
  900-episode reserved-map confirmation later.
- Option B: treat the interval evidence, strict 94%, new wrong declaration, and
  inert request channel as insufficient; do not freeze; close or redesign.

Recommendation: Option A only if the exposed result is explicitly treated as a
development basis for a later independent confirmation. Option B if the
decision requires the current run to demonstrate the targets at 95%
confidence. Do not describe either option as a confirmatory pass.
