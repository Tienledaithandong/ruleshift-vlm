# VLM Decision Next Gate

**Current gate:** CPU infrastructure complete; model inference not authorized.
**Required action:** a human GO or NO-GO on a bounded development pilot.
**Not authorized by this document:** any model call, GPU run, confirmatory
analysis, reserved-map use, or v0.6 candidate.

## 1. Where the project stands

The symbolic candidate-development cycle is closed. v0.5 is retained as a
development reference, not promoted as confirmatory. The active-request
manipulation produced no observed behavioral difference, and the reserved-map
confirmation is deferred.

A separate decision-quality benchmark now exists on branch
research/vlm-decision-microbenchmark:

- 118 development contexts with exact reference labels;
- deterministic CPU baselines;
- an RGB decision card and paired nuisance/diagnostic controls;
- leakage tests and a public-only cache boundary;
- a nine-arm experiment matrix;
- a feasibility protocol with stop rules.

The CPU evidence is sufficient to decide whether to run a model. It is not
sufficient to claim that a model reasons well.

## 2. The decision requested

Choose exactly one:

**A. GO for a bounded development inference pilot.** Authorize the nine arms on
the 118 development contexts, with the frozen checkpoint and parser, under the
stop rules in the feasibility protocol. This is a development/feasibility run.
It does not promote v0.5 and does not open reserved maps.

**B. NO-GO.** Keep the CPU benchmark as a development artifact. Do not run a
model. Revisit only with a new, explicit research question and a new
authorization.

**C. REVISE.** Do not run yet. Ask for a specific change to the design, the
model choice, the controls, or the stop rules, then return to this gate.

There is no option that silently converts the CPU packet into a confirmatory
result.

## 3. Recommendation

The recommendation is **A, conditional on a separate human GO**: run a bounded
development pilot if the reviewer accepts the leakage boundary, the nine-arm
matrix, and the stop rules. The pilot is small enough to expose interface and
gross decision failures before any larger investment.

The recommendation is **NO-GO for confirmation** regardless of the pilot
outcome. A development pilot cannot establish a population effect, cannot
validate v0.5, and cannot justify reserved maps.

## 4. What a GO would cover

- Nine arms: three information conditions crossed with three response
  contracts.
- 118 development contexts, one deterministic pass.
- A design estimate of about 1,536 requests across the base, nuisance, and
  diagnostic blocks.
- Frozen Qwen2.5-VL-3B revision 66285546d2b821cf421d4f5eb2576359d3770cd3, local
  files only, no downloads.
- CPU/GPU execution only as authorized by the human; no automatic merge.
- Offline scoring with the exact reference, public-only prompts, and separate
  declaration scoring.
- A written result that reports invalid/fallback, action distributions, and
  per-stratum behavior before any aggregate claim.

A GO does **not** cover:

- reserved maps 210-239;
- confirmatory endpoints or p-values;
- v0.6 development;
- changes to frozen protocols or v0.5 artifacts;
- prompt tuning against labels;
- automatic merge or PR approval;
- treating declaration accuracy as control reward.

## 5. Feasibility gates for the pilot

These are engineering/feasibility gates, not scientific confirmation:

1. **Leakage gate.** No forbidden field appears in any prompt bytes, image
   metadata, cache key, or public JSONL row.
2. **Parser gate.** At least 80% of responses parse to a valid action for at
   least six of nine arms. Below that, report interface failure rather than
   decision quality.
3. **Non-degeneracy gate.** No arm is at least 95% one action. A degenerate arm
   is labeled as such and is not used to claim decision quality.
4. **Control gate.** Every visual arm has paired base/nuisance cards with the
   same context and label; every diagnostic row has a fresh reference label.
5. **Reference gate.** Every scored context has an exact label and a feasible
   action set; no row is scored against a missing or stale reference.
6. **Provenance gate.** Checkpoint revision, prompt template, parser, context
   hash, label hash, and analysis script are recorded before the first request.

If any gate fails, stop and report. Do not relax the gate after seeing results.

## 6. Descriptive scientific questions for the pilot

The pilot should answer, with counts and paired comparisons:

- Does an explicit public posterior improve decisions over prior-only text?
- Does the RGB card preserve decision quality relative to posterior text?
- Does asking for a short rationale change actions, and in which direction?
- Does a separate declaration field change control behavior?
- Do arms avoid both reflexive probing and premature commitment?
- Are failures concentrated in near ties, high noise, or budget exhaustion?
- Does the nuisance card change actions more than the semantic diagnostic does?

These are descriptive questions. They are not confirmatory hypotheses, and the
118-context development set is not a probability sample.

## 7. Interpretation rules

- A raw accuracy gain is not enough. Report over-probe, under-probe, and
  tool-selection accuracy.
- An arm that always commits is not a good decision-maker, even if it is close
  to the commit-majority baseline.
- An arm that always probes is not a good decision-maker, even if it is close to
  the probe baseline.
- A declaration score is reported separately. It cannot rescue a bad control
  policy.
- A visual effect is not averaged away by a text result. Report the interface
  gap.
- No result from this pilot is confirmatory, regardless of effect size.

## 8. Stop conditions

Stop immediately and do not continue the run if:

- a forbidden field or evaluator label appears in a prompt, image, or cache key;
- the checkpoint revision or local-only policy cannot be verified;
- a download is attempted;
- maps 210-239 or any reserved-map identifier appears;
- v0.6 or a new symbolic candidate appears;
- the parser or analysis plan changes after the first request;
- the run is being described as confirmatory or as a v0.5 promotion.

## 9. Artifacts to preserve

- v0.5 candidate commit 751a3414ad8815087f64d85ad5a1523af11077f4.
- v0.5 docs commit bc79ce1d7a54aea38fa7502452d646450121c09a.
- PR #22 head a3e614b093dfd8237898cb19ac83cab92b8bad50.
- PR #23 draft, including its published secondary text; the closure document
  supplies the m3-s2 terminal-posterior correction without rewriting PR #23.
- v0.5 records and summary hashes recorded in the closure document.
- The CPU benchmark source, tests, and docs on this branch.
- Generated benchmark artifacts under results/ are not committed by default.

## 10. Sign-off block

A human reviewer must fill this in before any model request:

- Decision: GO / NO-GO / REVISE
- Authorized scope: bounded development pilot only / none / revised scope
- Model revision verified: yes / no
- Local-files-only verified: yes / no
- Reserved maps excluded: yes / no
- Prompt and parser frozen: yes / no
- Stop rules accepted: yes / no
- Reviewer name and date:
- Notes:

Until this block is completed, the correct state is **NO-GO**.
