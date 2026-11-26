# VLM Decision Feasibility Protocol

**Status:** CPU feasibility packet complete. Model inference not started.
**Decision required:** separate human GO before any VLM/GPU request.
**Scope:** development decision benchmark only.

## 1. Purpose

This protocol decides whether a small frozen-VLM feasibility pilot is worth
running, and what must be true before it can be run. It is not a confirmatory
protocol and it does not authorize reserved-map work.

The CPU work in this packet establishes that the benchmark exists, is
deterministic, has exact labels, and has a testable leakage boundary. It cannot
establish that a VLM will perform well.

## 2. Preconditions for any inference

All of the following must be satisfied and recorded before the first request:

1. Human GO is explicit and separate from this document.
2. The model checkpoint and revision are frozen. The candidate is Qwen2.5-VL-3B
   at revision 66285546d2b821cf421d4f5eb2576359d3770cd3.
3. Local files only is true; downloads are disabled.
4. The context, label, prompt-template, and code hashes are recorded.
5. The exact nine arms and their prompts are frozen.
6. The development split is used; maps 210-239 remain unauthorized.
7. The output schema and parser are frozen.
8. A public-only cache policy is in force.
9. The analysis script and stop rules are frozen before the first request.
10. The run is treated as development/feasibility evidence, never as
    confirmation.

If any precondition is missing, the correct action is NO-GO.

## 3. CPU validation record

Commands were run from the repository root with the local source tree on the
import path. The CPU-only suite does not load a model.

| Check | Result |
| --- | --- |
| Full test suite | 479 passed, exit 0 |
| New decision/visual tests | 16 passed |
| Frozen-backend accounting test | 1 passed |
| Builder determinism, two clean output directories | byte-identical contexts, labels, and audit |
| VLM inference | not run |
| GPU use | none |
| Reserved maps | none |

The full suite was run with a workspace-local temporary directory because the
host temp directory was not writable in the standing sandbox. This is an
environment detail, not a benchmark result.

## 4. Deterministic artifacts

Current development artifacts:

| File | SHA-256 |
| --- | --- |
| contexts.jsonl | 82cc81e16c4a64a34f1ad1f9c1c4733bd07643c6a690683560b693cbbd80945b |
| labels.jsonl | 876170892f53aa6090c17b0cf3cbdc0099f9eb38a57f8f19554410f7c7c7c357 |
| audit.json | eba1e80bcb9c4bd6c5a43efce63eebcf2bfdd73be55a40a65a2a2357e9c2bc9a |
| manifest.json | 4355bc13f1f44dd03139e8b1b5895744d675e403926238ffff0db2bf7d798266 |

The manifest stores the three file hashes above and the image metadata. The
manifest hash includes local image path strings and is therefore a local
provenance value; the three content hashes are the stable cross-run values.

The second build produced the same three content hashes, so the JSONL and audit
are byte-deterministic on this host after UTF-8/LF writing.

## 5. Leakage checks

- Every public context calls the field scanner before it enters the library.
- The scanner rejects evaluator fields such as latent_rule, reference,
  q_commit, q_sign, q_blue, best_action, optimal_actions, declaration_truth,
  label, and stratum.
- Tests assert that public payloads contain none of those keys and that the
  serialized public JSONL does not contain them either.
- Labels are a separate file. The builder does not merge labels into contexts.
- The response cache remains public-only; it rejects responses whose text
  contains forbidden fields.
- Visual cards are constructed only from public context fields. There is no
  label overlay.
- The diagnostic card is a new public context with a new exact posterior and a
  new label. It is not a hidden-label reveal.

A passing scanner is necessary, not sufficient. The later inference run must
also scan the final prompt bytes and the cache key before sending anything.

## 6. Reference-label coverage

| Check | Count |
| --- | ---: |
| Contexts | 118 |
| q_commit present | 118 |
| q_sign present (feasible) | 52 |
| q_blue present (feasible) | 38 |
| Both probes feasible | 20 |
| Unique optimal action | 117 |
| Two-way optimal tie | 1 |
| DECLARE_RULE reference gate | 44 |
| ABSTAIN reference gate | 74 |
| Reference-gate declared correct | 42 |
| Reference-gate declared incorrect | 2 |

The one two-way tie is retained rather than removed. The reference-gate
accuracy of 42/44 is a fixture property of the posterior-sampled latent rule,
not a claim that the gate is a truth oracle.

## 7. Candidate decisions in the development set

| Decision | Count |
| --- | ---: |
| COMMIT best | 81 |
| PROBE_SIGN best | 23 |
| PROBE_BLUE best | 14 |
| commit_preferred class | 76 |
| probe_preferred class | 20 |
| near_tie class | 21 |
| tie class | 1 |
| budget-exhausted stratum | 8 |

The exact best-action counts and the decision-class counts differ because some
near ties still have a unique deterministic best action. The set has a
commit-majority, probe-minority structure: 81 commits and 37 probes. A model
must handle both, not just the majority.

## 8. Baselines for interpretation

| Baseline | Optimal-action accuracy | Mean regret |
| --- | ---: | ---: |
| always_commit | 0.6864 | 0.4863 |
| always_probe | 0.6949 | 0.2698 |
| random_valid | 0.6949 | 0.3362 |
| information_gain | 0.9068 | 0.0429 |
| exact reference | 1.0000 | 0.0000 |

The information-gain baseline is the meaningful public heuristic benchmark. A
VLM arm that does not approach it is not yet demonstrating useful
probe-versus-commit reasoning. The exact reference is a sanity check, not a
model baseline.

## 9. Visual controls

The builder produced 118 image rows and 344 PNG files:

- 118 base cards;
- 118 nuisance cards;
- 108 diagnostic cards (10 contexts could not accept a blue diagnostic without
  exceeding budget).

Checks:

- all 118 base digests are unique;
- all 118 nuisance digests are unique;
- no nuisance card equals its base card pixel-for-pixel;
- nuisance keeps the same public context and label;
- diagnostic changes the observed blue role, reconditions the posterior, and
  produces a fresh reference label.

Per-image SHA-256 digests and paths are recorded in manifest.json under images.
The aggregate SHA-256 of the image metadata payload is
06f8f6b8913d67573336ba0dced367bd1615198c3a2c88983479696b68c95365.

The 108 diagnostic contexts divide into 90 commit-preferred, 13
probe-preferred, and 5 near-tie labels. Their exact best actions are 94 COMMIT
and 14 PROBE_SIGN; none is PROBE_BLUE after the blue observation. This is a
property of the paired perturbation, not a model result.

The card is 512x320. It contains the map, six candidate-rule descriptions,
prior/posterior bars, evidence, budget, costs, error rate, and horizon. It does
not contain an evaluator label.

## 10. Feasibility decision rule

**GO for a small development inference pilot** only if all preconditions in
section 2 hold, and only after a separate human authorization. The pilot should
use the nine-arm matrix, the 118 development contexts, the frozen parser, and
the stop rules below.

**NO-GO for confirmation** in all cases at this stage. A development pilot can
inform design; it cannot promote v0.5, authorize v0.6, or justify reserved-map
use.

The recommendation of this packet is therefore: the CPU infrastructure is
ready; the scientific inference is not yet authorized; the next action is a
human GO/NO-GO on a bounded development pilot, not an automatic model run.

## 11. Stop rules for a future pilot

Stop and report, rather than continue and cherry-pick, if:

- the frozen checkpoint or revision cannot be verified;
- any download is attempted;
- invalid/fallback exceeds 20% for an arm;
- the public prompt or cache key contains a forbidden field;
- the visual nuisance control changes decisions at a rate that indicates the
  arm is reading presentation rather than facts;
- the parser or schema changes after the first request;
- a reserved map, confirmatory endpoint, or v0.6 candidate appears in the run;
- the analysis plan is changed after seeing outcomes.

## 12. Remaining scientific risks

- **Small development set.** 118 contexts are enough to expose interface and
  gross decision failures, not to estimate a population effect.
- **Synthetic declaration truth.** The latent rule is sampled from the
  posterior. Declaration accuracy measures agreement with that fixture, not
  real rule recovery.
- **Macro reference.** Commit steps and roles are public macro quantities; the
  reference does not model primitive execution failures.
- **Visual/text mismatch.** The image and text conditions carry the same public
  facts but are not guaranteed to be equally legible. Nuisance/diagnostic
  controls measure part of this, not all of it.
- **Model contamination.** The checkpoint may have seen related grid or
  benchmark material. No claim of clean generalization is possible from a
  development pilot.
- **Prompt sensitivity.** A single prompt template can understate or overstate
  model capability. The matrix varies access and response contract, not an
  exhaustive prompt space.
- **Evaluator degree of freedom.** The near-tie margin, tie order, and
  declaration gate are design choices. They are frozen here and must not be
  tuned after seeing model results.
- **No power calculation.** This packet does not support a confirmatory sample
  size or a p-value claim.

## 13. What happens next

1. Human reviews this packet and the closure document.
2. Human issues GO or NO-GO for the bounded development pilot.
3. If GO: freeze prompts, run the nine arms on 118 contexts, evaluate offline,
   and report invalid/fallback and per-stratum behavior before any aggregate
   claim.
4. If NO-GO: keep the CPU benchmark as a development artifact and do not run a
   model.
5. In either case: do not merge automatically, do not touch reserved maps, and
   do not develop v0.6.
