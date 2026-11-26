# VLM Prompt Anchoring × Belief Conditioning — Next Gate

**Experiment:** `vlm-prompt-belief-diagnostic-v1`  
**Status:** COMPLETE — all four arms fail  
**Decision date:** 2026-11-26  

---

## 1. Decision

**CLOSE the current 3B feasibility line.**

The pre-registered all-fail rule was met:

- no arm reached 8/12 optimal actions;
- no arm produced a correct action in each of the three optimal-action classes;
- every arm was constant (`COMMIT` only);
- no arm beat `always_commit` on mean regret (all exactly equalled it);
- all four arms were 12/12 valid, so the failure is behavioral, not a parser failure.

The model is functionally `always_commit` under both belief conditions and both output formats. The diagnostic did not produce a viable candidate for visual smoke.

---

## 2. What is not authorized

The following remain **stopped** and must not be started from this result:

- resuming the original 48-request smoke or its remaining 36 requests;
- an RGB/visual smoke with this 3B checkpoint;
- the nine-arm pilot;
- maps 210–239;
- a confirmatory run on this exposed subset;
- v0.6 or any automatic prompt-tuning loop;
- automatic merge, PR #22 changes, or PR #23 changes;
- any model substitution, quantization, fine-tuning, or precision change inside this experiment.

No further model requests are authorized by this packet.

---

## 3. Why closing is the correct action

Three explanations were considered at the start:

1. **Prompt-format anchoring** — the original anchored footer might be causing the collapse.
2. **Belief-computation limitation** — the model might not have the posterior available.
3. **Probe-versus-commit decision limitation** — the model might have the information but fail to choose a probe.

The matched 2×2 produced zero action changes for both prompt-side interventions:

- neutral format did not change any action (B2 − B1 = 0; B4 − B3 = 0);
- explicit posterior did not change any action (B3 − B1 = 0; B4 − B2 = 0).

Thus the first two hypotheses are not sufficient explanations on this sample, and the observed policy is consistent with a decision-stage commit bias. Because the model was not internally probed, the experiment cannot claim that it “has correct beliefs but chooses commit.” It can claim only that supplying the posterior and removing the anchor do not repair the action.

That is enough to reject this checkpoint for the planned visual-smoke gate. Re-running the same model with cosmetic prompt changes would be post hoc tuning on an exposed subset and is not authorized.

---

## 4. Recommended next step: separate authorization only

If the research line is to continue, the human must separately authorize a **new, preregistered feasibility experiment** with a stronger model. This packet does not authorize it and does not select a model.

Such a proposal should:

1. Use a new experiment identifier, new output directory, new freeze, and a new source dataset.
2. Pin the model and revision; use local files only; no training, fine-tuning, quantization, or substitution.
3. Preserve the same 12-context matched diagnostic first, or a separately justified small development set.
4. Keep the same exact parser, action menu, feasible-action constraints, scoring, and request cap.
5. Include a fresh CPU preflight, leakage checks, and a stub dry run before any real call.
6. Record actual parameter dtype, compute dtype, attention implementation, autocast, emulation, and native-BF16 capability.
7. Treat constant-action behavior in an arm as a scientific outcome, not a technical stop.
8. Report all four arms and all paired comparisons, including null results.
9. Use the same development eligibility gate before any RGB feasibility work.
10. Stop after the report and decision packet; no automatic visual run.

A stronger model is a reasonable candidate because the failure is at the action-selection stage and the 3B model has no partial successes. But “stronger model” is a hypothesis, not a guarantee. The new experiment must be allowed to fail.

---

## 5. Gate for any future visual-smoke proposal

A future model earns a small RGB feasibility proposal only if, on the frozen development set:

| Criterion | Threshold |
|---|---|
| Valid actions | at least 11/12 |
| Optimal actions | at least 8/12 |
| Correct action in each optimal class | at least 1 in `COMMIT`, `PROBE_SIGN`, `PROBE_BLUE` |
| Action distribution | non-constant |
| Mean regret | below `always_commit` on the same subset |

Comparison against `information_gain` remains descriptive; it is not a substitute for the gate. Passing this development gate is **not** confirmation and does not authorize a full visual experiment by itself.

---

## 6. Provenance and closure state

| Item | Value |
|---|---|
| New experiment | `vlm-prompt-belief-diagnostic-v1` |
| Real requests | 48 |
| Automatic retries | 0 |
| Valid responses | 48 |
| Arms eligible | none |
| Kaggle kernel | https://www.kaggle.com/code/tiennamlee/ruleshift-vlm-prompt-belief-diagnostic |
| Authoritative run | `results/kaggle_belief_output_v3/results/vlm_prompt_belief_diagnostic/real/` |
| Freeze | `sha256:351c772ebfe3fc26d19bd42d910888f6aa393970a5d2f04ead77ad1706f7c968` |
| Protocol | `sha256:c4b7017b06c2742c6b446c8541235d2d486d693a1b45d8fad5ae2064a4c85a12` |
| Original smoke | remains closed A1 feasibility failure; untouched |

**End state:** the 3B feasibility line is closed. No visual inference, original continuation, nine-arm pilot, maps 210–239, confirmatory run, v0.6 work, prompt tuning, or merge is authorized. The report and this decision packet are the stopping point.
