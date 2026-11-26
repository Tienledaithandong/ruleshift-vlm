# VLM Decision Pilot Decision Packet

**Decision status: AWAITING A SEPARATE HUMAN DECISION.**

**Current recommendation: STOP AT THE SMOKE. DO NOT PROMOTE TO A PILOT, A2/A3 CONTINUATION, OR V0.6.**

This packet follows a completed bounded smoke that stopped at the frozen A1 feasibility gate. It is not a request to promote the smoke and it is not authorization for a pilot. The symbolic phase remains closed; v0.5 remains a development reference, not confirmatory; the 900-episode reserved-map confirmation remains deferred; no v0.6 work occurred.

## 1. What was authorized

The human decisions were:

- **CONDITIONAL GO FOR BOUNDED VLM SMOKE**, with CPU infrastructure accepted.
- **GO FOR KAGGLE GPU EXECUTION** after the local laptop GPU failed at model load.

The authorized scope was:

- 12 frozen development contexts: four COMMIT, four PROBE_SIGN, four PROBE_BLUE.
- A1 prior/evidence text (12), A2 posterior text (12), A3 posterior RGB card (12).
- Six RGB nuisance controls and six RGB semantic diagnostic controls.
- Maximum 48 model requests including failures and retries; no automatic retries.
- Exact Qwen/Qwen2.5-VL-3B-Instruct revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- bf16, no quantization, no offload, no substitution, no training or fine-tuning.
- Direct-action JSON only; no rationale or declaration arms.
- Stop after the smoke and decision packet.

Only infrastructure/path-portability changes were authorized on Kaggle. No scientific change was authorized.

## 2. What happened

| Gate | Result |
|---|---|
| Full CPU suite, before Kaggle execution | PASS: 488 tests |
| Frozen manifest, prompts, images, labels | PASS |
| Checkpoint exact revision and aggregate hash | PASS |
| Kaggle hardware and BF16 load | PASS: Tesla T4, bf16, 3.754B parameters |
| Leakage scan | PASS: zero hits |
| Actual processor image forwarding in preflight | PASS: 24 RGB cards, real tensors |
| 48-request stub dry run | PASS: 48/48 synthetic, zero model calls |
| A1 model requests | 12/12 completed, 12/12 valid |
| A1 action distribution | 12 COMMIT, 0 PROBE_SIGN, 0 PROBE_BLUE |
| A1 non-constant gate | **FAIL** |
| A1 feasibility gate | **FAIL** |
| A2 / A3 / nuisance / diagnostic | **NOT RUN** |
| Request cap | 12 of 48; stopped early by gate |
| Retries | 0 |
| GPU generation time | 24.1888 s |

The run did **not** fail at model load. The checkpoint loaded in 9.6181 s of wall time; checkpoint verification took 42.9810 s; peak CUDA reservation was 8,032,092,160 bytes on a 15,636,037,632-byte T4. The stop was the frozen scientific gate, not an OOM or parser failure.

Kaggle kernel: https://www.kaggle.com/code/tiennamlee/ruleshift-frozen-vlm-smoke-48 (version 4, terminal status COMPLETE).

## 3. What the evidence actually is

The smoke produced a genuine A1 execution:

- 12 actual Qwen requests, 12 valid direct JSON responses, zero fallback.
- Every response was `{"selected_action":"COMMIT"}`.
- Optimal-action accuracy: 0.3333 (4/12).
- Mean regret: 2.0928; maximum regret: 5.025.
- Under-probe rate: 0.6667 (8/12); over-probe rate: 0.0.
- The 12 contexts contain 4 optimal COMMIT, 4 optimal PROBE_SIGN, and 4 optimal PROBE_BLUE decisions.

These numbers are exactly the always-commit CPU baseline. A1 is therefore **degenerate**, not merely imperfect. The pre-registered gate correctly stopped the run.

The result is **not** evidence about A2, A3, the RGB card, nuisance invariance, diagnostic responsiveness, OCR, chart reading, or multimodal posterior use. None of those requests ran. The A3 image-forwarding check in preflight proves only that the processor built an image tensor; it is not a model observation.

A plausible contributor is example/schema anchoring: the frozen instruction ends with the literal `{"selected_action":"COMMIT"}` example and asks the model to replace `COMMIT`. All 12 outputs reproduced that exact object with 8 output tokens. This is a hypothesis, not a proven cause; one prompt format and 12 contexts cannot separate anchoring from a genuine COMMIT policy or a capability limitation. Prompt changes were not authorized and were not made.

## 4. Decision options

### Option A - STOP/HOLD at the smoke (recommended)

- Accept the bounded smoke as complete with a hard A1 gate failure.
- Preserve all frozen artifacts, raw responses, checksums, logs, and the local failed run.
- Do not issue A2/A3 or control requests.
- Do not tune the prompt, parser, menu, or decoding to rescue A1.
- Do not proceed to the full nine-arm 1,536-request pilot, the 118-context continuation, reserved maps, confirmatory evaluation, or v0.6.
- Keep v0.5 as a development reference only.

This is the lowest-risk and most defensible option. It records a real negative feasibility signal without spending more requests on an uninformative post-gate comparison.

### Option B - Authorize a new, separately pre-registered prompt-format smoke

- Treat this as a **new scientific protocol**, not a retry of the current smoke.
- Specify the revised A1 response format (for example, no action-specific literal example, an order-balanced action list, or an explicit format-only control) before generating any new requests.
- Create a new freeze, new request manifest, new preflight, and a new request cap.
- Keep the same model, revision, precision, and direct-action contract unless separately changed.
- Preserve the current 12-request result as prior evidence; do not merge the two runs into one analysis.

This option is only appropriate if the human wants to test the schema-anchoring hypothesis. It must not be launched automatically.

### Option C - Authorize a broader diagnostic study of prompt format and action anchoring

- This is a new experiment, larger than the current 48-request smoke.
- It requires its own preregistration, freeze, and budget.
- It must not be described as a continuation of the current smoke, and its results must not be merged with the current A1 result.

### Option D - Authorize a larger VLM pilot or confirmation

**Not supported by this result and not authorized.** A1 is constant and A2/A3 did not run. There is no evidence base for a 1,536-request pilot, a 118-context continuation, reserved maps 210-239, confirmatory evaluation, or v0.6.

## 5. Explicitly not authorized by this packet

- Any A2/A3 or control request under the current freeze.
- Any prompt tuning, parser change, menu change, action reordering, or decoding change.
- Any model, revision, precision, quantization, or offload change.
- The full nine-arm 1,536-request pilot.
- The 118-context continuation.
- Reserved maps 210-239.
- Confirmatory evaluation.
- v0.6 development.
- Any change to PR #22 or PR #23.
- Any automatic merge.

## 6. Preservation statement

The following remain preserved and unchanged:

- Candidate commit `751a3414ad8815087f64d85ad5a1523af11077f4`.
- Docs commit `bc79ce1d7a54aea38fa7502452d646450121c09a`.
- PR #22 head `a3e614b093dfd8237898cb19ac83cab92b8bad50`.
- PR #23 as published.
- Frozen requests, images, labels, manifests, checksums, and the frozen protocol.
- The local laptop failed run at `results/vlm_decision_smoke/real/`.
- The Kaggle raw run, preflight, dry run, hardware record, and kernel log.

No automatic merge or continuation has occurred.

## 7. Required human decision

Choose one:

- [ ] **A. STOP/HOLD.** Close the bounded smoke; preserve the A1 gate failure; do not continue.
- [ ] **B. Authorize a new pre-registered prompt-format smoke.** Provide the revised protocol and a new freeze; do not reuse the current 48-request result as if it were the same experiment.
- [ ] **C. Authorize a broader diagnostic study.** Provide the new preregistration and budget.
- [ ] **D. Something else.** Specify it explicitly; do not infer a larger pilot from this packet.

If B or C is chosen, specify the operator, the new frozen prompt, the new request budget, the output path, and confirmation that the current artifacts remain preserved. No option authorizes the full 1,536-request pilot, the 118-context continuation, maps 210-239, confirmatory evaluation, automatic merge, or v0.6.
