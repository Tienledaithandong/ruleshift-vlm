# VLM Prompt Anchoring × Belief Conditioning Diagnostic — Results

**Experiment:** `vlm-prompt-belief-diagnostic-v1`  
**Status:** COMPLETE — all 48 pre-registered requests executed; no technical stop  
**Decision:** all four arms fail the pre-registered visual-smoke eligibility gate; close the current 3B feasibility line  
**Date:** 2026-11-26  
**Kaggle kernel:** https://www.kaggle.com/code/tiennamlee/ruleshift-vlm-prompt-belief-diagnostic  
**Kaggle source dataset:** https://www.kaggle.com/datasets/tiennamlee/ruleshift-vlm-prompt-belief-source  

---

## 1. What was run

The frozen 2×2 diagnostic was executed exactly once, with no automatic retries:

| Arm | Belief input | Output instruction | Requests |
|---|---|---|---:|
| B1 | prior/evidence text | original anchored footer | 12 |
| B2 | prior/evidence text | neutral schema-only footer | 12 |
| B3 | explicit posterior text | original anchored footer | 12 |
| B4 | explicit posterior text | neutral schema-only footer | 12 |
| **Total** | | | **48** |

All four arms used the same 12 development contexts, the same feasible-action sets, the same reference labels, the same parser, the same model checkpoint/revision, deterministic decoding, and `max_new_tokens=32`. All prompts were text-only; no image was supplied.

The run completed all four arms. A constant-action result in one or more arms is treated as a scientific outcome, not a technical stop.

---

## 2. Primary result

**Every one of the 48 responses selected `COMMIT`.** All 48 were valid JSON with no parser fallback. Consequently, every arm has the same science metrics:

| Metric | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| Valid actions | 12/12 | 12/12 | 12/12 | 12/12 |
| Valid rate | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Invalid or fallback | 0 | 0 | 0 | 0 |
| `COMMIT` / `PROBE_SIGN` / `PROBE_BLUE` | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| Distinct valid actions | 1 | 1 | 1 | 1 |
| Optimal-action accuracy | 0.333333 | 0.333333 | 0.333333 | 0.333333 |
| Correct `COMMIT` (of 4) | 4 | 4 | 4 | 4 |
| Correct probe actions (of 8) | 0 | 0 | 0 | 0 |
| Mean regret, all | 2.0928107724292278 | 2.0928107724292278 | 2.0928107724292278 | 2.0928107724292278 |
| Max regret | 5.025 | 5.025 | 5.025 | 5.025 |
| Over-probe rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Under-probe rate | 0.6666666666666666 | 0.6666666666666666 | 0.6666666666666666 | 0.6666666666666666 |
| Tool-selection accuracy | undefined | undefined | undefined | undefined |
| Visual-smoke eligible | **false** | **false** | **false** | **false** |

### Accuracy by optimal-action class

| Optimal class | n | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|---:|
| `COMMIT` | 4 | 4/4 = 1.000000 | 4/4 = 1.000000 | 4/4 = 1.000000 | 4/4 = 1.000000 |
| `PROBE_SIGN` | 4 | 0/4 = 0.000000 | 0/4 = 0.000000 | 0/4 = 0.000000 | 0/4 = 0.000000 |
| `PROBE_BLUE` | 4 | 0/4 = 0.000000 | 0/4 = 0.000000 | 0/4 = 0.000000 | 0/4 = 0.000000 |

**Tool-selection accuracy is undefined**, not zero: no valid probe action was emitted, so there is no probe-target denominator. `score_records` correctly returns `null`.

The arms are exactly equal to the `always_commit` CPU baseline on accuracy and mean regret. They are worse than `information_gain` by `0.5833333333333333` accuracy and `2.0094774390958943` mean regret.

---

## 3. Pre-registered eligibility checks

Each criterion is shown for all arms; the values are identical because every action was `COMMIT`.

| Criterion | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| At least 11/12 valid | true | true | true | true |
| At least 8/12 optimal | false | false | false | false |
| At least one correct in each optimal class | false | false | false | false |
| Non-constant action distribution | false | false | false | false |
| Mean regret below `always_commit` | false (equal) | false (equal) | false (equal) | false (equal) |
| **Eligible for visual smoke consideration** | **false** | **false** | **false** | **false** |

**All arms fail.** The all-fail rule in the protocol is therefore triggered.

---

## 4. Paired comparisons

No context changed action between any two arms. Every paired action-change count is zero.

| Pair | Meaning | Action changes / 12 | Accuracy Δ (left − right) | Mean-regret Δ (left − right) |
|---|---|---:|---:|---:|
| B2 − B1 | prompt format without explicit posterior | 0 | 0.0 | 0.0 |
| B4 − B3 | prompt format with explicit posterior | 0 | 0.0 | 0.0 |
| B3 − B1 | explicit posterior under anchored format | 0 | 0.0 | 0.0 |
| B4 − B2 | explicit posterior under neutral format | 0 | 0.0 | 0.0 |

**Descriptive interaction** `(B4 − B3) − (B2 − B1)` is exactly `0.0` for both accuracy and mean regret.

Thus the experiment found no prompt-format effect, no posterior-text effect, and no interaction on this sample. It also found no partial effect: the model did not switch a single action.

---

## 5. Per-context actions

`COMMIT` is the reference-optimal action in the first four contexts; `PROBE_SIGN` in contexts 5–8; `PROBE_BLUE` in contexts 9–12. The model committed in every case in every arm.

| # | Context ID | Reference best | B1 | B2 | B3 | B4 | Regret of `COMMIT` |
|---:|---|---|---|---|---|---|---:|
| 1 | `vlm-dev-layout-a-base-rotate-50-expensive-0.00-` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 0.0 |
| 2 | `vlm-dev-layout-b-base-hazard-50-cheap-0.05-sign:The red tile is not the goal.` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 0.0 |
| 3 | `vlm-dev-layout-c-base-rotate-50-default-0.30-blue:hazard` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 0.0 |
| 4 | `vlm-dev-layout-a-hazard-confident-cheap-0.30-sign:The red tile is not the goal.` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 0.0 |
| 5 | `vlm-dev-layout-a-base-hazard-50-cheap-0.15-blue:safe` | `PROBE_SIGN` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 5.025 |
| 6 | `vlm-dev-layout-b-all-uniform-cheap-0.30-blue:goal` | `PROBE_SIGN` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 3.9448275862068956 |
| 7 | `vlm-dev-layout-b-sign-confident-cheap-0.15-blue:hazard` | `PROBE_SIGN` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 3.8418032786885226 |
| 8 | `vlm-dev-layout-c-base-hazard-90-default-0.05-blue:goal` | `PROBE_SIGN` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 1.2787234042553202 |
| 9 | `vlm-dev-layout-c-hazard-sign-50-cheap-0.05-sign:The red tile is not the goal.` | `PROBE_BLUE` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 3.649374999999999 |
| 10 | `vlm-dev-layout-a-base-hazard-sign-uniform-cheap-0.05-sign:The red tile is not the goal.` | `PROBE_BLUE` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 3.5762499999999973 |
| 11 | `vlm-dev-layout-c-hazard-sign-50-default-0.00-sign:The red tile is not the goal.` | `PROBE_BLUE` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 3.0250000000000004 |
| 12 | `vlm-dev-layout-b-hazard-confident-cheap-0.05-` | `PROBE_BLUE` | `COMMIT` | `COMMIT` | `COMMIT` | `COMMIT` | 0.7727499999999985 |

---

## 6. Raw response format

The output format changed the tokenization but not the action:

| Arm | Output instruction | Unique raw texts | Example | Output tokens |
|---|---|---:|---|---:|
| B1 | anchored | 1 | `{"selected_action":"COMMIT"}` | 8 each |
| B2 | neutral | 1 | pretty-printed JSON (3 lines) | 12 each |
| B3 | anchored | 1 | `{"selected_action":"COMMIT"}` | 8 each |
| B4 | neutral | 1 | pretty-printed JSON (3 lines) | 12 each |

The neutral footer therefore produced pretty-printed JSON with four additional output tokens, but no different action. All 48 parsed to `COMMIT` with no errors.

---

## 7. CPU baselines on the exact 12-context subset

Recomputed in the same frozen environment and reproduced in the Kaggle run's `cpu_baselines.json`.

| Baseline | Optimal accuracy | Mean regret | Max regret | Over-probe | Under-probe | Tool selection |
|---|---:|---:|---:|---:|---:|---:|
| `exact` | 1.0000000000000000 | 0.0000000000000000 | 0.0000000000000000 | 0.0000000000000000 | 0.0000000000000000 | 1.0000000000000000 |
| `information_gain` | 0.9166666666666666 | 0.08333333333333341 | 1.0000000000000009 | 0.08333333333333333 | 0.0000000000000000 | 0.8888888888888888 |
| `always_probe` | 0.5833333333333334 | 0.4648125000000001 | 2.0000000000000000 | 0.3333333333333333 | 0.0000000000000000 | 0.5833333333333334 |
| `random_valid` | 0.5000000000000000 | 1.2285481321839080 | 5.0250000000000000 | 0.2500000000000000 | 0.2500000000000000 | 0.6250000000000000 |
| `always_commit` | 0.3333333333333333 | 2.0928107724292278 | 5.0250000000000000 | 0.0000000000000000 | 0.6666666666666666 | — |

Class accuracy of the baselines (same subset):

| Baseline | `COMMIT` | `PROBE_SIGN` | `PROBE_BLUE` |
|---|---:|---:|---:|
| `exact` | 1.00 | 1.00 | 1.00 |
| `information_gain` | 0.75 | 1.00 | 1.00 |
| `always_probe` | 0.00 | 1.00 | 0.75 |
| `random_valid` | 0.25 | 0.50 | 0.75 |
| `always_commit` | 1.00 | 0.00 | 0.00 |

---

## 8. Runtime and precision audit

| Item | Value |
|---|---|
| GPU | Tesla T4, compute capability 7.5, 15,636,037,632 bytes total |
| GPU count visible | 2; request used device 0 |
| Parameter dtype | `torch.bfloat16` (824 tensors) |
| Model config dtype | `torch.bfloat16` |
| Parameters | 3,754,622,976 |
| Attention implementation | `sdpa` |
| Software | Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2 |
| Autocast | disabled; `autocast_dtype=torch.float16` is only a configured default |
| Explicit precision switch | none (`precision_changed=false`) |
| BF16 hardware-native | **false**: capability 7.5; T4 has no native BF16 tensor cores |
| `torch.cuda.is_bf16_supported()` flag | true (not evidence of native hardware) |
| Model-load wall | 3.766759430 s |
| Real generation wall | 111.779667388 s (sum of request latencies) |
| GPU time (sum of per-request `gpu_time_s`) | 111.571374894 s |
| Real execution command wall | 164.01 s |
| Full kernel wall | 277.856475111 s |
| Peak allocated after load | 7,628,753,920 bytes |
| Generation peak allocated | 7,840,860,160 bytes |
| Generation peak reserved | 8,097,103,872 bytes |
| Peak allocated / GPU total | 50.15% |

The audit does **not** claim native BF16. Parameters and configured compute remain BF16, but on T4 this is not native BF16 tensor-core execution; the audit records capability 7.5 explicitly. No precision condition was changed.

### Per-arm runtime

| Arm | Input tokens (total / mean) | Output tokens (total / mean) | Latency total | Mean latency | GPU time | Peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 8,049 / 670.75 | 96 / 8.0 | 26.878 s | 2,239.868 ms | 26.801 s | 7,840,860,160 |
| B2 | 8,073 / 672.75 | 144 / 12.0 | 27.466 s | 2,288.864 ms | 27.425 s | 7,840,860,160 |
| B3 | 8,852 / 737.67 | 96 / 8.0 | 27.553 s | 2,296.101 ms | 27.507 s | 7,840,860,160 |
| B4 | 8,876 / 739.67 | 144 / 12.0 | 29.882 s | 2,490.140 ms | 29.839 s | 7,840,860,160 |
| **All** | **33,850 / 705.21** | **480 / 10.0** | **111.780 s** | **2,328.743 ms** | **111.571 s** | **7,840,860,160** |

All latencies are descriptive; the balanced order controls first-position exposure per arm but does not eliminate warm-up or T4 contention effects. Latency is not a scientific endpoint.

---

## 9. Accounting and provenance

| Item | Value |
|---|---|
| Planned requests | 48 |
| Attempted requests | 48 |
| Completed `ok` responses | 48 |
| Failed responses | 0 |
| Parsed actions | 48 |
| Automatic retries | 0 |
| Stopped | none |
| Synthetic stub outputs in real run | false |
| All arms complete | true |
| Technical stop | none |
| Real model calls in this experiment | **48** |
| Model calls in failed Kaggle v1/v2 infrastructure attempts | **0 each** |
| Original smoke A1 calls (historical only) | 12; not part of these 48 |

Raw-log consistency audit (independent resolver):

| Check | Result |
|---|---|
| Planned == request rows | true |
| Attempted == response rows | true |
| Completed == `ok` response rows | true |
| Summary `model_calls` == response rows | true |
| Parsed rows == `ok` response rows | true |
| Overall | **true** |

### Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `freeze.json` | `sha256:351c772ebfe3fc26d19bd42d910888f6aa393970a5d2f04ead77ad1706f7c968` |
| `freeze.sha256` target | same |
| Protocol | `sha256:c4b7017b06c2742c6b446c8541235d2d486d693a1b45d8fad5ae2064a4c85a12` |
| Request manifest | `sha256:a3271a58e158c11e538126d4581e851c3fe1c5b22dc67707b938714ad6159a8b` |
| Prompts | `sha256:75d1ecd5229654835d6f5153cb395417c565101c2fd94c08e536accb5acb600e` |
| Prompt manifest | `sha256:a27375a7a038c6dfaf4ecb5db49b522787b9509fdb91266a9ae1a925da60cefe` |
| Public payloads | `sha256:46e14106f66ee2dce987b0137ad157d34401463860c1548719cf21ac4395a8f2` |
| Reference labels | `sha256:c44b695dfaf14df4e922993ee440e16bb6d8422df1684f3a6a081d9617e6d45d` |
| Analysis plan | `sha256:c8c08cbf9a2b74e77d9d70587a4334c74b80bbb5bc29d61fe879c964f491eb04` |
| Model/runtime contract | `sha256:3a3daa61d6112127c0cf4358debd32b34976abc32ccfaa89972160af9597d089` |
| Checkpoint manifest | `sha256:84ac05e1a20687b803aed751a1626d64b3ce071e095fdc0aac732b2425a0fbcc` |
| Checkpoint aggregate | `sha256:02bdb4ccb82eb3aceaed1608e1e7c55a81ab3e6125a20f6cccc690bbf1254c75` |

B1 used the exact original A1 prompt, and its total input-token count is exactly the original A1 total, **8,049 tokens**. The 12 B1 responses are the same compact `COMMIT` output. The new B1 is therefore a matched replication of the historical A1 outcome, not a substitute for it.

---

## 10. Infrastructure history and corrections

Three Kaggle executions were attempted. Only the third executed any model request.

| Attempt | Outcome | Real model calls |
|---|---|---:|
| v1 | Source-integrity manifest included `dataset-metadata.json`, which was not in the uploaded archive; stopped before preflight. | 0 |
| v2 | Source archive omitted required `configs/`; import of `ruleshift.envs` failed at `missing RuleShift config dev.yaml`; stopped during preflight. | 0 |
| v3 | Source archive locally preflighted, then completed all 48 requests. | 48 |

Corrections made before the real run:

1. The source manifest no longer includes upload-only dataset metadata.
2. Required runtime `configs/` were added to the source archive.
3. The exact staged source was preflighted locally before upload.
4. The execution wrapper records `bf16_hardware_native` from compute capability (`major >= 8`), not from `torch.cuda.is_bf16_supported()`.
5. The runner writes `parsed.jsonl` and top-level `model_calls` so summary accounting can be checked against raw logs. This was an infrastructure/logging change; no prompt, context, label, parser, scorer, model, or decoding setting changed. The earlier pre-fix freeze is preserved at `results/vlm_prompt_belief_diagnostic/frozen_pre_parsed_log/`; the authoritative freeze is `results/vlm_prompt_belief_diagnostic/frozen/` with hash `351c...`.
6. The historical artifact-collector defect was corrected in the old Kaggle wrapper: it now prefers `OUTPUTS/real/real` and falls back to `OUTPUTS/real`, instead of silently missing the nested run. Regression tests cover nested and flat layouts plus summary/raw-log mismatch rejection. The new runner emits the flat `real/` layout. The historical old manifest itself was not rewritten.
7. The artifact resolver handles both layouts for audits, and the new runner exposes `parsed.jsonl` plus top-level request counts so independent accounting matches the raw logs.

The original frozen smoke artifacts under `results/vlm_decision_smoke/` were not modified. The original artifact manifest remains the historical manifest produced by that run.

---

## 11. Exact artifact locations

| Artifact | Path |
|---|---|
| Kaggle kernel | https://www.kaggle.com/code/tiennamlee/ruleshift-vlm-prompt-belief-diagnostic |
| Kaggle execution manifest | `results/kaggle_belief_output_v3/KAGGLE_EXECUTION_MANIFEST.json` |
| Kaggle kernel log | `results/kaggle_belief_output_v3/ruleshift-vlm-prompt-belief-diagnostic.log` |
| Authoritative raw run | `results/kaggle_belief_output_v3/results/vlm_prompt_belief_diagnostic/real/` |
| Requests | `.../real/requests.jsonl` |
| Responses | `.../real/responses.jsonl` |
| Parsed actions | `.../real/parsed.jsonl` |
| Accounting | `.../real/request_accounting.json` |
| Summary | `.../real/summary.json` |
| Runtime audit | `.../real/runtime_audit.json` |
| Frozen plan | `results/vlm_prompt_belief_diagnostic/frozen/` |
| CPU preflight | `results/vlm_prompt_belief_diagnostic/preflight/` |
| 48-request stub dry run | `results/vlm_prompt_belief_diagnostic/dry_run/` |
| Extracted metrics | `results/vlm_prompt_belief_diagnostic/analysis/metrics.json` |
| Failed v1 output | `results/kaggle_belief_output/` |
| Failed v2 output | `results/kaggle_belief_output_v2/` |

The Kaggle manifest lists 23 output artifacts totaling 967,621 bytes for the v3 run.

---

## 12. Interpretation

**The pre-registered diagnostic did not rescue the 3B decision behavior.**

- **Prompt-format anchoring:** removed. Changing only the footer from the anchored action example to a neutral schema did not change any action (B2 − B1 = 0 changes; B4 − B3 = 0).
- **Belief conditioning:** removed as a sufficient explanation. Supplying an explicit posterior did not change any action (B3 − B1 = 0; B4 − B2 = 0).
- **Probe-versus-commit decision:** remains the locus consistent with the data. The model emitted a constant `COMMIT` action even when the public posterior was explicit and the output schema was neutral. However, this design cannot distinguish (a) the model ignoring the posterior text, (b) the model representing it but failing to map it to a probe action, and (c) a strong commit prior at the action-selection stage.

Because no action changed in any arm, the 2×2 is degenerate on this sample. The correct conclusion is **not** that the model has correct beliefs; it is that the supplied prompt variations had **zero behavioral effect** and the model's action policy collapsed to `COMMIT`.

The result is consistent with the original A1 observation and adds two matched controls that also fail. It does not support moving to visual inference with this 3B model.

---

## 13. Limitations

- Only 12 development contexts, all previously exposed; this is feasibility evidence, not confirmation.
- One model, one revision, one deterministic decoding configuration, one T4 session.
- Text-only; no image or rendered decision card.
- `max_new_tokens=32`; no sampling, no chain-of-thought, no prompt ensemble.
- All arms are constant, so paired differences have zero variance and no meaningful interval; no significance test is reported.
- `tool_selection_accuracy` is undefined because no probe was emitted; it is not a zero-accuracy estimate.
- T4 BF16 is not hardware-native; the audit records this and does not claim native BF16.
- The three Kaggle attempts include two infrastructure failures; both had zero model calls. The real evidence is only v3's 48 requests.

---

## 14. Decision

**STOP the current 3B feasibility line.**

All four arms fail the pre-registered gate. The model is functionally `always_commit` under every tested prompt/belief condition, and the matched diagnostic does not produce a viable visual-smoke candidate. No RGB visual smoke should be run with this checkpoint, and no original 48-request continuation, nine-arm pilot, maps 210–239, confirmatory run, or v0.6 work is authorized by this result.

See `docs/VLM_PROMPT_BELIEF_DIAGNOSTIC_ANALYSIS.md` for the scientific interpretation and `docs/VLM_PROMPT_BELIEF_NEXT_GATE.md` for the decision packet.
