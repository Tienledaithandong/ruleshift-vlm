# VLM Decision Smoke Results

**Status: EXECUTION SUCCEEDED THROUGH A1; STOPPED BY THE FROZEN A1 FEASIBILITY GATE. NO A2, A3, OR CONTROL EVIDENCE WAS PRODUCED.**

This is a bounded development feasibility smoke, not a pilot and not confirmatory evidence. The result is a genuine 12-request Qwen execution followed by a mandatory hard stop, not a model-load failure and not a completed nine-arm comparison.

## 1. Bottom line

The exact frozen 48-request smoke ran on a private Kaggle T4 kernel. The checkpoint loaded, the processor accepted the prompts, and A1 issued 12 real model requests. All 12 responses were valid JSON and parsed without fallback. Every response selected `COMMIT`.

The pre-registered A1 gate requires both at least 11 valid actions and a non-constant action distribution. A1 met the validity condition but failed the non-constant condition, so the frozen runner stopped before A2. No A2, A3, nuisance, or diagnostic request was sent.

| Item | Result |
|---|---:|
| Planned requests | 48 |
| Attempted model requests | 12 |
| Completed model requests | 12 |
| Model calls | 12 |
| Retries | 0 |
| Request failures | 0 |
| Valid A1 responses | 12/12 |
| A1 actions | 12 COMMIT, 0 PROBE_SIGN, 0 PROBE_BLUE |
| A1 gate | FAIL (`non_constant = false`) |
| Stop reason | `A1_feasibility_gate_failed` |
| A2 / A3 / controls | NOT RUN |
| Actual images consumed by the model | 0 |
| Stub dry-run requests | 48 (zero model calls) |
| GPU generation time | 24.1888 s |
| Model load wall time | 9.6181 s |
| Checkpoint verification | 42.9810 s |
| Peak CUDA allocated | 7,789,882,368 bytes |
| Peak CUDA reserved | 8,032,092,160 bytes |

Kaggle kernel: https://www.kaggle.com/code/tiennamlee/ruleshift-frozen-vlm-smoke-48

Kernel version: **4**. Terminal status: **COMPLETE**. The kernel's real-execute command returned exit code 1 because the hard gate stopped the run; the executor recorded the kernel as completed with `allow_failure=True`.

## 2. What the A1 result means

A1 is the text-only prior/evidence arm. It supplies the public rule prior and public evidence but not the numerical posterior or an image. On the frozen 12-context development subset, the model always committed:

- Optimal actions were 4 COMMIT, 4 PROBE_SIGN, and 4 PROBE_BLUE.
- The model chose COMMIT on all 12, including all 8 contexts whose exact reference optimum was a probe.
- Its optimal-action accuracy was 0.3333 (4/12) and its mean regret was 2.0928.
- These values are exactly the `always_commit` CPU baseline: accuracy 0.3333, mean regret 2.0928.
- The under-probe rate was 0.6667 (8/12); the over-probe rate was 0.0.
- The valid-action rate was 1.0; invalid and fallback counts were zero.

This is degenerate behavior under the frozen prompt. It is evidence that the model did not produce the action variation the gate required; it is not evidence that Qwen cannot use the public facts, because a single prompt format, 12 contexts, and one decoding configuration cannot separate model policy from schema or example anchoring.

One observed detail is relevant to interpretation: the frozen direct-action instruction ends with the literal example `{"selected_action":"COMMIT"}` and tells the model to replace `COMMIT` with a feasible action. All 12 outputs reproduced `{"selected_action":"COMMIT"}` exactly, with 8 output tokens each. Example/schema anchoring is a plausible contributor, but this run does not establish causation. Prompt changes are a scientific change and were not made.

## 3. A1 per-request results

`Q` values are the exact public expected control values; `regret` is the exact loss relative to the best feasible action.

| Request | Context | Best action | Q(COMMIT) | Q(PROBE_SIGN) | Q(PROBE_BLUE) | Selected | Regret |
|---|---|---|---:|---:|---:|---|---:|
| A1-01 | `vlm-dev-layout-a-base-rotate-50-expensive-0.00-` | COMMIT | 9.900 | 7.900 | 6.900 | COMMIT | 0.000 |
| A1-02 | `vlm-dev-layout-b-base-hazard-50-cheap-0.05-sign:The red tile is not the goal.` | COMMIT | 9.950 | null | 8.950 | COMMIT | 0.000 |
| A1-03 | `vlm-dev-layout-c-base-rotate-50-default-0.30-blue:hazard` | COMMIT | 9.750 | 8.750 | null | COMMIT | 0.000 |
| A1-04 | `vlm-dev-layout-a-hazard-confident-cheap-0.30-sign:The red tile is not the goal.` | COMMIT | 8.839 | null | 7.839 | COMMIT | 0.000 |
| A1-05 | `vlm-dev-layout-a-base-hazard-50-cheap-0.15-blue:safe` | PROBE_SIGN | 3.900 | 8.925 | null | COMMIT | 5.025 |
| A1-06 | `vlm-dev-layout-b-all-uniform-cheap-0.30-blue:goal` | PROBE_SIGN | 3.950 | 7.895 | null | COMMIT | 3.945 |
| A1-07 | `vlm-dev-layout-b-sign-confident-cheap-0.15-blue:hazard` | PROBE_SIGN | 4.589 | 8.430 | null | COMMIT | 3.842 |
| A1-08 | `vlm-dev-layout-c-base-hazard-90-default-0.05-blue:goal` | PROBE_SIGN | 7.552 | 8.831 | null | COMMIT | 1.279 |
| A1-09 | `vlm-dev-layout-c-hazard-sign-50-cheap-0.05-sign:The red tile is not the goal.` | PROBE_BLUE | 4.850 | null | 8.499 | COMMIT | 3.649 |
| A1-10 | `vlm-dev-layout-a-base-hazard-sign-uniform-cheap-0.05-sign:The red tile is not the goal.` | PROBE_BLUE | 4.950 | null | 8.526 | COMMIT | 3.576 |
| A1-11 | `vlm-dev-layout-c-hazard-sign-50-default-0.00-sign:The red tile is not the goal.` | PROBE_BLUE | 4.850 | null | 7.875 | COMMIT | 3.025 |
| A1-12 | `vlm-dev-layout-b-hazard-confident-cheap-0.05-` | PROBE_BLUE | 7.750 | 7.945 | 8.523 | COMMIT | 0.773 |

Sum of A1 regret: 25.113729269150735.

## 4. Arms not run

| Arm or control | Planned | Executed | Evidence |
|---|---:|---:|---|
| A1 prior/evidence text | 12 | 12 | 12 valid responses; degenerate COMMIT policy; gate failed |
| A2 posterior text | 12 | 0 | NOT RUN |
| A3 posterior RGB card | 12 | 0 | NOT RUN; no model image |
| RGB nuisance controls | 6 | 0 | NOT RUN |
| RGB semantic diagnostics | 6 | 0 | NOT RUN |

The A3 image-forwarding check in preflight is real, but it only proves that the processor produced an image tensor; it is not a model response. The visual card was never consumed by the model. No claim about visual posterior reading, OCR, chart use, nuisance invariance, diagnostic responsiveness, or the A2/A3 comparison is supported.

## 5. Frozen setup and preflight

| Item | Value |
|---|---|
| Freeze SHA-256 | `sha256:7ce10ec762564ce25ae738119514031d9b2552a2550a926473b1fb0e4b404eaa` |
| Request manifest SHA-256 | `sha256:5558a8c4c75449887c870eef67476bde0ae2caa8674b9071dff5300655ddf4a0` |
| Reference labels SHA-256 | `sha256:90bb55c07cfefdcf03e28573edbe8c7d56bdb8947c2b68f21b272d5116184551` |
| Checkpoint aggregate SHA-256 | `sha256:02bdb4ccb82eb3aceaed1608e1e7c55a81ab3e6125a20f6cccc690bbf1254c75` |
| Checkpoint | 11 files, 7,520,892,432 bytes, revision `66285546d2b821cf421d4f5eb2576359d3770cd3` |

Kaggle preflight: **PASS**.

- 48 request rows; A1/A2/A3 = 12 each; nuisance/diagnostic = 6 each.
- 24 RGB cards, all 512x320 PNG; pixel digests matched and the frozen request manifest matched exactly.
- 12 unique base images; 6 unique nuisance images; base and nuisance disjoint.
- Nuisance public payloads and labels unchanged; all six diagnostic optimal sets changed.
- Zero forbidden-label leakage in prompts and public payloads.
- A1/A2 processor probes were text-only; A3, nuisance, and diagnostic probes reached actual image tensors.
- Base, nuisance, and diagnostic vision tensors were distinct.
- The processor-only check did not load or call the model.

The executed request manifest and images were the frozen artifacts, not a re-render. This mattered because the Kaggle image encoder produced different PNG bytes from the local Windows encoder. The portability shim made the frozen PNGs authoritative for the image pixels and the frozen PNG hash, while the pixel digest and all non-image fields remained strict. This is an infrastructure/path-portability change; it did not change prompts, actions, labels, images, parser, scorer, or gates.

## 6. CPU baselines on the same 12 contexts

| Policy | Optimal-action accuracy | Mean regret | Max regret |
|---|---:|---:|---:|
| Exact public reference | 1.0000 | 0.0000 | 0.000 |
| Information-gain probe | 0.9167 | 0.0833 | 1.000 |
| Always probe | 0.5833 | 0.4648 | 2.000 |
| Random valid action | 0.5000 | 1.2285 | 5.025 |
| Always commit | 0.3333 | 2.0928 | 5.025 |
| **A1 observed** | **0.3333** | **2.0928** | **5.025** |

A1 is numerically identical to the always-commit baseline on this subset. That is the central scientific result of the smoke and the reason the gate stops the run.

## 7. Execution accounting and runtime

| Item | Value |
|---|---:|
| Kernel | `tiennamlee/ruleshift-frozen-vlm-smoke-48`, version 4 |
| Kernel terminal status | COMPLETE |
| GPU | Tesla T4, device 0 (of 2 visible) |
| GPU total memory | 15,636,037,632 bytes |
| GPU free before load | 15,526,068,224 bytes |
| CUDA / driver | CUDA 13.0; driver 580.178.04 |
| PyTorch / transformers / Pillow | 2.11.0+cu130 / 5.20.2 / 13.0.0 |
| Model parameters | 3,754,622,976 |
| Load | 3.8476 s GPU load; 9.6181 s wall |
| Checkpoint verification | 42.9810 s |
| Real-execute wall | 83.2613 s |
| GPU generation time | 24.1888 s |
| Mean GPU time per request | 2.0157 s |
| Mean response latency | 2,020.8 ms |
| Latency range | 1,884.1-2,700.8 ms |
| Input tokens | 652-751; mean 670.75; total 8,049 |
| Output tokens | 8 per response; total 96 |
| Peak CUDA allocated | 7,789,882,368 bytes |
| Peak CUDA reserved | 8,032,092,160 bytes |
| Retries | 0 |
| Model calls | 12 |

The kernel wall time was approximately 2 minutes 40 seconds. The `real_execute` command ran for 83.26 s; the preflight ran for 21.51 s and the 48-request stub dry run for 5.77 s.

## 8. Artifacts

Primary local artifact directory: `results/kaggle_smoke_output/`.

| Artifact | Path |
|---|---|
| Raw run (authoritative) | `results/kaggle_smoke_output/results/vlm_decision_smoke/real/real/` |
| Raw requests | `.../real/real/requests.jsonl` |
| Raw responses | `.../real/real/responses.jsonl` |
| Parsed actions | `.../real/real/parsed.jsonl` |
| Arm summaries | `.../real/real/arm_summaries.json` |
| Run summary | `.../real/real/summary.json` |
| Accounting | `.../real/real/request_accounting.json` |
| Load and memory | `.../real/real/model_load.json` |
| Checksums | `.../real/real/checksums.json` |
| Preflight | `results/kaggle_smoke_output/results/vlm_decision_smoke/kaggle_preflight/preflight.json` |
| Stub dry run | `results/kaggle_smoke_output/results/vlm_decision_smoke/kaggle_dry_run/dry_run/` |
| CPU baselines | `results/kaggle_smoke_output/results/vlm_decision_smoke/cpu_baselines.json` |
| Kaggle hardware | `results/kaggle_smoke_output/results/vlm_decision_smoke/kaggle_hardware.json` |
| Executor manifest | `results/kaggle_smoke_output/KAGGLE_EXECUTION_MANIFEST.json` |
| Kernel log | `results/kaggle_smoke_output/ruleshift-frozen-vlm-smoke-48.log` |

Independent checksum verification passed: 10/10 files in `real/real`, 10/10 files in the stub dry run, and 1/1 preflight file.

`KAGGLE_EXECUTION_MANIFEST.json` reports `outputs` as null/zero because its summary helper looked at the wrong nesting level (`real/` rather than `real/real/`). This is an artifact-collection defect, not a missing run. The authoritative raw `summary.json` and `request_accounting.json` both report 12 model calls, 12 completed requests, and zero retries.

## 9. Gates

| Gate | Outcome |
|---|---|
| Source freeze and request manifest | PASS |
| Checkpoint revision and aggregate hash | PASS |
| Preflight: request count, images, pixel digests, leakage | PASS |
| Actual processor image forwarding (preflight) | PASS |
| 48-request stub dry run | PASS; 48/48 synthetic, zero model calls |
| Model load on Kaggle T4 | PASS |
| A1 valid-action gate (at least 11 of 12) | PASS; 12/12 |
| A1 non-constant action gate | FAIL; one distinct valid action |
| A1 feasibility gate overall | FAIL |
| A2 / A3 / nuisance / diagnostic | NOT RUN |
| Request cap | 12 of 48; stopped early by gate |

## 10. What this run supports and does not support

Supported:

- The exact frozen checkpoint loads in bf16 on a Kaggle T4 and can issue valid direct-action outputs.
- The preflight, frozen-image portability path, parser, cache, accounting, and checksum path worked.
- The A1 prompt produced 12 valid, non-fallback responses with zero request failures.
- The A1 behavior was degenerate: all COMMIT, identical to the always-commit baseline.

Not supported:

- Any A2 or A3 result; neither arm ran.
- Any claim that the model can read the visual posterior card.
- Any claim about posterior use, OCR, chart reading, nuisance invariance, or diagnostic responsiveness.
- Any claim that Qwen is generally incapable of the task, or that a prompt change would fix it.
- Any promotion to the 1,536-request pilot, the 118-context continuation, reserved maps, confirmatory evaluation, or v0.6.

## 11. Status relative to the larger program

The symbolic phase remains closed. v0.5 remains a development reference, not confirmatory. The 900-episode reserved-map confirmation remains deferred. Maps 210-239 remain reserved and unauthorized. No v0.6 development occurred. No full nine-arm 1,536-request pilot, 118-context continuation, confirmatory evaluation, prompt tuning, or automatic merge occurred.

The smoke is complete for the authorization that existed. Its outcome is a scientific stop at the A1 gate, recorded in `docs/VLM_DECISION_SMOKE_FAILURE_ANALYSIS.md` and `docs/VLM_DECISION_PILOT_DECISION_PACKET.md`.
