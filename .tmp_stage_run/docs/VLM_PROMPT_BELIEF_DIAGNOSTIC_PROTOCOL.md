# VLM Prompt Anchoring x Belief Conditioning Diagnostic Protocol

**Experiment identifier:** `vlm-prompt-belief-diagnostic-v1`

**Status:** FROZEN BEFORE INFERENCE.

**Scope:** One bounded development diagnostic on the 12 previously exposed development contexts. This is not confirmatory evidence, not a pilot, and not a test of visual posterior reading.

## 1. Relationship to the closed smoke

The original `vlm-decision-smoke-v1` is closed as an A1 feasibility failure. Its 12 genuine model responses, hashes, result documents, local failure record, Kaggle versions 1-3 infrastructure records, and version 4 run are preserved unchanged. The remaining 36 requests from that protocol will not be resumed.

The original A1 responses are retained as a historical comparison only. They are not substituted for the new matched B1 replication.

This experiment is separately preregistered and uses a new experiment identifier, freeze, source dataset, kernel, and output directory.

## 2. Objective

Distinguish three candidate explanations for the original all-COMMIT A1 behavior:

1. **Prompt-format anchoring:** the literal `{"selected_action":"COMMIT"}` example dominates the response.
2. **Belief-computation limitation:** the model does not use the supplied prior/evidence well, even when the posterior is made explicit in text.
3. **Probe-versus-commit decision limitation:** the model represents the facts but does not select a probe when the expected value favors inspection.

The design is a matched 2x2 factorial. It estimates descriptive effects; it does not identify a causal mechanism with 12 contexts.

## 3. Frozen 12-context development subset

The contexts and exact reference labels are the same 12 used by the closed smoke. No reserved-map, held-out, or confirmatory context is included.

| Context ID | Exact best action | Optimal set | Decision margin |
|---|---|---|---:|
| `vlm-dev-layout-a-base-rotate-50-expensive-0.00-` | COMMIT | `COMMIT` | 2.000000 |
| `vlm-dev-layout-b-base-hazard-50-cheap-0.05-sign:The red tile is not the goal.` | COMMIT | `COMMIT` | 1.000000 |
| `vlm-dev-layout-c-base-rotate-50-default-0.30-blue:hazard` | COMMIT | `COMMIT` | 1.000000 |
| `vlm-dev-layout-a-hazard-confident-cheap-0.30-sign:The red tile is not the goal.` | COMMIT | `COMMIT` | 1.000000 |
| `vlm-dev-layout-a-base-hazard-50-cheap-0.15-blue:safe` | PROBE_SIGN | `PROBE_SIGN` | 5.025000 |
| `vlm-dev-layout-b-all-uniform-cheap-0.30-blue:goal` | PROBE_SIGN | `PROBE_SIGN` | 3.944828 |
| `vlm-dev-layout-b-sign-confident-cheap-0.15-blue:hazard` | PROBE_SIGN | `PROBE_SIGN` | 3.841803 |
| `vlm-dev-layout-c-base-hazard-90-default-0.05-blue:goal` | PROBE_SIGN | `PROBE_SIGN` | 1.278723 |
| `vlm-dev-layout-c-hazard-sign-50-cheap-0.05-sign:The red tile is not the goal.` | PROBE_BLUE | `PROBE_BLUE` | 3.649375 |
| `vlm-dev-layout-a-base-hazard-sign-uniform-cheap-0.05-sign:The red tile is not the goal.` | PROBE_BLUE | `PROBE_BLUE` | 3.576250 |
| `vlm-dev-layout-c-hazard-sign-50-default-0.00-sign:The red tile is not the goal.` | PROBE_BLUE | `PROBE_BLUE` | 3.025000 |
| `vlm-dev-layout-b-hazard-confident-cheap-0.05-` | PROBE_BLUE | `PROBE_BLUE` | 0.577750 |

Each arm receives four contexts from each exact best-action class. The evaluator reference labels are frozen in `reference_labels.json`; the displayed table is a summary.

## 4. Four arms

| Arm | Belief condition | Output instruction | Prompt body source |
|---|---|---|---|
| B1 | prior/evidence text, no explicit posterior | original anchored instruction | exact original A1 body |
| B2 | prior/evidence text, no explicit posterior | neutral schema-only instruction | exact original A1 body |
| B3 | explicit posterior text | original anchored instruction | exact original A2 body |
| B4 | explicit posterior text | neutral schema-only instruction | exact original A2 body |

B1 is a byte-for-byte replication of the original A1 prompt for the same context. B3 is a byte-for-byte replication of the original A2 prompt for the same context. B2 and B4 replace only the final output instruction of B1 and B3 respectively. No other prompt element, context payload, feasible-action menu, candidate-rule menu, parser, decoding setting, or generation budget changes.

### Anchored footer (B1 and B3, verbatim)

```text
Return ONLY one compact JSON object with exactly this key: {"selected_action":"COMMIT"} (replace COMMIT with one feasible action). Do not include reasoning, markdown, code fences, or extra keys.
```

### Neutral footer (B2 and B4, verbatim)

```text
Return ONLY one compact JSON object with exactly one key, "selected_action". Its value MUST be one of the feasible actions listed above. Do not include reasoning, markdown, code fences, or any other key.
```

The neutral footer describes the JSON key and the requirement to choose a feasible action. It contains no COMMIT, PROBE_SIGN, PROBE_BLUE, or other concrete action value, and no correct-action example. The feasible-action menu elsewhere in the prompt necessarily lists the three action names as the available choice set; that is a menu, not an answer example.

No reasoning hint, chain-of-thought request, rule declaration, confidence request, or correct-action example is added.

## 5. Model and runtime contract

| Field | Frozen value |
|---|---|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Checkpoint | Exact verified 11-file snapshot, aggregate `sha256:02bdb4ccb82eb3aceaed1608e1e7c55a81ab3e6125a20f6cccc690bbf1254c75` |
| Device | Kaggle CUDA GPU |
| Parameter dtype | bfloat16 |
| Attention | SDPA |
| Local files only | `local_files_only=True`; no network weight download |
| Decoding | greedy, `do_sample=False`, seed `20261126` |
| Generation budget | `max_new_tokens=32` |
| Inputs | text only; no image tensor, no visual inference |
| Training/fine-tuning | none |
| Quantization/offload/substitution | none |

The effective runtime must be verified on the Kaggle GPU before the first request. The run records the actual parameter dtype, the actual compute condition (no autocast unless it is observed and reported), the attention implementation, device capability, and CUDA/PyTorch flags.

**T4 BF16 audit.** A Tesla T4 is compute capability 7.5 and does not provide native BF16 tensor-core support. The diagnostic will not label the run as native BF16 merely because execution succeeds. It will record:

- actual model parameter dtype;
- whether CUDA autocast is enabled and its dtype;
- the relevant compute dtype used by the frozen backend;
- attention implementation;
- `torch.cuda.is_bf16_supported()` as reported by the runtime;
- device capability and a hardware-native flag derived from capability major >= 8;
- any explicit casting, autocast, or precision fallback observed.

The precision condition is fixed to the same bfloat16 parameter load used by the closed smoke. No FP16, FP32, quantized, or offloaded condition is introduced. If the audit shows a silent precision change, the run must stop before any request.

## 6. Deterministically balanced request order

There are 12 blocks, one per context, in the frozen `BASE_IDS` order. Each block contains one request per arm. For block index `i` (zero-based), the arm order is:

```text
(B1, B2, B3, B4)[i % 4:] + (B1, B2, B3, B4)[:i % 4]
```

This gives each arm 12 requests, gives each arm first position exactly three times, and prevents arm-major execution. Request indices are 1-48 in the resulting sequence. Request IDs are `B1-01` through `B4-12`; each arm-context pair appears exactly once.

## 7. Frozen prompt hashes

Prompt text is not reconstructed at execution time. B1 and B3 are checked byte-for-byte against the original A1 and A2 prompts. B2 and B4 are checked byte-for-byte against a footer-only replacement. The full prompt text is frozen in `prompts.jsonl`; this table is the independent prompt-hash ledger.

| Index | Request | Arm | Belief | Format | Label | Chars | Prompt SHA-256 |
|---:|---|---|---|---|---|---:|---|
| 1 | B1-01 | B1 | prior_evidence | anchored | COMMIT | 2520 | `sha256:165b0cb7621d01b5ce19ab6a143fda7101f6498a2adc3e05821869a162caa715` |
| 2 | B2-01 | B2 | prior_evidence | neutral | COMMIT | 2529 | `sha256:438a79768a0725352ab480f55e4f27f69b7ca86f17688a42eafd0922df98ed14` |
| 3 | B3-01 | B3 | explicit_posterior | anchored | COMMIT | 2642 | `sha256:088e45ea76bc3aea53633deecc0c3b3639d0176af23892d81f9dff14873a7e9e` |
| 4 | B4-01 | B4 | explicit_posterior | neutral | COMMIT | 2651 | `sha256:c2c40fd2501ab6cebb29b5437a417b0eca4677e65d1d86054720a83730b199a4` |
| 5 | B2-02 | B2 | prior_evidence | neutral | COMMIT | 2569 | `sha256:71c9e9d49c22816904b10531cabcb936e699595b41f92fc00bc2cf44494a5de4` |
| 6 | B3-02 | B3 | explicit_posterior | anchored | COMMIT | 2682 | `sha256:814d0ad05d5646635bc8f355c6c0bb53625321be938d40436e5441feb6a9c1c5` |
| 7 | B4-02 | B4 | explicit_posterior | neutral | COMMIT | 2691 | `sha256:051426bd052824386144e3598bd502cf18d7f2b2b3662f11eb2810d276fb52b4` |
| 8 | B1-02 | B1 | prior_evidence | anchored | COMMIT | 2560 | `sha256:87e2101a01435871dbf7f1799587357cc25abe0e413bcaf2b5e24bba78a4a08e` |
| 9 | B3-03 | B3 | explicit_posterior | anchored | COMMIT | 2643 | `sha256:7ab9c5294d2fe412cfa73b50559bc606ce8182898fde87f3e4d31b43287b1a70` |
| 10 | B4-03 | B4 | explicit_posterior | neutral | COMMIT | 2652 | `sha256:8b3ede7a0b99b6d6a20a262c40c14f6faebaf29910a602bcbd8b237127b831ad` |
| 11 | B1-03 | B1 | prior_evidence | anchored | COMMIT | 2521 | `sha256:ff6f891e539bdfdd00be54729940646cbf43ba38b4c029e1020f6927bbb5435c` |
| 12 | B2-03 | B2 | prior_evidence | neutral | COMMIT | 2530 | `sha256:1c1eb18ea1dfb4fcd096b08593243a771ab51f311a675ef8f29579abf4131a71` |
| 13 | B4-04 | B4 | explicit_posterior | neutral | COMMIT | 2720 | `sha256:dbc656c94bce675ead8a88a2cdec658463436de98676f776ca57a76dd7117ad2` |
| 14 | B1-04 | B1 | prior_evidence | anchored | COMMIT | 2559 | `sha256:f9cb73b0810af77dea605a4b24f10c5a1a6c6ef6fe9272f21c8d0e64ab2f1416` |
| 15 | B2-04 | B2 | prior_evidence | neutral | COMMIT | 2568 | `sha256:7cb1a9656b044841264e05e8ae511628592bc7635ca1117a1597db496e245dca` |
| 16 | B3-04 | B3 | explicit_posterior | anchored | COMMIT | 2711 | `sha256:a94528c494b976c558e01c24a09dc37e0cfc21a99ead65d5c3f84a6503ae1343` |
| 17 | B1-05 | B1 | prior_evidence | anchored | PROBE_SIGN | 2526 | `sha256:a91a40ebe5087a9738be77a072e10ccd4d362396043bbaba0659ccba7eab96cb` |
| 18 | B2-05 | B2 | prior_evidence | neutral | PROBE_SIGN | 2535 | `sha256:8b77b48cc046971e79371a34bf5980592a95c69f2892159a624b469818652309` |
| 19 | B3-05 | B3 | explicit_posterior | anchored | PROBE_SIGN | 2648 | `sha256:6779f9b5dee571bb17398eedb058b7cc114e4ea2fb1394460b1ad66f4640d7dc` |
| 20 | B4-05 | B4 | explicit_posterior | neutral | PROBE_SIGN | 2657 | `sha256:c4ad6a5e4d6d286385eaa15fa2c3a841ecd3b2c3a6a06d2a0317decca30a035e` |
| 21 | B2-06 | B2 | prior_evidence | neutral | PROBE_SIGN | 2630 | `sha256:14d782cddcc39f533f9a2a0d026d5145bb69c2e1d1820dad4728af21c180b6d0` |
| 22 | B3-06 | B3 | explicit_posterior | anchored | PROBE_SIGN | 2838 | `sha256:7e81ee24b85863def925d0b2c3c207c2edaf039467744b9cab8ee9f96cef6efd` |
| 23 | B4-06 | B4 | explicit_posterior | neutral | PROBE_SIGN | 2847 | `sha256:7b0405fb5deaba9832d0581c1dbe38115acc141ccd54e8b25b2392a200bec5b1` |
| 24 | B1-06 | B1 | prior_evidence | anchored | PROBE_SIGN | 2621 | `sha256:62830ea66ee4310612f2714ba105606d00f2f59c077d3dc416ad95f76e1a0fbb` |
| 25 | B3-07 | B3 | explicit_posterior | anchored | PROBE_SIGN | 2697 | `sha256:074846ebb3eb3da53473db0b689394c727affaedfd64280d53020d6e1d5b380d` |
| 26 | B4-07 | B4 | explicit_posterior | neutral | PROBE_SIGN | 2706 | `sha256:31cb480c2085fb1ea1724c8f95f53166a6bdfd1999a41a665a656dcbdb6813ce` |
| 27 | B1-07 | B1 | prior_evidence | anchored | PROBE_SIGN | 2528 | `sha256:ab4b6237d39e8807d00cbc0dddb4448f294c99a7efe6689e275da58e16972fe7` |
| 28 | B2-07 | B2 | prior_evidence | neutral | PROBE_SIGN | 2537 | `sha256:2d6e8dd3afa5ef7677c85a19754378fba5d54ca861d7758f9f9400b6e1edbc73` |
| 29 | B4-08 | B4 | explicit_posterior | neutral | PROBE_SIGN | 2682 | `sha256:b65f9ceb248a48d9eda8e669b88ee1ec24a55a300e480ca5bc69dbeaa0dc96e7` |
| 30 | B1-08 | B1 | prior_evidence | anchored | PROBE_SIGN | 2520 | `sha256:93194e7e5acfa24b309bb7a4b2d92b7ccc72f40c4adcb2ff61bee355e502985a` |
| 31 | B2-08 | B2 | prior_evidence | neutral | PROBE_SIGN | 2529 | `sha256:ad4106790a90a0c256fd473292630be3cb148b73d4faa0b367d324dc67df497d` |
| 32 | B3-08 | B3 | explicit_posterior | anchored | PROBE_SIGN | 2673 | `sha256:d19cc7e639a7f27c63cf0e383c6f60236e16a7f843764cfdca42d7247d3de796` |
| 33 | B1-09 | B1 | prior_evidence | anchored | PROBE_BLUE | 2554 | `sha256:ec1e9860ee27f89d158bd3bc12e02acdca338444a16a61a9f123369d29d4c654` |
| 34 | B2-09 | B2 | prior_evidence | neutral | PROBE_BLUE | 2563 | `sha256:1ba601fd37732408cf221c7bf75b53697416e0b383c1180f6e7ffd9eb14371c4` |
| 35 | B3-09 | B3 | explicit_posterior | anchored | PROBE_BLUE | 2676 | `sha256:31382c1b4aebf0808d7c67209b7c657b0d562dcf034455b6ead87203a80341cc` |
| 36 | B4-09 | B4 | explicit_posterior | neutral | PROBE_BLUE | 2685 | `sha256:8c686b236109f0c5ecd6cfda78dd4fc7ea540b8d475825f487bf41c6dbcaa0f2` |
| 37 | B2-10 | B2 | prior_evidence | neutral | PROBE_BLUE | 2614 | `sha256:1843dc38196f0513fea4698e7864a8afeaedcad65478cc310bf9bc638aa613c5` |
| 38 | B3-10 | B3 | explicit_posterior | anchored | PROBE_BLUE | 2727 | `sha256:4a36354d34fa0f7043490386d3e3deb048ef395b45f1c18a62780bc8b07fa22f` |
| 39 | B4-10 | B4 | explicit_posterior | neutral | PROBE_BLUE | 2736 | `sha256:36bf4a1332e85d55f00f8d1af13f3f737140a23488a8100cb310614843252fff` |
| 40 | B1-10 | B1 | prior_evidence | anchored | PROBE_BLUE | 2605 | `sha256:fa3194fa798c5a6f9b2173ff42a10817f526940e7b6cb3e028245391d09958b7` |
| 41 | B3-11 | B3 | explicit_posterior | anchored | PROBE_BLUE | 2675 | `sha256:f091cccf49540e05e240301b8c6ca6c75a947117867aeb497a00d22fe973e8a9` |
| 42 | B4-11 | B4 | explicit_posterior | neutral | PROBE_BLUE | 2684 | `sha256:8be444d8dec36c13773689228f2baee1135649629aac2447e877e165e6db3601` |
| 43 | B1-11 | B1 | prior_evidence | anchored | PROBE_BLUE | 2553 | `sha256:7131522020eeb2072b2955f455f4605adf83ec8f2b1f539230890a1552c151ed` |
| 44 | B2-11 | B2 | prior_evidence | neutral | PROBE_BLUE | 2562 | `sha256:bdba0b69823aa22c5b3dfd1d0ba709f75b06a6196e5d794bb23ea39680c96480` |
| 45 | B4-12 | B4 | explicit_posterior | neutral | PROBE_BLUE | 2652 | `sha256:5a6ca2561d618ddd3ce50123f7eff07b800fe619cfa425115cd575ca0007a819` |
| 46 | B1-12 | B1 | prior_evidence | anchored | PROBE_BLUE | 2521 | `sha256:6fd3d3c87cbd2e3d2705f957cde1265ec47919df4389314a331d42906d5a7590` |
| 47 | B2-12 | B2 | prior_evidence | neutral | PROBE_BLUE | 2530 | `sha256:d1d3eb470a2b65659ea9b5c159748e246294cb28be00439eed52663bb090f089` |
| 48 | B3-12 | B3 | explicit_posterior | anchored | PROBE_BLUE | 2643 | `sha256:8cbb030603d9d23f2532a84515f9fb7baaeb96c5cc78c785d73d53d2a1fc83fd` |

## 8. Parser and scoring

The parser is the frozen direct-action parser `vlm-smoke-direct-parser-v1`. It accepts only one JSON object with `selected_action` and validates the value against the request feasible-action menu. A parse failure becomes an explicit fallback to COMMIT with `valid=false` and `fallback=true`; fallbacks remain visible in every denominator.

Scoring uses the exact finite-horizon, cost-aware reference `vlm-decision-reference-v1`. Every arm is scored against the same exact label for its context. The primary per-arm values are:

- valid response rate;
- invalid and fallback counts;
- action distribution;
- optimal-action accuracy, overall and per exact best-action class;
- mean and maximum decision regret;
- over-probe and under-probe rates;
- tool-selection accuracy on valid probe selections;
- input and output token counts;
- GPU generation time and peak CUDA memory.

All five CPU baselines (`always_commit`, `always_probe`, `random_valid`, `information_gain`, `exact`) are recomputed on the exact 12-context subset with seed `20261126`.

## 9. Paired comparisons and descriptive interaction

All comparisons are paired at `context_id`:

| Comparison | Question |
|---|---|
| B2 - B1 | Prompt-format effect without an explicit posterior |
| B4 - B3 | Prompt-format effect with an explicit posterior |
| B3 - B1 | Explicit-posterior effect under the anchored format |
| B4 - B2 | Explicit-posterior effect under the neutral format |

For accuracy and mean regret, report the paired difference, the number of action changes, and the per-context paired rows. The descriptive interaction is:

```text
(B4 - B3) - (B2 - B1)
```

This is an interaction of differences, not a causal estimate. With 12 previously exposed contexts, it is descriptive only. No p-value, confidence interval, or confirmatory claim is planned.

## 10. Eligibility criteria

An arm is eligible for consideration in a separately authorized visual smoke only if **all** of these development criteria hold:

1. at least 11 of 12 responses are valid;
2. at least 8 of 12 selected actions are exact-optimal;
3. at least one correct action occurs in each of the three exact best-action classes (COMMIT, PROBE_SIGN, PROBE_BLUE);
4. the valid action distribution is non-constant;
5. mean regret is strictly lower than the `always_commit` baseline on the same contexts.

Passing is a feasibility result, not evidence of superiority. Every arm that passes is also compared descriptively with `information_gain`; that stronger baseline is not relaxed and no threshold is changed after seeing results.

If all four arms fail, the recommendation is to close the current 3B feasibility line and consider a separately authorized stronger-model experiment. If an arm passes, the next document is a small RGB feasibility proposal for human review, not an automatic visual run.

## 11. Execution and technical stopping conditions

The maximum is **48 new real model requests**, four arms x 12 contexts. Failed and retried requests count against the cap. There are **no automatic retries**.

Constant-action behavior in one arm is a scientific outcome, not a technical stopping condition. All four arms are evaluated unless one of these technical conditions occurs:

- a source, protocol, prompt, label, manifest, or freeze hash mismatch;
- a checkpoint revision, file-set, size, or aggregate-hash mismatch;
- a public-leakage check failure;
- an unexpected image or multimodal tensor;
- CUDA, model-load, processor, or generation failure;
- a request exception or an input-token accounting mismatch;
- evidence that the frozen bfloat16 precision condition changed silently;
- a request-budget violation or inability to account for every attempted call.

Invalid model JSON is not a technical stop. It is parsed as a fallback, counted, and the next preregistered request proceeds. No prompt, parser, threshold, model, revision, or decoding setting may change after the first real request.

## 12. Artifact and freeze requirements

Before inference, `scripts/prepare_vlm_prompt_belief.py` must write a new freeze under:

```text
results/vlm_prompt_belief_diagnostic/frozen/
```

and refuse any non-empty target. The freeze contains the protocol hash, exact prompts, prompt ledger, public payloads, reference labels, request manifest, analysis plan, model/runtime contract, checkpoint manifest, and source hashes. `freeze.sha256` covers `freeze.json`. The executor must verify the freeze and the rebuilt 48-request plan before any real request.

New output directories are `results/vlm_prompt_belief_diagnostic/preflight/`, `dry_run/`, and `real/`. The historical directories and the original artifact manifest are never overwritten.

## 13. Infrastructure correction

The historical Kaggle collector searched `results/vlm_decision_smoke/real/`, while the runner had written `results/vlm_decision_smoke/real/real/`. This was an artifact-collection defect: the run itself was present and the raw summary was correct.

The correction is:

- the collector resolves the nested `real/real/` layout first and the flat `real/` layout second;
- the corrected path is covered by a regression test that compares summary counters with raw request, response, and parsed logs;
- the original `KAGGLE_EXECUTION_MANIFEST.json` and all historical artifacts remain byte-for-byte unchanged;
- this correction is documented separately from the new experimental result and does not alter any scientific field.

## 14. Prohibitions

- no visual inference in this experiment;
- no continuation of the closed 36-request smoke;
- no full nine-arm pilot;
- no reserved maps 210-239;
- no confirmatory experiment;
- no v0.6;
- no automatic prompt-tuning loop;
- no automatic merge or modification of PR #22 or PR #23;
- no training, fine-tuning, quantization, offload, model substitution, or precision change.

## 15. Frozen declaration

This protocol, the exact 48 prompts, the request order, the parser, the scoring rules, and the eligibility criteria are frozen before the first new model request. No model-generated information is used to choose or modify any arm.

The only permitted post-freeze changes are infrastructure or path-portability fixes that do not alter prompts, contexts, labels, parser behavior, scoring, model revision, precision, or analysis. Any such fix must be documented.