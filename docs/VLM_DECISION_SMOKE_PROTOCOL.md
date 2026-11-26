# VLM Decision Smoke Protocol

**Status:** FROZEN BEFORE FIRST MODEL REQUEST — bounded development feasibility smoke only.

**Human decision:** CONDITIONAL GO FOR BOUNDED VLM SMOKE. CPU infrastructure accepted. This is not a confirmation run, not a pilot result, and not authorization for a larger stage.

**Date of protocol:** 2026-11-26. This document is written before any Qwen generation. The freeze manifest records its hash; it must not be edited after the first model request.

## 1. Authority and scope

The authorized smoke is limited to:

- 12 development contexts selected and frozen before inference: four COMMIT, four PROBE_SIGN, and four PROBE_BLUE.
- Three direct-action arms: A1 (prior + evidence text), A2 (posterior text), and A3 (posterior RGB card).
- Six RGB nuisance controls and six RGB semantic diagnostic controls.
- At most 48 model requests in total, including failures and retries. No automatic retry is implemented.
- The exact local checkpoint Qwen/Qwen2.5-VL-3B-Instruct at revision 66285546d2b821cf421d4f5eb2576359d3770cd3.
- Direct-action JSON only. No rationale, declaration, abstention, or confidence arms.
- Development feasibility reporting only.

Not authorized: the full nine-arm 1,536-request pilot, a 118-context continuation, the 900-episode reserved-map confirmation, maps 210–239, confirmatory evaluation, prompt/parser tuning after the first request, model substitution, quantization, offload, training, fine-tuning, automatic merge, or v0.6 development.

Preserved without modification: candidate commit 751a3414ad8815087f64d85ad5a1523af11077f4; docs commit bc79ce1d7a54aea38fa7502452d646450121c09a; frozen commits and manifests; PR #22 head a3e614b093dfd8237898cb19ac83cab92b8bad50; PR #23 as published; all v0.5 historical results and decisions.

## 2. Branch and regression reconciliation

The current microbenchmark branch is research/vlm-decision-microbenchmark at a51929d3acadc1f3d9b8d404081e31f94ee82afd. Its parent is f4b43a4ede7f76ab9d5b1f45178fe648ae419042, which is also the merge base with the v0.5 branch (docs commit bc79ce1d7a54aea38fa7502452d646450121c09a). The branches are siblings from f4b43a4; the microbenchmark branch intentionally does not carry the v0.5/identification/declaration modules. It does not rebase, reset, or replace either branch.

Test-count reconciliation:

- f4b43a4 baseline: 462 collected tests.
- a51929d (current microbenchmark): 479 collected tests, +17. The additions are 16 tests in tests/test_vlm_decision.py and 1 test in tests/test_vlm_backend_accounting.py.
- v0.5 bc79ce1: 528 collected tests, +66 from f4b43a4. The v0.5 change adds eight test modules plus one fixture: test_declaration_diagnosis, test_declaration_equivalence, test_grid_audits, test_grid_identification, test_grid_identification_runner, test_grid_identification_v2, test_grid_map_analysis, and test_grid_v05_policy.
- Therefore 528 - 479 = 49 = 66 - 17. The 49-test difference is v0.5-only coverage, not 49 missing current-scope regressions.
- This smoke adds nine CPU tests in tests/test_vlm_smoke.py, bringing the expected current count to 488. The full current suite must pass before the freeze; all inherited tests plus the new smoke tests are in scope. No v0.5 module is imported or skipped silently.

The 479-versus-528 comparison is about the two branch scopes at a51929d and bc79ce1. After this smoke the current count is expected to be 488; the remaining 40-test difference is still v0.5-only coverage (66 v0.5 additions versus 17 existing plus 9 smoke additions).

## 3. Frozen model and runtime

Model: Qwen/Qwen2.5-VL-3B-Instruct.

Revision: 66285546d2b821cf421d4f5eb2576359d3770cd3.

Local snapshot: C:/Users/Lenovo/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3.

The snapshot has 11 files and 7,520,892,432 bytes. The preparation script hashes every file and verifies the model.safetensors.index.json weight map. The aggregate checkpoint hash recorded in the freeze is:

sha256:02bdb4ccb82eb3aceaed1608e1e7c55a81ab3e6125a20f6cccc690bbf1254c75

The actual runner re-verifies the checkpoint before loading. Local-files-only is mandatory. If the checkpoint is absent, unreadable, hash-mismatched, or not the exact revision, the run stops and reports a blocker. No download, model substitution, quantization, CPU offload, or fallback backend is permitted.

Observed load probe, performed without generation: processor loaded; model loaded in 19.8 s on CPU, moved to CUDA in 9.3 s; 3,754,622,976 parameters; 7,628,753,408 CUDA bytes allocated; 7,874,805,760 CUDA bytes reserved; GPU total 8,585,216,000 bytes. The free-VRAM margin is small. The first actual request may fail from memory pressure. If it does, the failed request is preserved, counted, and the run stops. There is no retry and no workaround that changes the model or precision.

Runtime observed: Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, transformers 5.23.0, NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB total), driver 581.86, about 15.19 GB RAM. These are runtime facts, not guarantees for the run.

## 4. Frozen 12-context development subset

The subset is static and was selected before any model call. The selection rule was: exactly four contexts for each exact best action; each with a unique exact optimum; at least two feasible actions; base decision margin at least 0.5; then coverage of layouts and evidence types, with one moderate-margin PROBE_BLUE context retained because it admits a stronger semantic diagnostic. No model output influenced this list.

| ID | Exact best action | Reference margin | Feasible actions |
|---|---|---:|---|
| vlm-dev-layout-a-base-rotate-50-expensive-0.00- | COMMIT | 2.0 | 3 |
| vlm-dev-layout-b-base-hazard-50-cheap-0.05-sign:The red tile is not the goal. | COMMIT | 1.0 | 2 |
| vlm-dev-layout-c-base-rotate-50-default-0.30-blue:hazard | COMMIT | 1.0 | 2 |
| vlm-dev-layout-a-hazard-confident-cheap-0.30-sign:The red tile is not the goal. | COMMIT | 1.0 | 2 |
| vlm-dev-layout-a-base-hazard-50-cheap-0.15-blue:safe | PROBE_SIGN | 5.025 | 2 |
| vlm-dev-layout-b-all-uniform-cheap-0.30-blue:goal | PROBE_SIGN | 3.9448 | 2 |
| vlm-dev-layout-b-sign-confident-cheap-0.15-blue:hazard | PROBE_SIGN | 3.8418 | 2 |
| vlm-dev-layout-c-base-hazard-90-default-0.05-blue:goal | PROBE_SIGN | 1.2787 | 2 |
| vlm-dev-layout-c-hazard-sign-50-cheap-0.05-sign:The red tile is not the goal. | PROBE_BLUE | 3.649375 | 2 |
| vlm-dev-layout-a-base-hazard-sign-uniform-cheap-0.05-sign:The red tile is not the goal. | PROBE_BLUE | 3.57625 | 2 |
| vlm-dev-layout-c-hazard-sign-50-default-0.00-sign:The red tile is not the goal. | PROBE_BLUE | 3.025 | 2 |
| vlm-dev-layout-b-hazard-confident-cheap-0.05- | PROBE_BLUE | 0.57775 | 3 |

All twelve have a unique exact optimum and at least two feasible actions. The minimum base margin is 0.57775; the other eleven are at least 1.0. Exact expected-control values and latent labels are stored separately in the frozen reference labels and are never placed in prompts or public cache keys.

## 5. Frozen controls

Nuisance controls use the same context, exact label, map, posterior, evidence, costs, and feasible actions as their base; only palette and border change. They are paired with the base A3 request for:

- The first COMMIT context.
- The second COMMIT context.
- The first PROBE_SIGN context.
- The second PROBE_SIGN context.
- The first PROBE_BLUE context.
- The fourth PROBE_BLUE context (the moderate-margin context).

The nuisance invariant is checked three ways: identical public payload, identical reference label, and distinct rendered pixel digest.

Semantic diagnostic controls replace the observed blue-cell role, recompute the exact posterior from the public model, and require a different exact optimal-action set. Their fresh labels are not copied from the base. The six pairs are:

| Base ID | Diagnostic role | Base best | Fresh best | Fresh margin |
|---|---|---:|---:|---:|
| vlm-dev-layout-a-base-hazard-50-cheap-0.15-blue:safe | goal | PROBE_SIGN | COMMIT | 0.031 |
| vlm-dev-layout-b-all-uniform-cheap-0.30-blue:goal | hazard | PROBE_SIGN | COMMIT | 0.3177 |
| vlm-dev-layout-b-sign-confident-cheap-0.15-blue:hazard | safe | PROBE_SIGN | COMMIT | 0.8926 |
| vlm-dev-layout-c-base-hazard-90-default-0.05-blue:goal | hazard | PROBE_SIGN | COMMIT | 0.9647 |
| vlm-dev-layout-c-hazard-sign-50-cheap-0.05-sign:The red tile is not the goal. | goal | PROBE_BLUE | COMMIT | 0.0 (only COMMIT feasible) |
| vlm-dev-layout-b-hazard-confident-cheap-0.05- | hazard | PROBE_BLUE | PROBE_SIGN | 1.0947 |

Four diagnostics have both COMMIT and a probe feasible; B1 is forced to COMMIT and B4 changes to a feasible PROBE_SIGN. The controls are descriptive. They test whether the observed action tracks a real semantic change, not whether the model is generally correct.

## 6. Information conditions and prompts

A1 — prior/evidence text. The prompt contains the public map, candidate-rule menu, prior, observed evidence, sign-side facts, inspection error, tool costs, remaining budget, horizon, feasible actions, commit steps, success reward, hazard penalty, and step cost. It does not contain the exact posterior.

A2 — posterior text. The same public facts plus the exact public posterior. This is the text comparison for the visual posterior.

A3 — posterior RGB card. The model receives the actual 512×320 RGB card. The card is the sole source of the map, candidate-rule menu, prior/updated-belief bars, observed evidence, costs, budget, and inspection error. The text supplies only payoff constants, commit steps, feasible actions, action semantics, and the output contract. It does not contain the numerical posterior or the evidence table.

All arms use the same direct-action contract:

Return ONLY one compact JSON object with exactly one key, selected_action, whose value is one feasible action. No reasoning, markdown, code fence, extra key, declaration, or abstention.

The action menu is limited to the feasible actions for that context. COMMIT selects the maximum-belief candidate rule's goal color. PROBE_SIGN inspects the public signpost clause. PROBE_BLUE inspects the blue-cell role under the displayed inspection error.

The prompt renderer, parser, scorer, and image renderer are frozen before the first model request. No prompt improvements, parser loosening, threshold changes, or action-menu changes may occur after that point. The frozen manifest records the prompt hash for every request.

## 7. Parser and scoring

Parser: exact direct JSON only. It accepts exactly the selected_action key and validates against the request's feasible menu. Any missing key, extra key, malformed JSON, code-fence wrapping after the parser's fixed cleanup, or infeasible action becomes a deterministic COMMIT fallback, flagged as fallback and invalid. Fallbacks are never counted as valid optimal decisions.

Primary denominators:

- Valid action rate: valid non-fallback responses / all attempted requests in the completed arm.
- Optimal-action accuracy, valid-only: correct valid decisions / valid non-fallback decisions.
- Optimal-action accuracy, all-attempt: correct valid decisions / all completed attempts; fallback is failure.
- Mean regret, valid-only and all-attempt; fallback regret and fallback count are reported separately.
- Over-probe and under-probe rates among valid responses only.
- Tool-selection accuracy among valid PROBE responses only.
- Action counts and distinct valid actions.
- Input/output token totals and means; latency; GPU time; CUDA peak memory.

Baselines are recomputed on the same 12 contexts: always_commit, always_probe, random_valid (fixed seed 20261126), information_gain, and exact. The exact reference is the expected-control-value reference, not a model.

## 8. Budget, order, and stopping

Planned order: A1 (12), A2 (12), A3 (12), nuisance (6), diagnostic (6). Maximum 48 model requests. One attempt per request, no automatic retry, no reuse of a prior cache entry. Every attempt is counted before the call; a failure counts. The runner refuses to resume a non-empty run directory.

Hard gates, evaluated after each completed direct arm:

- At least 11 of 12 valid non-fallback responses.
- At least two distinct valid actions across the 12 base contexts, i.e. not a constant-action policy.
- For A2 and A3, at least one correct COMMIT and at least one correct PROBE among valid responses.
- A3 technical precondition: all 12 A3 requests dispatched, every A3 request carried an actual image tensor, and at least one valid parsed response. Controls run only if this passes.
- Request cap: no more than 48; no budget violation.
- Zero leakage: no evaluator-label term in prompts or public payloads.
- Frozen checkpoint, request manifest, prompts, images, labels, parser, and scorer verified.

If A1 fails a gate, A2 and A3 are not run. If A2 fails, A3 is not run. If A3 fails, controls are not run. If a model call raises, the failed request is recorded, the run stops, and no retry occurs. A partial run is a result about the smoke, not permission to continue.

## 9. Preflight and freeze

Before any model generation:

1. Freeze the 48-request manifest, prompts, public payloads, 24 images, reference labels, analysis plan, checkpoint manifest, parser/scorer versions, and source hashes.
2. Run a complete stub-backed dry run of all 48 requests with zero model calls. The stub uses the same prompts and actual image path.
3. Prove the actual processor receives A1/A2 as text-only and A3/nuisance/diagnostic as image tensors, with distinct base/nuisance/diagnostic pixel tensors.
4. Scan prompts and public payloads for evaluator labels; require zero hits. A1 must have no posterior; A2 must have the posterior; A3 text must not contain the numerical posterior.
5. Verify the checkpoint manifest, request counts, image dimensions, base/nuisance pixel distinctness, nuisance fact equality, diagnostic optimal-set changes, and all file checksums.
6. Record a freeze hash and refuse execution if any frozen source, protocol, manifest, prompt, image, label, parser, scorer, or analysis-plan hash changes.

The stub dry run is not VLM evidence. Its 48 responses are synthetic and cannot be promoted.

## 10. Artifacts and reproducibility

Frozen artifacts: results/vlm_decision_smoke/frozen/ (request_manifest.json, prompts.jsonl, public_payloads.jsonl, images/, images_manifest.jsonl, reference_labels.json, analysis_plan.json, checkpoint_manifest.json, freeze.json, freeze.sha256).

Preflight: results/vlm_decision_smoke/preflight/.

Actual run: results/vlm_decision_smoke/real/ with requests.jsonl, responses.jsonl, parsed.jsonl, cache.jsonl, execution_state.json, model_load.json, arm_summaries.json, request_accounting.json, summary.json, checksums.json. Raw responses, image digests, vision-tensor digests, parsed outputs, scores, runtime metadata, and checksums are preserved. Failure artifacts are preserved if the run stops.

No smoke artifact is committed unless explicitly intended; results/ is generated. Documentation may summarize aggregates and exact request counts.

## 11. Reporting and limitations

The report will state the exact number of model calls and GPU time, including failures and retries (retries are zero by design). It will separate valid from fallback decisions and report baselines on the same subset. It will report all six direct-arm validity gates and every control pair.

Limitations to state explicitly:

- Twelve contexts are a development feasibility sample, not a benchmark estimate.
- A1/A2/A3 compare information conditions; they do not isolate every source of model error.
- RGB-versus-text differences can reflect OCR, chart comprehension, or prompt format rather than better decision-making.
- Nuisance invariance can reflect insensitivity to pixels as easily as robustness.
- Diagnostic action changes are descriptive; four of six have a probe alternative and two are constrained.
- The small free-VRAM margin may cause a technical stop.
- No result here authorizes the full pilot, the 118-context continuation, the reserved maps, confirmation, or v0.6.

Any larger stage requires a separate human decision. No automatic merge or continuation.

## 12. Freeze declaration

Preparation must verify the full current CPU suite (expected 488 tests), the six prior-phase documents remain intact, this protocol is present and hashed, the stub dry run is complete, leakage is zero, the actual image path is verified, and the checkpoint/manifest hashes match. The results, failure-analysis, and decision-packet documents are written after the run and do not alter this protocol. Only then may the first model request occur. Once it occurs, this protocol is historical and may not be revised.
