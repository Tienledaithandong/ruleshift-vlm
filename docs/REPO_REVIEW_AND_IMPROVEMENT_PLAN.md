# Repository Review and Improvement Plan

**Review date:** 2026-11-22

**Reviewed branch/commit:** `arena/ce7ceae6-ruleshift-vlm` / `0c6c6240c2e7fbeb66d80d0e5676667484734de6`

**Review type:** implementation, experiment-integrity, and reproducibility review. No runtime code was changed as part of this review; this file records findings and proposed work.

---

## Update 2026-11-24 (current branch `feat/ruleshift-grid`)

This document is retained as the historical 2026-11-22 review of the original
harness. Its findings were **not re-audited line-by-line** in the 2026-11-24
pass, so the Family A/B items below remain open until explicitly verified. For
the additive RuleShift-Grid work, the binding sources are the approved spec
`docs/superpowers/specs/2026-11-23-ruleshift-grid-design.md`, the implementation
plan `docs/superpowers/plans/2026-11-23-ruleshift-grid.md` (all boxes ticked),
and PR #3.

Current status on this branch:

- 385 tests pass; `preflight.py` still prints NO-GO, as expected while
  `configs/preregistered_test.yaml` remains `DRAFT_NOT_FROZEN`.
- RuleShift-Grid adds Tier A (stochastic grid environment, partial tools,
  change-point belief, public A* reference, matched-budget baselines) and a
  text-only Tier B VLM scaffold without modifying Family A/B code.
- Post-review fixes on this branch: the generator's solvability gate now blocks
  the signpost, the structural audit re-checks solvability independently of the
  generator gate, and `information_gain_per_tool_call` now uses tool-only
  information via the shared `effective_map` blocking helper, and the
  change-point belief/expected-IG likelihoods use the same geometry. A 2026-11-24
  reviewer pass found no Critical issues; its one Important finding (raw-map
  belief likelihoods near blockers) is fixed in this branch, along with three
  minor schema/ID issues. The remaining minor (`hazard_safety_threshold` is not
  threaded from config) is deferred. Tier B was subsequently wired to send the
  public image with a single merged RGB+text prompt (`grid_v2`) at a 2048-token
  budget; a real Kaggle smoke remains a remote gate.
- The required 100-map x 5-seed scientific run has **not** been executed with
  the frozen VLM. All smoke/audit output remains a pipeline check, never a
  finding. No local GPU job has been run.

The "193 tests" and "204 tests" figures below are historical counts; README now
reports 385.

---

## Executive summary

The repository has a useful CPU research harness, explicit public/privileged data separation, and broad tests. The local validation is green. It is **not yet ready for claims about a visual agent or the proposed value-aware probing method**: the model currently receives text rather than RGB or the full public history, and the selector documents a one-step limitation that prevents it from valuing information across repeated decisions in ordinary horizons. There are also evaluation-budget, action-validation, and route-cost mismatches that should be resolved before a real closed-loop comparison.

Treat all generated stub reports as pipeline checks only. The test configuration remains `DRAFT_NOT_FROZEN`; this review does not recommend freezing it.

## Validation performed

- `.venv/bin/python -m pytest -q` — **204 tests passed**.
- `.venv/bin/python scripts/reproduce_all.py` — passed; artifacts and reported numbers are explicitly synthetic.
- `--help` smoke test — all 15 `scripts/*.py` entry points passed.
- Agent permission AST scan — zero violations.
- `scripts/preflight.py` — **NO-GO**, as expected in this sandbox: no local model checkpoint or `torch`/`transformers`, and the test config is draft.
- Whitespace check on this plan — clean; `git diff --check` was clean on tracked files before this untracked review document was added.

Passing tests establish the tested contracts; they do not exercise a real VLM checkpoint or prove the experiment is scientifically ready.

## Findings and priorities

Priority definitions: **P1** = resolve before a real validation/closed-loop run or any method claim; **P2** = resolve before broadening the benchmark or relying on the workflow for external reproduction.

### P1 — The model input does not include RGB or the full action/outcome history

`PublicObservation` contains `public_text` and hashes, not an image or serialized history (`src/ruleshift/public_api/interface.py:30-45`). `render_outcomes_prompt` fills both `public_history` and `public_text` from the same current text (`src/ruleshift/agents/structured_vlm.py:20-40`); `image_hash` is set from `public_obs_hash` (`:59-67`). The Transformers backend sends only a text message (`src/ruleshift/agents/vlm_backend.py:220-230`). Consequently, past evidence is not available to a stateless model call except for what remains in the current `last=...` summary, and no pixel reaches the VLM. The Colab setup correctly calls this text-only P1 mode, but the README's broader RGB/history description can be read as current behavior.

**Plan:** add an explicit public observation payload for RGB and the complete sanitized action/outcome history; pass the image through the selected processor; hash the actual image bytes and history in the request/cache key. Add tests proving that changed pixels or prior public outcomes change the model input and fingerprint, while privileged labels never enter either. Until then, label all real pilots as text-only diagnostics, not visual-agent evaluation.

### P1 — M1's current value estimate cannot value information over repeated decisions

`q_hat_from_prediction` explicitly documents that it uses a one-step continuation and that the continuation cancels between probe and commit whenever both fit in the horizon (`src/ruleshift/agents/value_selector.py:133-154`). `tests/test_value_selector.py` intentionally pins this behavior. The estimator therefore does not value a probe for improving several later deliveries, and does not model evidence becoming stale after a shift during probe travel. The VLM's `candidate_probabilities` field is also not used by `_StubMethod`; its belief is initialized uniformly and updated with a hand-written consistency heuristic instead (`src/ruleshift/evaluator/closed_loop.py:96-160`).

**Plan:** specify the decision horizon and a public-only belief/value approximation that covers repeated task opportunities, macro duration, probe noise, and the public shift prior. Decide whether the candidate probabilities are inputs to that method or should be removed from the contract. Compare the approximation against the evaluator's exact microtask solver on enumerated small horizons, and add tests where a probe should and should not win for information value—not only because a lockout branch is truncated. Keep the exact evaluator isolated from agent code.

### P1 — The closed-loop runner may stop well before the declared primitive-step budget

`run_episode` is bounded by `for decision in range(MAX_DECISIONS)` with `MAX_DECISIONS = 12` (`src/ruleshift/evaluator/closed_loop.py:35,267-318`). Once the six-call cap is spent, it emits `WAIT`; those one-step fallbacks can consume only a small fraction of the configured 60-step episode before the decision loop ends. A call-cap-zero sanity run ends at 12 primitive steps, not 60. Current tests assert only `steps <= max_env_steps`, so this mismatch passes.

**Plan:** make the stopping rule explicit and preregistered. Prefer continuing with the documented fallback until environment truncation/termination; if a separate macro-decision limit is required, configure and report it as a second shared budget. Add tests using an always-WAIT policy and call-cap exhaustion to assert the intended stopping condition and ensure methods receive comparable primitive-step opportunity.

### P1 — Value estimates use start-position route costs when macro durations depend on the current position

`RuleShiftEnv.public_view()` publishes `probe_detour_steps` and start-to-gate travel (`src/ruleshift/envs/gate_lock.py:160-174`). But `MacroExecutor.plan()` routes from the agent's current position (`src/ruleshift/envs/macro.py:69-89`), and a failed commit can leave the agent at the gate. `public_macro_durations()` always estimates a probe as `2 * detour + 1` and a commit as `commit_travel + 1` (`src/ruleshift/agents/value_selector.py:85-98`). Those formulas are only exact from the start tile, so selector costs can diverge from actual primitive costs in later decisions.

**Plan:** publish current-state public cost-to-go values for each macro, or expose enough public geometry/position to compute them with one shared pure route-cost function. Cross-check estimates against the executor for the start, gate, and station positions, including interrupted/truncated macros. Keep cost values public and test the actual reward sum against the estimate.

### P1 — Output validation checks the selected action, but not every action the selector may choose

`parse_structured_output()` checks that `selected_action` is in the current menu, but does not check each `action_predictions[*].action_id` or `estimated_future_best_action` against that menu (`src/ruleshift/agents/vlm_backend.py:248-270`). The schema allows the union of both families' macro IDs (`src/ruleshift/public_api/schemas.py:54-64`). A selector can therefore choose a different-family action from a prediction; the Family A environment then silently maps it to `WAIT`, while the runner's union-based check does not count it as invalid (`src/ruleshift/evaluator/closed_loop.py:282-285`).

**Plan:** validate all output action IDs against the observation's `available_actions`, and validate again at the environment boundary. Either reject the payload to the deterministic fallback or log/count the invalid prediction explicitly. Add a regression test where `selected_action` is valid but a higher-valued prediction is not in the menu.

### P2 — Cache fingerprints omit model revision; reproducibility manifests include generated trees

`ModelRequest.fingerprint()` includes `model_id` but no model revision (`src/ruleshift/agents/vlm_backend.py:28-54`), even though the backend has a separate `revision` setting and the permission contract says cache identity includes revision. Reusing the same cache after changing a checkpoint revision can therefore return old generations. Separately, `check_reproducibility.collect_hashes()` recursively hashes the workspace while excluding only a short list of directories (`scripts/check_reproducibility.py:25-40`). In this workspace, the generated manifest contained **2,846** entries, including `.venv/` and `kaggle_stage/`, with duplicate prompt hashes from the staged copy.

**Plan:** include immutable model revision and generation/backend configuration in request identity and stored cache metadata. Build source hashes from tracked source/config files (or explicitly exclude ignored/generated directories), and test that an ignored staging directory cannot change the manifest. Record the exact dependency and runtime versions used for a real run.

### P2 — Family B is an environment/analytic transfer scaffold, not yet a VLM benchmark

The Family B environment is tested, but the full closed-loop/held-out runner and structured selector are Family A-oriented. `run_main_eval.py` writes an analytical Family B reference rather than Family B agent episodes. The structured schema also hard-codes red/blue/green candidate probabilities (`src/ruleshift/public_api/schemas.py:23-25,175-195`), while the Family B environment advertises `stable`/`slippery` regimes. The Family B transfer script itself documents that exact labels cover only a single-decision abstraction.

**Plan:** either label Family B clearly as environment-plus-analytical-reference scaffolding, or implement family-aware schemas, a compatible public-only selector, a shift-aware evaluator reference, and paired closed-loop tests before making cross-family claims.

### P2 — Run status and setup documentation disagree

`README.md` and `docs/RUNBOOK.md` say no real model evaluation has run, while `docs/AGENT_HANDOFF_PROMPT.md` claims a Kaggle pilot completed and reports measurements; `docs/KAGGLE_HANDOFF_PROMPT.md` still says no experiment has run. The current tracked tree contains no raw real-run artifacts to resolve this. Other stale statements include `docs/problem_spec.md` saying the environment is not implemented, `src/ruleshift/evaluator/__init__.py` saying closed-loop metrics are still to come, and the README tree using `project_0/` and reporting 193 tests (204 are collected here).

**Plan:** verify any claimed remote run from its raw manifest/decisions, model revision, kernel URL, and checksums; then make one status page the source of truth and update or archive contradictory handoffs. Do not copy unverified metrics into the README or treat them as validation results.

### P2 — Real-backend readiness is not fully gated or tested

The local preflight reports a missing CUDA device as informational (`src/ruleshift/agents/model_check.py`) and does not receive the configured device/dtype. The Transformers backend maps unknown dtype strings to BF16 and does not accept the configured `local_files_only` policy (`src/ruleshift/agents/vlm_backend.py:202-214`). In this sandbox the real backend could not be executed, so the passing suite covers the stub, not checkpoint loading or GPU compatibility.

**Plan:** make preflight fail when the requested device/dtype is unsupported, verify the exact pinned revision is locally available when downloads are disabled, pass the offline/cache settings through to `from_pretrained`, reject unknown dtypes, and add mocked loader tests plus a small opt-in GPU smoke test. Keep the default CPU test suite independent of model weights.

## Recommended order of work

1. **Resolve status and provenance (P2, before external reporting).** Verify the handoff's remote-run claim against artifacts, then publish one consistent status/source-of-truth record.
2. **Close correctness and budget gaps (P1).** Fix action-menu validation, current-position macro costs, and episode stopping semantics. Add focused regression tests first.
3. **Complete the agent's public input and value path (P1).** Provide RGB and full history; define a multi-step public-only belief/value estimator; validate it against exact microtasks and outcome noise/shift cases.
4. **Make real inference reproducible (P2, before another pilot).** Add revision-aware cache keys, robust local/device preflight, and manifest hashing that excludes generated trees.
5. **Decide Family B scope and update docs/CI (P2).** Either clearly bound claims to Family A, or build a family-aware end-to-end path; refresh status docs, test Python 3.11 and 3.13, and make dependency locking reproducible.
6. **Only then plan validation and freeze.** Run a real text-only or visual pilot with its mode clearly named; review output diversity and failures; tune only on validation; freeze test scenarios only after the preregistered validation gate passes.

## Acceptance gate before a method claim

- A real model request contains the exact permitted public RGB/history (or the result is explicitly limited to text-only input).
- Every compared method obeys the same declared primitive-step and model-call budgets.
- Agent-selected actions are all in the active environment menu, and logged costs match executor costs.
- The value estimator shows correct probe/task choices on independent hand-computed and exact microtask cases with multi-step information value.
- Cache identity includes the pinned model revision and all response-affecting inputs; manifest contents are stable across clean and locally staged workspaces.
- Raw real-run artifacts and source/checkpoint provenance are available, and the documentation consistently labels synthetic vs empirical outputs.
- `configs/preregistered_test.yaml` remains draft until the validation and freeze criteria in `docs/RUNBOOK.md` are met.
