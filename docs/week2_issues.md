# Week 2 Issue Backlog (ordered)

These are the Week 2 implementation issues, in dependency order. Each has an
acceptance criterion that is a test, not a narrative claim.

**Status (end of Week 2): implemented.** W2-01 through W2-06 and W2-08 through
W2-10 are complete and covered by the passing suite. W2-07 delivered the
privileged trace adapter and import isolation only — the Bayes filter and exact
solver are deliberately Week 3 work. The full suite runs clean with
`python -m pytest -q`; `python scripts/smoke_env.py --seed 42 --family gate`
prints a public trace and a separately labelled evaluator trace.

---

### W2-01 — Python package, CI and environment checker scaffold

- Add `gymnasium`, `numpy`, `Pillow`, `pyyaml` to a new optional `env` extra;
  keep the core importable on CPU without optional deps.
- Create `src/ruleshift/envs/` with `__init__.py` exporting nothing to agents.
- Add `scripts/smoke_env.py --seed 42 --family gate`.
- **Acceptance:** `pytest -q` still green; `gymnasium.utils.env_checker.check_env`
  runs on a minimal stub env; CI workflow passes.

### W2-02 — Semantic RNG (`envs/events.py`)

- Implement keyed streams `(scenario_id, event_type, primitive_step, attempt_index)`
  hashed with `(master_seed, key)`.
- Event types: `SHIFT_TRIGGER`, `SHIFT_TARGET`, `PROBE_FEEDBACK`,
  `HAZARD_OUTCOME`, `OBSERVATION_NOISE`.
- **Acceptance:** `tests/test_rng.py` — repeated streams identical; independent
  semantic keys do not alias; a policy consuming extra draws does not change
  another event's stream.

### W2-03 — Public view and evaluator state adapters (`envs/observations.py`)

- Implement `public_view()` and `evaluator_state()` with disjoint key sets.
- Public payload uses exactly `PUBLIC_OBSERVATION_FIELDS`.
- **Acceptance:** `validate_public_observation(public_view()) == []`;
  `set(public_view()) & PRIVILEGED_FIELDS == set()`; no privileged key in any
  prompt serialization candidate.

### W2-04 — Family A core environment (`envs/gate_lock.py`)

- Gymnasium `reset/step/render`, 5×5–7×7 layout, default rule `red`, repeated
  deliveries, accepted `+10`, per-step `-0.05`, lockout `k ∈ {0,3,8}` with
  cost-once semantics, no hidden penalty channel.
- At most one shift; no-shift realisations; hazard over the full horizon, no
  narrow window, no phase feature.
- **Acceptance:** master-plan critical tests 1, 2, 4, 7, 14; env checker green.

### W2-05 — Macro executor (`envs/macro.py`)

- Expand public macro ids to primitive steps using only public geometry.
- `COMMIT_*` = public distance to gate + 1; `PROBE_*` = out + test + back;
  `WAIT` = 1; `GO_TO_STATION` = public distance.
- Handle in-macro shifts: commit/probe outcome resolved by the mechanic active
  at the relevant primitive step.
- **Acceptance:** tests 5 and 13; `MAX_ENV_STEPS` counts primitives; no free
  travel, test, lockout tick or reward.

### W2-06 — Probe station and feedback noise

- Diagnostic station uses the same active mechanic as the gate; symmetric flip
  `eps ∈ {0, 0.1, 0.25}` via semantic RNG; no second hidden rule.
- **Acceptance:** test 4 (same mechanic), noise rates reproduce empirical flip
  frequency within tolerance over many seeded draws.

### W2-07 — Evaluator/oracle isolation (time-boxed interface only)

- Define (do not yet solve) the `evaluator` package skeleton and the separate
  `evaluator_only.jsonl` adapter; ensure no import path from `agents` reaches it.
- **Acceptance:** `tests/test_agent_permissions.py` still green after the new
  packages exist; a deliberate bad import is caught (scanner self-test).

### W2-08 — Determinism, cost invariants and smoke tests

- `tests/test_env.py`, `tests/test_macro_costs.py`, `scripts/smoke_env.py`.
- **Acceptance:** tests 1, 3, 5, 11, 12; at least one dev config where probe is
  predicted advantageous and one where it is predicted disadvantageous
  (verified by a hand cost calculation first, exact solver in Week 3).

### W2-09 — Configuration plumbing

- Load `configs/dev.yaml` into a validated config object; keep `test_locked`
  scenarios absent until Week 6.
- **Acceptance:** determinism across process restarts; `dev.yaml` contains no
  frozen test scenario and no shift-window shortcut.

### W2-10 — Gate W2 review

- Run the full critical test list (§7 of the master plan) against the built env.
- **Pivot if** any privileged field leaks, any macro miscounts steps, or both
  probe and task actions are always optimal. Fix the problem definition; do
  **not** add a VLM in response.

---

## Deferred (not Week 2)

- Family B `envs/bridge_hazard.py` with genuinely changed transition law (W8).
- Bayes filter, exact solver, `Delta_probe` (W3).
- Renderer RGB array and cache-key hashing (W4).
- VLM backend, structured selector, DIVA-style baseline (W4–W6).
