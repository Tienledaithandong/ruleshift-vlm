# Week 4 — Offline pilot dry run (SYNTHETIC)

**SYNTHETIC DRY RUN. No VLM was run.** The backend used here is a deterministic
stub (`STUB_SYNTHETIC`). These numbers exercise the harness only; they are NOT
evidence about any vision-language model and must not appear in a paper.

Public-text (P1) microtask decisions: 9.
Methods: direct.
Distinct public requests: 9. Model calls: 9.
Exact reference labels use the specified context design and remaining budget.
Decision regret includes the executed WAIT fallback on invalid responses.
Public manifest: `manifest.jsonl` (public fields only).
Privileged labels: `evaluator_only.jsonl` (never joined into model input).

| Diagnostic | Value |
|---|---:|
| JSON valid rate | 1.000 |
| Schema-valid action rate | 1.000 |
| Correct-argmax rate (chosen macro is exact Q* optimal) | 0.444 |
| Invalid JSON / invalid action / WAIT fallbacks | 0 / 0 / 0 |
| Over-probe rate (probed when exact label says do not) | 0.111 |
| Under-probe rate (did not probe when label says probe) | 0.000 |
| Mean decision regret vs exact Q* (covered decisions only) | 5.000 |
| Decisions with an exact label (full macro menu) | 1.000 |
| Mean output tokens | 2.0 |
| Cache hit rate | 0.000 |

## Probe decisions by stratum

Probe rate is `PROBE_*` choices over that stratum's contexts. In the
`delta_probe<0` row it is the over-probe rate; in the `delta_probe>=0` row it
is the complement of under-probing. Regret is averaged over covered decisions.

| Stratum | n | Probes | Probe rate | Correct argmax | Mean regret |
|---|---:|---:|---:|---:|---:|
| all | 9 | 3/9 | 0.333 | 0.444 | 5.000 |
| delta_probe>=0 | 3 | 2/3 | 0.667 | 0.667 | 3.333 |
| delta_probe<0 | 6 | 1/6 | 0.167 | 0.333 | 5.833 |
| detour=0 | 3 | 1/3 | 0.333 | 0.667 | 1.667 |
| detour=2 | 3 | 1/3 | 0.333 | 0.333 | 5.000 |
| detour=5 | 3 | 1/3 | 0.333 | 0.333 | 8.333 |

## Failure taxonomy measured by this harness

1. Invalid JSON / parse failures — counted, deterministic fallback to `WAIT`.
2. Invalid macro ids — counted separately from parse failures.
3. Over-probing — probe chosen with a negative label outside the 1e-9 tie band.
4. Under-probing — non-probe chosen with a positive label outside the tie band.
5. This direct-action pilot records the chosen macro.
6. Cost accounting — tokens, calls, latency and cache hits per decision.

A real frozen-VLM pilot requires an available checkpoint. When one exists,
swap `StubVLMBackend` for `TransformersVLMBackend`; only then do these rows
become empirical observations, and the synthetic label must be removed.
