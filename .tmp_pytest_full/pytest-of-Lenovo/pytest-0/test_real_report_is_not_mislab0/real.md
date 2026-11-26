# Week 7 — Family A closed loop (REAL BACKEND)

**REAL BACKEND: a frozen model produced the model-side decisions.** This is a development-grid run unless it was separately frozen; do not treat it as a held-out or locked evaluation.

All methods use the same public macro menu, the same macro executor and the same primitive-step budget. Costs are charged once in the shared reward stream. The 6-call cap and ``WAIT`` fallback apply to model-side methods; scripted baselines make no model calls.

| Comparison | Scenarios | Mean paired difference | 95% CI |
|---|---:|---:|---|

 episodes.jsonl contains public decision rows only. evaluator_only.jsonl is a separate privileged trace and is never joined into model input.
