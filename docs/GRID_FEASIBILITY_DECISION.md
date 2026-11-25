# RuleShift-Grid feasibility decision

**Status: CLOSED for the frozen 3B grid.** This record closes the 100-map x
5-seed program on `Qwen/Qwen2.5-VL-3B-Instruct`. It does not claim a
confirmatory result, and it does not authorize any Kaggle chunk.

## Decision

Do not run chunk 1, a weekly slice, or the 5,500-episode grid. The policy
cannot satisfy the strict full-menu contract reliably, and the failures that
remain after schema repair are behavioural: the model does not revise its
belief from public history, does not call tools, and does not solve the
base-rule episode.

## Evidence

| Run | Prompt | Strict full-menu valid | Parser fallbacks | Blocked repeats | Tool decisions | Base success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke-13 | v6 | 10.8% (13/120) | 107 | 0/27 | 0 | 0/1 |
| smoke-14 | v7 | 75.8% (91/120) | 29 | 0/17 | 0 | 0/1 |
| smoke-15 | v8 | 26.7% (32/120) | 88 | 0/30 | 0 | 0/1 |
| smoke-16 | v9 | 22.5% (27/120) | 93 | 0/25 | 0 | 0/1 |

Every run is a two-episode scaffold, not a scientific result. The blocked
counts use the recorded public outcome, not a same-cell inference; the guard
works. The technical gate failed every time because at least one response
fell back.

The schema failure is arithmetic, not omitted ids. v7 emitted all six ids but
often used `0.33,0.33,0.17,0.17,0.17,0.17` (sum 1.34); v8 and v9 repeated the
same pattern. A normalization audit of the cached responses gives only 7
unique vectors in v7 (81/119 are the same vector) and 5 in v9 (88/119 are the
same vector). Normalizing would make the schema check green without making
the beliefs responsive.

The 55-episode dry chunk (Gate B) makes the same point at larger n: 2,143
unique model calls all return the unchanged step-0 prior, 0/10 base-rule
episodes succeed, and the policy never uses a tool. See
`docs/GRID_CHUNK0_ANALYSIS.md`.

## What was repaired

- Public wall and adjacent-direction text in the observation.
- A deterministic guard against repeating the latest blocked action.
- A semantic gate that uses the recorded outcome and requires belief change,
  tool use and base-rule success.
- Prompt versions through `grid_v7` (the best observed 75.8% full-menu rate).
  v8 and v9 are retained as rejected experiment records.

## What was not repaired

- The 3B model’s probability arithmetic under the strict six-key contract.
- Belief revision from public history.
- Tool use (`INSPECT` / `READ_SIGN`).
- Base-rule success.

## Disposition

- Smoke artifacts and the completed map-0 chunk are instrumentation only; they
  must not be promoted to a result or merged as confirmatory data.
- No new Kaggle chunk is scheduled. `configs/preregistered_test.yaml` remains
  `DRAFT_NOT_FROZEN`; the preregistration is not READY.
- The branch keeps the repaired observation/guard code and the failing prompt
  history so a later attempt starts from evidence, not a retry loop.

## Reopen criteria

Reopen only if all of these hold:

1. A new interface or model is chosen explicitly (for example normalized
   weights, constrained decoding, a two-stage action/belief readout, or a
   larger frozen model); a prompt-only retry is not sufficient.
2. One two-episode smoke passes **both** the technical and semantic gates,
   with zero parser fallbacks, belief change, at least one tool decision, and
   base-rule success.
3. A 10-map slice is projected within the remaining GPU quota, and a human
   marks a new preregistration READY with a fresh freeze manifest.

Until then, the correct output is this feasibility record, not another run.
