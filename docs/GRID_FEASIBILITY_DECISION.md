# RuleShift-Grid feasibility decision

**Status: CLOSED for the frozen 3B grid.** This record closes the 100-map x
5-seed program on `Qwen/Qwen2.5-VL-3B-Instruct`. It does not claim a
confirmatory result, and it does not authorize any Kaggle chunk.
The [negative feasibility note](GRID_NEGATIVE_FEASIBILITY_NOTE.md) summarizes
the diagnostic evidence and cost.

## Decision

Do not run chunk 1, a weekly slice, or the 5,500-episode grid. The policy
cannot satisfy the strict full-menu contract reliably. Even with valid or
normalized menus, the artifacts show no reliable belief revision from public
history. The policy never calls a tool and fails every observed base-rule
episode.

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

v6 omitted candidate IDs. The v7–v9 cached responses supplied all six IDs, but
invalid menus failed the sum check. The recurring v8 and v9 vector
`0.33,0.33,0.17,0.17,0.17,0.17` sums to 1.34. A normalization audit of the
cached responses gives 7 distinct vectors in v7 (81/119 share one vector),
4 in v8 (82/119), and 5 in v9 (88/119). Normalization repairs the sums of
complete menus while leaving the observed concentration high.

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
3. A resized grid is projected within the remaining GPU quota, and a human
   marks a new preregistration READY with a fresh freeze manifest.

Until then, the correct output is this feasibility record, not another run.
