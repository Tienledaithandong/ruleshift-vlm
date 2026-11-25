# Grid chunk 0 — behavioural analysis

> **Scaffold, not a scientific result.** This is the 55-episode dry chunk for map 0.
> It is useful for debugging the harness and the policy, not for a benchmark claim.

Status: Gate B passes the current technical checks, but the policy fails every
semantic check that matters. Do not launch chunk 1, a weekly slice, or the full
grid on the basis of this run.

## 1. Scope and data

- 55 episodes, one per scenario: `m0-s0..s4` × 11 scenario types.
- 3,086 public decision rows; 2,143 unique model calls and 943 cache hits.
- All 55 start at `(1, 5)`; the image and public observation at step 0 are
  identical across every scenario (the map is fixed, the rule is hidden).
- The model is `Qwen/Qwen2.5-VL-3B-Instruct` at a pinned revision, and every
  output was schema-valid with zero parser fallbacks.

Artifacts:

- `results/grid_chunks/grid_real_decisions.m000.jsonl`
- `results/grid_chunks/grid_real_evaluator_only.m000.jsonl`
- `results/grid_chunks/response_cache.m000.jsonl`
- `results/grid_chunks/baselines_m000.jsonl` (CPU reference; ignored by Git)

## 2. What the model actually did

### 2.1 The belief vector never changed

All **2,143 unique** cached responses contain exactly the same six-number
vector:

`base 0.17, remap_swap 0.17, remap_rotate 0.17, slippery 0.17,
hazard_goal_swap 0.16, sign_composition 0.16`

The cache keys are unique (2,143 distinct fingerprints; 2,125 distinct history
hashes and 671 distinct image hashes), so this is not a cache collision. The
model emitted its step-0 prior unchanged at every step, including after every
move and every `blocked` outcome. It explicitly ignored the prompt rule:
"Once ANY history exists ... update the probabilities ... do not return the
same uniform vector unchanged."

Consequently `final_rule_estimate` is `base` in **55/55** episodes. The true
rule is `base` in only 10/55, so the estimate is a constant, not an inference.
Its 10/55 agreement (18.2%) is just the prevalence of `base`, close to the 16.7%
chance rate for six rules.

### 2.2 The action policy is a fixed script, not map-conditioned control

All 55 episodes start identically and all 55 choose `LEFT` at steps 0, 1 and 2.
At step 3 the policy switches to `DOWN` in 45 episodes and stays `LEFT` in 10.
Across all 3,086 decisions the actions are:

| Action | Count | Share |
|---|---:|---:|
| `DOWN` | 2,586 | 83.8% |
| `LEFT` | 486 | 15.7% |
| `UP` | 14 | 0.5% |
| `RIGHT` | **0** | 0% |
| `INSPECT` / `READ_SIGN` / `INTERACT` | **0** | 0% |

`RIGHT` is never emitted, although every goal marker is to the right of the
start (`red/blue` at x=5, `green` at x=2; start x=1). No tool is ever used,
even where `READ_SIGN` is available (103 decisions). The model therefore never
collects the evidence that could distinguish the six rules.

At the start cell `(1,5)`, `LEFT` and `DOWN` are walls for every identity-move
rule (`base`, `slippery`, `hazard_goal_swap`, `sign_composition`). The fixed
`LEFT` prefix can only move under `remap_swap` (LEFT→RIGHT) or `remap_rotate`
(LEFT→UP), or by a slip outcome. The model is not choosing a move that is useful
for the visible map; it is emitting an action template.

### 2.3 The policy locks into blocked actions

There are 2,659 blocked transitions in the chunk. In **2,589 of them (97.4%)**
the next action is the same blocked action again, directly contradicting the
prompt's "if the last entry ends with blocked ... choose a different legal
action now" rule.

- All 49 failed episodes run the full 60 steps.
- Mean blocked steps per failed episode: **51.96** (success mean: 18.83).
- 33/55 episodes have at least 50 blocked steps; 48/55 have at least 40.
- Mean distinct cells visited: 4.64. 27/55 visit at most four cells; 9/55 visit
  at most two.
- Three episodes spend all 60 steps at one cell.

The environment is not mislabelling these as moves: the public history says
`blocked`, and the runner records the position change. The policy simply
repeats the action anyway. This is the dominant failure mode, not a rare edge
case.

### 2.4 The six successes are rule-flip or slip luck

| True rule | VLM success | Successful paths |
|---|---:|---|
| `base` | 0/10 | — |
| `remap_swap` | 4/10 | 7–37 steps, almost all `LEFT` |
| `remap_rotate` | 0/5 | — |
| `slippery` | 0/10 | — |
| `hazard_goal_swap` | 2/10 | 39–46 steps, `LEFT`×3 then `DOWN` |
| `sign_composition` | 0/10 | — |

The `remap_swap` successes work because the hidden rule turns the fixed `LEFT`
into `RIGHT`, moving the agent along the bottom toward the red tile. The
`hazard_goal_swap` successes work because repeated `DOWN` at the bottom wall is
occasionally converted into a sideways slip, eventually drifting to the blue
tile. In both cases the action sequence is essentially constant: the six
successful episodes have 0, 0, 1, 1, 1 and 2 action changes. There is no
evidence of closed-loop navigation or rule inference.

## 3. CPU baseline comparison

On the same 55 scenarios and config, the public CPU baselines in
`scripts/export_grid_baselines.py` produce:

| Method | Success | Wilson 95% CI |
|---|---:|---:|
| `no_adaptation` | 38/55 (69.1%) | 56.0–79.7% |
| `public_advisor` | 38/55 (69.1%) | 56.0–79.7% |
| `random_probe` | 38/55 (69.1%) | 56.0–79.7% |
| `info_gain` | 25/55 (45.5%) | 33.0–58.5% |
| **grid VLM** | **6/55 (10.9%)** | **5.1–21.8%** |

Per true rule, `no_adaptation` gets 10/10 on `base`, 10/10 on `slippery`, 5/10
on `hazard_goal_swap`, 5/10 on `sign_composition`, 5/10 on `remap_swap`, and
3/5 on `remap_rotate`; the VLM gets 0/10, 0/10, 2/10, 0/10, 4/10 and 0/5
respectively. The comparison is not a paired scientific test (the CPU policies
receive the public map object, while the VLM receives wall geometry through
pixels), but the base-rule result is decisive as an execution floor: a simple
planner solves every base episode, while the VLM solves none.

The identical 38/55 for the first three methods is expected on this map: they
follow the same base plan, and only `random_probe` differs when a public
discrepancy triggers a tool probe; that mostly does not happen before the goal.
`info_gain` spends steps probing and does worse here.

## 4. Diagnosis

**Confirmed:**

1. The model does not update beliefs from public history or movement outcomes.
2. The action policy collapses to a fixed `LEFT` prefix and then a `DOWN`
   loop; it repeats blocked actions and never uses `RIGHT` or a tool.
3. The observed successes are explained by remapped movement or random slip,
   not by goal-directed planning.
4. This is not a parser or fallback problem: JSON is valid, probabilities sum
   to 1.0, and the environment labels blocked transitions correctly.

**Likely contributing causes:**

1. **The textual observation has no wall geometry.** `public_text` lists
   markers, key, door and signpost, but not walls; the action mask still
   includes all four movement directions at a wall. If the 3B model cannot
   resolve the 224×224 image, it has no textual way to know `LEFT`/`DOWN` are
   blocked. This is consistent with every observed choice, though the log
   alone cannot prove image non-use.
2. **The prompt is a long multi-constraint instruction.** It repeats "LEFT"
   several times (including "Never default to LEFT" and "Do not always choose
   LEFT"), asks for six probabilities plus arithmetic, a legal action and a
   tool target, and forbids the blocked repeat. The model satisfies the JSON
   schema but ignores the semantic constraints. A 3B model may be latching
   onto the most repeated action token rather than parsing the map.
3. **The action mask is not a mobility mask.** All four movement actions are
   "legal" even when they are walls, so the mask provides no grid guidance.

## 5. Why Gate B passed

Gate B checks JSON validity, zero fallbacks, at least two distinct actions and
some movement. This run satisfies those checks while being semantically
degenerate: three actions, no `RIGHT`, no tools, one belief vector, and 97.4%
immediate repetition of blocked actions. The gate is necessary but not
sufficient.

A semantic gate should reject a chunk when any of these holds:

- the belief vector is identical at every step (or at least the final estimate
  never varies across true rules);
- the same action is repeated after a `blocked` outcome above a small
  threshold (for example >5%);
- the first action is identical across all scenarios with identical start
  observations (or the model never emits an action that can move toward the
  visible goal);
- no tool is ever used on ambiguous-rule episodes;
- a trivial public baseline solves base-rule episodes while the VLM solves
  none.

These are debugging gates for a scaffold, not preregistered scientific
criteria. They should be agreed before the next smoke.

## 6. What to do next

1. **Stop GPU work.** Do not launch chunk 1 or the full grid; this policy
   cannot produce a meaningful adaptation measurement.
2. **Fix the observation/action contract first.** Expose public wall or
   neighbor status in the text observation (or make the movement mask exclude
   visibly blocked directions), and add a deterministic guard that forbids the
   last blocked action. Keep these changes explicit and versioned.
3. **Simplify the prompt.** Remove repeated `LEFT` cues, separate action choice
   from belief reporting, and ask for integer candidate weights rather than
   six-decimal arithmetic on every step. Re-run a 2–3 episode smoke only.
4. **Add the semantic gates above** to `run_grid_real_pilot.py` and tests, so a
   later chunk cannot pass on JSON validity alone.
5. **Re-cost the design.** The measured chunk is 6.39 GPU-hours; the full
   5,500-episode grid remains about 640 GPU-hours (~24 weeks at 27 h/week).
   Even after the policy is fixed, the design needs resizing or faster
   inference; a 10-map dev slice would be ~64 GPU-hours.

## 7. Implementation status of the repair

The observation/action contract and semantic gate have been repaired; the full
test suite is green. The prompt repair did not work: no two-episode smoke
passed the strict full-menu contract. These changes are not evidence that the
policy now adapts, and they do not authorize any run beyond a 2-episode smoke.

- `build_public_text()` now emits `adjacent=[UP=...,DOWN=...,LEFT=...,RIGHT=...]`
  and the complete public `walls=[...]` set, both read from `PublicGridMap`.
  These are public map data; no hidden rule or evaluator value is exposed.
- `grid_v7.txt` gives the wall/adjacent contract, makes the latest blocked
  outcome operational, repeats the full-menu prior only at step 0, and states
  the strict JSON schema.
- `GridVLMPolicy` applies a deterministic public guard when the chosen action
  equals the action whose latest public outcome was `blocked`. It prefers a
  visible-free neighbour and a public key/red-goal target. The guard is
  recorded as `last_guard_applied` and in public rows as `guard_applied`.
- The runner copies `hypothesis_probabilities` into public rows for auditing.
- `semantic_gate()` rejects constant beliefs, blocked-action repeat rates above
  5%, zero tool use when a tool is available, and zero success on final
  base-rule episodes. `main()` requires both the technical and semantic gates.
- Only 2-episode smokes are permitted. Smoke-13 (v6) omitted `base`
  (10.8% valid); smoke-14 (v7) emitted all six ids but many summed to 1.50
  (75.8% valid); smoke-15 (v8) added an arithmetic example and fell to 26.7%;
  smoke-16 (v9) combined the all-ids and v5 arithmetic text and reached only
  22.5%. All four had parser fallbacks, so Gate A failed. None is evidence. The
  55-episode chunk and the 5,500-episode grid remain blocked.
- The active prompt is back at `grid_v7`, the best observed full-menu rate.
  v8 and v9 are retained as rejected experiment records. The binding blocker
  is the 3B model’s probability arithmetic under the strict six-key contract;
  the next step is an interface decision, not another prompt-only smoke.

## 8. Reproduction notes

- Gate B technical counts and action distributions are computed directly from
  the two JSONL files above.
- The baseline file was produced with `export_grid_baselines` on the 55
  `m0-*` scenarios from `build_grid_scenarios(100, 5, config)`.
- The report in `reports/grid_chunks/grid_real_pilot.m000.md` is the kernel's
  own scaffold report; this analysis does not promote it to a result.
