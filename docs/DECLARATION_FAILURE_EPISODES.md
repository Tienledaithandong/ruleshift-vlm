# Declaration failure episode index

This appendix indexes every one of the 37 v4 target episodes (33 unresolved, 4 incorrect) and its matched v3 run. The complete per-decision public posterior, action, tool target, tool value, remaining budget, safe-commit eligibility, and declaration event for all 74 traces are in `docs/declaration_failure_traces_v1.jsonl`.

The table is a reading aid, not a substitute for the trace. `Crit` is the number of decisions at which top >= 0.90 and margin >= 0.10; `Elig` is the number at which the frozen policy safe-commit branch returned true. The v4 unresolved cases all have `Elig = 0`.

## Unresolved episodes

| Episode | True | Primary cause | Steps | Crit/Elig | Final top / margin / run | Tool units | v3 status | v3 steps |
|---|---|---|---:|---:|---:|---:|---|---:|
| m0-s3 / slippery | slippery | stability | 6 | 1/0 | 1.0000 / 1.0000 / 1 | 0 | correct_commit | 6 |
| m1-s0 / remap_swap | remap_swap | stability | 2 | 1/0 | 0.9969 / 0.9938 / 1 | 0 | correct_commit | 2 |
| m1-s1 / remap_swap | remap_swap | stability | 2 | 1/0 | 0.9969 / 0.9938 / 1 | 0 | correct_commit | 2 |
| m1-s1 / slippery | slippery | stability | 3 | 1/0 | 1.0000 / 1.0000 / 1 | 1 | correct_commit | 5 |
| m1-s2 / remap_swap | remap_swap | stability | 2 | 1/0 | 0.9969 / 0.9938 / 1 | 0 | correct_commit | 2 |
| m1-s3 / remap_swap | remap_swap | stability | 2 | 1/0 | 0.9969 / 0.9938 / 1 | 0 | correct_commit | 2 |
| m1-s4 / remap_swap | remap_swap | stability | 2 | 1/0 | 0.9969 / 0.9938 / 1 | 0 | correct_commit | 2 |
| m5-s0 / sign_composition | sign_composition | stability | 4 | 1/0 | 0.9744 / 0.9487 / 1 | 3 | correct_commit | 4 |
| m5-s1 / sign_composition | sign_composition | stability | 3 | 1/0 | 0.9744 / 0.9487 / 1 | 3 | correct_commit | 3 |
| m5-s2 / sign_composition | sign_composition | stability | 3 | 1/0 | 0.9744 / 0.9487 / 1 | 3 | correct_commit | 3 |
| m5-s3 / sign_composition | sign_composition | stability | 3 | 1/0 | 0.9744 / 0.9487 / 1 | 3 | correct_commit | 3 |
| m5-s4 / sign_composition | sign_composition | stability | 3 | 1/0 | 0.9744 / 0.9487 / 1 | 3 | correct_commit | 3 |
| m6-s0 / slippery | slippery | stability | 3 | 1/0 | 1.0000 / 1.0000 / 1 | 1 | correct_commit | 12 |
| m8-s1 / slippery | slippery | stability | 4 | 1/0 | 1.0000 / 1.0000 / 1 | 0 | correct_commit | 4 |
| m1-s0 / slippery | slippery | terminal-only | 3 | 0/0 | 1.0000 / 1.0000 / 2 | 0 | unresolved | 3 |
| m3-s0 / remap_rotate | remap_rotate | terminal-only | 2 | 0/0 | 0.9419 / 0.8953 / 1 | 0 | correct_commit | 4 |
| m3-s1 / base | base | terminal-only | 5 | 0/0 | 0.9714 / 0.9430 / 6 | 1 | unresolved | 5 |
| m3-s1 / remap_rotate | remap_rotate | terminal-only | 2 | 0/0 | 0.9419 / 0.8953 / 1 | 0 | correct_commit | 4 |
| m3-s1 / slippery | slippery | terminal-only | 7 | 0/0 | 1.0000 / 1.0000 / 4 | 1 | unresolved | 7 |
| m3-s2 / base | base | terminal-only | 5 | 0/0 | 0.9714 / 0.9430 / 6 | 1 | unresolved | 5 |
| m3-s2 / remap_rotate | remap_rotate | terminal-only | 2 | 0/0 | 0.9419 / 0.8953 / 1 | 0 | correct_commit | 5 |
| m3-s2 / slippery | slippery | terminal-only | 2 | 0/0 | 0.9419 / 0.8953 / 1 | 0 | unresolved | 2 |
| m3-s3 / base | base | terminal-only | 5 | 0/0 | 0.9714 / 0.9430 / 6 | 1 | unresolved | 5 |
| m3-s3 / remap_rotate | remap_rotate | terminal-only | 2 | 0/0 | 0.9419 / 0.8953 / 1 | 0 | correct_commit | 4 |
| m3-s4 / remap_rotate | remap_rotate | terminal-only | 3 | 0/0 | 0.9474 / 0.8947 / 2 | 0 | correct_commit | 4 |
| m1-s1 / remap_rotate | remap_rotate | insufficient | 3 | 0/0 | 0.7330 / 0.5882 / 2 | 0 | correct_commit | 10 |
| m1-s4 / hazard_goal_swap | hazard_goal_swap | insufficient | 2 | 0/0 | 0.5000 / 0.0000 / 2 | 3 | unresolved | 2 |
| m6-s0 / base | base | insufficient | 4 | 0/0 | 0.7168 / 0.4336 / 1 | 1 | correct_commit | 6 |
| m6-s1 / base | base | insufficient | 2 | 0/0 | 0.8686 / 0.7399 / 3 | 1 | unresolved | 2 |
| m6-s2 / base | base | insufficient | 2 | 0/0 | 0.8686 / 0.7399 / 3 | 1 | unresolved | 2 |
| m6-s3 / base | base | insufficient | 6 | 0/0 | 0.5698 / 0.1696 / 1 | 3 | correct_commit | 11 |
| m6-s4 / base | base | insufficient | 4 | 0/0 | 0.7168 / 0.4336 / 1 | 1 | correct_commit | 6 |
| m8-s4 / hazard_goal_swap | hazard_goal_swap | insufficient | 8 | 0/0 | 0.5000 / 0.0000 / 6 | 3 | unresolved | 8 |

Summary: 14 stability, 11 terminal-only, 8 insufficient. The 14 stability cases were all v3 correct declarations. The 11 terminal-only cases split 5 v3 correct / 6 v3 unresolved. The 8 insufficient cases split 4 v3 correct / 4 v3 unresolved.

## Incorrect declarations

| Episode | True | Primary cause | Declared | Declared step | Confidence | v4 steps / termination | Tool units | v3 status |
|---|---|---|---|---:|---:|---|---:|---|
| m0-s3 / sign_composition | sign_composition | inspect-noise | hazard_goal_swap | 8 | 0.9744 | 60 / max_decisions | 3 | wrong_commit |
| m1-s1 / base | base | premature | slippery | 3 | 0.9143 | 7 / success | 1 | wrong_commit |
| m2-s2 / base | base | premature | slippery | 5 | 0.9343 | 12 / success | 1 | wrong_commit |
| m5-s1 / hazard_goal_swap | hazard_goal_swap | inspect-noise | sign_composition | 3 | 0.9744 | 60 / max_decisions | 3 | wrong_commit |

The same four episodes are the final incorrect declarations in v3 and v4, with the same declared rules. In v4 two are explained by a contradictory noisy `INSPECT` (the policy likelihood includes the public 5% error rate) and two are premature commitments on ambiguous movement/sign evidence; the two premature v4 cases recovered to the true terminal MAP. v3 recovered one of those two and ended `m2-s2` at the wrong MAP after the decision limit. Full pre-declaration evidence is in the JSONL appendix.

## Reading the trace file

Each JSONL row contains:

- `scenario_id`, `scenario_type`, `map_seed`, `run_seed`, `method`, `status`, `true_rule`, `task_success`, `env_steps`, `termination_reason`, `terminal_map`, `terminal_posterior`;
- `events`: time-stamped declarations emitted at the safe-commit branch;
- `decisions`: one record per `choose()` call, with the complete posterior, top two probabilities, margin, leader run, safe-commit eligibility, remaining tool budget, action, target, target color, observed and true tool values, and any event emitted at that decision;
- `diagnosis`: the primary cause and secondary factors assigned by the pure classifier.

Terminal MAP is recorded separately from declarations. It is never counted as a declaration.
