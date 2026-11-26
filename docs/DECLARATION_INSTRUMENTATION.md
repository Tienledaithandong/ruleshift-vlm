# Declaration instrumentation and behavioural equivalence

Date: 2026-11-25. Candidate policy commit: `e65e1915503246f0f07ed15c4884ac6952925ee8`.
Behavioural baseline: `f4b43a4ede7f76ab9d5b1f45178fe648ae419042`.
Evaluator coverage correction: `a46ed21` (does not change policy actions).

## 1. Why instrumentation was added

The policies previously selected a terminal MAP and a plan, but they emitted no
explicit "the active rule is X" event. Scoring terminal MAP as an identification
would conflate a posterior snapshot with a declaration. The candidate therefore
needs a declaration API that records the policy's own commitment at the moment it
is made. The API is observational only.

## 2. Exact hook semantics

In `DiagnosticBeliefPolicy` (v3) and `DiagnosticBeliefPolicyV4` (v4), the hook is
called only in the existing safe-commit branch. It records:

- `step`: the current environment-step counter at the decision;
- `declared_rule`: the rule selected by the existing safe-commit criterion;
- `confidence`: the top posterior probability at that decision;
- `posterior_margin`: top minus second posterior probability;
- `source`: `policy.safe_commit`.

The hook does not call the environment, change the policy's action, change probe
selection, change posterior updates, or change the plan. Adjacent duplicate
declarations of the same rule are suppressed so a commitment is one event rather
than one event per subsequent step. Non-adjacent returns to a rule are recorded.
Events are cleared on `reset()`.

Fallback commitments made by v4's plan fallback are **not** recorded. A fallback
is not an explicit identification. Terminal MAP is never inspected by the hook,
and no event is synthesized after termination from the terminal posterior.

The evaluator accepts declarations only within the episode horizon and only for
known rules. It requires finite probabilities in [0, 1], ordered steps, a
nonnegative top-second margin, and a settled rule with confidence ≥ 0.90 and
margin ≥ 0.10. The final declaration is scored; earlier wrong declarations are
counted separately. An unsettled declaration remains an unresolved episode.

## 3. Behavioural equivalence fixture

The fixture `tests/fixtures/v04_equivalence_maps0_9_seeds0_4.json` was generated
from the pre-instrumentation policy at the behavioural baseline. It covers maps
0–9 and seeds 0–4, restricted to the 300 static identification scenarios per
method. For each method and scenario it stores the action-trace SHA-256, task
success, final rule estimate, environment steps, tool units and declaration
events. The events in the fixture are empty because the old code had no API.

The test `tests/test_declaration_equivalence.py` reruns every scenario with the
instrumented v3 and v4 policies and compares each behavioural field. It also
checks the event contract (step ordering, known rules, confidence/margin ranges,
adjacent distinct rules), checks that no_adaptation has no declaration API, and
checks that the fixture is complete.

Observed result at the candidate commit:

    v3: 300/300 scenarios identical
    v4: 300/300 scenarios identical
    behavioural differences: 0

Focused command and result:

    python -m pytest -q tests/test_declaration_equivalence.py
    3 passed

The equivalence tests establish that the new observation is inert. They do not
establish that the declarations are accurate; that is a separate property and is
reported below.

## 4. Development declaration measurements

These are exposed maps 0–9, seeds 0–4 (300 static episodes per method). They are
development evidence, not confirmatory results and not used to tune the policy.

| Method | Events | Episodes with event | Correct | Wrong | Unresolved | Unconditional accuracy | Settled coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 | 294 | 288 | 284 | 4 | 12 | 94.67% | 96.0% |
| v4 | 268 | 267 | 263 | 4 | 33 | **87.67%** | 89.0% |
| no_adaptation | 0 | 0 | 0 | 0 | 0 (unavailable 300) | unavailable | 0.0% |

The earlier draft's `declaration_coverage` field counted API availability and
therefore reported 100% for v4 despite 33 episodes with no settled declaration.
That was a reporting bug, corrected at `a46ed21`:

- `declaration_event_episodes` / `declaration_coverage`: episodes with at least
  one valid declaration event;
- `settled_declaration_episodes` / `settled_declaration_coverage`: episodes with
  a settled declaration;
- `declaration_available_coverage`: episodes whose policy exposes the API;
- `commitment_accuracy`: correct settled declarations / all episodes, with
  unresolved and unavailable in the denominator.

For v4, all 33 unresolved episodes have no declaration event; event coverage is
267/300 and unconditional accuracy is 263/300. For v3, all 12 unresolved
episodes have no event, and 4 wrong declarations remain in the denominator.

The candidate also records earlier wrong declarations: v4 has 1 episode with an
earlier wrong declaration, v3 has 6. A later correct declaration does not erase
the earlier error; both fields are reported.

## 5. Development gate implications

Using the frozen gate definitions on the exposed bank:

- G1 (v4 explicit accuracy ≥ 95%): **fail**, 263/300 = 87.67%;
- G2 (v4 base task 150? Development has 50): 50/50, provisional pass at this
  smaller exposed base;
- G3 (v4 hazard declarations ≥ 90%): 47/50 = 94%, provisional pass;
- G4 (v4 sign declarations ≥ 90%): 44/50 = 88%, **fail**;
- G5 (zero raw zero-tool hazard/sign successes): 0, pass on the exposed bank;
- G6 (tool budget): no violation observed in the candidate's 300 static episodes.

These are 50-per-rule development subsets, not the frozen 150-per-rule
confirmatory endpoints. The correct reading is that the candidate is at material
risk of failing its primary identification gate, not that the gate has already
been adjudicated on the holdout.

## 6. Reproduction

Commands used:

    $env:PYTHONPATH='.;src'
    python scripts/build_grid_manifest.py --maps 0-9 --seeds 0-4 --scope static_identification --out results/p0_dev_static_manifest.json
    python scripts/run_grid_identification.py --manifest results/p0_dev_static_manifest.json --mode development --methods no_adaptation,diagnostic_belief_v3,diagnostic_belief_v4 --out results/p0_dev_static_e65e191 --code-commit e65e191
    python -m pytest -q tests/test_declaration_equivalence.py

The raw development records are in the ignored archive
`results/archive/p0_2026-11-25/p0_dev_static_e65e191/`.
