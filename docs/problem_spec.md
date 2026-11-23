# RuleShift — Problem Specification and Environment Contract (v0.1)

**Status:** specified, **not implemented**. Week 1 delivers this contract; the
Gymnasium environment, renderer, macro executor and Bayes reference are Week 2+
work (see `week2_issues.md`). All numeric values below are *development
hypotheses to tune on dev*, not results and not frozen test settings.

**Governing documents:** `AGENT_TASK_WEEK1_RuleShift.md` (task brief) and
`RuleShift_VLM_Research_12_Week_Master_Plan.md` (master plan). Where this
document is more specific, it resolves ambiguities in those documents.

---

## 1. Question and scope

Frozen vision-language agents act in a partially observable gridworld whose
hidden transition law may change **once, without warning, inside an episode**.
At each decision the agent chooses between a **task action** (try to complete
the delivery) and a **probe** (pay physical travel and test time to obtain
noisy evidence about the active law). A probe has no task reward; its value is
entirely in how it changes *future* decisions. We study whether agents choose
probes whose control value justifies their real cost.

Two mechanic families are in scope:

- **Family A — Gate/Lock** (first MVP): three candidate gate-acceptance laws
  (RED / BLUE / GREEN).
- **Family B — Hazard/Bridge** (second MVP): 2–3 candidate crossing regimes
  that change **transition probabilities**, not just reward labels.

Closed-set only: the candidate descriptions are public; open-ended rule
invention is out of scope.

---

## 2. Notation and state

Hidden state at primitive step `t`:

```
x_t = (p_t, inventory_t, gate_state_t, deliveries_t, m_t, c_t, ...)
```

| Symbol | Meaning | Visibility |
|---|---|---|
| `p_t` | agent grid position | public |
| `inventory_t` | held cards / items | public |
| `gate_state_t` | `{open, locked, lockout_remaining}` after a rejection | public |
| `deliveries_t` | completed delivery count | public |
| `m_t` | active mechanic id (`red`/`blue`/`green` for A; regime id for B) | **evaluator only** |
| `c_t` | 1 iff the single allowed shift has occurred, else 0 | **evaluator only** |
| `h_t` | hidden hazard sample source (seed-derived) | **evaluator only** |
| `z_t` | stochastic outcome sample of the current primitive step | **evaluator only** |

Agent history `hist_t = (o_0, a_0, r_0, o_1, ..., o_t)` is public by definition.
The agent knows the *set* of candidate mechanics through natural-language
descriptions `D(m)` and knows that a shift is *possible*, but never knows the
realized `m_t`, the shift time, the hazard realization, the transition kernel,
or any reference action value.

---

## 3. Family A — Gate/Lock

### 3.1 Layout and task

- Grid size 5×5 to 7×7 (dev-tunable). Public fixed geometry: start tile, gate
  tile, delivery tile, and a diagnostic station tile. Geometry never encodes
  the active rule.
- The agent holds one task card. Repeated delivery tasks occur within the same
  episode: after an accepted delivery, the agent respawns at the start with a
  new card and the same (possibly shifted) active mechanic.
- The gate sprite and every public observation are **identical across active
  mechanics**. A shift alone never changes a visible sprite or text.

### 3.2 Candidate laws (public text, no likelihood table)

| id | Public description (example) |
|---|---|
| `red` | "The gate currently accepts a red card." |
| `blue` | "The gate currently accepts a blue card." |
| `green` | "The gate currently accepts a green card." |

The agent starts each episode believing the old default is `red`, but does not
know whether a shift already happened. These descriptions are knowledge, not
privileged labels: they do not say which rule is active now.

### 3.3 Gate outcome and lockout

A `COMMIT_<color>` task action is evaluated against `m_t` **at the primitive
step on which the commit is executed**:

- **Accepted**: `+10` delivery reward, delivery counter increments, respawn.
- **Rejected**: no task reward. The attempt consumes its primitive step and the
  gate enters lockout for `k` further primitive steps, `k ∈ {0, 3, 8}` (dev).
  During lockout, commit actions are invalid and must fall back to `WAIT`.

**Cost-once rule:** a rejected commit is penalised *only* through (a) the step
cost of the attempt and (b) the step costs of the `k` forced lockout steps.
There is no additional subtracted "wrong-answer penalty". This avoids double
counting the same lockout penalty.

---

## 4. Family B — Hazard/Bridge (specified, implemented later)

- Route choice: a shortcut bridge with hidden slip/failure risk versus a safe
  detour. Regimes `stable`, `slippery` (and optionally `icy`) define real
  transition probabilities `P(next | state, action, regime)`; these exist only
  inside `env` and the evaluator.
- A sensing-station probe estimates the active regime with the same error model
  as Family A. Attempting the crossing may itself reveal hazard information,
  so passive learning from ordinary task outcomes remains enabled.
- Failure costs recovery time via real primitive steps, not a hidden penalty.
- Public text is qualitative ("the bridge may be stable or slippery; a bad
  crossing may cost recovery time"), never a probability table.

**Family B must change `P(s'|s,a)`, not merely reward labels.** A build that
only relabels rewards does not satisfy this contract.

---

## 5. Observations and the public view

`o_t` consists of:

1. A local **RGB** rendering (Week 2) whose digest `public_obs_hash` is cached.
2. A **public text view** (`public_text`) describing exactly what is visible —
   used for the P1 diagnostic input mode; it contains no invisible state.
3. The **public history** and its digest `public_history_hash`.
4. The **available macro-action ids** and the **public cost description**.
5. The natural-language candidate-rule descriptions.

The canonical agent-visible payload has exactly these keys (enforced by
`ruleshift.public_api.schemas.validate_public_observation`):

```
observation_id, public_obs_hash, public_history_hash, public_text,
available_actions, candidate_rules, public_costs
```

`last_outcome` inside `public_text` is the outcome of the most recent
*informative* primitive op. A probe's return leg therefore does not erase the
station's accept/reject report: the decision-boundary observation still carries
it, so a probing agent can update its belief afterwards.

`public_costs` publishes non-negative **magnitudes** only — `primitive_step`,
`probe_detour` (one-way station distance), `commit_travel` (public Manhattan
distance to the gate), `lockout` and `delivery_reward`. Consumers must convert
them to a signed cost model exactly once; a positive per-step "cost" would
reward taking more steps.

No other key is permitted. In particular, a `shift_time`, `mechanic_id`,
`oracle_belief`, `Q*`, `delta_probe` or hidden-RNG field must never be present.
`reset`/`step` `info` dictionaries returned to generic agent code must satisfy
the same allowlist; privileged data is delivered through a **separate
evaluation-only adapter** and stored in `evaluator_only.jsonl`.

---

## 6. Actions and the common macro executor

### 6.1 Decision interface (public macro ids)

```
COMMIT_RED, COMMIT_BLUE, COMMIT_GREEN,   # task category
PROBE_RED,  PROBE_BLUE,  PROBE_GREEN,    # probe category
WAIT,
GO_TO_STATION
```

The model may only return one of these ids; invalid ids and parse failures use
a prespecified fallback (default `WAIT`) and are logged (`json_valid`,
`invalid_action`). All methods share this identical menu.

### 6.2 Macro semantics and duration

A macro expands to a sequence of primitive moves/traces controlled by a common
executor. The executor may use **only public fixed geometry** (e.g. Manhattan
distance from `p_t` to the station). It must never read `m_t`, `c_t`, the shift
schedule or the transition kernel.

| Macro | Primitive duration (before in-macro events) |
|---|---|
| `COMMIT_*` | travel to gate (public distance) + 1 commit step |
| `PROBE_*` | travel to station + 1 test step + travel back to start |
| `WAIT` | 1 step |
| `GO_TO_STATION` | travel to station (public distance) |

`MAX_ENV_STEPS` counts **primitive steps**, not macro decisions. `d` denotes
the public one-way station detour; a probe from the start therefore costs
`2d + 1` primitive steps, with the exact route and cost identical for all
compared agents. There is no free teleport and no free round trip.

A shift can occur between any two primitive steps, **including inside a
macro**. Therefore:

- The commit outcome is decided by the mechanic active at the commit step.
- The probe feedback is generated by the mechanic active at the test step.
- The evaluator model and the real environment must produce the same result
  under the same conditions (master plan test 13).

Week 2 uses the smallest deterministic macro implementation first, then adds
noise/shift and cross-checks against simulation. If interruptible macros are
ever introduced, the same behavior is exposed to every agent.

---

## 7. Reward, cost, and step semantics

`step(action)` follows the Gymnasium contract:
`(obs, reward, terminated, truncated, info)`.

- **Per primitive step:** `step_cost = -0.05` (dev), applied exactly once for
  every primitive step consumed by any macro, including travel, the probe
  test, forced lockout ticks and `WAIT`.
- **Accepted delivery:** `+10` (dev).
- **Probe:** no task reward; only its real step costs.
- **No separate penalty** is stacked on lockout (see §3.3).
- **Truncation:** `truncated = True` when `MAX_ENV_STEPS` (dev default 60) is
  reached; `terminated` on episode-specific terminal conditions. Total
  undiscounted return is the primary endpoint, so costs appear once in the
  same reward stream.
- `reset(seed, options) -> (obs, info)` initialises the scenario, RNG streams,
  default rule `red`, and an unshifted state.

**Invariant:** the undiscounted return equals `sum(reward_t)` over primitive
steps; there is no hidden penalty channel.

---

## 8. Shift process (single, unannounced, with no-shift episodes)

- Begin with `m = red`. Conditional on not having shifted yet, each primitive
  step triggers a shift with hazard `h` (dev: low/high values); at most one
  shift per episode. When it occurs, choose `blue` or `green` from the
  shift-outcome stream.
- Some realisations have **no shift**; this is a valid negative control and is
  *not* signalled to the agent.
- Shift times must not be concentrated in a narrow window known to the agent,
  and no "episode progress", "fraction of budget remaining" or shift-risk
  feature derived from the hidden schedule is exposed. The raw elapsed
  primitive-step count is part of public history and is allowed.
- The shift does not change any sprite or public text; only subsequent action
  outcomes reveal it. Evaluator-only trace records the actual `shift_time`.
- Report both delay-from-actual-shift and the opportunity-conditioned delay;
  never use only the latter.

---

## 9. Feedback noise (probe and task)

- Station feedback is generated by the **same active mechanic** that governs
  task actions. There is no second inconsistent hidden rule.
- Symmetric error rate `eps ∈ {0, 0.1, 0.25}` (dev): with probability `eps` the
  accept/reject label is flipped; otherwise it is correct. With noise, feedback
  is informative but not conclusive.
- The noise draw is keyed semantically (§10) so paired comparisons do not
  depend on how many draws other policies happened to consume.

---

## 10. Randomness and determinism

Use **semantic event keys** for every stochastic draw:

```
key = (scenario_id, semantic_event_type, relevant_primitive_step, attempt_index)
```

Semantic event types include `SHIFT_TRIGGER`, `SHIFT_TARGET`, `PROBE_FEEDBACK`,
`HAZARD_OUTCOME`, `OBSERVATION_NOISE`. Streams are derived by hashing
`(master_seed, key)` into a local PRNG. This permits paired comparisons even
when agents take different actions and consume draws in different orders.

Tests must show:

1. `reset(seed)` + a fixed action sequence gives identical public observations,
   rewards, termination, **and** identical privileged evaluator trace across
   repeated rollouts.
2. Common random numbers do not make divergent policies perfectly paired —
   pair by `scenario_id` and report scenario-level variance.

---

## 11. Exact Bayes reference — restricted

The evaluator may maintain `b_t(x_t)` (or `b_t(c_t, m_t)` when public task state
is known) and update with

```
b_{t+1}(x') ∝ P(o_{t+1} | x', a_t) Σ_x P(x' | x, a_t) b_t(x)
```

using evaluator-only likelihoods and the shift prior. Finite-horizon Q-values
for **enumerated microtasks only**:

```
Q*_h(b,a) = E[ r(b,a,x') + V*_{h-d(a)}(b') ]
V*_h(b)   = max_a Q*_h(b,a)
Delta_probe(b,h) = max_{a in PROBE} Q*_h(b,a) - max_{a in TASK} Q*_h(b,a)
```

`d(a)` must be the full macro-duration distribution including in-macro chance
events; no separate `-cost` term when the reward already contains costs.
`Delta_probe` is an evaluator-only label. Reference results on larger tasks
must be labelled **approximate** (truncated lookahead / MCTS); a controller
that knows the true mechanic is a separate **privileged oracle** upper bound.
The phrase "Bayes-optimal regret" is reserved for genuinely exact cases.

**Exactness acceptance:** dynamic programming matches independent brute-force
tree search at horizons 1–3 within `1e-8`; ties and terminals inspected; at
least one config with `Delta_probe > 0` and one with `Delta_probe < 0`.

---

## 12. Proposed method constraint (no hidden model borrowing)

The planned VLM method consumes only public RGB, public history, candidate
descriptions and public cost text. The selector may do arithmetic on
**VLM-provided** estimates and public constants, but must not access simulator
likelihoods, `env.copy().step`, Bayes `Q*`, oracle belief or true rules.

### 12.1 M1 one-step selector — baseline/diagnostic, not proven effective

The reference one-step selector (`ValueAwareSelector`, alias `M1`) uses

```
Qhat(a|h) = E_{m~b_hat, o'~Phat}[ rhat(h,a,m,o') + Vhat_1(h + (a,o')) ]
```

with `Vhat_1` a **one-step** continuation `max_a rhat(a | posterior)`. This is a
deliberately shallow baseline, added so that probes (zero immediate reward) are
not systematically undervalued by pure immediate-reward arithmetic.

**Known limitation, documented rather than hidden:** `Vhat_1` is added to
*every* outcome branch, including successful commits, so future commits are
double-counted relative to a probe whose information pays off across several
future decisions. When both probe and commit durations fit inside the horizon,
the continuation largely cancels and the comparison collapses to probe cost
versus commit expected immediate reward. Consequently M1 one-step can select
an immediate commit even when the exact finite-horizon optimum is a probe
(see `tests/test_m1_microtasks.py::test_microtask_A_probe_valuable_but_m1_picks_commit`).

Because of this, M1 one-step is presented as a **baseline/diagnostic** for
Week 1-2 development, not as a claim of an effective probe-vs-task method.
`Delta_probe` and probe-vs-commit conclusions must not be drawn from M1 one-step
results alone.

### 12.2 Multi-step Vhat — experimental, diagnostics only

`MultiStepValueAwareSelector` (ablation `M1_multi_step`) replaces `Vhat_1` with
a finite-horizon recursion `V(b, steps_left)` over public predictions only:
probe feedback updates the public belief and consumes `probe_duration` steps;
a commit consumes `commit_duration` (+ lockout on reject) but the agent may
keep acting while steps remain. No simulator likelihood, no oracle, no hidden
transitions are used.

This ablation is included in the Week 6 comparison matrix (`ABLATION_NAMES`)
so that its behavior can be measured against M1 one-step on dev scenarios with
hand-computed exact optima. It remains **experimental**: outcomes are assumed
fixed from the initial VLM payload (no re-query per recursion step), and the
belief updater is a soft heuristic that does not model feedback noise or
staleness after a shift (see section 15 Q4).

---

## 13. Acceptance tests (to implement in Week 2+)

Mapped from master plan §7; each will be an automated test.

1. `reset(seed)` determinism for public and privileged traces.
2. Shift alone does not alter visible sprites/text.
3. No privileged field in `public_view`, actions, prompts or cache payloads.
4. Probe feedback governed by the same active mechanic as task actions.
5. Every macro consumes the documented primitive steps; no free steps/reward.
6. A value-aware reference controller changes preferred card when belief flips.
7. Passive task outcomes update belief (no straw-man that disables learning).
8. Oracle mode is the only place true mechanic is read.
9. One-step hand-computed Bayes update matches implementation.
10. Exact solver Q/V match brute force at horizons 1–3 within `1e-8`.
11. At least one `Delta_probe > 0` and one `Delta_probe < 0` dev config.
12. Same image + different history → different cache key.
13. Mid-macro shift matches evaluator model and actual env.
14. No use of true shift time at agent runtime.
15. VLM JSON parser handles invalid/missing/conflicting fields deterministically.

CI also runs `gymnasium.utils.env_checker.check_env(env)` once the env exists.

---

## 14. Explicit non-goals for Week 1

No neural model, VLM runner, Bayes oracle, environment, renderer, replay cache,
or result plot is implemented here. All numbers are dev hypotheses. No claim
of novelty or publishability is made.

---

## 15. Open scientific questions (do not silently assume answers)

1. How should task-action **diagnosticity** be quantified preregistered (a
   specific expected-information-gain formula under a fixed reference belief)?
2. What exact reward/lockout magnitudes yield both probe-advantageous and
   probe-disadvantageous dev cells without looking at test data?
3. For Family B, what slip/failure probabilities are non-trivially inferable
   but not trivially visible from one normal move?
4. How should the continuation-value estimator be bounded so it cannot
   manufacture probe value from optimistic VLM predictions?
5. What is the minimum sample size for a stable paired CI at the chosen reward
   scale (power/variance pilot, not a guessed number)?
