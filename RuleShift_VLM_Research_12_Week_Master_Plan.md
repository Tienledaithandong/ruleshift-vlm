# MASTER RESEARCH & EXECUTION PLAN — RuleShift / Agentic VLM

**Working title:** *Do Visual Agents Know When an Experiment Is Worth Its Cost? Task-Aware Probing under Hidden Dynamics Shifts*  
**Version:** 1.0 — 21 November 2026  
**Horizon:** 12 weeks (proposed execution schedule, not a guaranteed paper deadline)  
**Resource constraints:** CPU as the primary development environment; limited free Kaggle/Colab GPU for frozen VLM inference; optional single GPU with 24 GB VRAM; no full VLM fine-tuning, no multi-agent debate, no long RL training.  
**Scope:** closed-set hidden mechanics, unannounced *within-episode* changes, visual observations, online probe-vs-act decisions, inference-time intervention.

> **Research Question (frozen):** When hidden environment dynamics change without warning, do frozen vision-language agents choose probes whose control value justifies their cost, and can structured counterfactual outcome reasoning—without access to the environment's generative model—reduce probe-decision regret and improve closed-loop return?

## TÓM TẮT TRIỂN KHAI (TIẾNG VIỆT)

- **Tuần 1:** rà prior art (DIVA, ZendoWorld, EVOLvE, NeurIPS 2024, OneLife), viết `problem_spec.md`, khóa ranh giới giữa thông tin agent và evaluator.
- **Tuần 2:** xây Gymnasium `RuleShiftEnv` chạy CPU với macro-action; unit tests cho shift, probe và chi phí.
- **Tuần 3:** Bayesian filtering, solver chính xác cho microtask rất nhỏ, `Delta_probe`, các baseline không dùng VLM, phase diagram có vùng nên/không nên probe.
- **Tuần 4:** pilot 50–100 quyết định với frozen VLM, lưu cache và phân loại over-/under-probing.
- **Tuần 5–6:** belief + counterfactual outcome prediction + value-aware selector; so direct VLM và DIVA-style trên validation; khóa test.
- **Tuần 7–9:** closed-loop trên hai họ mechanics, held-out scenarios, cùng ngân sách môi trường và VLM, báo cáo mean return + paired CI.
- **Tuần 10–12:** phân tích perception/belief/prediction/control, robustness/external check, tái lập và viết paper.

**Lưu ý quan trọng:** baseline Bayes chỉ dùng để đo/đối chiếu; VLM không được đọc simulator, oracle belief hay transition probabilities thật. Không dùng reward tức thời đơn độc để định giá probe vì lợi ích chủ yếu nằm ở hành động sau khi nhận thông tin. Đây là kế hoạch mục tiêu, không đảm bảo paper sẽ được hội nghị chấp nhận.

---

## 0. Success criteria, non-goals, and scientific honesty

The project succeeds as a **research prototype** if it produces (i) a fully specified and verified reproducible environment, (ii) exact Bayes references on *microtasks only* and correctly labeled approximate references for larger tasks, (iii) a meaningful offline diagnosis of VLM exploration decisions, (iv) closed-loop evaluation versus strong baselines, and (v) a transparent account of positive and negative results.

**Target paper contribution (a hypothesis to validate, NOT a guaranteed novelty claim):** characterize where visual agents over-/under-probe under changing mechanics and test an inference-time method that predicts counterfactual action outcomes and selects value-aware probes *without calling the ground-truth simulator*. Scientific distinction from existing work must be re-evaluated after a careful literature review.

**Hard boundaries:**
1. The **agent-side pipeline** never reads the simulator state, transition code, likelihood tables, oracle belief, `mechanic_id`, `shift_time`, hidden RNG, or reference-planner Q-values.
2. The **evaluator-only** exact Bayes planner may access the simulator and full generative model, but only to produce reference labels and diagnostic metrics. The evaluation data interface is separate from agent input.
3. No method is allowed to learn from privileged features unintentionally recorded in its observation JSON, prompt, cache key visible to the model, or tools.
4. All methods get the same public action affordances, physical step budget, reward definition, observations, and per-episode cap on VLM calls; record actual calls, tokens, latency, and invalid outputs.
5. Avoid the phrase "Bayes-optimal regret" on a setting where the reference is merely MCTS / short-horizon / approximate.
6. At least one negative-control regime must make probes *not worth their cost*. Do not tune environments specifically to make the proposed method win.
7. Do not call inference-time belief updates "training" or claim open-world rule invention in this closed-set setup.

**Out of scope until the core result passes gates:** PPO, end-to-end VLM RL, fine-tuning, video world models, GAN/world-model generators, agent debate, fully open-set rule discovery, complex FPS/MOBA environments.

## 1. Hypotheses and preregistered comparisons

| ID | Testable hypothesis | Independent variable | Primary response |
|---|---|---|---|
| H1 (diagnostic) | Frozen VLMs make systematic probe-decision errors as probe cost, feedback noise and task-action diagnosticity vary. | Cost × noise × diagnosticity | Offline action-decision regret, probe selection confusion matrix |
| H2 (method) | Structured outcome predictions + value-aware selection improve decisions over direct VLM and a DIVA-style decision-aware prompting baseline. | Inference-time method, SAME input and call cap | Held-out mean closed-loop episodic return |
| H3 (explanation) | Errors can be localized to visual perception, belief inference, outcome prediction or action choice. | Input modes + diagnostic interventions | Conditional errors across stages |
| H4 (robustness; optional) | Positive results persist with hidden mechanics shifts, no-shift controls, new layouts and a second mechanic family. | Held-out shift schedule/layout/family | Paired return difference with interval estimates |

**Primary endpoint:** mean **total undiscounted episodic return** under a fixed maximum of primitive environment steps on a locked test distribution; higher is better. Reward must include actual time and probe costs once.  
**Secondary:** offline decision regret against **exact** Q* on microtasks; Δ_probe sign classification; post-shift return (shift time only evaluator knows); false probe rate in no-shift episodes; time to first post-shift successful task; probes/episode; physical steps spent on probes; VLM calls/tokens/latency/invalid JSON; outcome prediction accuracy; belief scoring rule (e.g., Brier or NLL when labels available to evaluator).  
**Important:** H1 is descriptive and will not alone imply a novel algorithm. H2 is the main method claim and requires closed-loop evidence.

## 2. Verified prior art to read before asserting novelty

Read *the actual papers*, not just summaries; record setting, input modality, hidden state, budget, planning role, oracle privileges, metrics, strongest baseline, and remaining gap in `research/literature_matrix.csv`.

1. **DIVA** — *Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs* (2026): https://arxiv.org/abs/2610.32181 ; official code https://github.com/seunghan96/diva . This directly threatens generic "value of information for an LLM" novelty. Adapt a **DIVA-style** baseline to the same action/observation interface and disclose differences rather than claiming exact reproduction.
2. **ZendoWorld** — *Playing ZendoWorld: Challenging AI Agents on Active Visual Concept Induction* (2026): https://arxiv.org/abs/2607.08233 ; https://github.com/ml-research/ZendoWorld . Already benchmarks visual hypothesis induction and experiments; contrast *task return under shifting dynamics* against rule recovery/information acquisition.
3. **EVOLvE** — *Evaluating and Optimizing LLMs For In-Context Exploration* (ICML 2025): https://proceedings.mlr.press/v267/nie25b.html . Exploration and regret prior art; clarify sequential control difference.
4. **Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning** (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html . Language hypotheses plus probabilistic inference and experiment design.
5. **One Life to Learn** (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/a24d08f2eecc6c96f4a0ad37a74ddeca-Abstract-Conference.html . Symbolic world-model learning under limited interactions; compare learned world models vs known candidate descriptions.
6. **WorldCoder** and **PoE-World**: locate canonical paper/code and verify year/method before using in a paper. Do not put unverified bibliographic metadata into manuscript.

**Novelty sentence to test, not assume:** "Unlike visual rule-discovery benchmarks or decision-aware evidence acquisition without environment control, we measure and improve value-aware *physical probing* that consumes task time and can affect future decisions while the latent transition law itself may switch mid-episode."

## 3. Scientific definitions and environment contract

### 3.1 Formal state and action definition

A partially observable environment has hidden state `x_t = (p_t, inventory_t, gate_state_t, deliveries_t, m_t, c_t, ...)`, where `m_t` is the active mechanic and `c_t` records whether the single allowed shift has occurred. Agent history `h_t=(o_0,a_0,r_0,o_1,...,o_t)` is public. `o_t` consists only of a rendered local RGB observation or a *non-privileged* text/state view for diagnostic modes; publicly disclosed task and candidate-rule descriptions are allowed. Underlying dynamics transition according to `P(x_{t+1}|x_t,a_t)`.

Candidate set `M` is known through **natural-language descriptions**, not likelihood tables. At most one unannounced within-episode shift in MVP. All methods know the *possibility* of shifts, but not the true shift schedule or active rule. Some episodes have **no shift**.

The full primitive action space can contain navigation, `use_card(color)`, probe and `wait`. The model's **decision interface** is a common macro-action menu, e.g. `COMMIT_RED`, `COMMIT_BLUE`, `COMMIT_GREEN`, `PROBE_RED`, `PROBE_BLUE`, `PROBE_GREEN`, `WAIT`, possibly `GO_TO(...)`. Every policy is evaluated through the identical macro executor; the executor may use only **public, fixed map geometry**, never the current mechanic or shift state. One macro may consume multiple primitive steps; the environment can shift during macro execution. In the MVP, return the observation, action-outcome and elapsed steps after the macro; if implementing interruptible macros, expose that identical behavior to all agents.

**Compute accounting:** 
- `MAX_ENV_STEPS` limits primitive physical steps, not macro decisions.
- Each moved tile, failure lockout tick and probe test incurs the documented actual cost once.
- A `probe` cannot teleport the agent or get a free roundtrip. A scripted controller can automate route following as long as its costs and behavior are identical across compared agents.
- A VLM output is restricted to a *public* valid macro ID; invalid action and parsing failures have prespecified fallback and are counted.

### 3.2 Family A — Gate/Lock (first MVP)

Small gridworld with delivery goal and three candidate gate-acceptance laws: RED / BLUE / GREEN. Agent starts knowing the old default RED but not whether a shift has happened or to which alternative. The gate sprite and public observation do not reveal the active rule. Repeated delivery tasks occur within the same episode. Wrong gate input costs time and may lock the gate for `k` primitive steps. A separate diagnostic station produces accept/reject evidence based on the *same rule*, with no task reward, and costs physical travel plus testing.

**Difficulty requirement:** some settings make direct gate attempts optimal (probe dominated), others make visiting the probe station optimal (when lockout is expensive), and noisy feedback introduces additional trade-offs. Gate responses should be informative but not necessarily conclusive when feedback noise is enabled.

**Illustrative defaults (TUNE ONLY ON DEV, not facts):** 5×5 to 7×7 map, `MAX_ENV_STEPS=60`, delivery success `+10`, per primitive step `-0.05`, lockout in `{0, 3, 8}`, probe detour `0–6` steps, station feedback error in `{0, 0.1, 0.25}`. If wrong gate input only causes lockout, do *not* separately subtract the same penalty again. Decide exactly how a wrong use consumes a step and reward before coding.

### 3.3 Family B — Hazard/Bridge (second MVP)

Small route-selection environment with two crossing regimes, e.g. a shortcut with hidden slip/failure risk vs a safe route. A small diagnostic action at a sensing station can estimate the active hazard regime, consuming time but reducing risk of a costly route error; doing the task itself may also reveal the hazard. Use at most 2–3 candidate regimes at the beginning and ensure failure risks are not trivially inferable from sprites or a single normal move. The *actual* transition probabilities exist only inside `env` and `evaluation`, not in the agent. The agent gets natural-language descriptions such as "bridge may be stable or slippery; attempting the crossing may cost recovery time" rather than a code-readable table. This family must genuinely change **transition dynamics**, not merely reward labels.

### 3.4 Held-out generalization (future)

Third family or external environment only after core experiments: simple MiniGrid variant with partial RGB view, or a mechanically different toy task. If using MiniGrid, the standard encoded `image` observation is not RGB; for a visual evaluation, use `RGBImgPartialObsWrapper` or your own local RGB renderer. Preserve agent access restrictions and identical action costs.

### 3.5 Observation, reset, and RNG independence

Implement Gymnasium API:
- `reset(seed: int|None, options: dict|None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `render(mode="rgb_array")`
- `observation_space`, `action_space` and `max_episode_steps`

Separate `env.public_view()` from `env.evaluator_state()`. For paper-ready evaluation, avoid putting `mechanic_id`, shift timestamps, latent hazard, Q* or any private information in the `info` dictionary passed through generic agent code. Supply privileged evaluator data via **a separate evaluation-only adapter**; even if debugging info is returned by Gymnasium, strip it before constructing agent prompts and caches.

Use **semantic event keys** for deterministic randomness, e.g. keyed by `(scenario_id, semantic_event_type, relevant primitive step, attempt_index)`. This supports paired comparisons even when agents take different actions and consume random draws in different orders. Test repeated rollouts of the same policy/config are identical. Do not pretend common random numbers make divergent policies perfectly paired—pair by scenario and report variance.

### 3.6 Shift generator and information leakage

MVP: beginning rule RED; 0 or 1 hidden shift to BLUE/GREEN. On development, use a per-primitive-step hazard `h` conditional on not yet shifted; test with held-out `h`, layouts and no-shift realizations. The agent may know the family of hazard process but not exact hidden random outcomes. If the shift generator is uniform over a narrow time window, absolute time can become a strong predictive cue; avoid giving unnecessary phase information or preschedule deterministic shifts across all episodes. Log actual shift time **only evaluator-side**. No-shift/unchanged-rule episode is a valid negative control. Never compute a detector's delay from the "first identifiable opportunity" alone: also report delay from actual shift, and the opportunity-conditioned metric only for analysis.

## 4. Exact Bayes reference — carefully restricted

### 4.1 Belief and update

Belief on evaluator's microtask is `b_t(x_t)` or, if public task state is fully known, `b_t(c_t,m_t)`. Update with Bayesian filtering using **evaluator-only** transition and observation likelihoods, including chance of hidden shifts. Formally:

`b_{t+1}(x') ∝ P(o_{t+1}|x',a_t) Σ_x P(x'|x,a_t) b_t(x)`.

Observation noise must be modeled correctly. The reference may know the scenario-generating distribution, but it cannot know the realized hidden mechanic or shift time unless explicitly labeled *privileged oracle*.

### 4.2 Bellman equation and probe value

For tiny microtasks, compute finite-horizon optimal Q via exhaustive recursion/dynamic programming over reachable discrete histories or beliefs:

```
Q*_h(b,a) = E[ r(b,a,x') + V*_{h-d(a)}(b') ]
V*_h(b)   = max_a Q*_h(b,a)
```

Here `d(a)` is the primitive duration of a macro-action; more precisely, the recursion must use the full macro-duration distribution and all in-macro chance events. No separate `-cost` term if the reward already includes the cost. Discount may be omitted for the fixed-horizon episodic-return endpoint.

```
Delta_probe(b,h) = max_{a in PROBE} Q*_h(b,a)
                   - max_{a in TASK} Q*_h(b,a)
```

`Delta_probe > 0` => probe is optimal among probe-vs-task categories; `Delta_probe < 0` => task action is preferable, modulo ties/numerical precision. Treat action values from exact solver as **evaluator-only labels**, never put them in VLM prompts.

**Exactness acceptance conditions:** enumerate tiny instances, compare dynamic programming to independent brute-force tree search for horizon ≤3, assert numerical match within 1e-8, inspect ties and terminal cases, test that higher probe cost can make probe dominated in at least some cases. Limit exact solver to small maps/horizons; truncated lookahead and Monte Carlo tree search in large environments must be labeled **approximate**. A controller that knows the actual mechanic is a *privileged oracle*, a different upper bound.

### 4.3 Technical pitfall to catch early

An evaluator can compute exact Q* only if the **observation and action process** is fully specified, including costs and whether shifts can happen inside a macro. If you abstract the macro duration inconsistently between environment and solver, `Delta_probe` labels become wrong. Prefer the smallest deterministic macro implementation in week 2, then add noise/shift and cross-check by simulation.

## 5. Agent architecture (NO access to simulator)

```
public RGB observation + public action/outcome history + textual candidate rules
                                  |
                              Frozen VLM
                                  |
              +-------------------+------------------------+
              |                   |                        |
       estimated beliefs   predicted outcomes       predicted task value
          b_hat(m)         for each (m, action)     / short-horizon proxy
              +-------------------+------------------------+
                                  |
                   history-only consistency updater
                                  |
                     value-aware deterministic selector
                                  |
                     VALID MACRO ACTION ID -> environment
                                  |
                   real observed outcome appended to history
```

**No hidden model borrowing:** the selector can perform arithmetic on **VLM-provided** predictions and public costs, but must NOT query `env.step` on copied states, import simulator transition functions, use Bayes Q* tables, privileged or scripted hazard likelihoods, or oracle belief. Public map distance used by the common macro executor is acceptable only if it is common, transparent, and independent of the hidden mechanics.

### 5.1 Intervention variants

- `B0 Direct`: one VLM call, returns one action ID and optional free-form rationale. No output oracle belief.
- `B1 Structured belief`: VLM returns candidate probabilities and selects an action directly.
- `B2 Predict outcomes`: VLM estimates outcomes of task/probe actions under candidate mechanics and selects action itself.
- `B3 DIVA-style VoI`: adaptation of the paper's decision-informed evidence-acquisition rule to these physical actions; report that it is adapted, not identical, and include physical time costs.
- `M1 Proposed`: VLM estimated belief + counterfactual outcome predictions + predicted immediate reward / continuation implications, history-only consistency updates + a deterministic value-aware selector.
- `M1 minus belief`, `M1 minus outcome`, `M1 minus cost`, `M1 minus consistency`, `M1 minus selector`: ablations.
- `B4 Self-consistency`: multiple VLM samples aggregated, counting every extra sample toward model-call/tokens budget.
- `B5 Oracle belief` **diagnostic upper bound only**, never primary result.
- `B6 Non-VLM`: never-probe/passive, random probe, max-information-gain using **evaluator reference** where appropriately labeled, return-aware Bayes reference, privileged true-rule oracle (distinct label).

### 5.2 Critical value estimation correction

A selector that computes **only** `Σ_m b_hat(m) * immediate_reward(a,m) - cost(a)` will *usually undervalue probes*, since probing may have zero immediate reward and value comes through changing future decisions. The proposed method therefore must estimate **at least a one-step continuation value**, e.g. from VLM-generated conditional outcomes and future action values:

```
Qhat(a | h) = E_{m ~ b_hat, o' ~ Phat(o' | h,a,m)}
              [ rhat(h,a,m,o') + Vhat(h + (a,o')) ]
```

All `Phat`, `rhat` and `Vhat` used by the **method** come from publicly provided constants and/or VLM predictions; **never** use simulator likelihoods. A possible minimal method: ask VLM to predict which task action would be chosen and its value after each potential probe outcome, then calculate expected improvement minus real physical cost. Require exactly the same call cap as the baselines. If this is computationally expensive, use a restricted candidate-action menu and batch predictions in one model call rather than silently dropping continuation value.

### 5.3 Structured output JSON schema (target)

```json
{
  "candidate_probabilities": {"red": 0.25, "blue": 0.50, "green": 0.25},
  "action_predictions": [
    {
      "action_id": "PROBE_BLUE",
      "under_rules": [
        {"rule_id": "red", "possible_outcomes": [{"observation": "reject", "p": 0.95}, {"observation": "accept", "p": 0.05}]},
        {"rule_id": "blue", "possible_outcomes": [{"observation": "accept", "p": 0.95}, {"observation": "reject", "p": 0.05}]}
      ],
      "estimated_future_best_action": "COMMIT_BLUE"
    }
  ],
  "selected_action": "PROBE_BLUE"
}
```

This is **a schema example**, not an accurate prediction for all mechanics or rules. Validate sums ≈1, numeric bounds, action validity, enum membership, missing keys and output parse reliability. Do not include a single `true_rule` field in model-visible input. Treat confidences as estimates, not calibrated probabilities unless tested.

### 5.4 Input modes to isolate causal failure sources

- `P0 RGB-public` (primary): rendered local RGB + history of only seen observations, actions, rewards, and candidate rules in natural language.
- `P1 Text-public` (diagnostic): textual description of exactly the publicly observable content of the same frames, with no invisible state.
- `P2 Oracle-belief` (diagnostic **upper-bound**, not claimed method): same as P0 plus evaluator belief; must be clearly segmented in tables.
- `P3 Perfect-outcome / perfect-belief interventions` (diagnostic only): replace a single predicted component by the evaluator's true distribution to attribute errors; never report as an agent-valid method.

Pair `P0` vs `P1` to estimate perception contribution. Pair `P0` vs `P2/P3` to locate belief/outcome/control failures. No exact causal decomposition claim from observational correlations alone; diagnostic interventions give stronger evidence.

### 5.5 VLM call control and caching

Cache by immutable fingerprint of model name/revision, decoding config/seed, prompt template version, task family, public RGB image hash(es), public history, candidate rule descriptions and available actions. **Do not cache by current frame alone**: identical images can require different actions given different histories. For stochastic decoding, preserve all samples separately. Cache files contain no hidden state or actual test labels in the prompt payload.

Use fixed `max_new_tokens`, image resolution, context length, deterministic decoding for main runs; count token totals when API/model exposes counts. Print/log per-call latency, output type, parse success, retries and fallback. Main evaluation uses a per-episode VLM call cap (e.g. 6) plus actual physical step budget. Baselines that need 3 samples per decision spend 3 calls.

**Model recommendation to test, not guaranteed to fit every Colab GPU:** Qwen2.5-VL-3B-Instruct (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) as a small anchor; Qwen3-VL-4B-Instruct (https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) as a newer alternative; try 4-bit or supported lower-precision settings depending on available runtime, image tokens and framework version. Verify VRAM and compatibility before launching full runs. Do not promise that a 7–8B VLM or any free API tier is always available; use one stronger model for a *small supplementary* sample if resources permit.

## 6. Pre-registration and split construction

### 6.1 Splits are by *scenario*, not frame

Define each `scenario_id` with family, map/layout, mechanics set, probe-cost/noise/diagnosticity config, hazard process seed, shift outcome seed and observation-noise seed. Never mix different frames from the same scenario into train/dev and test. Distinguish `train` (environment design; no VLM tuning), `dev` (debug), `validation` (prompt and method selection), and `test_locked` (one-time main evaluation). Exact solver labels are allowed to guide diagnostic analysis but NOT to optimize test prompts.

### 6.2 Proposed grid and affordable evaluation

Symbolic CPU initial grid:
- `probe_detour` physical steps: `[0, 2, 5]` (actual route must match)
- observation noise: `[0.00, 0.10, 0.25]`
- task-action diagnosticity: `low`, `medium`, `high` (quantify numerically via expected information gain of task actions under a preregistered belief distribution)
- shift: no-shift, low-hazard, high-hazard
- at least two mechanic families in the end-to-end study

Not all `3×3×3×3×2` conditions need VLM inference. On CPU, cover the grid thoroughly. On VLM pilot, **stratify** 12 cells with positive/negative/marginal Δ_probe, noise and hazard; lock the selection before testing the proposed method. For full frozen-VLM evaluation, target a budget-feasible subset with held-out scenario seeds and multiple layouts; expand to a broader set only after measuring runtime.

**Indicative, adjustable scale (budget targets, not power guarantees):**
- Unit/CPU: 500–2,000 seeded symbolic episodes per inexpensive baseline; exact microtask enumeration as affordable.
- Initial offline VLM pilot: ~50–100 *distinct* decisions, focusing on varied Δ_probe signs and margins.
- First closed-loop VLM test: 25–50 paired scenarios × 2–3 models/methods, cap 4–6 VLM calls per episode.
- Final VLM subset: ~100–200 held-out paired scenarios across a small set of primary comparisons if compute allows. Example 120 scenarios × 3 VLM methods × 4 calls = **1,440 model calls**, before extra sampling and ablations. Set the actual cap after benchmarking inference speed/cost.
- Use hierarchy-aware CIs: resample at the independent scenario/layout or seed-cluster level, not each frame as an independent observation; report 95% paired bootstrap interval on `method_return − baseline_return`.

Create `configs/preregistered_test.yaml` and freeze its Git commit before inspecting held-out results. Save `experiment_manifest.json` with package versions, model ID/revision, environment revision, prompt hash, generation parameters, RNG seeds, compute environment and hardware profile.

## 7. Critical tests (must pass before VLM)

### Deterministic functional tests

1. `reset(seed)` + fixed action sequence gives identical public observations/rewards/termination and identical privileged evaluator trace.
2. A mechanics shift does not directly alter visible sprites or text; *without an affected action*, public views from otherwise identical states are identical.
3. No privileged fields appear in `public_view`, action candidates, prompt serialization, or cache payload.
4. Probe result is generated by the **same active mechanic** governing task action, including noise semantics; no separate inconsistent hidden rule.
5. Every macro action consumes expected primitive steps; no free movement, probe, lockout wait or extra reward.
6. If latent belief changes from mostly RED to mostly BLUE, a value-aware *reference* controller can change its preferred card action in a controlled test; if it doesn't, debug before adding VLM.
7. Passive agent updates beliefs based on ordinary task outcomes; never disable passive learning for a straw-man baseline.
8. Oracle knows true mechanic only in expressly privileged upper-bound mode.
9. One-step hand-computed Bayes update matches reference implementation.
10. `ExactBayesSolver` Q/V match independent exhaustive brute force at horizon 1, 2, 3, within numerical tolerance.
11. At least one config with `Delta_probe > 0` and one config with `Delta_probe < 0`; if not, redesign public costs/reward and keep test split locked after design.
12. Histories that have the same image but different observed past outcomes produce **different cache keys**.
13. Macro-shift mid-execution yields the same outcome in evaluator model and actual env under the same conditions.
14. Confidence and statistics do not use unobservable true shift time at agent runtime.
15. VLM JSON parser handles invalid/missing/conflicting fields with deterministic, logged behavior.

**Validation method:** automated `pytest`, property tests across random seeds, small exact enumerations, and `gymnasium.utils.env_checker.check_env(env)`.

## 8. Detailed 12-week execution plan

### Week 1 — Literature + problem specification + preregistration draft

**Day 1:** Create GitHub repo and `research/reading_notes.md`. Read DIVA and its official code README; write a 1-page comparison covering what counts as a check, cost, uncertainty and the final decision.  
**Day 2:** Read ZendoWorld paper and inspect its environment/repo; write why physical probing and hidden dynamics differ from purely recovering a visual rule.  
**Day 3:** Read EVOLvE, Piriyakulkij et al., and One Life to Learn at least at method/experiments level; fill literature matrix.  
**Day 4:** Write `docs/problem_spec.md` with full formal state, agent/evaluator privileges, macro semantics, shift/no-shift process, noise model and rewards.  
**Day 5:** Write `docs/benchmark_protocol.md` and initial scenario/split schemas; code simple renderer mock.  
**Day 6:** Review novelty with mentor or self-review: formulate 3 hypothetical reviewer objections; define precise baselines and expected failure modes.  
**Day 7:** Freeze design version `v0.1` and create issues/tasks for Week 2.

**Deliverables:** `literature_matrix.csv`, `problem_spec.md`, `benchmark_protocol.md`, `pre_registration_draft.md`.  
**Gate W1:** Explain in a concise paragraph why this differs from DIVA, ZendoWorld and EVOLvE **without claiming to be first in value-of-information**.

### Week 2 — CPU environment, macro executor and tests

Implement family A and public-vs-privileged interfaces. Add reference `RuleShiftEnv`, semantic RNG, renderer, at-most-one shift, no-shift scenario, `COMMIT_*` and `PROBE_*` actions with real physical step costs. Include simple random and passive baseline. Test deterministic paths and reward invariants. Integrate Gymnasium environment checker and CI.  
**Deliverables:** `envs/gate_lock.py`, `envs/macro.py`, `envs/observations.py`, `tests/test_env_*.py`, `scripts/smoke_env.py`.  
**Gate W2:** 100% passing critical environment tests; no leaked privileged fields; at least two distinct conditions with probe advantaged/disadvantaged are achievable.

### Week 3 — Bayes reference + exact microtask + phase diagram

Implement evaluator-only shift-aware Bayesian filtering and exhaustive finite-horizon solver for tiny settings; test against independent brute force for horizon ≤3; create `delta_probe(b,h)`. Implement `never_probe`, `random_probe`, `max_information_gain`, `return_aware_reference` and `true_mechanic_oracle` (clearly labeled). Generate CPU phase diagram with probe detour vs feedback noise vs task-action diagnosticity. For full environment use **approximate** reference as necessary.  
**Deliverables:** `eval/bayes_filter.py`, `eval/exact_solver.py`, `eval/approx_reference.py`, `scripts/phase_diagram.py`, `reports/w3_phase_diagram.pdf/png`, solver tests.  
**Gate W3:** Correct exact small cases and a non-trivial phase diagram. If probe always wins/loses, redesign development configurations before locking test scenarios.

### Week 4 — Early frozen VLM pilot

Add public RGB observation and text-only accessible equivalent. Select 50–100 diverse offline decision contexts from development scenarios with labels Q* from exact microtask. Try a small frozen VLM anchor; if available compare a second model. Use identical valid action menu and public history, no oracle belief. Measure JSON validity, valid actions, over-probe, under-probe, outcome errors and token/call costs. Cache all responses.  
**Deliverables:** `agents/vlm_backend.py`, `agents/prompt_direct.txt`, `eval/offline_audit.py`, `data/offline_pilot_manifest.jsonl`, `reports/w4_failure_taxonomy.md`.  
**Gate W4:** Working VLM inference and reproducible diagnostic results with at least one meaningful error mode OR evidence of ceiling performance (then create harder dev cases without changing frozen test set).

### Week 5 — Structured reasoning and value-aware selector

Implement structured belief prompting and outcome-prediction prompting, use only public history for soft consistency updating; batch counterfactual predictions for candidate macros; compute approximate expected continuation from VLM predictions and public task costs. Implement value-aware action selector; compare to direct VLM and zero/one-step naive selector on **validation**. Do not import any evaluator-only Bayes code into agent package.  
**Deliverables:** `agents/structured_vlm.py`, `agents/consistency.py`, `agents/value_selector.py`, JSON schema and tests.  
**Gate W5:** Agent validity and no leakage verified; cost-aware method outputs different actions for sensible changes in probe cost, without using ground-truth model.

### Week 6 — Strong baselines + ablation + freeze test plan

Add DIVA-style adapted baseline, self-consistency with matched actual model-call budget, no-cost ablation, no-belief/outcome/consistency ablations. Keep gate/action budget fixed per comparison. Tune prompts only on validation. Run pairwise validation and error analysis. Lock test scenarios, prompt versions and method code; save Git commit hashes.  
**Deliverables:** `configs/preregistered_test.yaml`, `research/ablation_matrix.csv`, `reports/w6_validation.md`, scripts to compute metrics and CIs.  
**Gate W6:** The method shows promising **validation** return or reduced decision regret over direct VLM AND DIVA-style adapted baseline; otherwise revise or explicitly pivot to strong diagnostic analysis.

### Week 7 — Closed-loop family A

Run all main methods through the exact same macro executor in real episodes. Test no-shift, mid-episode shift, easy/difficult feedback, probe advantageous/disadvantageous. Record episode JSONL including ONLY separate privileged evaluator trace. Calculate total return and paired differences with CIs, episode-wise error taxonomy, calls/tokens/latency.  
**Deliverables:** `eval/closed_loop.py`, `reports/w7_family_A.md`, trajectories, first main return plot.  
**Gate W7:** No leakage and reliable rollouts; effect does not reverse solely because of cost accounting or closed-loop distribution shift.

### Week 8 — Family B and cross-family checks

Implement bridge/hazard family with distinct transition law, two regimes or three candidate rules. Rerun small family-B CPU references; use a deliberately reduced set of VLM conditions if compute constrained. Add held-out maps and visual distractor/noise where semantics remain constant.  
**Deliverables:** `envs/bridge_hazard.py`, tests, `reports/w8_transfer.md`.  
**Gate W8:** Study is not reducible to a single color-card lookup task; identify where the mechanism transfers or fails.

### Week 9 — Main closed-loop held-out evaluation

With prompts/methods frozen, evaluate locked held-out scenario set; compute paired bootstrap confidence intervals over independent scenario/layout units. Include VLM direct, structured baseline, DIVA-style adaptation and proposed method. Report all negative results; account for number of invalid JSON outputs and cost per episode. Assess hazard mismatch between development and test.  
**Deliverables:** `results/main_*.parquet/csv`, `reports/w9_main_results.md`, manuscript tables and scripts.  
**Gate W9:** Decide whether support for H2 survives **after control-cost accounting**, matched model budgets, and robust CIs.

### Week 10 — Causal diagnostics + one external check if feasible

Perform surgical counterfactual diagnostics offline: replace only belief with oracle belief, only outcomes with evaluator outcomes, or only perception with public text representation. Show where errors arise. If feasible port to MiniGrid RGB partial observation or add third diagnostic-mechanics family **without changing existing results**.  
**Deliverables:** `reports/w10_error_decomposition.md`, optional `envs/external_adapter.py`.  
**Gate W10:** Can state which component is the bottleneck; no unsupported generality claims.

### Week 11 — Statistical stress tests + reproducibility

Confirm all seeds, prompts, model revisions, license constraints and dependencies. Perform paired inference, confidence intervals (optionally `rliable`), sensitivity to calibration and shift prior, report result variance and limitations. Run clean installation/reproduction from README on CPU for all symbolic components.  
**Deliverables:** clean repo, `requirements.lock` or `uv.lock`, CI, tables and plots, reproduction notebook if needed.

### Week 12 — Write paper + external feedback

Draft abstract (only after numbers), introduction, formal setup, related work, baseline comparison, main method, controlled benchmark design, offline and closed-loop results, failure analysis, compute budget, limitations, responsible claim wording. Prepare artifact checklist and ask a mentor for scientific review. Target venue based on actual evidence; workshop is a realistic fallback.  
**Deliverables:** `paper/draft.tex` or `.md`, 6–8 page initial draft, `paper/figures/`, detailed appendix, research portfolio README.

## 9. Go/no-go and pivots — must be written before results

| Checkpoint | Continue if | Pivot if |
|---|---|---|
| W1 novelty | Distinguishing control-within-episode mechanics shift from DIVA/ZendoWorld/EVOLvE is convincing | Reframe as a rigorous replication/benchmark with a new controlled setting; avoid novelty overclaim |
| W3 environment | Exact microtask verified; probe profitable and unprofitable regimes both exist; no leakage | Fix problem definition, do NOT add VLM |
| W4 pilot | VLM makes testable decision errors or phase-specific anomalies | If uniformly near-perfect, increase *development* complexity or focus on perception/compositional mechanics |
| W6 method | Structured method improves validation against direct and strong VoI prompts with no privileged input | Focus on decision-error characterization / baseline reproduction; do not claim algorithm contribution |
| W9 paper | Main closed-loop gain survives matched budgets with CI and several task configurations | Write analysis/workshop paper, add stronger model/check more environments, or stop method claim |

Do not treat *any single* positive difference in 10 episodes as a publishable result. Predefine minimal practically meaningful effect, e.g. ≥5% relative improvement in normalized return **only if defined safely near zero**, or a fixed absolute return threshold appropriate to the reward scale; perform pilot power/variance assessment before choosing final sample size.

## 10. Experiment orchestration and repository layout

```text
ruleshift-vlm/
├── README.md
├── pyproject.toml
├── uv.lock                       # or pinned requirements
├── LICENSE
├── .github/workflows/tests.yml
├── docs/
│   ├── problem_spec.md
│   ├── agent_evaluator_permissions.md
│   ├── benchmark_protocol.md
│   └── preregistration.md
├── research/
│   ├── literature_matrix.csv
│   ├── reading_notes.md
│   └── ablation_matrix.csv
├── configs/
│   ├── dev.yaml
│   ├── validation.yaml
│   ├── preregistered_test.yaml
│   └── models.yaml
├── src/ruleshift/
│   ├── envs/                   # hidden transition code; AGENT must not import
│   │   ├── gate_lock.py
│   │   ├── bridge_hazard.py
│   │   ├── macro.py
│   │   ├── observations.py
│   │   └── events.py
│   ├── public_api/             # only obs, available actions, history, public costs
│   │   ├── interface.py
│   │   └── schemas.py
│   ├── agents/                # MUST NOT import envs or evaluator
│   │   ├── direct.py
│   │   ├── vlm_backend.py
│   │   ├── structured.py
│   │   ├── consistency.py
│   │   ├── value_selector.py
│   │   └── baselines.py
│   ├── evaluator/             # privileged evaluator-only oracle
│   │   ├── bayes_filter.py
│   │   ├── exact_solver.py
│   │   ├── approx_reference.py
│   │   └── metrics.py
│   └── prompts/
│       ├── direct_v1.txt
│       ├── belief_v1.txt
│       ├── outcomes_v1.txt
│       └── diva_style_v1.txt
├── scripts/
│   ├── smoke_env.py
│   ├── test_exact_solver.py
│   ├── generate_microtasks.py
│   ├── phase_diagram.py
│   ├── run_offline_audit.py
│   ├── run_closed_loop.py
│   ├── aggregate_results.py
│   └── check_leakage.py
├── tests/
│   ├── test_env.py
│   ├── test_rng.py
│   ├── test_macro_costs.py
│   ├── test_exact_solver.py
│   ├── test_belief_update.py
│   ├── test_agent_permissions.py
│   └── test_prompt_cache.py
├── results/                    # generated; mostly ignored in git
├── reports/
│   ├── w3_phase_diagram.md
│   ├── w4_vlm_pilot.md
│   └── w9_main_results.md
└── paper/
    ├── draft.md
    └── figures/
```

**Target CLI (to implement; commands will NOT run until agent creates modules):**

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/smoke_env.py --seed 42 --family gate
python scripts/test_exact_solver.py --horizon 3
python scripts/phase_diagram.py --config configs/dev.yaml
python scripts/run_offline_audit.py --config configs/validation.yaml --model qwen-small
python scripts/run_closed_loop.py --config configs/preregistered_test.yaml --method structured
python scripts/aggregate_results.py --input results/ --paired-ci
```

**Suggested dependencies:** Python 3.11; `gymnasium`, `numpy`, `scipy`, `pydantic`, `Pillow`, `pytest`, `matplotlib`, `pandas`, `pyyaml`; optionally `transformers`, `torch`, `accelerate`, `bitsandbytes`/supported quantization runtime, `rliable`. Pin tested versions after first successful smoke runs; avoid CUDA-specific dependencies in CPU core.

## 11. Logging schema and quality controls

Save one row per decision and one row per episode; test keys/columns explicitly.

`decisions.jsonl` PUBLIC view columns: `run_id`, `scenario_id`, `agent_method`, `episode_id`, `decision_id`, `public_obs_hash`, `public_history_hash`, `available_actions`, `chosen_action`, `physical_steps_elapsed`, `actual_reward`, `vlm_calls_used`, `input_tokens`, `output_tokens`, `latency_ms`, `json_valid`, `invalid_action`, `model_id`, `prompt_hash`, `run_seed`.

`evaluator_only.jsonl`: same keys plus `true_rule_before_after`, `shift_time`, `Bayes_belief`, `exact_Q_values_where_available`, `delta_probe`, `actual_noise_outcome`, `privileged_oracle_return`. **Never join evaluator-only columns into model-facing datasets.** Restrict read permissions or load paths at execution layer; CI static checks ensure `src/ruleshift/agents` does not import `src/ruleshift/evaluator` or `envs`.

`episodes.csv`: return, success count, probes attempted, failed commits, physical steps, calls/tokens/latency, family, split, noise, cost, diagnosticity, shift/no-shift, prompt/model revision.

Generate standard plots:
1. Mean total episodic return (95% paired CI) by agent method.
2. Probe-decision error probability vs **exact** Δ_probe bins on microtasks.
3. Probe-cost × noise phase diagram (split by mechanic family/diagnosticity).
4. Over-/under-probe confusion matrix by method.
5. Closed-loop return difference by environment family and shift/no-shift.
6. Actual model-call cost vs return (Pareto analysis).
7. Error attribution from RGB vs public-text vs oracle belief/outcome diagnostics.

## 12. Research risks and mitigation

| Risk | Specific mitigation |
|---|---|
| Oracle is secretly used by method | Package-level import ban; separate public/evaluator interfaces; prompt JSON leak test; no `env.copy().step()` in agent |
| Immediate-value selector never probes | Include counterfactual observation->posterior->future best action value; do not use only immediate reward |
| Too-easy `RuleShift-Lock` | Ensure both positive/negative Δ_probe and meaningful cost/noise/diagnosticity boundaries; add bridge family |
| Bayes "optimal" label unjustified | Exact only for enumerated short-horizon microtask; approximate clearly named elsewhere |
| VLM decisions exploit known shift schedule | No privileged shift time, random no-shift, held-out hazards, adversarial timing checks |
| Poor sampling/statistics | Independent scenario-level replication, paired bootstrap CI, effect sizes, held-out locked test |
| Large model uses hidden reward via evaluator | Keep agent reward and public task costs explicit; module cannot access Q* or simulator likelihoods |
| Prompt-tuning on test | Test manifest frozen at W6; only validation for prompts; report all tested variants |
| Free GPU unavailable | All CPU components run independently; run smaller offline audits; use supported optional API only when available |
| Method may be DIVA-like already | Strong DIVA-style adapted baseline; stronger novelty claim requires dynamic control-specific insight |
| Result negative | Preserve failure taxonomy and honest limits; target analysis/workshop venue if warranted |

## 13. Reading, tooling, and deliverable links

- Gymnasium custom environment: https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
- Gymnasium API / env checker: https://gymnasium.farama.org/api/utils/
- MiniGrid RGB partial observation: https://minigrid.farama.org/api/wrappers/
- Small VLM reference (model card): https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- Alternative VLM (model card): https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- RLiable: https://github.com/google-research/rliable
- DIVA paper and code: https://arxiv.org/abs/2610.32181 | https://github.com/seunghan96/diva
- ZendoWorld paper and code: https://arxiv.org/abs/2607.08233 | https://github.com/ml-research/ZendoWorld
- EVOLvE ICML 2025: https://proceedings.mlr.press/v267/nie25b.html
- Piriyakulkij et al., NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html
- One Life to Learn, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/a24d08f2eecc6c96f4a0ad37a74ddeca-Abstract-Conference.html

## 14. Week 1 agent-ready acceptance checklist

- [ ] Git repository initialized with Python package and `pytest -q` CI.
- [ ] `docs/problem_spec.md` includes exact hidden/public variables, action protocol, reward, noise, single-shift/no-shift process and state reset.
- [ ] `docs/agent_evaluator_permissions.md` lists forbidden fields and unit tests.
- [ ] `research/literature_matrix.csv` covers at least the five verified papers above.
- [ ] `docs/benchmark_protocol.md` defines primary metric, scenario splits and expected output figures.
- [ ] `configs/dev.yaml` contains a *development-only* cost/noise/diagnosticity grid.
- [ ] `README.md` explains the problem in English and explicitly says there are no claimed results yet.
- [ ] Week-2 issues created (public API, core gate environment, macro executor, RNG, tests).

**Final rule: EVIDENCE BEFORE SCALE.** Build one environment that is provably faithful, one reference that is provably exact on tiny cases, and one VLM experiment that directly tests probe-vs-act decisions. Only then grow the benchmark or system.
