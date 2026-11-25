# Belief-state research note: from prompt repair to explicit state

**Status: research memo, no GPU run, no confirmatory result, Phase 1 blocked.**
This is the follow-up to the closed frozen-3B feasibility study. It records a
CPU-only diagnostic and the next experiment worth running. The original 10-map
replication failed its acceptance gates; after benchmark repair it passes six
of seven, but the 100% base gate still fails at 49/50. See
[`GRID_PHASE0_10MAP.md`](GRID_PHASE0_10MAP.md). This memo does not reopen the
5,500-episode grid.

## 1. What the VLM traces actually contain

The map-0 chunk has 3,086 decision rows over 55 episodes. I replayed every
recorded movement through `GridBelief` using only the public map, door state and
the next recorded position. This is an oracle-quality state-update test: the
filter sees the real landing cell and never sees the true rule. It tests
whether a correct Bayesian update could recover the rule from the evidence the
VLM actually gathered.

| Subset | Top-1 rule | Mean final entropy |
| --- | ---: | ---: |
| all 55 episodes | 32/55 (eventual rule) | 1.02 bits |
| `remap_swap` | 5/5 | 0.00 bits |
| `remap_rotate` | 5/5 | 0.00 bits |
| `slippery` | 5/5 | 0.21 bits |
| `base` | 20/30 | 1.31 bits |
| `hazard_goal_swap` | 0/5 | 1.59 bits |
| `sign_composition` | 0/5 | 1.59 bits |

Across all 55 episodes, 19 end below 0.1 bits, 3 are between 0.1 and 1.55,
and 33 end at or above 1.55 bits. The split is structural, not a filter bug.
`base`, `hazard_goal_swap` and `sign_composition` share the same movement
distribution, and `hazard_goal_swap` and `sign_composition` also share the same
sign clauses. Movement can never separate those three. `slippery` and the two
remaps are movement-identifiable, and the replay identifies them immediately.

The VLM made zero tool decisions in the chunk and in all four smokes. So the
information needed to separate the movement-equivalent rules was never
collected. This is the part that schema repair, normalization and more prompt
iteration cannot fix.

## 2. The missing evidence has a cheap deterministic protocol

Under the public menu, two tools are sufficient for the three
movement-equivalent rules, and the costs fit the budget exactly:

1. `READ_SIGN` on side 0 (cost 1).
   - `base` gives “The red tile is not the hazard.”
   - `hazard_goal_swap` and `sign_composition` give “The red tile is not the goal.”
2. If the sign says “not the goal”, `INSPECT` the blue cell (cost 2).
   - `hazard_goal_swap` gives `goal`.
   - `sign_composition` gives `safe`.

The tool budget is 3, so this is the exact affordable diagnostic pair. A direct
check with `GridBelief` over those three hypotheses gives 1.00 posterior for
`base`, and 0.974 for the correct rule in the other two cases under the public
0.05 `INSPECT` noise. This is a decision-tree test, not an EIG search that can
wander.

The existing `info_gain` baseline optimizes expected IG per unit cost at every
step and scored 25/55 in the map-0 comparison, below the 38/55 of
`no_adaptation`. The likely reason is that it probes for small gains instead of
committing to the one diagnostic sequence. That is a testable hypothesis, not a
conclusion yet.

## 3. What the literature says

- **Belief-State Engine (BSE)** argues that an LLM agent is a
  history-conditioned policy with no explicit belief over hidden state, and
  fixes this with an external Bayesian posterior exposed to the LLM instead of
  the raw log. It proves the pair is a sound Markov policy on the belief MDP.
  This is the closest architectural match to our failure.
  [arXiv:2609.10036](https://arxiv.org/abs/2609.10036)
- **Agent-BRACE** decouples a belief-state model from a policy model and
  conditions actions on a compact verbalized belief rather than the full
  history. It is training-based; our setting is frozen and training-free.
  [arXiv:2605.11436](https://arxiv.org/abs/2605.11436)
- **The Perceptual Bandwidth Bottleneck** frames active visual reasoning as
  sequential experimental design and selects observations by expected
  information gain. This supports a targeted probe policy over a generic
  “always probe” heuristic.
  [arXiv:2605.01345](https://arxiv.org/abs/2605.01345)
- **Structured Uncertainty guided Clarification** uses expected value of
  perfect information to decide whether a question is worth asking. The same
  commit-versus-probe decision applies to our tool budget.
  [arXiv:2511.08798](https://arxiv.org/abs/2511.08798)
- **Let Me Speak Freely?** and **Capacity, Not Format** find that strict output
  formats can degrade reasoning, especially for smaller models. That is
  consistent with the v7–v9 pattern: the six-key menu is a format burden on a
  3B model, not a neutral readout.
  [ACL 2024](https://aclanthology.org/2024.emnlp-industry.91/),
  [arXiv:2606.09410](https://arxiv.org/abs/2606.09410)
- **MET-Bench** and **VSTAT** document entity/state-tracking failures in
  vision-language models. They make the negative result less surprising and
  motivate an external state module rather than a longer prompt.
  [arXiv:2502.10886](https://arxiv.org/pdf/2502.10886v2),
  [arXiv:2606.03920](https://arxiv.org/abs/2606.03920)

## 4. Proposed next experiment

Build a CPU reference first; do not spend GPU quota on another prompt variant.

### Phase 0 — implemented and evaluated (CPU dev)

`DiagnosticBeliefPolicy` now lives in `agents/grid_baselines.py` and is wired
into `scripts/export_grid_baselines.py` as `diagnostic_belief`. It uses the
existing `GridBelief`, the public A* planner and a fixed decision tree:

1. Update the filter from every public transition.
2. If the posterior has a decisive gap, plan to the MAP goal with A*.
3. Otherwise navigate to an informative sign face and `READ_SIGN` (side 0
   preferred; side 1 when side 0 is walled off).
4. If the sign is “not the goal”, navigate within two cells of blue and
   `INSPECT` it.
5. Plan to the MAP goal and commit.

While walking to a tool, every marker is treated as a wall: any marker can be
the goal under some hypothesis, so crossing one can end the episode before
the evidence is collected.

It does not probe after movement has already identified a rule. On the same
55 map-0 scenarios (5 seeds x 11 types) the paired CPU result is:

| Method | Success | Env steps | Tool units |
| --- | ---: | ---: | ---: |
| `diagnostic_belief` | **54/55** | 473 | 60 |
| `no_adaptation` | 38/55 | 1,365 | 0 |
| `public_advisor` | 38/55 | 1,365 | 0 |
| `random_probe` | 38/55 | 1,365 | 0 |
| `info_gain` | 25/55 | 2,685 | 120 |

The paired counts against `no_adaptation` are 38 both, **16 only
diagnostic**, 0 only baseline, and 1 neither. The one shared failure is the
same noisy inspect episode. By initial rule: `base` 5/5 (1 tool each),
`remap_swap` 5/5 (0), `remap_rotate` 5/5 (0), `slippery` 5/5 (0),
`hazard_goal_swap` 5/5 (3), `sign_composition` 4/5 (3). There were no false
alarms, and tool spend was only 0, 1 or 3 units.

The single failure, `m0-s3-identification-sign_composition`, is a 5%
`INSPECT` error: blue was truly `safe`, the tool returned `goal`, and the
filter correctly followed that public observation to `hazard_goal_swap`. The
policy then navigated to blue (the wrong goal under the true rule) and timed
out. This is irreducible with one noisy inspect and a 3-unit budget; it is not
a protocol or belief-update bug.

Caveat: this is one map and a development set. The 20/20 change-detection
episodes are not evidence of post-shift adaptation: `no_adaptation` also scores
20/20, because those episodes finish before the shift window. The gain is in
identification, especially the ten movement-equivalent `hazard_goal_swap` and
`sign_composition` scenarios.

**Exploratory five-map check (maps 0-4, 275 episodes).** The same policy scores
**268/275** versus **200/275** for `no_adaptation`; identification is **147/150**
versus **75/150**. The paired counts are 194 both, 74 only diagnostic, 6 only
baseline, and 1 neither. The three identification failures are inspect noise
(one map-0 sign, one map-3 hazard, one map-3 sign). The six baseline-only wins
are two identification episodes where the baseline happened to choose the true
goal and four post-shift change episodes; the policy does not re-probe after a
shift and often has already spent its budget. Change detection is therefore
96/100 versus 100/100, a scoped adaptation limitation, not an identification
failure. No-shift is 25/25 for both.

The five-map check exposed two bugs that map 0 did not: a signpost can have both
side-0 faces walled off (the policy now uses informative side 1), and a shortest
tool route can cross a candidate goal and terminate before the probe (tool
routing now treats every marker as blocked). After both fixes, map 4
`hazard_goal_swap` and `sign_composition` collect the sign and blue evidence
instead of succeeding by accident. The primary map-0 numbers were unchanged.
This is still a development check, not a preregistered result.

**10-map replication v0.1: gates failed.** The 10-map CPU run (550 episodes x 2
methods) scores **518/550 (94.2%)** for `diagnostic_belief` versus **382/550
(69.5%)** for `no_adaptation`. Identification is **276/300 (92.0%)**, hazard
**39/50 (78.0%)**, sign **38/50 (76.0%)**, and base **49/50 (98.0%)**; there are
**20 zero-tool hazard/sign episodes**. The tool cap and paired-gain gates pass,
but the identification, hazard, sign, base, and zero-tool gates fail. Two
benchmark defects are material: the 10 map indices contain only 8 unique
geometries (the 100-map audit predicts 62 unique), and maps 7/8 have no
reachable informative sign face, producing 20 zero-tool episodes (19 failures
and one accidental success). One base failure on map 5 is a real face-selection
limitation: the policy prefers the nearer but less discriminative side.

**Repaired benchmark v0.3: six gates pass, base still fails.** Map retries are
collision-free, the generator requires a reachable side-0 face (side 0 is the
only face that separates `base` from the hazard/sign hypotheses), and the policy
now implements side-0 priority instead of nearest-face selection. The 100-map
audit is 100/100 unique with no missing side-0 face. The v0.3 CPU run scores
**534/550 (97.1%)** versus **378/550 (68.7%)** for `no_adaptation`; identification
is **287/300 (95.7%)**, hazard **49/50**, sign **49/50**, base **49/50**, and there
are **zero** zero-tool hazard/sign episodes. Paired overall: 364 both, 170
diagnostic-only, 14 baseline-only, 2 neither (`p = 3.13e-35`). The remaining base
failure is one stochastic navigation episode on map 2, not a benchmark
affordance defect: the policy commits to `slippery` after base slips and misses
the 60-step cap. No gate was relaxed and no face rule was tuned after seeing
this result. Per the stop rule, Phase 1 was not started; no VLM smoke and no
55-episode run were performed. Full evidence, commands, and limitations:
[`GRID_PHASE0_10MAP.md`](GRID_PHASE0_10MAP.md).

**Phase 1 — VLM on a compact belief (GPU, only if Phase 0 works).** Replace the
six-key probability menu with a small high-level action interface: `COMMIT`,
`PROBE_SIGN`, `PROBE_BLUE`. Put the current posterior and the available
high-level actions in the prompt; the symbolic executor handles navigation.
This tests whether the 3B model can act as a belief-MDP policy when the
arithmetic and memory burdens are removed. It is not another prompt tweak: the
interface is different.

Run one two-episode smoke. It must pass both the existing technical gate and
the semantic gate (belief change, at least one tool decision, base-rule
success). Only then run one 55-episode map. A resized 3-map x 2-seed grid is
about 7.7 GPU-hours; the 5,500-episode grid remains closed.

**Stop rule.** If the symbolic diagnostic policy cannot beat the no-adaptation
baseline on the 55-map dev set, the benchmark’s tool/budget design is the
problem, not the VLM. If the VLM still makes zero tool decisions with a compact
posterior, the 3B model cannot be the belief-MDP policy; report that and keep
the symbolic upper bound. Do not scale either failure.

## 5. Framing if this works

The paper is not “we fixed the prompt.” It is a diagnostic and an architectural
result: a frozen small VLM fails active rule identification because it neither
computes a posterior nor acquires the evidence that separates observationally
equivalent rules; an external belief state plus a three-unit diagnostic
protocol supplies both. The negative VLM-only result and the symbolic upper
bound are both part of the contribution.
