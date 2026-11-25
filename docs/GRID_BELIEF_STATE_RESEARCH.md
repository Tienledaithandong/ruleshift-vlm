# Belief-state research note: from prompt repair to explicit state

**Status: research memo, no GPU run and no confirmatory result.** This is the
follow-up to the closed frozen-3B feasibility study. It records a CPU-only
diagnostic and the next experiment worth running. It does not reopen the
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

**Phase 0 — `DiagnosticBeliefPolicy` (CPU).** Use the existing `GridBelief`,
the public planner and a fixed decision tree:

1. Update the filter from every public transition.
2. If the movement-equivalent ambiguity is already resolved, plan to the MAP
   goal with A*.
3. Otherwise navigate to sign side 0 and `READ_SIGN`.
4. If the sign is “not the goal”, navigate within two cells of blue and
   `INSPECT` it.
5. Plan to the MAP goal and commit.

Evaluate on the same 55 map-0 scenarios as the existing baselines. The
comparison is paired and CPU-only: `no_adaptation` 38/55, `info_gain` 25/55,
`public_advisor` 38/55. Report success, final rule accuracy, tool units,
false-alarm rate and steps. A useful result is not “beats 38/55 by one”; it is
“identifies the 10 movement-equivalent hazard/sign episodes without losing the
base episodes or exceeding the 3-unit tool budget.”

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
