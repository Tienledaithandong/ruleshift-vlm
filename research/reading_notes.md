# Reading Notes — Week 1

**Verification level:** bibliographic metadata, abstracts and official READMEs
were fetched from the canonical URLs listed in `literature_matrix.csv` on
2026-11-21. **Full texts were not read in Week 1.** Every matrix row is flagged
`METADATA_AND_ABSTRACT_ONLY__FULLTEXT_NOT_READ`. No bibliographic claim, year,
venue, page range or result number was invented. WorldCoder and PoE-World are
mentioned in the master plan but were **not** located or verified; do not cite
them until canonical sources are checked.

---

## 1. DIVA — what it actually claims

From the abstract and official README: LLMs fall into an "information trap",
paying for checks that cannot change their decision or are not worth their
cost. DIVA elicits a small decision model from the LLM (probability of a bad
outcome and how each check would shift it), values each check by the imagined
decision improvement net of its cost, and stops when no check is worth it.
DIVA-U averages over uncertain check strengths; DIVA-DP plans remaining
acquisitions with exact dynamic programming. Training-free. Four domains, 20
LLMs. The synthetic environments give the reliability of each check and an
exact DP optimum; real-data settings use UCI records and SEC filings.

**What matters for us:** the closest conceptual threat. Any claim to "value of
information for an LLM" is already occupied. Our differentiation must be the
*physical control* setting: checks cost primitive steps in the same reward
stream, taking a check changes where the agent can act next, and the law being
probed can change mid-episode. We must implement a DIVA-style adapted baseline
with physical costs rather than claim an exact reproduction.

## 2. ZendoWorld

Interactive visual rule induction: agents see rendered scenes, propose new
scenes, receive feedback and refine a hypothesised logical rule. Reported
findings include that predicting labels for observed examples does not imply
recovering the rule, that perception and induction are distinct bottlenecks
for different agent classes, and that VLM agents propose near-uninformative
experiments. Human data are collected. Official repo uses Blender rendering,
Prolog rules and a game runner.

**What matters for us:** it benchmarks *rule recovery and experiment quality*,
not closed-loop task return under a switching dynamics law. Our probe is a
physical detour with a real step cost, and the "rule" is a transition law that
governs control, not just a label to be named.

## 3. EVOLvE (ICML 2025)

Stateless contextual and non-contextual bandits. Measures LLM exploration via
cumulative regret; improves it with algorithm-guided inference support and
algorithm distillation. Regret is explicitly linked to model size.

**What matters for us:** exploration/regret prior art, but stateless bandits
have no hidden state to track, no physical probe action, and no law that
changes within an episode. We use regret as a *microtask diagnostic*, with
closed-loop return as the primary endpoint.

## 4. Piriyakulkij et al. (NeurIPS 2024)

Natural-language rule inference by doing experiments; LLM plus Monte Carlo
probabilistic inference, online belief updates and information-theoretic
experiment design; human-model comparison on a Zendo-style task. Finding:
fitting human data requires fuzzy probabilistic rules and approximately
Bayesian updates.

**What matters for us:** strong prior art for language-based hypothesis
revision and experiment design. Our candidate laws are given as text; our
contribution must be the value-aware *control* decision, not hypothesis
revision itself.

## 5. One Life to Learn (ICLR 2026)

Learns symbolic world models of stochastic dynamics from limited unguided
interaction in Crafter-OO, using conditionally activated programmatic laws in
a probabilistic programming framework. Metrics: state ranking and state
fidelity, plus planning utility. Exposes a structured symbolic state and pure
transition function.

**What matters for us:** complementary. It *learns* laws from symbolic state;
we assume a small public candidate set and test whether a frozen VLM pays a
physical cost to disambiguate a law that can change mid-episode. Do not claim
world-model learning or open-ended rule discovery.

---

## 6. Gate W1 novelty paragraph (to test, not to assume)

> Unlike visual rule-discovery benchmarks that score rule recovery or
> information gain, and unlike decision-aware information acquisition for
> text-only LLM checks, we measure and seek to improve **value-aware physical
> probing under changing dynamics**: a probe consumes task time in the same
> reward stream, can change the agent's later control options, and the law that
> governs task outcomes may switch once without warning inside the episode. We
> do not claim to be first in value of information; we claim a controlled
> setting where the *control value* of evidence, not its information gain, is
> the quantity that must justify its cost.

This paragraph is a hypothesis about framing. It must be re-checked after full
texts are read and after the Week 6 comparison with the DIVA-style baseline.

## 7. Three hypothetical reviewer objections (write answers before results)

1. **"This is DIVA with pixels and a gridworld."** Response must show that
   actionable control, physical cost in the shared reward stream and
   mid-episode law shifts create a decision problem that DIVA's checks do not
   model — and must present the adapted DIVA baseline with matched budgets.
2. **"The probe decision is fully solvable by a Bayes filter; the VLM is
   redundant."** Response: exact solution is restricted to enumerated
   microtasks and used only as a diagnostic label; the method must not read
   the generative model, and the main claim concerns VLM behaviour under cost
   and noise, not solving the POMDP.
3. **"The effect is an artifact of cost accounting."** Response: report
   cost-once invariants, identical macro executors, matched call caps, and the
   result both with and without the physical-cost component labelled, without
   repositioning the endpoint after seeing numbers.

---

## 8. Sources checked

- arXiv abstract pages for DIVA (2610.32181) and ZendoWorld (2607.08233).
- Official GitHub READMEs for `seunghan96/diva` and `ml-research/ZendoWorld`.
- PMLR proceedings page for EVOLvE (v267, pp. 46346–46376, 2025).
- NeurIPS 2024 proceedings abstract for Piriyakulkij et al.
- ICLR 2026 proceedings abstract for One Life to Learn.

All fetched 2026-11-21. These are pointer-level verifications, not full readings.
