# RuleShift / Agentic VLM Research — Project Brief

Use this document as the project context before proposing or writing code. It
describes the repository that already exists, its research question, its
interfaces, its limits, and how new work should fit. Treat the source code as
the final authority if this brief and the implementation differ. Do not
silently replace the project with a new prototype.

## 1. Project in one paragraph

This is a CPU-first, reproducible research harness for studying whether a
frozen vision-language model (VLM) can make useful, cost-aware information
gathering decisions when an environment's hidden mechanics may change during
an episode. The benchmark models an agent choosing between a task action and
a probe: a probe consumes real environment steps, returns noisy evidence, and
may improve later decisions. The main scientific target is the value of
probing and structured counterfactual outcome reasoning under matched costs
and model-call budgets. It is not currently a real-time game-playing system,
an open-ended game agent, or a trained world-model controller.

## 2. Research question and ambition

### Current repository question

> When hidden environment dynamics change without warning, do frozen
> vision-language agents choose probes whose control value justifies their
> cost, and can structured counterfactual outcome reasoning—without access to
> the environment's generative model—reduce probe-decision regret and improve
> closed-loop return?

The agent should choose *whether information is worth acquiring*, given its
public history, the available actions, and their public costs. Probes are not
free API calls: they consume primitive environment steps in the same reward
stream as task actions. The evaluator may calculate the true value of a probe
for analysis, but the agent may not see that value or the simulator's hidden
transition model.

### Longer-term research direction

The user's broader interest is Agentic VLMs for visual games: multimodal
reasoning, tool use, adaptation to unseen mechanics, and a fast controller
with occasional slower reasoning. Uncertainty-triggered world-model reasoning,
sparse tool/critic calls, and distillation are possible future directions.
They are **not yet the implemented main method or a result**. Treat them as
extensions to evaluate against the current question, not as permission to
replace the benchmark, interfaces, or research question without saying so.

## 3. Benchmark and environment

The project uses small controlled Gymnasium environments so hypotheses can be
tested cheaply and reproducibly before attempting commercial games or large
models.

### Family A — Gate/Lock

- The agent repeatedly chooses a task commitment (for example,
  `COMMIT_RED`) or a probe (for example, `PROBE_BLUE`).
- The hidden mechanic determines which color is accepted. The old default is
  red; at most one unannounced shift can change it during an episode. A valid
  no-shift episode is also included.
- A successful commitment earns the task reward. A failed commitment and its
  lockout consume primitive steps. A probe travels to a public station, gets
  noisy feedback, and returns; it has no hidden free reward.
- All agents use the same public macro-action menu and shared executor.

### Family B — Bridge/Hazard

- The agent chooses a safe route, a risky bridge, or a sensing-station probe.
- Stable and slippery regimes change the *transition probability* of a
  crossing, rather than merely relabeling a reward. Failures consume recovery
  time.
- This is a second mechanic family for checking transfer, not an invitation
  to introduce an unrelated action system.

The canonical implementations are under `src/ruleshift/envs/`. The
Gemini-generated grid with physical `UP/DOWN/LEFT/RIGHT/INTERACT` controls is
a separate prototype, not the canonical environment.

## 4. Current architecture

```text
configs/                 development, model, runtime, and draft test settings
docs/                    problem, permissions, benchmark, setup, and runbook
research/                literature and ablation planning
src/ruleshift/public_api/ agent-visible observations, schemas, and action ids
src/ruleshift/envs/      hidden simulator mechanics and public renderers
src/ruleshift/agents/    public-only policies, prompts, backends, cache, selectors
src/ruleshift/evaluator/ privileged references, scoring, and experiment runners
scripts/                 preflight, smoke, closed-loop, audits, and reports
tests/                   contracts, environment, leakage, and method checks
prototypes/              isolated experiments that are not the main implementation
```

The separation between `agents` and `evaluator` is a core research constraint.
The environment/evaluator owns true mechanics, transition probabilities,
shift timing, exact posterior labels, and exact action values. The agent gets
only a `PublicObservation`, public action outcomes/history, candidate-rule
descriptions, and public cost magnitudes. Agent code must not import
`ruleshift.envs` or `ruleshift.evaluator`, inspect hidden state, or simulate
counterfactual actions using the real environment.

The public macro protocol includes the Family A actions
`COMMIT_RED/BLUE/GREEN`, `PROBE_RED/BLUE/GREEN`, `WAIT`, and
`GO_TO_STATION`; Family B advertises its own subset. A macro's travel, probe,
lockout, and step costs are counted consistently. Public costs are
non-negative magnitudes and must be converted into signed reward costs once.

## 5. Implemented capabilities

The repository already contains:

- Gymnasium Family A and Family B environments, deterministic semantic random
  event streams, renderers, public/evaluator trace separation, and a common
  macro contract.
- A public-only VLM request/output schema, deterministic stub backend,
  optional frozen Transformers backend, response cache keyed by the full
  public request context, and deterministic invalid-output fallback.
- History-based consistency updates, direct and cost-aware selectors,
  one-step continuation estimates, DIVA-style adapted and self-consistency
  baselines, and M1 ablations.
- Evaluator-only Bayes/reference calculations, an exact solver for tiny
  microtasks, a clearly labelled approximate reference for larger tasks,
  paired scenario-level intervals, closed-loop and held-out evaluation,
  diagnostics, and stress sweeps.
- `src/ruleshift/agents/change_detection.py`: an additive CPU-only
  standardized-residual/CUSUM utility. It is **not connected to a trained
  world model or the experiment runner**.

These are code capabilities, not evidence that the research hypothesis has
been supported. Stub outputs are synthetic plumbing checks, never VLM results.

## 6. Important current limitation: visual input

The environment can render RGB observations, and public observations carry an
image digest. However, the current VLM request/backend path sends prompt text;
the image digest is not image content. The current real pilot is therefore a
text-input pilot, not an end-to-end visual reasoning evaluation. Do not claim
the existing model path consumes images.

If adding real visual input, design it explicitly across the public
observation contract, request fingerprint/cache, backend processor, prompt,
permissions, logging, and tests. The image must be derived only from public
state. Do not just add a `PIL.Image` to an agent while leaving cache identity
or the backend text-only.

## 7. Current research and compute status

- A real public-text Qwen pilot ran on Kaggle. The v12 archive records 81
  decisions, 27 distinct requests and a constant action. The revised offline
  diagnostic, P0 and paired real validation remain pending. See
  [research progress](docs/RESEARCH_STATUS.md).
- A deterministic stub is used for code paths and synthetic dry runs. Never
  report its behavior as model evidence.
- The test configuration is intentionally `DRAFT_NOT_FROZEN`. Do not create
  or freeze held-out test scenarios before validation and a recorded code
  revision.
- Resource assumptions: ordinary Kaggle/Colab sessions are the primary
  accessible compute; at most one lab GPU with about 24 GB VRAM may be
  available. Do not assume multiple GPUs, long RL runs, or large-scale VLM
  fine-tuning.
- Prefer small CPU experiments, frozen pretrained models, cached outputs,
  modest context sizes, matched call caps, and short validation runs. A small
  learned transition model is only a proposed experiment and must have an
  explicit data split and cost budget.
- `configs/models.yaml` records the pinned Qwen anchor and the unpinned
  alternative. The local Kaggle archives document which model actually ran.

## 8. Metrics and experimental discipline

Use metrics that answer the actual question and report costs beside quality:

- total return / task completion;
- probe decision regret where exact labels exist;
- detection/adaptation delay when applicable;
- false probe or false alarm rate on no-shift episodes;
- primitive steps, probe count, VLM calls, input/output tokens, and latency;
- paired confidence intervals clustered by independent scenario, not by
  treating frames or seeds from one scenario as independent.

Keep public agent traces and evaluator-only labels in separate files. Tune on
development/validation scenarios only. Report negative results and all
relevant ablations. Do not assume a desired trend (for example, that CUSUM
beats disagreement) before measuring it.

## 9. Status of the Gemini code in this repository

`prototypes/gemini_ruleshift_grid/` contains four scripts copied from the
Gemini conversation: a second grid environment, an MLP ensemble script, an
image/SoM and mock-VLM script, and a separate closed-loop runner. They are
preserved additively and do not replace the canonical code under
`src/ruleshift/`. Their action protocol, environment mechanics, and
open-ended `R4_COMPOUND` rule composition differ from the current benchmark.
The scripts have not been validated as part of the main project; do not mix
their results with the canonical reports or assume the mock backend proves
VLM performance.

Ideas from that conversation that may be developed as extensions include
standardized transition residuals, sequential CUSUM, sparse slow-reasoner
calls, real image keyframe strips, and compositional mechanics. Before
integrating an idea, explain how it maps to the current public macro contract
and research question. If the idea requires a genuinely different benchmark,
keep it as an isolated prototype or propose a deliberate benchmark revision.

## 10. Rules for generating code for this project

When asked to write or modify code, follow these rules:

1. Read the relevant existing modules and tests before editing. Search for an
   existing capability first; do not reimplement caching, metrics, baselines,
   environments, or selectors that already exist.
2. Preserve current files and behavior unless the user explicitly asks for a
   replacement or breaking change. Prefer a small additive module or a
   backward-compatible extension at the canonical interface.
3. Do not create another environment, action vocabulary, or project root for
   the main implementation. A separate prototype is acceptable only when it
   is clearly isolated and documented.
4. Keep agent code public-only. Hidden labels, true transition kernels,
   evaluator beliefs, exact Q values, and shift timestamps stay evaluator
   side. Never leak them through prompts, cache keys, logs, or training data
   presented as agent-observable.
5. Keep claims honest: mark code as implemented, proposed, synthetic, or
   empirically validated. Never insert expected numbers as results.
6. Respect the compute limit: CPU-first, frozen model by default, one GPU up
   to roughly 24 GB when available, no assumed multi-GPU training.
7. When adding an experiment, state its question, inputs, outputs, data split,
   metrics, resource needs, and how it will be compared to existing baselines.
8. Do not automatically run costly downloads, training, API calls, or held-out
   experiments. Make execution steps explicit and keep setup from downloading
   model weights implicitly.

## 11. Recommended project-specific reply when code is requested

Before proposing code, state briefly which existing modules it will extend
and whether the requested idea is already present. Then implement against the
canonical interfaces, preserve unrelated files, and summarize the changed
files, assumptions, and validation status. If a requested design conflicts
with the current research question or permission contract, explain the
conflict and offer a compatible extension path instead of silently building
a second project.
