# Gemini RuleShift-Grid prototype

These four scripts contain the exploratory gridworld from the Gemini
conversation. The main Gate/Lock benchmark lives in `src/ruleshift`. The two
environments use different actions, rewards, shift processes and evaluation
protocols.

## Files

- `ruleshift_env.py` — RuleShift-Grid v2 Gymnasium environment and Week 1
  baselines.
- `run_week2_ensemble.py` — transition MLP ensemble and residual/CUSUM
  comparisons.
- `week3_vlm_system.py` — pixel/Set-of-Mark rendering, compound rule,
  reasoner wrapper, and SQLite cache.
- `run_week4_closed_loop.py` — combined benchmark runner.

Imports work from this directory and through the
`prototypes.gemini_ruleshift_grid` namespace from the repository root.
The simulator uses Gymnasium and NumPy, the renderer also uses Pillow, and
the ensemble and combined runner import PyTorch. To check all four modules
from the repository root:

```powershell
python -m compileall -q prototypes/gemini_ruleshift_grid
python -c "from prototypes.gemini_ruleshift_grid import ruleshift_env, run_week2_ensemble, week3_vlm_system, run_week4_closed_loop"
```

The Week 2 and Week 4 script entry points train ensembles on CPU. The Week 3
demo uses an in-process heuristic and writes a PNG and SQLite cache in its
working directory. The combined runner also defaults to this mock. Mock
decisions and external API responses use separate cache keys.

## Scientific scope

- The Week 1 planner and belief baselines inspect the simulator transition
  model. The oracle and oracle-advice policies also read the active rule.
  Their performance describes model-informed or privileged references.
- The mock reads symbolic public transitions and ignores image pixels.
  Its calls measure the reasoner interface, and its actions provide no
  evidence about perception or a frozen VLM.
- The combined demo compares an uncapped per-step caller against a gated
  caller. It reports success, primitive steps and calls separately.
  Differences in success do not establish a benefit at matched budget.
- The detector evaluation fixes a shift time and conditions on episodes
  surviving beyond it. Its AUROC and alarm delays concern that selected
  population. Independent evaluation on varied shift times and complete
  episodes is needed for population-level claims.
- The compound mechanic uses the same axis remap as R1 (UP ↔ LEFT,
  DOWN ↔ RIGHT) together with R3 tile semantics. A CPU smoke check verifies
  this transition, feature shape, rendering and mock cache replay. Ensemble
  training quality, real API responses and VLM performance remain unverified.

