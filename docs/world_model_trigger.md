# Optional uncertainty-triggered slow reasoning

This is an additive research extension inspired by the Gemini notes. The
repository's existing value-aware probing and VLM evaluation remain the main
pipeline. The new `ruleshift.agents.change_detection` module provides a
public-signal CUSUM building block; it does not yet connect a learned world
model to the closed-loop runner.

## What is available

- `standardized_residual_score(...)` converts a model's predicted transition
  mean and variance plus an observed transition into an RMS standardized
  residual.
- `CUSUMDetector` accumulates evidence over steps and reports when a caller's
  calibrated threshold is exceeded. The caller can use that event to request a
  slow reasoner.
- `calibrate_cusum_threshold(...)` sets an episode-level threshold from
  no-shift development episodes. With too few calibration episodes to support
  the requested false-alarm rate, it returns infinity instead of implying a
  reliable operating point.

The module uses only the Python standard library and accepts agent-provided
predictions and public outcomes. It does not import an environment, evaluator,
PyTorch, or a VLM backend.

## Planned experiment, subject to validation

1. Fit a small transition predictor on development data only, initially on
   CPU or one 24 GB GPU. Keep the VLM frozen and do not fine-tune it.
2. Compare raw prediction error, standardized residual, and ensemble
   disagreement on held-out no-shift and mechanics-shift episodes.
3. Calibrate the CUSUM threshold only from no-shift development episodes,
   then report false alarms, detection delay, success/return, VLM calls,
   tokens, and latency together.
4. Add an event-triggered VLM method to the runner only after the current
   public observation/history contract can provide the transition inputs it
   needs and the trigger has a validated operating point. Keep per-episode
   model-call caps matched across methods.

These are experiment steps, not expected outcomes. No detector, VLM, GPU run,
or result is claimed by adding this utility.

## Standalone Gemini prototype files

The full code from the Gemini conversation is preserved under
`prototypes/gemini_ruleshift_grid/` as four standalone files. That prototype
defines a second environment and action protocol, with physical movement,
fixed color tiles, and `R4_COMPOUND` rule composition. The main project still
uses public macro actions, separate Gate/Lock and Bridge/Hazard families, and
a closed candidate set. Keeping the scripts isolated avoids replacing the
existing implementation or bypassing its permission and logging contracts.

The repository already has a public-only response cache, structured model
outputs, value-aware selection, call caps, and an optional frozen-model
backend. The prototype's image keyframe strip and open-ended `commit_rule`
tools are not connected to the main runner: the current model request contains
prompt text and an image hash, not image bytes, and the current public API does
not permit open-set rule creation. Integrating them would require a separate
benchmark revision.

