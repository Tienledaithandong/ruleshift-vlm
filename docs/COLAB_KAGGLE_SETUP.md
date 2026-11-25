# Hosted runtime preparation

The [research checkpoint](RESEARCH_STATUS.md) describes the archived Kaggle
public-text pilot. The CPU suite and artifact review work on a local machine.
Remote execution is disabled in the checked-in runtime configuration.

## Kaggle source and execution

The script kernel copies a private source **dataset**, independent of the
GitHub clone path. Model weights come from the pinned Hugging Face revision in
`configs/models.yaml`. A GitHub token is unnecessary for the dataset kernel.
Keep Kaggle credentials outside this repository.

For a future approved pilot:

1. Set `execution.allow_remote_run` in `configs/runtime.yaml` for the run snapshot. On the submitting machine, check `python scripts/kaggle_preflight.py --strict`. It checks local CLI/credential presence and the opt-in setting. The accelerator belongs to the target runtime.
2. Run `python scripts/stage_kaggle_dataset.py` to recreate the local `kaggle_stage/ruleshift-source` snapshot. Review it and version that dataset before submitting the kernel.
3. `kaggle/kernel/kernel-metadata.json` specifies the source dataset and enables the accelerator. Submitting this kernel starts a remote job. The kernel installs the pilot dependencies and requests the pinned checkpoint download explicitly.
4. Save artifacts in a new attempt directory. Check output provenance and the Week 4 diagnostics listed in the [runbook](RUNBOOK.md).

## Colab

`notebooks/colab_setup.ipynb` checks a source checkout and the CPU suite.
`notebooks/colab_pilot.ipynb` is the real P1 experiment. For a future approved
run, use an accessible checkout, set the runtime opt-in in that checkout and
install the VLM extra. The pilot cell uses the pinned revision and explicitly
allows a checkpoint download. The primary P0 input still requires a validated
RGB path.

Both platforms use `scripts/run_real_pilot.py`. A pilot uses development inputs
and the draft test configuration. Locked evaluation is a later stage.
