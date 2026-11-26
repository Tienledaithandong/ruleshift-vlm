# Kaggle VLM Smoke Runbook

**Status: COMPLETE.** Kernel version 4 executed the frozen smoke on a private Kaggle T4, issued 12 real A1 requests, and stopped at the frozen A1 feasibility gate. A2, A3, and controls did not run. The scientific result is recorded in [VLM_DECISION_SMOKE_RESULTS.md](VLM_DECISION_SMOKE_RESULTS.md); this document records the execution path, artifacts, and infrastructure fixes.

**Scope:** one private Kaggle kernel, one frozen checkpoint, at most 48 model requests. No prompt, parser, scorer, gate, model, revision, precision, quantization, offload, or arm change is permitted here. The Kaggle run was the same frozen experiment as the local attempt, not a new pilot.

## 1. Why this execution path exists

The local laptop run at `results/vlm_decision_smoke/real/` is preserved as a historical infrastructure failure. It stopped during model load with `cudaErrorDevicesUnavailable` before any model request. The laptop GPU was not retried. The human decision was **GO FOR KAGGLE GPU EXECUTION**.

## 2. Frozen inputs

| Item | Value |
|---|---|
| Source commit | `a51929d3acadc1f3d9b8d404081e31f94ee82afd` |
| Merge base / v0.5 reference | `f4b43a4ede7f76ab9d5b1f45178fe648ae419042` |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Checkpoint files | 11 |
| Checkpoint bytes | 7,520,892,432 |
| Aggregate checkpoint hash | `sha256:02bdb4ccb82eb3aceaed1608e1e7c55a81ab3e6125a20f6cccc690bbf1254c75` |
| Request manifest | `sha256:5558a8c4c75449887c870eef67476bde0ae2caa8674b9071dff5300655ddf4a0` |
| Reference labels | `sha256:90bb55c07cfefdcf03e28573edbe8c7d56bdb8947c2b68f21b272d5116184551` |
| Freeze | `sha256:7ce10ec762564ce25ae738119514031d9b2552a2550a926473b1fb0e4b404eaa` |
| Frozen request order | A1-01..12, A2-01..12, A3-01..12, N1..N6, D1..D6 |

Only A1 (prior/evidence text), A2 (posterior text), and A3 (posterior RGB card) were authorized. A3 must carry the actual 512x320 RGB image; the text must not contain the numerical posterior. Controls were six nuisance and six semantic-diagnostic requests.

## 3. Private Kaggle artifacts

| Artifact | Reference | Notes |
|---|---|---|
| Source dataset | `tiennamlee/ruleshift-vlm-smoke-source` | Exact frozen source plus `SOURCE_MANIFEST.json`; **158 files** |
| Model dataset | `tiennamlee/ruleshift-qwen25vl3b-frozen` | Exact 11 checkpoint files; aggregate hash verified before upload |
| Dedicated kernel | `tiennamlee/ruleshift-frozen-vlm-smoke-48` | Private script kernel; GPU enabled; internet enabled for dependency setup only |
| Kaggle URL | https://www.kaggle.com/code/tiennamlee/ruleshift-frozen-vlm-smoke-48 | Version 4 is the executed version |
| Accelerator | `NvidiaTeslaT4` | T4 x2; 15,636,037,632 bytes per GPU; only device 0 used |

Kaggle exposed `bf16_native: true` on the T4, contrary to the initial assumption. The frozen loader still used `torch.bfloat16`; no precision change was made. The selected GPU had 15,526,068,224 bytes free before load.

Observed stack: Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, transformers 5.20.2, accelerate 1.15.0, Pillow 13.0.0, NumPy 2.1.3, driver 580.178.04.

## 4. Kernel stages

1. Locate the source and model roots under `/kaggle/input`; support extracted datasets and a single top-level directory.
2. Verify all 158 staged source files against `SOURCE_MANIFEST.json`; skip generated `__pycache__`, `.pytest_cache`, and `.pyc` entries.
3. Copy the source to `/kaggle/working/ruleshift-vlm-smoke`.
4. Normalise the checkpoint to a directory literally named `66285546d2b821cf421d4f5eb2576359d3770cd3` via symlinks to the flat model dataset. The frozen verifier requires that basename.
5. Set `RULESHIFT_SNAPSHOT_PATH`, `RULESHIFT_FROZEN_DIR`, `PYTHONPATH`, `PYTHONHASHSEED=0`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_OFFLINE=1`.
6. Run CPU preflight: 48 requests, 24 images, leakage scan, processor image-forwarding checks.
7. Run the 48-request stub dry run with zero model calls.
8. Run the real executor. The real executor verifies the checkpoint again, loads bf16 on CUDA, writes raw responses, stops on the A1 gate, and exits 1 as a hard stop.
9. Recompute CPU baselines on the same 12 contexts and write the executor manifest.

### Portability shim

`sitecustomize.py` is generated only inside the Kaggle working copy. It makes three in-memory path/byte-identity adjustments:

- Normalise Windows backslash keys in `freeze.json` source hashes to Linux `/` for lookup.
- Point `verify_checkpoint_manifest` at the normalised snapshot directory.
- Make the frozen PNGs authoritative for image requests: load `results/vlm_decision_smoke/frozen/images/<request_id>.png` into `item.image`, keep the pixel digest strict, and use the frozen PNG byte hash for `image_png_sha256` when rebuilding the manifest record.

This is infrastructure-only. It does not alter prompts, labels, the request manifest, the image pixels, the parser, the scorer, or the gates. The frozen PNGs themselves are unchanged. The shim was necessary because Kaggle Pillow 13.0.0 encoded PNG bytes differently from the local Windows encoder even when the decoded pixels matched.

## 5. Version history

| Version | Outcome | Calls | Record |
|---|---:|---:|---|
| 1 | Source manifest included Kaggle-excluded bytecode | 0 | Infrastructure failure |
| 2 | Windows backslash freeze keys did not resolve on Linux | 0 | Infrastructure failure |
| 3 | Rebuilt manifest differed at preflight | 0 | Infrastructure failure |
| 4 | Preflight PASS; model loaded; A1 executed; A1 gate failed | 12 | Scientific stop |

Versions 1-3 contain no model requests. They are described in [VLM_DECISION_SMOKE_FAILURE_ANALYSIS.md](VLM_DECISION_SMOKE_FAILURE_ANALYSIS.md). Version 4 is the only run with model behavior.

## 6. Launch and monitoring

From the repository root, with the Kaggle credentials already configured:

```powershell
python -m kaggle kernels push -p kaggle_stage\vlm_smoke_kernel --accelerator NvidiaTeslaT4 -t 10800
python -m kaggle kernels status tiennamlee/ruleshift-frozen-vlm-smoke-48
python -m kaggle kernels output tiennamlee/ruleshift-frozen-vlm-smoke-48 -p results\kaggle_smoke_output
```

The output download command may report a nonzero process exit because the kernel log contains Unicode and the Windows console uses a legacy code page. Set `$env:PYTHONIOENCODING='utf-8'` and `[Console]::OutputEncoding=[System.Text.Encoding]::UTF8` before downloading. The downloaded artifacts themselves are UTF-8.

Kernel log is also downloaded as `results/kaggle_smoke_output/ruleshift-frozen-vlm-smoke-48.log`.

## 7. Artifacts and accounting

| Artifact | Location |
|---|---|
| Authoritative raw run | `results/kaggle_smoke_output/results/vlm_decision_smoke/real/real/` |
| Preflight | `results/kaggle_smoke_output/results/vlm_decision_smoke/kaggle_preflight/` |
| Stub dry run | `results/kaggle_smoke_output/results/vlm_decision_smoke/kaggle_dry_run/` |
| CPU baselines | `results/kaggle_smoke_output/results/vlm_decision_smoke/cpu_baselines.json` |
| Hardware | `results/kaggle_smoke_output/results/vlm_decision_smoke/kaggle_hardware.json` |
| Executor manifest | `results/kaggle_smoke_output/KAGGLE_EXECUTION_MANIFEST.json` |
| Local failed run | `results/vlm_decision_smoke/real/` (preserved, not overwritten) |

Run facts: 48 planned; 12 attempted; 12 completed; 12 model calls; 0 retries; 0 request failures; 12/12 valid A1; 12 COMMIT, 0 PROBE_SIGN, 0 PROBE_BLUE. A1 gate failed on `non_constant`. A2/A3/controls: 0. GPU generation: 24.1888 s. Model load: 9.6181 s wall. Checkpoint verification: 42.9810 s. Peak CUDA allocated: 7,789,882,368 bytes. Peak CUDA reserved: 8,032,092,160 bytes.

`KAGGLE_EXECUTION_MANIFEST.json` reports null/zero output fields because its summary helper looked at `real/` rather than `real/real/`. The raw `summary.json` and `request_accounting.json` are authoritative. This is an artifact-collection defect, not a missing run; it does not affect the raw evidence.

Independent checksum verification: 10/10 raw files, 10/10 dry-run files, and 1/1 preflight file passed.

## 8. Stop and reporting rules

- Stop at 48 requests, a failed request, a hard arm gate, a checkpoint/image/leakage failure, or a GPU/model-load failure.
- Do not retry automatically. Do not change the model, revision, precision, prompts, parser, scorer, menu, or gates.
- A hard A1 gate is a scientific stop; record it as such, not as an infrastructure failure and not as a completed multi-arm smoke.
- A Kaggle infrastructure failure before inference is infrastructure evidence, not VLM decision evidence.
- Report exact model-call count including failures; retries are zero by design.

## 9. Cleanup

Temporary staging helpers and local test outputs are not part of the scientific artifacts:

- `.tmp_pathfix/`
- `kaggle_stage/extract_sitecustomize.py`
- `kaggle_stage/make_source_manifest.py`
- `kaggle_stage/summarize_a1.py`
- `kaggle_stage/inspect_summary.py`
- `kaggle_stage/inspect_arms.py`
- `kaggle_stage/a1_table.py`
- `kaggle_stage/show_prompt.py`
- `kaggle_stage/verify_checksums.py`

Do not delete the frozen artifacts, the local failed run, the Kaggle raw run, or the kernel log. Do not commit `results/` or Kaggle credentials.

## 10. Current execution status

Kernel version 4 is COMPLETE at https://www.kaggle.com/code/tiennamlee/ruleshift-frozen-vlm-smoke-48. It ran 12 A1 requests and stopped at the frozen A1 gate. A2/A3/controls did not run. The next decision is in [VLM_DECISION_PILOT_DECISION_PACKET.md](VLM_DECISION_PILOT_DECISION_PACKET.md); the recommendation is STOP/HOLD.
