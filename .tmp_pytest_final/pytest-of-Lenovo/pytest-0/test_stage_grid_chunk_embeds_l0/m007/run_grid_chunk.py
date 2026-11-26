"""Kaggle GPU kernel: one 55-episode dry chunk of the frozen grid run.

A chunk is one map x five run seeds x eleven scenarios (55 episodes; the
measured 8.85 s/decision makes the worst case near 8.1 GPU-hours, not 2.2). Ten
such chunks form a weekly slice of ten maps. Do not set MAP_INDEX by hand:
scripts/stage_grid_chunk.py embeds the literal index in each staged copy and
fails loudly if the environment form is still present. Artifacts are stamped
with mNNN so chunks can be merged without overwriting each other.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/kaggle/working/ruleshift-vlm")


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


matches = sorted(Path("/kaggle/input").rglob("run_grid_real_pilot.py"))
if not matches:
    print("kaggle input tree:", [p.name for p in Path("/kaggle/input").iterdir()], flush=True)
    raise SystemExit("source dataset not found under /kaggle/input")
SOURCE = matches[0].parent.parent
print("source dataset:", SOURCE, flush=True)

if PROJECT.exists():
    shutil.rmtree(PROJECT)
shutil.copytree(SOURCE, PROJECT)

marker = PROJECT / "SOURCE_COMMIT"
if not marker.exists():
    raise SystemExit("SOURCE_COMMIT marker missing: source dataset is not traceable")
stamp = json.loads(marker.read_text(encoding="utf-8"))
commit = str(stamp.get("commit", ""))
print("source commit:", commit, stamp.get("short_commit"), flush=True)
print("source subject:", stamp.get("subject"), flush=True)
if not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise SystemExit(f"SOURCE_COMMIT does not hold a full commit id: {commit!r}")
expected = os.environ.get("RULESHIFT_EXPECTED_COMMIT", "").strip()
if expected and expected != commit:
    raise SystemExit(
        f"source commit mismatch: dataset has {commit}, expected {expected}"
    )
print("expected commit check:", expected or "skipped (RULESHIFT_EXPECTED_COMMIT unset)", flush=True)

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))

run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
run(["nvidia-smi"])

os.environ["RULESHIFT_KAGGLE"] = "1"

map_index = 7
if not 0 <= map_index < 100:
    raise SystemExit(f"MAP_INDEX must be in [0, 99], got {map_index}")
chunk_id = f"m{map_index:03d}"
report = "/kaggle/working/reports/grid_chunks/grid_real_pilot.md"
out = "/kaggle/working/results/grid_chunks"
decisions = f"{out}/grid_real_decisions.{chunk_id}.jsonl"
evaluator = f"{out}/grid_real_evaluator_only.{chunk_id}.jsonl"
chunk_report = f"/kaggle/working/reports/grid_chunks/grid_real_pilot.{chunk_id}.md"

run(
    [
        sys.executable,
        "scripts/run_grid_real_pilot.py",
        "--model-key", "qwen2.5-vl-3b",
        "--device", "cuda",
        "--maps", "100",
        "--seeds", "5",
        "--map-offset", str(map_index),
        "--map-count", "1",
        "--max-episodes", "55",
        "--chunk-id", chunk_id,
        "--out", out,
        "--report", report,
    ]
)

decision_rows = [
    json.loads(line)
    for line in Path(decisions).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
evaluator_rows = [
    json.loads(line)
    for line in Path(evaluator).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
scenario_ids = {str(row.get("scenario_id", "")) for row in evaluator_rows}
if len(evaluator_rows) != 55 or len(scenario_ids) != 55:
    raise SystemExit(
        f"chunk {chunk_id} incomplete: {len(evaluator_rows)} evaluator rows, "
        f"{len(scenario_ids)} unique scenarios (expected 55)"
    )
if not decision_rows:
    raise SystemExit(f"chunk {chunk_id} has no public decisions")
print(f"chunk {chunk_id}: {len(evaluator_rows)} episodes, {len(decision_rows)} rows", flush=True)
print("GRID CHUNK COMPLETE (scaffold, not a scientific finding)", flush=True)
for path in (chunk_report, decisions, evaluator):
    print(path, Path(path).exists(), flush=True)
