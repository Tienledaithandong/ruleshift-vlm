"""Stage one deterministic 55-episode Kaggle chunk kernel.

Kaggle kernel metadata has no environment-variable field, so each chunk gets a
copy of ``kaggle/kernel/run_grid_chunk.py`` with its map index written into the
script. This avoids the silent failure where every chunk would run map 0.

Usage:
    python scripts/stage_grid_chunk.py --map-index 0
    python -m kaggle kernels push -p kaggle_stage/grid_chunks/m000 \
        --accelerator NvidiaTeslaT4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_TEMPLATE = ROOT / "kaggle" / "kernel" / "run_grid_chunk.py"
DEFAULT_PREFIX = "tiennamlee/ruleshift-grid-chunk"
DATASET = "tiennamlee/ruleshift-source"
# Pinned to the image that produced the successful real smoke run.
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@"
    "sha256:fc720c15545eb096a510e1f085b17fc118387a007599424d953a4f327b71cacf"
)
ENV_LINE = 'map_index = int(os.environ.get("MAP_INDEX", "0"))'


def render_kernel(map_index: int) -> str:
    """Return the chunk kernel with a literal, validated map index."""

    if not 0 <= map_index < 100:
        raise ValueError(f"map index must be in [0, 99], got {map_index}")
    text = KERNEL_TEMPLATE.read_text(encoding="utf-8")
    if text.count(ENV_LINE) != 1:
        raise SystemExit(
            "chunk kernel must contain exactly one MAP_INDEX environment line"
        )
    return text.replace(ENV_LINE, f"map_index = {map_index}")


def build_stage(map_index: int, out: Path) -> Path:
    """Write the kernel script and metadata for ``map_index`` and return it."""

    rendered = render_kernel(map_index)
    chunk_id = f"m{map_index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_grid_chunk.py").write_text(rendered, encoding="utf-8")
    metadata = {
        "id": f"{DEFAULT_PREFIX}-{map_index:03d}",
        "title": f"RuleShift grid chunk {chunk_id}",
        "code_file": "run_grid_chunk.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": DOCKER_IMAGE,
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-index", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: kaggle_stage/grid_chunks/mNNN",
    )
    args = parser.parse_args()
    out = args.out or ROOT / "kaggle_stage" / "grid_chunks" / f"m{args.map_index:03d}"
    stage = build_stage(args.map_index, out)
    metadata = json.loads((stage / "kernel-metadata.json").read_text(encoding="utf-8"))
    print(f"staged chunk {args.map_index:03d} in {stage}")
    print(f"kernel id: {metadata['id']}")
    print(
        "next: python -m kaggle kernels push -p "
        f"{stage.relative_to(ROOT)} --accelerator NvidiaTeslaT4"
    )


if __name__ == "__main__":
    main()