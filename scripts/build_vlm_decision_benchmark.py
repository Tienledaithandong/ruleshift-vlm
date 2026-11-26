"""Build the CPU-only VLM decision-development benchmark.

This script never loads a VLM, never calls a model, and never touches a reserved
grid map. It writes public contexts, evaluator-side labels, an audit report, and
optional paired visual cards. Running it twice with the same source must produce
byte-identical JSONL and the same digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ruleshift.evaluator.vlm_decision import (
    DecisionLibrary,
    audit_library,
    build_development_library,
    reference_for,
)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return _sha256_bytes(data)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode(
        "utf-8"
    )
    path.write_bytes(data)
    return _sha256_bytes(data)


def _context_payload(library: DecisionLibrary) -> list[dict[str, Any]]:
    return [context.to_public_payload() for context in library.contexts]


def _label_payload(library: DecisionLibrary) -> list[dict[str, Any]]:
    return [label.as_payload() for label in library.labels]


def _safe_name(context_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in context_id
    )


def _image_rows(
    library: DecisionLibrary, image_dir: Path
) -> list[dict[str, Any]]:
    from PIL import Image

    from ruleshift.evaluator.vlm_decision_visual import (
        blue_role_perturbation,
        render_decision_card,
        visual_digest,
    )

    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for context in library.contexts:
        base = render_decision_card(context, variant="base")
        nuisance = render_decision_card(context, variant="nuisance")
        stem = _safe_name(context.context_id)
        base_path = image_dir / f"{stem}.base.png"
        nuisance_path = image_dir / f"{stem}.nuisance.png"
        Image.fromarray(base).save(base_path)
        Image.fromarray(nuisance).save(nuisance_path)
        row: dict[str, Any] = {
            "context_id": context.context_id,
            "base_path": str(base_path),
            "nuisance_path": str(nuisance_path),
            "base_digest": visual_digest(base),
            "nuisance_digest": visual_digest(nuisance),
            "nuisance_same_context": True,
        }
        observed = {
            item.value for item in context.evidence if item.kind == "blue"
        }
        diagnostic_roles = [
            role for role in ("goal", "hazard", "safe") if role not in observed
        ]
        for role in diagnostic_roles:
            try:
                diagnostic = blue_role_perturbation(context, role)
            except ValueError:
                continue
            image = render_decision_card(diagnostic, variant="base")
            path = image_dir / f"{stem}.diag-blue-{role}.png"
            Image.fromarray(image).save(path)
            label = reference_for(diagnostic)
            row["diagnostic"] = {
                "role": role,
                "context_id": diagnostic.context_id,
                "path": str(path),
                "digest": visual_digest(image),
                "best_action": label.best_action,
                "decision_class": label.decision_class,
            }
            break
        rows.append(row)
    return rows


def build(out: Path, *, with_images: bool) -> dict[str, Any]:
    library = build_development_library()
    out.mkdir(parents=True, exist_ok=True)
    contexts_path = out / "contexts.jsonl"
    labels_path = out / "labels.jsonl"
    audit_path = out / "audit.json"
    manifest_path = out / "manifest.json"
    contexts_hash = _write_jsonl(contexts_path, _context_payload(library))
    labels_hash = _write_jsonl(labels_path, _label_payload(library))
    audit = audit_library(library)
    audit_hash = _write_json(audit_path, audit)
    manifest: dict[str, Any] = {
        "name": "vlm-decision-development",
        "split": "development",
        "reference_version": library.manifest["reference_version"],
        "n_contexts": len(library.contexts),
        "library_manifest": dict(library.manifest),
        "files": {
            "contexts.jsonl": contexts_hash,
            "labels.jsonl": labels_hash,
            "audit.json": audit_hash,
        },
        "public_fields_note": (
            "contexts.jsonl contains only public map, rule, prior, posterior, "
            "evidence, cost, budget, and feasible-action facts; labels.jsonl is "
            "evaluator-side and must not enter a prompt or cache key."
        ),
        "reserved_maps_used": False,
        "vlm_inference": False,
        "images": None,
    }
    if with_images:
        manifest["images"] = _image_rows(library, out / "images")
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/vlm_decision_benchmark"))
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    manifest = build(args.out, with_images=not args.no_images)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
