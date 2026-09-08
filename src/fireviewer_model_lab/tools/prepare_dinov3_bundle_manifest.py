"""Adapt the existing FireWarning smoke-segmentation bundle to DINOv3.

The source bundle already contains image/mask pairs, split groups, provenance,
and a strong/weak annotation label. This adapter adds the DINOv3 multitask
fields (mask digest, mask-derived point, and explicit visual abstention) without
recreating or mutating the source payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_anchor(path: Path) -> tuple[float, float] | None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Pillow is required to adapt the segmentation bundle") from exc
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
        height, width = array.shape
        ys, xs = np.nonzero(array > 0)
    if not len(xs):
        return None
    return (
        float(xs.mean()) / width,
        float(ys.mean()) / height,
    )


def adapt(bundle_root: Path, output: Path) -> dict[str, Any]:
    source_root = bundle_root / "sources" / "boreal-forest-fire-segmentation-v1"
    source_manifest = source_root / "manifest.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    strength_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for line_number, line in enumerate(source_manifest.read_text(encoding="utf-8").splitlines(), 1):
        source = json.loads(line)
        sample_artifact = source_root / source["artifact"]["path"]
        sample = json.loads(sample_artifact.read_text(encoding="utf-8"))
        image = sample["image"]
        annotation = sample["annotation"]
        image_path = source_root / image["path"]
        mask_path = source_root / annotation["path"]
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing payload at source line {line_number}")
        anchor = _mask_anchor(mask_path)
        split = str(source["split"])
        strength = str(sample["annotation_strength"])
        strength_counts[strength] += 1
        split_counts[split] += 1
        rows.append(
            {
                "sample_id": source["sample_id"],
                "source_id": source["source_id"],
                "source_revision": "wildfire-smoke-segmentation-v1",
                "split": split,
                "split_group": source["split_group"],
                "image_relpath": image["path"],
                "image_sha256": image["sha256"],
                "mask_relpath": annotation["path"],
                "mask_sha256": annotation["sha256"],
                "mask_quality": "human_strong" if strength == "strong" else "sam_weak",
                "annotation_strength": strength,
                "license": source["license"],
                "sample_validation_status": "source_provided",
                "anchor_points": (
                    [{"kind": "smoke_centroid", "x": anchor[0], "y": anchor[1]}]
                    if anchor is not None
                    else []
                ),
                "visual_abstention_reason": (None if anchor is not None else "empty_smoke_mask"),
                "is_operational_incident": False,
                "source_line": line_number,
            }
        )
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_bundle": str(bundle_root.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "annotation_strength_counts": dict(sorted(strength_counts.items())),
        "promotion_ready": False,
        "promotion_blockers": [
            "weak_sam_masks_present",
            "smoke_only_point_supervision",
            "human_review_required",
        ],
    }
    (output / "adaptation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt the FireWarning smoke bundle to DINOv3")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adapt(args.bundle_root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
