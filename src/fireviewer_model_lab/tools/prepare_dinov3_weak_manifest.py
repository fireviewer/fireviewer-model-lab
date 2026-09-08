"""Materialize an explicit weak-supervision DINOv3 manifest from frozen COCO.

This is a training unblocker, not a replacement for human/SAM masks. Every
record is labelled ``bbox_rasterized_weak`` and downstream promotion must keep
that status visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_mask(path: Path, width: int, height: int, annotations: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Pillow is required to materialize weak DINOv3 masks") from exc
    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        category = int(annotation["category_id"]) + 1
        left = max(0, min(width - 1, round(x)))
        top = max(0, min(height - 1, round(y)))
        right = max(left, min(width - 1, round(x + box_width)))
        bottom = max(top, min(height - 1, round(y + box_height)))
        draw.rectangle((left, top, right, bottom), fill=category)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _read_coco(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"COCO document is not an object: {path}")
    return value


def materialize(coco_root: Path, output_root: Path) -> dict[str, Any]:
    coco_root = coco_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and (output_root / "manifest.jsonl").is_file():
        raise FileExistsError(f"refusing to overwrite completed output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    partial_manifest = output_root / "manifest.partial.jsonl"
    if partial_manifest.is_file():
        records = [
            json.loads(line)
            for line in partial_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed_ids = {str(record["sample_id"]) for record in records}
    split_counts: dict[str, int] = {}
    for coco_split, split in (("train", "train"), ("valid", "validation"), ("test", "test")):
        document = _read_coco(coco_root / coco_split / "_annotations.coco.json")
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in document.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image in document.get("images", []):
            image_id = int(image["id"])
            sample_id = f"fire-smoke-corpus-v1:{split}:{image_id}"
            if sample_id in completed_ids:
                continue
            image_annotations = annotations_by_image.get(image_id, [])
            image_path = coco_root / coco_split / str(image["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(f"COCO image is missing: {image_path}")
            mask_relpath = Path("masks") / split / f"{image_id:08d}.png"
            mask_path = output_root / mask_relpath
            if not mask_path.is_file():
                _write_mask(
                    mask_path,
                    int(image["width"]),
                    int(image["height"]),
                    image_annotations,
                )
            anchors = []
            for annotation in image_annotations:
                x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
                anchors.append(
                    {
                        "kind": "bbox_center_weak",
                        "category_id": int(annotation["category_id"]),
                        "x": (x + box_width / 2) / float(image["width"]),
                        "y": (y + box_height / 2) / float(image["height"]),
                    }
                )
            record = {
                "sample_id": sample_id,
                "source_id": "fire-smoke-detection-corpus-v1",
                "source_revision": "corpus-audit.json",
                "split": split,
                "split_group": f"fire-smoke-corpus-v1:{split}:{image_id}",
                "image_relpath": str(Path(coco_split) / str(image["file_name"])),
                "image_sha256": sha256_file(image_path),
                "license": "per-source-license-in-corpus-audit",
                "sample_validation_status": "source_provided",
                "annotations": image_annotations,
                "mask_relpath": mask_relpath.as_posix(),
                "mask_sha256": sha256_file(mask_path),
                "mask_quality": "bbox_rasterized_weak",
                "anchor_points": anchors,
                "visual_abstention_reason": (
                    "no_detection_annotation_weak" if not image_annotations else None
                ),
                "is_operational_incident": False,
            }
            records.append(record)
            with partial_manifest.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            completed_ids.add(sample_id)
        split_counts[split] = len(document.get("images", []))
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    partial_manifest.unlink(missing_ok=True)
    report = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest),
        "rows": len(records),
        "split_counts": split_counts,
        "mask_quality": "bbox_rasterized_weak",
        "source_coco_root": str(coco_root),
        "promotion_ready": False,
        "promotion_blockers": [
            "human_or_sam_masks_required",
            "weak_bbox_points_are_not_coordinate_authority",
        ],
    }
    (output_root / "materialization-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize weak DINOv3 masks from frozen COCO")
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.coco_root, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
