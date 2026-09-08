"""Build a compact, high-quality ground-view RF-DETR training corpus."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CLASSES = ("flame_visible", "smoke_visible")
CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}
SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}
SEQUENCE_CAPS = {"train": 8, "validation": 6, "test": 6}
ELITE_SEQUENCE_CAPS = {"train": 2, "validation": 1, "test": 1}
MIN_PROJECTED_SIDE = 6.0
MIN_AREA_RATIO = 0.0004
MAX_AREA_RATIO = 0.75
MAX_PHASH_DISTANCE = 3
POLICY_ID = "ground-premium-v1"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            yield value


def _phash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _label_signature(row: dict[str, Any]) -> str:
    labels = sorted(
        {
            str(annotation.get("class_name"))
            for annotation in row.get("annotations", [])
            if annotation.get("class_name") in CLASS_IDS
        }
    )
    return "+".join(labels) if labels else "negative"


def quality_score(row: dict[str, Any]) -> float | None:
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    if width < 512 or height < 360:
        return None
    if row.get("near_duplicate_of"):
        return None
    if row.get("sample_validation_status") not in {"source_provided", "validated"}:
        return None
    annotations = [
        item for item in row.get("annotations", []) if item.get("class_name") in CLASS_IDS
    ]
    if not annotations:
        return 1.0

    best_projected_side = 0.0
    best_area_ratio = 0.0
    for annotation in annotations:
        bbox = annotation.get("bbox_xywh")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x, y, box_width, box_height = (float(value) for value in bbox)
        if box_width <= 0 or box_height <= 0 or x < 0 or y < 0:
            return None
        if x + box_width > width + 1 or y + box_height > height + 1:
            return None
        area_ratio = (box_width * box_height) / (width * height)
        if area_ratio > MAX_AREA_RATIO:
            return None
        projected_side = min(box_width / width * 512, box_height / height * 512)
        best_projected_side = max(best_projected_side, projected_side)
        best_area_ratio = max(best_area_ratio, area_ratio)
    if best_projected_side < MIN_PROJECTED_SIDE or best_area_ratio < MIN_AREA_RATIO:
        return None
    labels = _label_signature(row)
    diversity_bonus = 3.0 if "+" in labels else 0.0
    return best_projected_side + 20.0 * math.sqrt(best_area_ratio) + diversity_bonus


def select_rows(
    rows: Iterable[dict[str, Any]],
    split: str,
    *,
    sequence_caps: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    caps = sequence_caps or SEQUENCE_CAPS
    if split not in caps:
        raise ValueError(f"Unknown split: {split}")
    groups: dict[tuple[str, str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != split:
            continue
        score = quality_score(row)
        phash = str(row.get("phash") or "")
        if score is None or not phash or len(phash) != 16:
            continue
        key = (
            str(row.get("source_id") or "unknown"),
            str(row.get("sequence_id") or row.get("split_group") or row["sample_id"]),
            _label_signature(row),
        )
        groups[key].append((score, row))

    selected: list[dict[str, Any]] = []
    seen_sha: set[str] = set()
    seen_fingerprint: set[str] = set()
    for key in sorted(groups):
        kept_phashes: list[str] = []
        candidates = sorted(
            groups[key],
            key=lambda item: (-item[0], str(item[1]["sample_id"])),
        )
        for score, row in candidates:
            sha = str(row.get("sha256") or "")
            fingerprint = str(row.get("visual_fingerprint") or "")
            phash = str(row["phash"])
            if sha in seen_sha or (fingerprint and fingerprint in seen_fingerprint):
                continue
            if any(_phash_distance(phash, kept) <= MAX_PHASH_DISTANCE for kept in kept_phashes):
                continue
            selected.append({**row, "premium_quality_score": round(score, 6)})
            seen_sha.add(sha)
            if fingerprint:
                seen_fingerprint.add(fingerprint)
            kept_phashes.append(phash)
            if len(kept_phashes) >= caps[split]:
                break
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def _receipt_index(source_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for local_split in SPLIT_DIRS.values():
        receipt_dir = source_root / "_materialization" / local_split
        image_dir = source_root / "_rfdetr_coco" / local_split
        for receipt in sorted(receipt_dir.glob("*.records.jsonl")):
            for row in _read_jsonl(receipt):
                sample_id = str(row["sample_id"])
                image = image_dir / str(row["file_name"])
                if sample_id in result:
                    raise ValueError(f"Duplicate materialized sample_id: {sample_id}")
                if not image.is_file() or image.stat().st_size == 0:
                    raise FileNotFoundError(image)
                result[sample_id] = (image, row)
    return result


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_premium_ground(
    source_root: Path,
    output: Path,
    *,
    sequence_caps: dict[str, int] | None = None,
    dataset_profile: str = "ground-premium",
    dataset_id: str = "fireviewer/fire-smoke-ground-premium-rfdetr-small-v1",
    policy_id: str = POLICY_ID,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Premium output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_manifests = {
        "fasdd_v9": source_root / "manifests" / "fasdd-cv" / "manifest.jsonl",
        "pyro_sdis_a1e553e": source_root / "manifests" / "pyro-sdis" / "manifest.jsonl",
    }
    rows: list[dict[str, Any]] = []
    for manifest in source_manifests.values():
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        rows.extend(_read_jsonl(manifest))
    receipts = _receipt_index(source_root)

    caps = sequence_caps or SEQUENCE_CAPS
    selected_by_split = {
        split: select_rows(rows, split, sequence_caps=caps) for split in SPLIT_DIRS
    }
    selected_ids = {
        str(row["sample_id"]) for selected in selected_by_split.values() for row in selected
    }
    missing = sorted(selected_ids.difference(receipts))
    if missing:
        raise ValueError(f"Selected samples missing from local payload: {len(missing)}")

    reports: dict[str, Any] = {}
    all_selected: list[dict[str, Any]] = []
    for split, local_split in SPLIT_DIRS.items():
        split_dir = output / "_rfdetr_coco" / local_split
        split_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = 1
        source_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        for row in selected_by_split[split]:
            sample_id = str(row["sample_id"])
            source_image, receipt = receipts[sample_id]
            destination = split_dir / source_image.name
            os.link(source_image, destination)
            image_id = len(images) + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": destination.name,
                    "width": int(receipt["width"]),
                    "height": int(receipt["height"]),
                }
            )
            for annotation in receipt["annotations"]:
                annotations.append({"id": annotation_id, "image_id": image_id, **annotation})
                annotation_id += 1
            source_counts[str(row["source_id"])] += 1
            label_counts[_label_signature(row)] += 1
            all_selected.append(row)
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": index, "name": name, "supercategory": "wildfire"}
                for index, name in enumerate(CLASSES)
            ],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        reports[local_split] = {
            "images": len(images),
            "annotations": len(annotations),
            "source_counts": dict(sorted(source_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
        }

    for source_id, destination_name in (
        ("fasdd_v9", "fasdd-cv"),
        ("pyro_sdis_a1e553e", "pyro-sdis"),
    ):
        _write_manifest(
            output / "manifests" / destination_name / "manifest.jsonl",
            [row for row in all_selected if row.get("source_id") == source_id],
        )

    audit = {
        "schema_version": 1,
        "dataset_profile": dataset_profile,
        "dataset_id": dataset_id,
        "source_dataset": "fireviewer/fire-smoke-ground-only-rfdetr-large-v1",
        "selection_policy": {
            "id": policy_id,
            "ground_view_only": True,
            "declared_near_duplicates_excluded": True,
            "minimum_dimensions": [512, 360],
            "minimum_projected_box_side_at_512": MIN_PROJECTED_SIDE,
            "minimum_box_area_ratio": MIN_AREA_RATIO,
            "maximum_box_area_ratio": MAX_AREA_RATIO,
            "maximum_phash_distance_within_sequence": MAX_PHASH_DISTANCE,
            "sequence_caps": caps,
            "payload_storage": "ntfs_hardlinks",
        },
        "classes": list(CLASSES),
        "splits": reports,
        "max_samples_per_split": None,
    }
    coco_root = output / "_rfdetr_coco"
    (coco_root / "_conversion_complete.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (output / "corpus-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output / "publication-manifest.json").write_text(
        json.dumps(
            {
                "dataset_profile": dataset_profile,
                "dataset_id": audit["dataset_id"],
                "redistribution": "inherits_per_source_licenses",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return audit


def prepare_elite_ground(source_root: Path, output: Path) -> dict[str, Any]:
    return prepare_premium_ground(
        source_root,
        output,
        sequence_caps=ELITE_SEQUENCE_CAPS,
        dataset_profile="ground-elite",
        dataset_id="fireviewer/fire-smoke-ground-elite-rfdetr-small-v1",
        policy_id="ground-elite-v1",
    )
