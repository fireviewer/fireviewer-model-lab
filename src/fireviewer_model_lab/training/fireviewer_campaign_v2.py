"""Build the shared FireViewer v2 segmentation corpus without copying sources.

The materialized dataset contains one relative manifest and derived masks only.
Original images remain in their immutable source directories so DINOv3 and
SegFormer consume the same payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

BOREAL_REVISION = "8420ddebfadd94babe6f930b484dc9e2391e9dbe"
FIRESENTRY_REVISION = "f8693204071a871562a3b4b4e24797a6a0d3ae3f"
FIRESENTRY_SPLITS = {
    "A": "train",
    "B": "validation",
    "C": "test",
    "D": "train",
    "E": "train",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes campaign root: {path}")
    return path.relative_to(root).as_posix()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _mask_anchor(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return None
    height, width = mask.shape
    return float(xs.mean()) / width, float(ys.mean()) / height


def _last_binary_video_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unable to open FireSentry mask video: {path}")
    last: np.ndarray | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        last = frame
    capture.release()
    if last is None:
        raise ValueError(f"FireSentry mask video contains no frame: {path}")
    gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    return (gray >= 128).astype(np.uint8) * 255


def _boreal_rows(campaign_root: Path, boreal_root: Path) -> list[dict[str, Any]]:
    source_manifest = boreal_root / "manifest.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        source = json.loads(line)
        artifact_path = boreal_root / str(source["artifact"]["path"])
        sample = json.loads(artifact_path.read_text(encoding="utf-8"))
        image = sample["image"]
        annotation = sample["annotation"]
        image_path = boreal_root / str(image["path"])
        mask_path = boreal_root / str(annotation["path"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing Boreal payload at line {line_number}")
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened.convert("L"))
        anchor = _mask_anchor(mask)
        strength = str(sample["annotation_strength"])
        rows.append(
            {
                "sample_id": str(source["sample_id"]),
                "source_id": "boreal-forest-fire-segmentation-v1",
                "source_revision": BOREAL_REVISION,
                "split": str(source["split"]),
                "split_group": str(source["split_group"]),
                "image_relpath": _relative(campaign_root, image_path),
                "image_sha256": str(image["sha256"]),
                "mask_relpath": _relative(campaign_root, mask_path),
                "mask_sha256": str(annotation["sha256"]),
                "mask_quality": "human_strong" if strength == "strong" else "sam_weak",
                "annotation_strength": strength,
                "sample_validation_status": "source_provided",
                "anchor_points": (
                    [{"kind": "smoke_centroid", "x": anchor[0], "y": anchor[1]}]
                    if anchor is not None
                    else []
                ),
                "visual_abstention_reason": None if anchor is not None else "empty_smoke_mask",
                "license": str(source["license"]),
                "redistribution_allowed": True,
                "is_operational_incident": False,
            }
        )
    return rows


def _firesentry_rows(
    campaign_root: Path,
    firesentry_root: Path,
    output_root: Path,
    frame_loader: Callable[[Path], np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region, split in FIRESENTRY_SPLITS.items():
        region_root = firesentry_root / f"Region {region}"
        visible_root = region_root / "Visible Light"
        mask_video_root = region_root / "Fire Mask Videos"
        mask_videos = sorted(mask_video_root.glob("video_*.mp4"))
        visible_images = sorted(visible_root.glob("*.jpg"))
        if not visible_images or not mask_videos:
            raise ValueError(f"FireSentry Region {region} has no usable RGB/mask sequence")
        visible_by_index = {int(path.stem): path for path in visible_images}
        for mask_video in mask_videos:
            try:
                index = int(mask_video.stem.removeprefix("video_"))
            except ValueError as exc:
                raise ValueError(f"invalid FireSentry mask filename: {mask_video.name}") from exc
            image_path = visible_by_index.get(index)
            if image_path is None:
                raise ValueError(
                    f"FireSentry Region {region} has no RGB observation for mask {mask_video.name}"
                )
            with Image.open(image_path) as opened:
                width, height = opened.size
            mask = frame_loader(mask_video)
            if mask.ndim != 2:
                raise ValueError(f"FireSentry mask frame is not grayscale: {mask_video}")
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = (mask >= 128).astype(np.uint8) * 255
            mask_path = (
                output_root
                / "masks"
                / "firesentry"
                / f"region-{region.lower()}"
                / f"{index:05d}.png"
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = mask_path.with_suffix(".partial.png")
            if not cv2.imwrite(str(temporary), mask):
                raise OSError(f"unable to write FireSentry mask: {temporary}")
            os.replace(temporary, mask_path)
            anchor = _mask_anchor(mask)
            rows.append(
                {
                    "sample_id": f"firesentry:region-{region.lower()}:{index:05d}",
                    "source_id": "FireSentry-Benchmark-Dataset",
                    "source_revision": FIRESENTRY_REVISION,
                    "split": split,
                    "split_group": f"firesentry:region-{region.lower()}",
                    "image_relpath": _relative(campaign_root, image_path),
                    "image_sha256": sha256_file(image_path),
                    "mask_relpath": _relative(campaign_root, mask_path),
                    "mask_sha256": sha256_file(mask_path),
                    "mask_quality": "sam2_video_last_frame_resized_weak",
                    "annotation_strength": "weak",
                    "sample_validation_status": "source_provided",
                    "anchor_points": (
                        [{"kind": "fire_centroid", "x": anchor[0], "y": anchor[1]}]
                        if anchor is not None
                        else []
                    ),
                    "visual_abstention_reason": None if anchor is not None else "empty_fire_mask",
                    "license": "upstream-license-not-provided",
                    "redistribution_allowed": False,
                    "is_operational_incident": False,
                    "temporal_alignment": "last_mask_frame_to_next_rgb_observation",
                }
            )
    return rows


def build_segmentation_corpus(
    *,
    campaign_root: Path,
    boreal_root: Path,
    firesentry_root: Path,
    output_root: Path,
    additional_manifests: tuple[Path, ...] = (),
    frame_loader: Callable[[Path], np.ndarray] = _last_binary_video_frame,
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    boreal_root = boreal_root.resolve()
    firesentry_root = firesentry_root.resolve()
    output_root = output_root.resolve()
    _relative(campaign_root, boreal_root)
    _relative(campaign_root, firesentry_root)
    _relative(campaign_root, output_root)
    rows = _boreal_rows(campaign_root, boreal_root)
    rows.extend(_firesentry_rows(campaign_root, firesentry_root, output_root, frame_loader))
    included_manifests: list[str] = []
    for additional_manifest in additional_manifests:
        additional_manifest = additional_manifest.resolve()
        _relative(campaign_root, additional_manifest)
        if not additional_manifest.is_file():
            raise FileNotFoundError(additional_manifest)
        included_manifests.append(_relative(campaign_root, additional_manifest))
        for line_number, line in enumerate(
            additional_manifest.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"additional manifest row is not an object: {additional_manifest}:{line_number}"
                )
            for key in ("image_relpath", "mask_relpath"):
                path = (campaign_root / str(row.get(key, ""))).resolve()
                _relative(campaign_root, path)
                if not path.is_file():
                    raise FileNotFoundError(
                        "additional manifest payload missing: "
                        f"{additional_manifest}:{line_number}:{key}"
                    )
            rows.append(row)
    rows.sort(key=lambda row: str(row["sample_id"]))
    manifest = output_root / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    source_counts = Counter(str(row["source_id"]) for row in rows)
    quality_counts = Counter(str(row["mask_quality"]) for row in rows)
    group_owners: dict[str, set[str]] = {}
    for row in rows:
        group_owners.setdefault(str(row["split_group"]), set()).add(str(row["split"]))
    leaking_groups = sorted(group for group, owners in group_owners.items() if len(owners) != 1)
    if leaking_groups:
        raise ValueError(f"segmentation split-group leakage: {leaking_groups}")
    blockers: list[str] = []
    if not any(source.startswith("Camp Swift") for source in source_counts):
        blockers.append("camp_swift_not_ingested")
    if not any(source.startswith("RxCADRE") for source in source_counts):
        blockers.append("rxcadre_not_ingested")
    if "FireSentry-Benchmark-Dataset" in source_counts:
        blockers.append("firesentry_upstream_license_not_provided")
    report = {
        "schema_version": 2,
        "dataset_family": "fireviewer-segmentation-shared-v2",
        "consumers": ["dinov3_multitask", "segformer_baseline"],
        "manifest": _relative(campaign_root, manifest),
        "manifest_sha256": sha256_file(manifest),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "mask_quality_counts": dict(sorted(quality_counts.items())),
        "split_group_leakage": leaking_groups,
        "source_revisions": {
            "boreal": BOREAL_REVISION,
            "firesentry": FIRESENTRY_REVISION,
        },
        "included_manifests": included_manifests,
        "training_ready": not any(
            blocker in blockers for blocker in ("camp_swift_not_ingested", "rxcadre_not_ingested")
        ),
        "publication_ready": not blockers,
        "publication_blockers": blockers,
    }
    _write_json(output_root / "report.json", report)
    return report
