"""Materialize Pyro-SDIS as conservative FireViewer multi-task supervision."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PYRO_SDIS_REVISION = "a1e553ec4d806f71fc6db744cc22bc3469487382"
TEACHER_ID = "fireviewer/segformer-b2-fire-smoke-baseline-v1"
TEACHER_REVISION = "f07112018627ada432d71621e7b7cfc68f278911"


def parse_yolo_boxes(value: str) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(value.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid Pyro-SDIS annotation line {line_number}")
        _, center_x, center_y, width, height = fields
        box = tuple(float(item) for item in (center_x, center_y, width, height))
        if not all(np.isfinite(item) and 0.0 <= item <= 1.0 for item in box):
            raise ValueError(f"out-of-range Pyro-SDIS annotation line {line_number}")
        boxes.append(box)
    return boxes


def station_from_camera(camera: str) -> str:
    head, separator, tail = camera.rpartition("-")
    return head if separator and tail.isdigit() and head else camera


def assign_split(split_group: str) -> str:
    bucket = (
        int.from_bytes(hashlib.blake2b(split_group.encode("utf-8"), digest_size=8).digest(), "big")
        % 100
    )
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def parse_captured_at(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H-%M-%S")
        except ValueError as error:
            raise ValueError(f"unsupported Pyro-SDIS capture date: {value}") from error


def build_session_groups(
    metadata: Iterable[dict[str, str]], *, maximum_gap_seconds: int = 1800
) -> dict[str, str]:
    grouped: dict[tuple[str, str], list[tuple[datetime, str]]] = defaultdict(list)
    for row in metadata:
        partner = str(row["partner"])
        camera = str(row["camera"])
        image_name = str(row["image_name"])
        captured = parse_captured_at(str(row["date"]))
        grouped[(partner, station_from_camera(camera))].append((captured, image_name))
    assignments: dict[str, str] = {}
    for (partner, station), entries in sorted(grouped.items()):
        entries.sort()
        session_start: datetime | None = None
        previous: datetime | None = None
        for captured, image_name in entries:
            if previous is None or (captured - previous).total_seconds() > maximum_gap_seconds:
                session_start = captured
            assert session_start is not None
            assignments[image_name] = (
                f"pyro-sdis:{partner}:{station}:{session_start.isoformat(timespec='seconds')}"
            )
            previous = captured
    return assignments


def boxes_to_roi(
    boxes: list[tuple[float, float, float, float]],
    shape: tuple[int, int],
    *,
    expansion_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    core = np.zeros((height, width), dtype=np.uint8)
    expanded = np.zeros_like(core)
    for center_x, center_y, box_width, box_height in boxes:
        x1 = int(np.floor((center_x - box_width / 2) * width))
        y1 = int(np.floor((center_y - box_height / 2) * height))
        x2 = int(np.ceil((center_x + box_width / 2) * width))
        y2 = int(np.ceil((center_y + box_height / 2) * height))
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        core[y1:y2, x1:x2] = 1
        margin_x = max(2, int(np.ceil(box_width * width * expansion_ratio)))
        margin_y = max(2, int(np.ceil(box_height * height * expansion_ratio)))
        expanded[
            max(0, y1 - margin_y) : min(height, y2 + margin_y),
            max(0, x1 - margin_x) : min(width, x2 + margin_x),
        ] = 1
    return core, expanded


def filter_teacher_mask(
    probability: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    *,
    threshold: float,
    minimum_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    core, valid = boxes_to_roi(boxes, probability.shape)
    candidate = ((probability >= threshold) & (valid > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    kept = np.zeros_like(candidate)
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        component_mask = labels == component
        if area >= minimum_pixels and bool(np.any(component_mask & (core > 0))):
            kept[component_mask] = 1
    return kept * 255, valid * 255


def smoke_base(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return None
    bottom = int(ys.max())
    band = xs[ys >= max(int(ys.min()), bottom - max(1, mask.shape[0] // 100))]
    return float(np.median(band)) / mask.shape[1], float(bottom) / mask.shape[0]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_png(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.png")
    if not cv2.imwrite(str(temporary), array):
        raise OSError(f"unable to write {temporary}")
    os.replace(temporary, path)
    return _sha256_bytes(path.read_bytes())


def _load_metadata(parquet_files: list[Path]) -> list[dict[str, str]]:
    import pyarrow.parquet as parquet

    rows: list[dict[str, str]] = []
    for parquet_file in parquet_files:
        table = parquet.read_table(
            parquet_file,
            columns=["annotations", "image_name", "partner", "camera", "date"],
        )
        rows.extend(table.to_pylist())
    if len({row["image_name"] for row in rows}) != len(rows):
        raise ValueError("Pyro-SDIS image_name is not unique")
    return rows


def _extract_image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and value.get("bytes"):
        return bytes(value["bytes"])
    raise ValueError("Pyro-SDIS parquet image has no embedded bytes")


def materialize_pyro_sdis(
    *,
    parquet_root: Path,
    campaign_root: Path,
    output_root: Path,
    batch_size: int = 8,
    threshold: float = 0.55,
    minimum_pixels: int = 24,
    device_name: str = "cuda",
) -> dict[str, Any]:
    import pyarrow.parquet as parquet
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    parquet_files = sorted(parquet_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no Pyro-SDIS parquet files below {parquet_root}")
    metadata = _load_metadata(parquet_files)
    session_groups = build_session_groups(metadata)
    metadata_by_name = {str(row["image_name"]): row for row in metadata}

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION)
    model = (
        SegformerForSemanticSegmentation.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION)
        .eval()
        .to(device)
    )

    output_root.mkdir(parents=True, exist_ok=True)
    partial_manifest = output_root / "manifest.partial.jsonl"
    completed: set[str] = set()
    if partial_manifest.is_file():
        for line in partial_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line)["sample_id"]))

    def process_batch(items: list[dict[str, Any]]) -> None:
        if not items:
            return
        images = [item["image"] for item in items]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=images[0].size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            probabilities = logits.sigmoid()[:, 0].cpu().numpy()
        with partial_manifest.open("a", encoding="utf-8", newline="\n") as stream:
            for item, probability in zip(items, probabilities, strict=True):
                image_name = item["image_name"]
                metadata_row = metadata_by_name[image_name]
                boxes = parse_yolo_boxes(str(metadata_row["annotations"]))
                split_group = session_groups[image_name]
                split = assign_split(split_group)
                relative_image = (
                    Path("sources")
                    / "pyro-sdis"
                    / "images"
                    / str(metadata_row["partner"])
                    / station_from_camera(str(metadata_row["camera"]))
                    / image_name
                )
                image_path = campaign_root / relative_image
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not image_path.is_file():
                    image_path.write_bytes(item["image_bytes"])

                if boxes:
                    mask, valid = filter_teacher_mask(
                        probability,
                        boxes,
                        threshold=threshold,
                        minimum_pixels=minimum_pixels,
                    )
                    anchor = smoke_base(mask)
                    abstention = None if anchor is not None else "teacher_box_consensus_empty"
                    if abstention is not None:
                        valid[:] = 0
                else:
                    mask = np.zeros(probability.shape, dtype=np.uint8)
                    valid = np.full(probability.shape, 255, dtype=np.uint8)
                    anchor = None
                    abstention = None
                stem = Path(image_name).stem
                relative_mask = Path("derived") / "pyro-sdis" / "masks" / f"{stem}.png"
                relative_valid = Path("derived") / "pyro-sdis" / "valid" / f"{stem}.png"
                mask_sha = _write_png(campaign_root / relative_mask, mask)
                valid_sha = _write_png(campaign_root / relative_valid, valid)
                row = {
                    "sample_id": f"pyro-sdis:{stem}",
                    "source_id": "pyronear-pyro-sdis",
                    "source_revision": PYRO_SDIS_REVISION,
                    "split": split,
                    "split_group": split_group,
                    "image_relpath": relative_image.as_posix(),
                    "image_sha256": _sha256_bytes(item["image_bytes"]),
                    "mask_relpath": relative_mask.as_posix(),
                    "mask_sha256": mask_sha,
                    "valid_mask_relpath": relative_valid.as_posix(),
                    "valid_mask_sha256": valid_sha,
                    "mask_quality": "segformer_b2_box_consensus_weak",
                    "annotation_strength": "weak" if boxes else "negative",
                    "sample_validation_status": "teacher_generated_weak",
                    "anchor_points": (
                        [{"kind": "smoke_column_base", "x": anchor[0], "y": anchor[1]}]
                        if anchor is not None
                        else []
                    ),
                    "visual_abstention_reason": abstention,
                    "license": "Apache-2.0",
                    "redistribution_allowed": True,
                    "is_operational_incident": True,
                    "sample_weight": 0.5 if boxes else 0.75,
                    "source_annotations_yolo": str(metadata_row["annotations"]),
                    "camera_id": str(metadata_row["camera"]),
                    "captured_at": str(metadata_row["date"]),
                }
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    pending: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        parquet_file_reader = parquet.ParquetFile(parquet_file)
        for batch in parquet_file_reader.iter_batches(batch_size=128):
            for raw in batch.to_pylist():
                image_name = str(raw["image_name"])
                sample_id = f"pyro-sdis:{Path(image_name).stem}"
                if sample_id in completed:
                    continue
                image_bytes = _extract_image_bytes(raw["image"])
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                if image.size != (1280, 720):
                    raise ValueError(
                        f"unexpected Pyro-SDIS dimensions for {image_name}: {image.size}"
                    )
                pending.append(
                    {
                        "image_name": image_name,
                        "image_bytes": image_bytes,
                        "image": image,
                    }
                )
                if len(pending) >= batch_size:
                    process_batch(pending)
                    pending.clear()
    process_batch(pending)

    rows = [
        json.loads(line)
        for line in partial_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: str(row["sample_id"]))
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    partial_manifest.unlink()
    split_counts = Counter(str(row["split"]) for row in rows)
    quality_counts = Counter(str(row["annotation_strength"]) for row in rows)
    report = {
        "schema_version": 1,
        "source_id": "pyronear-pyro-sdis",
        "source_revision": PYRO_SDIS_REVISION,
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "annotation_strength_counts": dict(sorted(quality_counts.items())),
        "session_groups": len(set(session_groups.values())),
        "manifest": str(manifest),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
