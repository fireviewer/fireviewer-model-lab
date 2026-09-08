"""Acquire a bounded HPWREN FIgLib slice and derive temporal smoke supervision."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from fireviewer_model_lab.training.pyro_sdis_multitask import TEACHER_ID, TEACHER_REVISION, assign_split, smoke_base
from fireviewer_model_lab.training.remote_zip import require_http_url

DEFAULT_OFFSETS = (
    -2400,
    -1800,
    -1200,
    -600,
    -300,
    -120,
    -60,
    0,
    60,
    120,
    300,
    600,
    1200,
    1800,
    2400,
)
IMAGE_LINK = re.compile(r"(?P<epoch>\d+)_(?P<offset>[+-]\d+)\.jpg$", re.IGNORECASE)
FIGLIB_BUNDLE_REVISION = "ndp-hpwren-figlib-2016-2020-2024-12-17"


def parse_sequence_page(html: str, *, sequence_url: str) -> list[dict[str, Any]]:
    links = re.findall(r"href\s*=\s*['\"]?([^'\"\s>]+)", html, flags=re.IGNORECASE)
    rows: list[dict[str, Any]] = []
    for href in links:
        decoded = urllib.parse.unquote(href)
        match = IMAGE_LINK.search(decoded)
        if match is None:
            continue
        rows.append(
            {
                "filename": Path(decoded).name,
                "epoch_seconds": int(match.group("epoch")),
                "offset_seconds": int(match.group("offset")),
                "url": urllib.parse.urljoin(sequence_url, href),
            }
        )
    if not rows:
        raise ValueError(f"FIgLib sequence page has no images: {sequence_url}")
    return sorted(rows, key=lambda row: int(row["offset_seconds"]))


def select_offsets(
    available: list[dict[str, Any]], targets: tuple[int, ...] = DEFAULT_OFFSETS
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for target in targets:
        closest = min(
            available,
            key=lambda row: (abs(int(row["offset_seconds"]) - target), int(row["offset_seconds"])),
        )
        filename = str(closest["filename"])
        if filename in used:
            continue
        used.add(filename)
        selected.append({**closest, "target_offset_seconds": target})
    return sorted(selected, key=lambda row: int(row["offset_seconds"]))


def visibility_role(offset_seconds: int) -> str:
    if offset_seconds <= -300:
        return "pre_onset_negative"
    if offset_seconds <= 0:
        return "onset_ambiguous"
    return "post_onset_positive_candidate"


def _download_one(url: str, destination: Path) -> dict[str, Any]:
    url = require_http_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return {"path": str(destination), "bytes": destination.stat().st_size, "status": "present"}
    partial = destination.with_suffix(destination.suffix + ".part")
    error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(  # noqa: S310 - URL validated above
                url, headers={"User-Agent": "FireViewer/1.0"}
            )
            with (
                urllib.request.urlopen(  # noqa: S310 - URL validated above
                    request, timeout=30.0
                ) as response,
                partial.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            error = None
            break
        except Exception as caught:  # urllib exposes several transport-specific errors
            error = caught
            partial.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(attempt + 1)
    if error is not None:
        raise error
    if partial.stat().st_size == 0:
        raise OSError(f"empty FIgLib image download: {url}")
    os.replace(partial, destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "status": "downloaded"}


def _resolve_sequence(sequence: dict[str, Any], offsets: tuple[int, ...]) -> list[dict[str, Any]]:
    sequence_url = require_http_url(str(sequence["sequence_url"]))
    error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(  # noqa: S310 - URL validated above
                sequence_url, headers={"User-Agent": "FireViewer/1.0"}
            )
            with urllib.request.urlopen(  # noqa: S310 - URL validated above
                request, timeout=30.0
            ) as response:
                html = response.read().decode("utf-8", errors="replace")
            error = None
            break
        except Exception as caught:
            error = caught
            if attempt < 2:
                time.sleep(attempt + 1)
    if error is not None:
        raise error
    selected = select_offsets(
        parse_sequence_page(html, sequence_url=str(sequence["sequence_url"])),
        offsets,
    )
    rows: list[dict[str, Any]] = []
    for image in selected:
        relative = (
            Path("sources")
            / "hpwren-figlib"
            / str(sequence["sequence_id"])
            / str(image["filename"])
        )
        split_group = str(sequence["split_group"])
        rows.append(
            {
                **image,
                "sequence_id": sequence["sequence_id"],
                "event_key": sequence["event_key"],
                "camera_id": sequence["camera_id"],
                "split_group": split_group,
                "split": assign_split(split_group),
                "cross_view_candidate": bool(sequence["cross_view_candidate"]),
                "visibility_role": visibility_role(int(image["offset_seconds"])),
                "image_relpath": relative.as_posix(),
            }
        )
    return rows


def acquire_figlib_selection(
    *,
    index_manifest: Path,
    campaign_root: Path,
    output_root: Path,
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
    workers: int = 8,
) -> dict[str, Any]:
    sequences = [
        json.loads(line)
        for line in index_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sequences:
        raise ValueError("FIgLib sequence index is empty")
    inventory: list[dict[str, Any]] = []
    skipped_sequences: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        sequence_futures = {
            executor.submit(_resolve_sequence, sequence, offsets): sequence
            for sequence in sequences
        }
        for future in as_completed(sequence_futures):
            sequence = sequence_futures[future]
            try:
                inventory.extend(future.result())
            except Exception as error:
                skipped_sequences.append(
                    {
                        "sequence_id": str(sequence["sequence_id"]),
                        "sequence_url": str(sequence["sequence_url"]),
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
    if not inventory:
        raise ValueError("no accessible FIgLib sequence produced selected images")

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in inventory:
            destination = campaign_root / str(row["image_relpath"])
            futures[executor.submit(_download_one, str(row["url"]), destination)] = row
        for future in as_completed(futures):
            result = future.result()
            futures[future]["bytes"] = int(result["bytes"])
            futures[future]["download_status"] = result["status"]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "acquisition.jsonl"
    inventory.sort(key=lambda row: (str(row["sequence_id"]), int(row["offset_seconds"])))
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inventory),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_id": "hpwren-figlib",
        "declared_sequences": len(sequences),
        "accessible_sequences": len({str(row["sequence_id"]) for row in inventory}),
        "skipped_sequences": sorted(skipped_sequences, key=lambda row: row["sequence_id"]),
        "images": len(inventory),
        "bytes": sum(int(row["bytes"]) for row in inventory),
        "workers": workers,
        "offsets": list(offsets),
        "role_counts": dict(
            sorted(Counter(str(row["visibility_role"]) for row in inventory).items())
        ),
        "manifest": str(manifest),
    }
    (output_root / "acquisition-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def acquire_figlib_archive(
    *,
    archive: Path,
    index_manifest: Path,
    campaign_root: Path,
    output_root: Path,
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
    delete_archive: bool = True,
) -> dict[str, Any]:
    sequences = {
        str(row["sequence_id"]): row
        for row in (
            json.loads(line)
            for line in index_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if not sequences:
        raise ValueError("FIgLib sequence index is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    members: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            match = IMAGE_LINK.search(normalized)
            if match is None:
                continue
            parts = Path(normalized).parts
            if len(parts) < 2:
                continue
            sequence_id = parts[-2]
            if sequence_id not in sequences:
                continue
            members[normalized] = info
            grouped[sequence_id].append(
                {
                    "filename": parts[-1],
                    "epoch_seconds": int(match.group("epoch")),
                    "offset_seconds": int(match.group("offset")),
                    "member_name": normalized,
                }
            )

        inventory: list[dict[str, Any]] = []
        for sequence_id, available in sorted(grouped.items()):
            sequence = sequences[sequence_id]
            selected = select_offsets(available, offsets)
            for image in selected:
                relative = Path("sources") / "hpwren-figlib" / sequence_id / str(image["filename"])
                destination = campaign_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and destination.stat().st_size > 0:
                    status = "present"
                else:
                    payload = bundle.read(members[str(image["member_name"])])
                    Image.open(BytesIO(payload)).verify()
                    partial = destination.with_suffix(".partial.jpg")
                    partial.write_bytes(payload)
                    os.replace(partial, destination)
                    status = "extracted"
                inventory.append(
                    {
                        **image,
                        "source_revision": FIGLIB_BUNDLE_REVISION,
                        "sequence_id": sequence_id,
                        "event_key": sequence["event_key"],
                        "camera_id": sequence["camera_id"],
                        "split_group": sequence["split_group"],
                        "split": assign_split(str(sequence["split_group"])),
                        "cross_view_candidate": bool(sequence["cross_view_candidate"]),
                        "visibility_role": visibility_role(int(image["offset_seconds"])),
                        "image_relpath": relative.as_posix(),
                        "bytes": destination.stat().st_size,
                        "download_status": status,
                    }
                )
    if not inventory:
        raise ValueError("FIgLib bulk archive produced no indexed image selection")
    if delete_archive:
        archive.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "acquisition.jsonl"
    inventory.sort(key=lambda row: (str(row["sequence_id"]), int(row["offset_seconds"])))
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inventory),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_id": "hpwren-figlib",
        "source_revision": FIGLIB_BUNDLE_REVISION,
        "archive_sequences": len(grouped),
        "selected_images": len(inventory),
        "selected_bytes": sum(int(row["bytes"]) for row in inventory),
        "reused_existing_images": sum(row["download_status"] == "present" for row in inventory),
        "archive_deleted": delete_archive,
        "network_requests_during_extraction": 0,
        "role_counts": dict(
            sorted(Counter(str(row["visibility_role"]) for row in inventory).items())
        ),
        "manifest": str(manifest),
    }
    (output_root / "acquisition-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _teacher_mask(probability: np.ndarray, *, threshold: float, minimum_pixels: int) -> np.ndarray:
    candidate = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    kept = np.zeros_like(candidate)
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if minimum_pixels <= area <= int(candidate.size * 0.35):
            kept[labels == component] = 1
    return kept * 255


def materialize_figlib(
    *,
    acquisition_manifest: Path,
    campaign_root: Path,
    output_root: Path,
    batch_size: int = 8,
    threshold: float = 0.60,
    minimum_pixels: int = 48,
    device_name: str = "cuda",
) -> dict[str, Any]:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    rows = [
        json.loads(line)
        for line in acquisition_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_id"])].append(row)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION)
    model = (
        SegformerForSemanticSegmentation.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION)
        .eval()
        .to(device)
    )
    output_rows: list[dict[str, Any]] = []
    for sequence_id, sequence_rows in sorted(by_sequence.items()):
        sequence_rows.sort(key=lambda row: int(row["offset_seconds"]))
        images = [
            Image.open(campaign_root / str(row["image_relpath"])).convert("RGB")
            for row in sequence_rows
        ]
        masks: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=batch_images[0].size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
            masks.extend(
                _teacher_mask(probability, threshold=threshold, minimum_pixels=minimum_pixels)
                for probability in logits.sigmoid()[:, 0].cpu().numpy()
            )
        nonempty = [bool(np.any(mask)) for mask in masks]
        for index, (row, _image, teacher_mask) in enumerate(
            zip(sequence_rows, images, masks, strict=True)
        ):
            role = str(row["visibility_role"])
            if role == "pre_onset_negative":
                mask = np.zeros_like(teacher_mask)
                valid = np.full_like(teacher_mask, 255)
                abstention = None
                strength = "temporal_negative"
            elif role == "onset_ambiguous":
                mask = np.zeros_like(teacher_mask)
                valid = np.zeros_like(teacher_mask)
                abstention = "smoke_onset_ambiguous"
                strength = "abstention"
            else:
                neighbor_support = sum(
                    nonempty[position]
                    for position in range(max(0, index - 1), min(len(nonempty), index + 2))
                )
                if nonempty[index] and neighbor_support >= 2:
                    mask = teacher_mask
                    valid = np.full_like(teacher_mask, 255)
                    abstention = None
                    strength = "weak"
                else:
                    mask = np.zeros_like(teacher_mask)
                    valid = np.zeros_like(teacher_mask)
                    abstention = "teacher_temporal_consensus_missing"
                    strength = "abstention"
            anchor = smoke_base(mask)
            stem = f"{sequence_id}-{int(row['offset_seconds']):+06d}"
            relative_mask = Path("derived") / "hpwren-figlib" / "masks" / f"{stem}.png"
            relative_valid = Path("derived") / "hpwren-figlib" / "valid" / f"{stem}.png"
            for relative, array in ((relative_mask, mask), (relative_valid, valid)):
                path = campaign_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".partial.png")
                if not cv2.imwrite(str(temporary), array):
                    raise OSError(f"unable to write {temporary}")
                os.replace(temporary, path)
            image_bytes = (campaign_root / str(row["image_relpath"])).read_bytes()
            mask_bytes = (campaign_root / relative_mask).read_bytes()
            valid_bytes = (campaign_root / relative_valid).read_bytes()
            output_rows.append(
                {
                    "sample_id": f"figlib:{stem}",
                    "source_id": "hpwren-figlib",
                    "source_revision": row.get("source_revision", "official-live-index"),
                    "split": row["split"],
                    "split_group": row["split_group"],
                    "image_relpath": row["image_relpath"],
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "mask_relpath": relative_mask.as_posix(),
                    "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                    "valid_mask_relpath": relative_valid.as_posix(),
                    "valid_mask_sha256": hashlib.sha256(valid_bytes).hexdigest(),
                    "mask_quality": "segformer_b2_temporal_consensus_weak",
                    "annotation_strength": strength,
                    "sample_validation_status": "teacher_generated_weak",
                    "anchor_points": (
                        [{"kind": "smoke_column_base", "x": anchor[0], "y": anchor[1]}]
                        if anchor
                        else []
                    ),
                    "visual_abstention_reason": abstention,
                    "license": "HPWREN-FIgLib-upstream-terms",
                    "redistribution_allowed": False,
                    "is_operational_incident": True,
                    "sample_weight": 1.0,
                    "camera_id": row["camera_id"],
                    "event_key": row["event_key"],
                    "onset_offset_seconds": row["offset_seconds"],
                    "cross_view_candidate": row["cross_view_candidate"],
                }
            )
        for image in images:
            image.close()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_id": "hpwren-figlib",
        "rows": len(output_rows),
        "sequences": len(by_sequence),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in output_rows).items())),
        "strength_counts": dict(
            sorted(Counter(str(row["annotation_strength"]) for row in output_rows).items())
        ),
        "manifest": str(manifest),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
