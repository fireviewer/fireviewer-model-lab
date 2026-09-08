"""Acquire selected FIReStereo FiresGL pairs and derive conservative fire masks."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from fireviewer_model_lab.training.pyro_sdis_multitask import smoke_base
from fireviewer_model_lab.training.remote_zip import RemoteZip, RemoteZipEntry

REGISTRY_PATH = Path(__file__).parent / "registries" / "dino-complements-v1.json"
SOURCE_REVISION = "54cb48b606e4f5b84219931ca6322a70264367ce"
SPLITS = {"firesgl-1": "train", "firesgl-2": "validation", "firesgl-3": "test"}


@dataclass(frozen=True)
class StereoPair:
    sequence_group: str
    archive_name: str
    frame_key: str
    left: RemoteZipEntry
    right: RemoteZipEntry


def _source(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    return next(
        source for source in registry["sources"] if source["source_id"] == "firestereo-firesgl"
    )


def stereo_pairs(
    entries: Iterable[RemoteZipEntry], *, sequence_group: str, archive_name: str
) -> list[StereoPair]:
    left: dict[str, RemoteZipEntry] = {}
    right: dict[str, RemoteZipEntry] = {}
    for entry in entries:
        normalized = entry.name.replace("\\", "/")
        if not normalized.lower().endswith(".png"):
            continue
        if "/img_left/" in normalized:
            key = normalized.replace("/img_left/", "/")
            left[key] = entry
        elif "/img_right/" in normalized:
            key = normalized.replace("/img_right/", "/")
            right[key] = entry
    return [
        StereoPair(sequence_group, archive_name, key, left[key], right[key])
        for key in sorted(left.keys() & right.keys())
    ]


def even_sample(items: list[StereoPair], maximum: int) -> list[StereoPair]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if len(items) <= maximum:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=maximum, dtype=np.int64)
    return [items[int(index)] for index in indices]


def _safe_frame_stem(pair: StereoPair) -> str:
    return f"{Path(pair.archive_name).stem}-{pair.frame_key.replace('/', '-')[:-4]}"


def _download_pair(
    remote: RemoteZip,
    pair: StereoPair,
    *,
    campaign_root: Path,
) -> dict[str, Any]:
    stem = _safe_frame_stem(pair)
    base = (
        Path("sources") / "firestereo-firesgl" / pair.sequence_group / Path(pair.archive_name).stem
    )
    left_relative = base / "left" / f"{stem}.png"
    right_relative = base / "right" / f"{stem}.png"
    result: dict[str, Any] = {}
    for side, entry, relative in (
        ("left", pair.left, left_relative),
        ("right", pair.right, right_relative),
    ):
        path = campaign_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > 0:
            payload = path.read_bytes()
            status = "present"
        else:
            payload = remote.read(entry)
            Image.open(BytesIO(payload)).verify()
            partial = path.with_suffix(".partial.png")
            partial.write_bytes(payload)
            os.replace(partial, path)
            status = "downloaded"
        result[f"{side}_relpath"] = relative.as_posix()
        result[f"{side}_bytes"] = len(payload)
        result[f"{side}_status"] = status
    return result


def acquire_firesgl_selection(
    *,
    campaign_root: Path,
    output_root: Path,
    registry_path: Path = REGISTRY_PATH,
    maximum_per_sequence: int = 1500,
    workers: int = 8,
) -> dict[str, Any]:
    source = _source(registry_path)
    remotes: dict[str, RemoteZip] = {}
    all_pairs: dict[str, list[StereoPair]] = defaultdict(list)
    asset_urls: dict[str, str] = {}
    archive_counts: dict[str, int] = {}
    for asset in source["assets"]:
        archive_name = str(asset["filename"])
        remote = RemoteZip(str(asset["url"]), timeout_seconds=180.0)
        if remote.size != int(asset["expected_bytes"]):
            raise ValueError(f"unexpected FIReStereo archive size: {archive_name}")
        pairs = stereo_pairs(
            remote.entries(),
            sequence_group=str(asset["sequence_group"]),
            archive_name=archive_name,
        )
        remotes[archive_name] = remote
        asset_urls[archive_name] = str(asset["url"])
        archive_counts[archive_name] = len(pairs)
        all_pairs[str(asset["sequence_group"])].extend(pairs)

    selected: list[StereoPair] = []
    for _sequence_group, pairs in sorted(all_pairs.items()):
        selected.extend(even_sample(pairs, maximum_per_sequence))

    output_root.mkdir(parents=True, exist_ok=True)
    partial_manifest = output_root / "acquisition.partial.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if partial_manifest.is_file():
        for line in partial_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[str(row["sample_key"])] = row

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for pair in selected:
            sample_key = f"{pair.sequence_group}:{_safe_frame_stem(pair)}"
            if sample_key in completed:
                continue
            future = executor.submit(
                _download_pair,
                remotes[pair.archive_name],
                pair,
                campaign_root=campaign_root,
            )
            futures[future] = (pair, sample_key)
        with partial_manifest.open("a", encoding="utf-8", newline="\n") as stream:
            for future in as_completed(futures):
                pair, sample_key = futures[future]
                row = {
                    "sample_key": sample_key,
                    "sequence_group": pair.sequence_group,
                    "archive_name": pair.archive_name,
                    "source_member_left": pair.left.name,
                    "source_member_right": pair.right.name,
                    "source_url": asset_urls[pair.archive_name],
                    "split": SPLITS[pair.sequence_group],
                    **future.result(),
                }
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                completed[sample_key] = row

    rows = sorted(completed.values(), key=lambda row: str(row["sample_key"]))
    expected_keys = {f"{pair.sequence_group}:{_safe_frame_stem(pair)}" for pair in selected}
    if set(completed) != expected_keys:
        raise ValueError("FIReStereo acquisition manifest does not match selected pairs")
    manifest = output_root / "acquisition.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    partial_manifest.unlink(missing_ok=True)
    report = {
        "schema_version": 1,
        "source_id": "firestereo-firesgl",
        "source_revision": SOURCE_REVISION,
        "available_pairs": sum(archive_counts.values()),
        "selected_pairs": len(rows),
        "selected_by_sequence": dict(
            sorted(Counter(str(row["sequence_group"]) for row in rows).items())
        ),
        "archive_pair_counts": archive_counts,
        "downloaded_bytes": sum(int(row["left_bytes"]) + int(row["right_bytes"]) for row in rows),
        "workers": workers,
        "manifest": str(manifest),
        "full_archives_downloaded": False,
    }
    (output_root / "acquisition-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _zip_entry(info: zipfile.ZipInfo) -> RemoteZipEntry:
    return RemoteZipEntry(
        name=info.filename,
        compressed_size=info.compress_size,
        uncompressed_size=info.file_size,
        compression_method=info.compress_type,
        crc32=info.CRC,
        local_header_offset=info.header_offset,
    )


def _extract_archive_pairs(
    archive: Path,
    pairs: list[StereoPair],
    *,
    campaign_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        for pair in pairs:
            stem = _safe_frame_stem(pair)
            base = (
                Path("sources")
                / "firestereo-firesgl"
                / pair.sequence_group
                / Path(pair.archive_name).stem
            )
            result: dict[str, Any] = {}
            for side, entry, relative in (
                ("left", pair.left, base / "left" / f"{stem}.png"),
                ("right", pair.right, base / "right" / f"{stem}.png"),
            ):
                path = campaign_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_file() and path.stat().st_size > 0:
                    payload_size = path.stat().st_size
                    status = "present"
                else:
                    payload = bundle.read(entry.name)
                    Image.open(BytesIO(payload)).verify()
                    partial = path.with_suffix(".partial.png")
                    partial.write_bytes(payload)
                    os.replace(partial, path)
                    payload_size = len(payload)
                    status = "extracted"
                result[f"{side}_relpath"] = relative.as_posix()
                result[f"{side}_bytes"] = payload_size
                result[f"{side}_status"] = status
            rows.append(
                {
                    "sample_key": f"{pair.sequence_group}:{stem}",
                    "sequence_group": pair.sequence_group,
                    "archive_name": pair.archive_name,
                    "source_member_left": pair.left.name,
                    "source_member_right": pair.right.name,
                    "split": SPLITS[pair.sequence_group],
                    **result,
                }
            )
    return rows


def acquire_firesgl_archives(
    *,
    campaign_root: Path,
    archive_root: Path,
    output_root: Path,
    registry_path: Path = REGISTRY_PATH,
    maximum_per_sequence: int = 1500,
    workers: int = 5,
    delete_archives: bool = True,
) -> dict[str, Any]:
    source = _source(registry_path)
    all_pairs: dict[str, list[StereoPair]] = defaultdict(list)
    archive_paths: dict[str, Path] = {}
    archive_counts: dict[str, int] = {}
    for asset in source["assets"]:
        archive_name = str(asset["filename"])
        archive = archive_root / archive_name
        if not archive.is_file() or archive.stat().st_size != int(asset["expected_bytes"]):
            raise ValueError(f"missing or incomplete FIReStereo archive: {archive}")
        with zipfile.ZipFile(archive) as bundle:
            entries = [_zip_entry(info) for info in bundle.infolist() if not info.is_dir()]
        pairs = stereo_pairs(
            entries,
            sequence_group=str(asset["sequence_group"]),
            archive_name=archive_name,
        )
        archive_paths[archive_name] = archive
        archive_counts[archive_name] = len(pairs)
        all_pairs[str(asset["sequence_group"])].extend(pairs)

    selected: list[StereoPair] = []
    for _, pairs in sorted(all_pairs.items()):
        selected.extend(even_sample(pairs, maximum_per_sequence))
    selected_by_archive: dict[str, list[StereoPair]] = defaultdict(list)
    for pair in selected:
        selected_by_archive[pair.archive_name].append(pair)

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    deleted: list[str] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(selected_by_archive))) as executor:
        futures = {
            executor.submit(
                _extract_archive_pairs,
                archive_paths[archive_name],
                pairs,
                campaign_root=campaign_root,
            ): archive_name
            for archive_name, pairs in selected_by_archive.items()
        }
        for future in as_completed(futures):
            archive_name = futures[future]
            archive_rows = future.result()
            rows.extend(archive_rows)
            if delete_archives:
                archive_paths[archive_name].unlink()
                deleted.append(archive_name)

    rows.sort(key=lambda row: str(row["sample_key"]))
    if len(rows) != len(selected):
        raise ValueError("FIReStereo local extraction did not materialize every selected pair")
    manifest = output_root / "acquisition.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_id": "firestereo-firesgl",
        "source_revision": SOURCE_REVISION,
        "available_pairs": sum(archive_counts.values()),
        "selected_pairs": len(rows),
        "selected_by_sequence": dict(
            sorted(Counter(str(row["sequence_group"]) for row in rows).items())
        ),
        "archive_pair_counts": archive_counts,
        "extracted_bytes": sum(int(row["left_bytes"]) + int(row["right_bytes"]) for row in rows),
        "deleted_archives": sorted(deleted),
        "network_requests_during_extraction": 0,
        "manifest": str(manifest),
    }
    (output_root / "acquisition-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def thermal_candidate_mask(
    image: np.ndarray, *, minimum_pixels: int = 16, maximum_fraction: float = 0.35
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("thermal image must be grayscale")
    values = image.astype(np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_hot = median + 4.0 * 1.4826 * max(mad, 1.0)
    lower_tail = float(np.quantile(values, 0.985))
    upper_tail = float(np.quantile(values, 0.995))
    threshold = max(lower_tail, min(robust_hot, upper_tail))
    candidate = (values >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    kept = np.zeros_like(candidate)
    maximum_pixels = int(candidate.size * maximum_fraction)
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if minimum_pixels <= area <= maximum_pixels:
            kept[labels == component] = 1
    return kept


def stereo_candidate_mask(
    left: np.ndarray,
    right: np.ndarray,
    *,
    maximum_disparity: int = 64,
    minimum_pixels: int = 16,
) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("stereo thermal images must have the same shape")
    left_mask = thermal_candidate_mask(left, minimum_pixels=minimum_pixels)
    right_mask = thermal_candidate_mask(right, minimum_pixels=minimum_pixels)
    kernel = np.ones((3, maximum_disparity * 2 + 1), dtype=np.uint8)
    right_support = cv2.dilate(right_mask, kernel, iterations=1)
    candidate = left_mask & (right_support > 0)
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    return candidate


def temporal_consensus_mask(masks: list[np.ndarray], index: int) -> np.ndarray:
    current = masks[index]
    support = np.zeros_like(current)
    kernel = np.ones((11, 11), dtype=np.uint8)
    for neighbor in (index - 1, index + 1):
        if 0 <= neighbor < len(masks):
            support |= cv2.dilate(masks[neighbor], kernel, iterations=1)
    return (current & (support > 0)).astype(np.uint8) * 255


def _write_png(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.png")
    if not cv2.imwrite(str(partial), array):
        raise OSError(f"unable to write {partial}")
    os.replace(partial, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_firesgl(
    *, acquisition_manifest: Path, campaign_root: Path, output_root: Path
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in acquisition_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_group"])].append(row)
    output_rows: list[dict[str, Any]] = []
    for sequence_group, sequence_rows in sorted(by_sequence.items()):
        sequence_rows.sort(key=lambda row: str(row["sample_key"]))
        left_images: list[np.ndarray] = []
        stereo_masks: list[np.ndarray] = []
        for row in sequence_rows:
            left = cv2.imread(str(campaign_root / str(row["left_relpath"])), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(campaign_root / str(row["right_relpath"])), cv2.IMREAD_GRAYSCALE)
            if left is None or right is None:
                raise FileNotFoundError(f"missing FIReStereo pair: {row['sample_key']}")
            left_images.append(left)
            stereo_masks.append(stereo_candidate_mask(left, right))
        for index, (row, _image) in enumerate(zip(sequence_rows, left_images, strict=True)):
            mask = temporal_consensus_mask(stereo_masks, index)
            if np.any(mask):
                valid = np.full_like(mask, 255)
                abstention = None
                strength = "sensor_derived"
            else:
                valid = np.zeros_like(mask)
                abstention = "stereo_temporal_fire_consensus_missing"
                strength = "abstention"
            anchor = smoke_base(mask)
            stem = str(row["sample_key"]).replace(":", "-")
            relative_mask = Path("derived") / "firestereo-firesgl" / "masks" / f"{stem}.png"
            relative_valid = Path("derived") / "firestereo-firesgl" / "valid" / f"{stem}.png"
            mask_sha = _write_png(campaign_root / relative_mask, mask)
            valid_sha = _write_png(campaign_root / relative_valid, valid)
            image_path = campaign_root / str(row["left_relpath"])
            output_rows.append(
                {
                    "sample_id": f"firesgl:{row['sample_key']}",
                    "source_id": "firestereo-firesgl",
                    "source_revision": SOURCE_REVISION,
                    "split": row["split"],
                    "split_group": f"firestereo:{sequence_group}",
                    "image_relpath": row["left_relpath"],
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "stereo_evidence_relpath": row["right_relpath"],
                    "mask_relpath": relative_mask.as_posix(),
                    "mask_sha256": mask_sha,
                    "valid_mask_relpath": relative_valid.as_posix(),
                    "valid_mask_sha256": valid_sha,
                    "mask_quality": "thermal_stereo_temporal_consensus_weak",
                    "annotation_strength": strength,
                    "sample_validation_status": "sensor_generated_weak",
                    "anchor_points": (
                        [{"kind": "fire_base", "x": anchor[0], "y": anchor[1]}]
                        if anchor is not None
                        else []
                    ),
                    "visual_abstention_reason": abstention,
                    "license": "FIReStereo-upstream-terms",
                    "redistribution_allowed": False,
                    "is_operational_incident": False,
                    "sample_weight": 1.5,
                    "stereo_baseline_m": 0.246,
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.jsonl"
    output_rows.sort(key=lambda row: str(row["sample_id"]))
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_id": "firestereo-firesgl",
        "source_revision": SOURCE_REVISION,
        "rows": len(output_rows),
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
