"""Hydrate a composed DINOv3 corpus on persistent storage and verify it globally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from fireviewer_model_lab.training.dinov3_corpus_identity import (
    benchmark_matches,
    decoded_pixel_sha256,
    load_benchmark_denylist,
    phash64_imagehash_v1,
    validate_canonical_event_splits,
    validate_composed_row_identities,
    validate_source_identity_contract,
)

SPLITS = frozenset({"train", "validation", "test"})
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
HFLoader = Callable[[str, str, str], Iterable[dict[str, Any]]]
S3Fetcher = Callable[[dict[str, Any]], bytes | Iterable[bytes]]
DEFAULT_COMPOSITION_REGISTRY = (
    Path(__file__).with_name("registries") / "dinov3-multitask-composition-v1.json"
)
DEFAULT_BENCHMARK_DENYLIST = (
    Path(__file__).with_name("registries") / "dinov3-independent-benchmark-denylist-v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"composition row is not an object: {line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _extension(value: Any, default: str) -> str:
    suffix = PurePosixPath(str(value or "")).suffix.casefold()
    return suffix if suffix in ALLOWED_EXTENSIONS else default


def _safe_destination(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"unsafe hydration destination: {relative}")
    resolved_root = Path(os.path.abspath(root))
    # The relative path is already constrained to safe path components. Keep the
    # containment check lexical so a remote volume does not receive two metadata
    # lookups for every one of the 100k+ expected artifacts.
    destination = resolved_root / Path(*posix.parts)
    try:
        destination.relative_to(resolved_root)
    except ValueError:
        raise ValueError(f"hydration destination escapes data root: {relative}")
    return destination


def _chunks(value: bytes | Iterable[bytes]) -> Iterator[bytes]:
    if isinstance(value, bytes):
        if value:
            yield value
        return
    for chunk in value:
        if chunk:
            yield chunk


def _materialize(
    destination: Path,
    expected_sha256: str,
    payload: bytes | Iterable[bytes],
    *,
    allow_reuse: bool = True,
) -> tuple[int, bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if allow_reuse and destination.is_file() and _sha256(destination) == expected_sha256:
        return destination.stat().st_size, True
    partial = destination.with_name(
        f"{destination.name}.partial-{os.getpid()}-{threading.get_ident()}"
    )
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with partial.open("wb") as handle:
            for chunk in _chunks(payload):
                digest.update(chunk)
                size += len(chunk)
                handle.write(chunk)
        if size <= 0 or digest.hexdigest() != expected_sha256:
            raise ValueError(f"hydrated payload hash mismatch: {destination.name}")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return size, False


def _default_hf_loader(repository: str, revision: str, split: str) -> Iterable[dict[str, Any]]:
    from datasets import Image as HFImage
    from datasets import load_dataset

    mirror = os.environ.get("DINO_HF_DATASET_MIRROR")
    if mirror:
        shards = sorted((Path(mirror) / "data" / split).glob("*.parquet"))
        if not shards:
            raise ValueError(f"pinned HF mirror has no shards for split: {split}")
        dataset = load_dataset(
            "parquet",
            data_files={split: [str(shard) for shard in shards]},
            split=split,
            streaming=True,
        )
    else:
        dataset = load_dataset(
            repository,
            revision=revision,
            split=split,
            streaming=True,
        )
    return dataset.cast_column("image", HFImage(decode=False))


def _build_s3_fetcher(region: str) -> S3Fetcher:
    import boto3
    from botocore.config import Config

    client = boto3.Session(region_name=region).client(
        "s3",
        config=Config(
            retries={"total_max_attempts": 5, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=120,
            max_pool_connections=32,
        ),
    )

    def fetch(locator: dict[str, Any]) -> Iterable[bytes]:
        parsed = urlsplit(str(locator.get("uri") or ""))
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError("invalid S3 artifact locator")
        response = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        body = response["Body"]
        try:
            yield from body.iter_chunks(chunk_size=1024 * 1024)
        finally:
            body.close()

    return fetch


def _prepare_rows(
    rows: list[dict[str, Any]], data_root: Path
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, tuple[str, str, Path]]],
    dict[tuple[str, str], tuple[dict[str, Any], Path]],
]:
    hydrated: list[dict[str, Any]] = []
    hf_targets: dict[tuple[str, str, str], dict[str, tuple[str, str, Path]]] = defaultdict(dict)
    s3_targets: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    for line_number, source in enumerate(rows, 1):
        row = dict(source)
        sample_id = str(row.get("sample_id") or "")
        split = str(row.get("split") or "")
        image_sha = str(row.get("image_sha256") or "").casefold()
        if (
            not sample_id
            or sample_id in sample_ids
            or split not in SPLITS
            or not _valid_sha(image_sha)
            or image_sha in image_hashes
            or row.get("sample_validation_status") != "strict_automated_validated"
            or row.get("validation_profile") != "fireviewer_multitask_composition_v1"
        ):
            raise ValueError(f"invalid composed row identity: {line_number}:{sample_id}")
        sample_ids.add(sample_id)
        image_hashes.add(image_sha)
        image_locator = row.get("image_locator")
        if not isinstance(image_locator, dict):
            raise ValueError(f"image locator missing: {sample_id}")
        image_extension = _extension(
            row.get("image_extension") or image_locator.get("extension"), ".jpg"
        )
        image_relative = f"images/{image_sha}{image_extension}"
        image_destination = _safe_destination(data_root, image_relative)
        kind = image_locator.get("kind")
        if kind == "hf_dataset_row":
            repository = str(image_locator.get("repository") or "")
            revision = str(image_locator.get("revision") or "")
            locator_split = str(image_locator.get("split") or "")
            locator_sample = str(image_locator.get("sample_id") or "")
            locator_sha = str(image_locator.get("sha256") or "").casefold()
            if (
                not repository
                or not revision
                or locator_split != split
                or locator_sample != sample_id
                or locator_sha != image_sha
            ):
                raise ValueError(f"HF locator contract mismatch: {sample_id}")
            key = (repository, revision, split)
            if locator_sample in hf_targets[key]:
                raise ValueError(f"duplicate HF locator sample id: {sample_id}")
            hf_targets[key][locator_sample] = (image_sha, image_relative, image_destination)
        elif kind == "s3_object":
            s3_targets.setdefault(
                ("image", image_sha),
                (image_locator, image_destination),
            )
        else:
            raise ValueError(f"unsupported image locator: {sample_id}:{kind}")
        row["image_relpath"] = image_relative

        implicit_zero = row.get(
            "mask_encoding"
        ) == "implicit_zero_from_explicit_negative" and row.get("annotation_strength") in {
            "negative",
            "temporal_negative",
        }
        segmentation_supervised = row.get("segmentation_supervised") is True
        mask_locator = row.get("mask_locator")
        if segmentation_supervised and not implicit_zero:
            if not isinstance(mask_locator, dict) or mask_locator.get("kind") != "s3_object":
                raise ValueError(f"supervised mask locator missing: {sample_id}")
            mask_sha = str(mask_locator.get("sha256") or "").casefold()
            if not _valid_sha(mask_sha):
                raise ValueError(f"invalid mask SHA-256: {sample_id}")
            mask_extension = _extension(mask_locator.get("extension"), ".png")
            mask_relative = f"masks/{mask_sha}{mask_extension}"
            mask_destination = _safe_destination(data_root, mask_relative)
            s3_targets.setdefault(("mask", mask_sha), (mask_locator, mask_destination))
            row["mask_relpath"] = mask_relative
            row["mask_sha256"] = mask_sha
        elif mask_locator is not None:
            raise ValueError(f"unsupervised row has a mask locator: {sample_id}")

        valid_locator = row.get("valid_mask_locator")
        if valid_locator is not None:
            if not isinstance(valid_locator, dict) or valid_locator.get("kind") != "s3_object":
                raise ValueError(f"invalid valid-mask locator: {sample_id}")
            valid_sha = str(valid_locator.get("sha256") or "").casefold()
            if not _valid_sha(valid_sha):
                raise ValueError(f"invalid valid-mask SHA-256: {sample_id}")
            valid_extension = _extension(valid_locator.get("extension"), ".png")
            valid_relative = f"valid-masks/{valid_sha}{valid_extension}"
            valid_destination = _safe_destination(data_root, valid_relative)
            s3_targets.setdefault(
                ("valid_mask", valid_sha),
                (valid_locator, valid_destination),
            )
            row["valid_mask_relpath"] = valid_relative
            row["valid_mask_sha256"] = valid_sha
        row["hydration_status"] = "sha256_materialization_pending"
        hydrated.append(row)
    return hydrated, hf_targets, s3_targets


def _hydrate_hf(
    targets_by_dataset: dict[tuple[str, str, str], dict[str, tuple[str, str, Path]]],
    loader: HFLoader,
    workers: int,
    known_existing_files: set[Path] | None = None,
) -> Counter[str]:
    if workers <= 0:
        raise ValueError("HF hydration workers must be positive")

    def hydrate_dataset(
        item: tuple[tuple[str, str, str], dict[str, tuple[str, str, Path]]],
    ) -> Counter[str]:
        (repository, revision, split), original_targets = item
        totals: Counter[str] = Counter()
        targets = dict(original_targets)
        for sample_id, (expected_sha, _relative, destination) in list(targets.items()):
            may_exist = (
                destination in known_existing_files
                if known_existing_files is not None
                else destination.is_file()
            )
            if may_exist and destination.is_file() and _sha256(destination) == expected_sha:
                totals["reused_files"] += 1
                totals["reused_bytes"] += destination.stat().st_size
                del targets[sample_id]
        if not targets:
            return totals
        for source in loader(repository, revision, split):
            sample_id = str(source.get("sample_id") or "")
            target = targets.get(sample_id)
            if target is None:
                continue
            expected_sha, _relative, destination = target
            declared_sha = str(source.get("sha256") or source.get("image_sha256") or "")
            image = source.get("image")
            payload = image.get("bytes") if isinstance(image, dict) else None
            if declared_sha.casefold() != expected_sha or not isinstance(payload, bytes):
                raise ValueError(f"HF binary row contract mismatch: {sample_id}")
            size, reused = _materialize(
                destination,
                expected_sha,
                payload,
                allow_reuse=destination in known_existing_files
                if known_existing_files is not None
                else True,
            )
            totals["reused_files" if reused else "downloaded_files"] += 1
            totals["reused_bytes" if reused else "downloaded_bytes"] += size
            del targets[sample_id]
            if not targets:
                break
        if targets:
            examples = sorted(targets)[:5]
            raise ValueError(
                f"HF pinned dataset did not yield {len(targets)} composed rows: {examples}"
            )
        return totals

    totals: Counter[str] = Counter()
    items = sorted(targets_by_dataset.items())
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(items)))) as executor:
        for result in executor.map(hydrate_dataset, items):
            totals.update(result)
    return totals


def _hydrate_s3(
    targets: dict[tuple[str, str], tuple[dict[str, Any], Path]],
    fetcher: S3Fetcher,
    workers: int,
    known_existing_files: set[Path] | None = None,
) -> Counter[str]:
    if workers <= 0:
        raise ValueError("S3 hydration workers must be positive")

    def transfer(item: tuple[tuple[str, str], tuple[dict[str, Any], Path]]) -> tuple[int, bool]:
        (_kind, expected_sha), (locator, destination) = item
        may_exist = (
            destination in known_existing_files
            if known_existing_files is not None
            else destination.is_file()
        )
        if may_exist and destination.is_file() and _sha256(destination) == expected_sha:
            return destination.stat().st_size, True
        return _materialize(
            destination,
            expected_sha,
            fetcher(locator),
            allow_reuse=may_exist,
        )

    totals: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for size, reused in executor.map(transfer, sorted(targets.items())):
            totals["reused_files" if reused else "downloaded_files"] += 1
            totals["reused_bytes" if reused else "downloaded_bytes"] += size
    return totals


def _image_signature(path: Path) -> dict[str, Any]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    width, height = image.size
    if width < 32 or height < 32:
        raise ValueError(f"image is too small: {path.name}:{width}x{height}")
    gray = image.resize((9, 8), Image.Resampling.BILINEAR).convert("L")
    values = np.asarray(gray, dtype=np.uint8).reshape(-1).tolist()
    dhash = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            dhash = (dhash << 1) | int(values[offset + column] > values[offset + column + 1])
    thumbnail = np.asarray(
        image.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.int16
    ).tobytes()
    return {
        "width": width,
        "height": height,
        "decoded_pixel_sha256": decoded_pixel_sha256(image),
        "phash64_imagehash_v1": phash64_imagehash_v1(image),
        "dhash": dhash,
        "thumbnail": thumbnail,
    }


def _strict_binary_array(
    array: np.ndarray[Any, Any], *, label: str, sample_id: Any
) -> np.ndarray[Any, Any]:
    if array.ndim != 2:
        raise ValueError(f"{label} is not single-channel: {sample_id}")
    if np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{label} uses a floating dtype: {sample_id}")
    values = np.unique(array)
    if not bool(np.all(np.isin(values, (0, 1, 255)))):
        raise ValueError(f"{label} has unknown values: {sample_id}")
    return array > 0


def _bottom_band_median(mask: np.ndarray[Any, Any]) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("cannot derive a base point from an empty mask")
    bottom = int(ys.max())
    band_height = max(1, round(mask.shape[0] * 0.02))
    band_x = xs[ys >= max(0, bottom - band_height)]
    return (
        round(float(np.median(band_x)) / max(1, mask.shape[1] - 1), 8),
        round(float(bottom) / max(1, mask.shape[0] - 1), 8),
    )


def _validate_row(row: dict[str, Any], data_root: Path) -> dict[str, Any]:
    image_path = _safe_destination(data_root, str(row["image_relpath"]))
    if _sha256(image_path) != str(row["image_sha256"]):
        raise ValueError(f"final image rehash failed: {row['sample_id']}")
    signature = _image_signature(image_path)
    mask_array: np.ndarray[Any, Any] | None = None
    binary_mask: np.ndarray[Any, Any] | None = None
    if row.get("mask_relpath"):
        mask_path = _safe_destination(data_root, str(row["mask_relpath"]))
        if _sha256(mask_path) != str(row["mask_sha256"]):
            raise ValueError(f"final mask rehash failed: {row['sample_id']}")
        with Image.open(mask_path) as opened:
            if opened.size != (signature["width"], signature["height"]):
                raise ValueError(f"image/mask dimensions differ: {row['sample_id']}")
            mask_array = np.asarray(opened)
        binary_mask = _strict_binary_array(
            mask_array,
            label="supervised mask",
            sample_id=row["sample_id"],
        )
        foreground_fraction = float(np.count_nonzero(binary_mask) / binary_mask.size)
        if foreground_fraction <= 0.0:
            raise ValueError(f"positive supervised mask is empty: {row['sample_id']}")
        if foreground_fraction > 0.95:
            raise ValueError(f"positive supervised mask is implausibly full: {row['sample_id']}")
    valid_binary: np.ndarray[Any, Any] | None = None
    if row.get("valid_mask_relpath"):
        valid_path = _safe_destination(data_root, str(row["valid_mask_relpath"]))
        if _sha256(valid_path) != str(row["valid_mask_sha256"]):
            raise ValueError(f"final valid-mask rehash failed: {row['sample_id']}")
        with Image.open(valid_path) as opened:
            if opened.size != (signature["width"], signature["height"]):
                raise ValueError(f"image/valid-mask dimensions differ: {row['sample_id']}")
            valid_array = np.asarray(opened)
        valid_binary = _strict_binary_array(
            valid_array,
            label="valid mask",
            sample_id=row["sample_id"],
        )
        if not np.any(valid_binary):
            raise ValueError(f"valid mask is empty: {row['sample_id']}")
        if binary_mask is not None and np.any(binary_mask & ~valid_binary):
            raise ValueError(f"positive mask escapes valid mask: {row['sample_id']}")
    presence_targets = row.get("presence_targets")
    if not isinstance(presence_targets, dict) or set(presence_targets) != {
        "flame_visible",
        "smoke_visible",
    }:
        raise ValueError(f"presence targets are invalid: {row['sample_id']}")
    points = row.get("anchor_points") or []
    if points and mask_array is None:
        raise ValueError(f"point-supervised row has no mask: {row['sample_id']}")
    annotation_strength = str(row.get("annotation_strength") or "")
    if annotation_strength in {"negative", "temporal_negative"}:
        if points or any(bool(value) for value in presence_targets.values()):
            raise ValueError(f"negative row has positive evidence: {row['sample_id']}")
        if row.get("mask_encoding") != "implicit_zero_from_explicit_negative":
            raise ValueError(f"negative row has no explicit zero-mask contract: {row['sample_id']}")
        if not str(row.get("visual_abstention_reason") or "").strip():
            raise ValueError(f"negative row has no abstention reason: {row['sample_id']}")
    elif binary_mask is not None and not any(bool(value) for value in presence_targets.values()):
        raise ValueError(f"positive mask contradicts presence targets: {row['sample_id']}")
    for point in points:
        x, y = float(point["x"]), float(point["y"])
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"point is out of bounds: {row['sample_id']}")
        kind = str(point.get("kind") or "")
        expected_presence = {
            "fire_base": "flame_visible",
            "smoke_column_base": "smoke_visible",
        }.get(kind)
        if expected_presence is None:
            raise ValueError(f"point kind is not admitted: {row['sample_id']}:{kind}")
        if presence_targets.get(expected_presence) is not True:
            raise ValueError(f"point contradicts presence target: {row['sample_id']}:{kind}")
        assert binary_mask is not None
        height, width = binary_mask.shape
        center_x = round(x * (width - 1))
        center_y = round(y * (height - 1))
        radius = max(2, round(max(width, height) * 0.02))
        window = binary_mask[
            max(0, center_y - radius) : min(height, center_y + radius + 1),
            max(0, center_x - radius) : min(width, center_x + radius + 1),
        ]
        if not np.any(window):
            raise ValueError(f"point is not supported by its mask: {row['sample_id']}")
        foreground_y = np.flatnonzero(np.any(binary_mask, axis=1))
        lower_quartile = float(np.quantile(foreground_y, 0.75))
        if center_y + radius < lower_quartile:
            raise ValueError(f"base point is too high in its mask: {row['sample_id']}")
        derivation = str(row.get("point_derivation") or "")
        if derivation in {
            "deterministic_binary_mask_bottom_band_median",
            "sensor_mask_bottom_band_median",
        }:
            expected_x, expected_y = _bottom_band_median(binary_mask)
            if abs(x - expected_x) > 1e-7 or abs(y - expected_y) > 1e-7:
                raise ValueError(
                    f"declared bottom-band-median point does not match mask: {row['sample_id']}"
                )
    return {"sample_id": row["sample_id"], "split": row["split"], **signature}


def _near_duplicate_report(signatures: list[dict[str, Any]]) -> dict[str, Any]:
    indexes: dict[tuple[int, int], list[int]] = defaultdict(list)
    evidence: list[dict[str, Any]] = []
    cross_split_pairs = 0
    within_split_pairs = 0
    segments = ((0, 22), (22, 21), (43, 21))
    for index, current in enumerate(signatures):
        value = int(current["dhash"])
        candidates: set[int] = set()
        for segment_index, (offset, bits) in enumerate(segments):
            key = (segment_index, (value >> offset) & ((1 << bits) - 1))
            candidates.update(indexes[key])
        current_thumb = np.frombuffer(current["thumbnail"], dtype=np.int16)
        for other_index in candidates:
            other = signatures[other_index]
            distance = (value ^ int(other["dhash"])).bit_count()
            if distance > 2:
                continue
            other_thumb = np.frombuffer(other["thumbnail"], dtype=np.int16)
            mean_absolute_error = float(np.abs(current_thumb - other_thumb).mean())
            if mean_absolute_error > 3.0:
                continue
            cross_split = current["split"] != other["split"]
            cross_split_pairs += int(cross_split)
            within_split_pairs += int(not cross_split)
            if len(evidence) < 100:
                evidence.append(
                    {
                        "left": other["sample_id"],
                        "left_split": other["split"],
                        "right": current["sample_id"],
                        "right_split": current["split"],
                        "dhash_distance": distance,
                        "thumbnail_mean_absolute_error": mean_absolute_error,
                    }
                )
        for segment_index, (offset, bits) in enumerate(segments):
            key = (segment_index, (value >> offset) & ((1 << bits) - 1))
            indexes[key].append(index)
    return {
        "cross_split_pairs": cross_split_pairs,
        "within_split_pairs": within_split_pairs,
        "evidence": evidence,
        "evidence_truncated": cross_split_pairs + within_split_pairs > len(evidence),
    }


def hydrate_composition(
    *,
    composition_manifest: Path,
    composition_report: Path,
    composition_integrity_receipt: Path,
    composition_registry: Path,
    benchmark_denylist_path: Path,
    data_root: Path,
    output_dir: Path,
    region: str = "eu-west-2",
    hf_workers: int = 3,
    s3_workers: int = 12,
    validation_workers: int = 8,
    hf_loader: HFLoader | None = None,
    s3_fetcher: S3Fetcher | None = None,
) -> dict[str, Any]:
    source_report = json.loads(composition_report.read_text(encoding="utf-8"))
    integrity_receipt = json.loads(composition_integrity_receipt.read_text(encoding="utf-8"))
    registry = json.loads(composition_registry.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 2:
        raise ValueError("unsupported composition registry schema")
    registry_sha256 = _sha256(composition_registry)
    identities = validate_source_identity_contract(registry)
    benchmark_boundary = registry.get("benchmark_boundary")
    if not isinstance(benchmark_boundary, dict):
        raise ValueError("benchmark boundary contract is missing")
    benchmark_denylist = load_benchmark_denylist(benchmark_denylist_path, benchmark_boundary)
    manifest_sha = _sha256(composition_manifest)
    source_hard_gates = source_report.get("hard_gates")
    source_quality_gates = source_report.get("quality_gates")
    if (
        source_report.get("integrity_gates_passed") is not True
        or source_report.get("schema_version") != 2
        or source_report.get("pilot_corpus_ready") is not True
        or source_report.get("publication_allowed") is not False
        or source_report.get("manifest_sha256") != manifest_sha
        or source_report.get("composition_registry_sha256") != registry_sha256
        or source_report.get("source_identity_contract_sha256") != identities.contract_sha256
        or source_report.get("benchmark_denylist_sha256") != benchmark_denylist.sha256
        or source_report.get("composition_integrity_receipt_sha256")
        != _sha256(composition_integrity_receipt)
        or not isinstance(source_hard_gates, dict)
        or source_hard_gates.get("perceptual_near_duplicate_pairs_max") != 0
        or not isinstance(source_quality_gates, dict)
    ):
        raise ValueError("composition receipt is not eligible for hydration")
    expected_integrity_receipt = {
        "schema_version": 2,
        "campaign_id": source_report.get("campaign_id"),
        "composition_registry_sha256": source_report.get("composition_registry_sha256"),
        "source_identity_contract_sha256": identities.contract_sha256,
        "benchmark_denylist_sha256": benchmark_denylist.sha256,
        "manifest_sha256": manifest_sha,
        "composition_rows": source_report.get("composition_rows"),
        "detection_revision": source_report.get("detection_revision"),
        "integrity_gates_passed": True,
        "pilot_corpus_ready": True,
        "professional_corpus_ready": source_report.get("professional_corpus_ready"),
        "publication_allowed": False,
    }
    if any(
        integrity_receipt.get(key) != value for key, value in expected_integrity_receipt.items()
    ):
        raise ValueError("composition integrity receipt contract failed")
    professional_corpus_ready = source_report.get("professional_corpus_ready") is True
    quality_gate_deficits = source_report.get("quality_gate_deficits")
    if not isinstance(quality_gate_deficits, dict):
        raise ValueError("composition receipt has no quality gate accounting")
    if professional_corpus_ready != (not quality_gate_deficits):
        raise ValueError("composition professional readiness contradicts quality deficits")
    source_rows = _read_jsonl(composition_manifest)
    if len(source_rows) != int(source_report.get("composition_rows", -1)):
        raise ValueError("composition row count differs from its report")
    if any(
        row.get("campaign_id") != source_report.get("campaign_id")
        or row.get("composition_registry_sha256")
        != source_report.get("composition_registry_sha256")
        for row in source_rows
    ):
        raise ValueError("composition rows are not bound to their registry campaign")
    validate_composed_row_identities(source_rows, registry=registry, identities=identities)
    source_event_leaks = validate_canonical_event_splits(source_rows)
    if source_event_leaks:
        raise ValueError(f"composition has canonical event leakage: {source_event_leaks[:5]}")
    raw_benchmark_matches = sorted(
        str(row.get("sample_id") or "")
        for row in source_rows
        if benchmark_matches(
            raw_sha256=str(row.get("image_sha256") or ""),
            denylist=benchmark_denylist,
        )["raw_sha256"]
    )
    if raw_benchmark_matches:
        raise ValueError(f"composition contains benchmark raw hashes: {raw_benchmark_matches[:5]}")
    data_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Network volumes make one stat call per expected file prohibitively slow.
    # Inventory the existing tree once, then skip reuse probes for fresh targets.
    known_existing_files = {
        Path(directory) / filename
        for directory, _subdirs, filenames in os.walk(data_root)
        for filename in filenames
    }
    rows, hf_targets, s3_targets = _prepare_rows(source_rows, data_root)
    # AWS login credentials are deliberately short-lived. Materialize the small
    # S3 overlay set first, before the much larger local HF extraction.
    s3_totals = _hydrate_s3(
        s3_targets,
        s3_fetcher or _build_s3_fetcher(region),
        s3_workers,
        known_existing_files,
    )
    hf_totals = _hydrate_hf(
        hf_targets,
        hf_loader or _default_hf_loader,
        hf_workers,
        known_existing_files,
    )

    def validate(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return _validate_row(row, data_root), None
        except Exception as exc:
            return None, f"{row.get('sample_id')}:{type(exc).__name__}:{exc}"

    signatures: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=validation_workers) as executor:
        for signature, error in executor.map(validate, rows):
            if error:
                if len(validation_errors) < 100:
                    validation_errors.append(error)
            elif signature is not None:
                signatures.append(signature)
    decoded_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signature in signatures:
        decoded_groups[str(signature["decoded_pixel_sha256"])].append(signature)
    decoded_duplicates = [group for group in decoded_groups.values() if len(group) > 1]
    decoded_duplicate_evidence = [
        [f"{item['split']}:{item['sample_id']}" for item in group]
        for group in decoded_duplicates[:100]
    ]
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row.get("split_group") or "")].add(str(row["split"]))
    group_leaks = sorted(group for group, splits in groups.items() if len(splits) > 1)
    near_duplicates = _near_duplicate_report(signatures)
    signature_by_sample = {str(signature["sample_id"]): signature for signature in signatures}
    decoded_benchmark_matches: list[str] = []
    phash_benchmark_matches: list[str] = []
    for row in rows:
        signature = signature_by_sample.get(str(row["sample_id"]))
        if signature is None:
            continue
        row["decoded_pixel_sha256"] = signature["decoded_pixel_sha256"]
        row["phash64_imagehash_v1"] = f"{int(signature['phash64_imagehash_v1']):016x}"
        matches = benchmark_matches(
            raw_sha256=str(row["image_sha256"]),
            decoded_sha256=str(signature["decoded_pixel_sha256"]),
            phash64=int(signature["phash64_imagehash_v1"]),
            denylist=benchmark_denylist,
        )
        if matches["decoded_sha256"]:
            decoded_benchmark_matches.append(str(row["sample_id"]))
        if matches["phash"]:
            phash_benchmark_matches.append(str(row["sample_id"]))
    hard_errors: list[str] = []
    if validation_errors:
        hard_errors.append(f"payload_or_semantic_validation_errors:{len(validation_errors)}")
    if decoded_duplicates:
        hard_errors.append(f"decoded_pixel_duplicate_groups:{len(decoded_duplicates)}")
    if group_leaks:
        hard_errors.append(f"split_group_leakage:{len(group_leaks)}")
    event_leaks = validate_canonical_event_splits(rows)
    if event_leaks:
        hard_errors.append(f"canonical_event_leakage:{len(event_leaks)}")
    if decoded_benchmark_matches:
        hard_errors.append(f"benchmark_decoded_hash_matches:{len(decoded_benchmark_matches)}")
    if phash_benchmark_matches:
        hard_errors.append(f"benchmark_phash_matches:{len(phash_benchmark_matches)}")
    if near_duplicates["cross_split_pairs"]:
        hard_errors.append(
            f"cross_split_near_duplicate_pairs:{near_duplicates['cross_split_pairs']}"
        )
    if near_duplicates["within_split_pairs"]:
        hard_errors.append(
            f"within_split_near_duplicate_pairs:{near_duplicates['within_split_pairs']}"
        )
    integrity_passed = not hard_errors
    gpu_smoke_ready = integrity_passed and source_report.get("pilot_corpus_ready") is True
    full_training_ready = integrity_passed and professional_corpus_ready
    for row in rows:
        row["hydration_status"] = (
            "sha256_and_semantic_validation_passed"
            if integrity_passed
            else "hydrated_global_gate_failed"
        )
    hydrated_manifest = output_dir / "hydrated_multitask_manifest.jsonl"
    _write_jsonl(hydrated_manifest, rows)
    report = {
        "schema_version": 2,
        "composition_manifest_sha256": manifest_sha,
        "composition_report_sha256": _sha256(composition_report),
        "composition_integrity_receipt_sha256": _sha256(composition_integrity_receipt),
        "campaign_id": source_report.get("campaign_id"),
        "composition_registry_sha256": source_report.get("composition_registry_sha256"),
        "source_identity_contract_sha256": identities.contract_sha256,
        "benchmark_denylist_sha256": benchmark_denylist.sha256,
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "hf_materialization": dict(sorted(hf_totals.items())),
        "s3_materialization": dict(sorted(s3_totals.items())),
        "verified_image_rows": len(signatures),
        "validation_errors": validation_errors,
        "decoded_pixel_duplicate_groups": len(decoded_duplicates),
        "decoded_pixel_duplicate_evidence": decoded_duplicate_evidence,
        "split_group_leakage": group_leaks,
        "canonical_event_leakage": event_leaks,
        "benchmark_raw_hash_matches": len(raw_benchmark_matches),
        "benchmark_decoded_hash_matches": len(decoded_benchmark_matches),
        "benchmark_phash_matches": len(phash_benchmark_matches),
        "benchmark_decoded_hash_match_samples": decoded_benchmark_matches[:100],
        "benchmark_phash_match_samples": phash_benchmark_matches[:100],
        "perceptual_near_duplicates": near_duplicates,
        "hard_gate_errors": hard_errors,
        "hydration_integrity_passed": integrity_passed,
        "quality_gate_deficits": quality_gate_deficits,
        "pilot_corpus_ready": source_report.get("pilot_corpus_ready") is True,
        "professional_corpus_ready": professional_corpus_ready,
        "ready_for_gpu_finite_loss_smoke": gpu_smoke_ready,
        "ready_for_full_training": full_training_ready,
        "training_ready": False,
        "publication_allowed": False,
        "hf_replacement_allowed": False,
        "reviews_admitted": False,
        "next_gate": (
            "gpu_finite_loss_smoke"
            if full_training_ready
            else (
                "corpus_expansion_before_full_training"
                if integrity_passed
                else "automatic_recomposition"
            )
        ),
        "manifest": hydrated_manifest.name,
        "manifest_sha256": _sha256(hydrated_manifest),
    }
    hydration_integrity_receipt = output_dir / "hydration_integrity_receipt.json"
    if integrity_passed:
        _write_json(
            hydration_integrity_receipt,
            {
                "schema_version": 2,
                "manifest_sha256": report["manifest_sha256"],
                "rows": len(rows),
                "campaign_id": report["campaign_id"],
                "composition_registry_sha256": report["composition_registry_sha256"],
                "source_identity_contract_sha256": identities.contract_sha256,
                "benchmark_denylist_sha256": benchmark_denylist.sha256,
                "composition_manifest_sha256": report["composition_manifest_sha256"],
                "composition_report_sha256": report["composition_report_sha256"],
                "composition_integrity_receipt_sha256": report[
                    "composition_integrity_receipt_sha256"
                ],
                "hydration_integrity_passed": True,
                "pilot_corpus_ready": report["pilot_corpus_ready"],
                "professional_corpus_ready": report["professional_corpus_ready"],
                "quality_gate_deficits": report["quality_gate_deficits"],
                "training_ready": False,
                "publication_allowed": False,
            },
        )
        report["hydration_integrity_receipt"] = hydration_integrity_receipt.name
        report["hydration_integrity_receipt_sha256"] = _sha256(hydration_integrity_receipt)
    else:
        report["hydration_integrity_receipt"] = None
        report["hydration_integrity_receipt_sha256"] = None
    _write_json(output_dir / "hydration_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--composition-report", type=Path, required=True)
    parser.add_argument("--composition-integrity-receipt", type=Path, required=True)
    parser.add_argument("--composition-registry", type=Path, default=DEFAULT_COMPOSITION_REGISTRY)
    parser.add_argument("--benchmark-denylist", type=Path, default=DEFAULT_BENCHMARK_DENYLIST)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--region", default="eu-west-2")
    parser.add_argument("--hf-workers", type=int, default=3)
    parser.add_argument("--s3-workers", type=int, default=12)
    parser.add_argument("--validation-workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = hydrate_composition(
        composition_manifest=args.composition_manifest,
        composition_report=args.composition_report,
        composition_integrity_receipt=args.composition_integrity_receipt,
        composition_registry=args.composition_registry,
        benchmark_denylist_path=args.benchmark_denylist,
        data_root=args.data_root,
        output_dir=args.output_dir,
        region=args.region,
        hf_workers=args.hf_workers,
        s3_workers=args.s3_workers,
        validation_workers=args.validation_workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
