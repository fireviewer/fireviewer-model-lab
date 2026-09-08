"""Strict, isolated audit of the immutable IJmond smoke segmentation release.

The source masks supervise smoke segmentation and presence.  A
``smoke_column_base`` point is emitted only when conservative image and mask
geometry proves that the human-revised smoke component reaches a visible
vertical emission source.  Every other valid positive remains an abstention
example; a mask bottom is never accepted as a point by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat
from scipy import fftpack, ndimage

SOURCE_ID = "ijmond-industrial-smoke-segmentation-v3"
SOURCE_FAMILY = "IJmond industrial smoke segmentation"
FIGSHARE_ARTICLE_ID = 31_847_188
FIGSHARE_VERSION = 3
FIGSHARE_FILE_ID = 63_066_070
SOURCE_DOI = "10.21942/uva.31847188.v3"
SOURCE_RECORD_URL = "https://doi.org/10.21942/uva.31847188.v3"
SOURCE_PAPER_URL = "https://arxiv.org/abs/2603.23754v1"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
ARCHIVE_FILENAME = "ijmond_seg.tar.gz"
ARCHIVE_DOWNLOAD_URL = f"https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}"
EXPECTED_ARCHIVE_BYTES = 266_159_976
EXPECTED_ARCHIVE_MD5 = "20eb0e868e1fb8575612b9a4b77367e1"
EXPECTED_ARCHIVE_SHA256 = "5a26dbad99b5e590608cdbb98f269f4a24266c020852401090f040f75bcb6343"
EXPECTED_RAW_IMAGES = 900
EXPECTED_RAW_POSITIVE_IMAGES = 893
EXPECTED_RAW_POLYGONS = 1_209
PROFESSIONAL_SOURCE_POINT_CAP = 750

CAMERAS = ("hoogovens_6_7", "kooks_1", "kooks_2")
MASK_VALUES = frozenset({0, 155, 255})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
MAX_PHASH_DISTANCE = 6
EPISODE_GAP_SECONDS = 30 * 60
SPLIT_TARGET_RATIOS = {"train": 0.70, "validation": 0.10, "test": 0.20}
# Absolute row-ratio deviation allowed after indivisible chronological episode groups.
SPLIT_RATIO_TOLERANCE = 0.05
MAX_MEMBER_COUNT = 100_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 16 * 1024**3
MAX_SINGLE_MEMBER_BYTES = 2 * 1024**3
MAX_JSON_BYTES = 64 * 1024**2

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
UTC_COMPAT = timezone.utc  # noqa: UP017 - SageMaker image currently uses Python 3.10
_DATE_TIME_PATTERNS = (
    re.compile(
        r"(?<!\d)(?P<year>20\d{2})[-_]?(?P<month>0[1-9]|1[0-2])[-_]?"
        r"(?P<day>0[1-9]|[12]\d|3[01])[tT_ -]+"
        r"(?P<hour>[01]\d|2[0-3])[-_:]?"
        r"(?P<minute>[0-5]\d)(?:[-_:]?(?P<second>[0-5]\d))?(?!\d)"
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def validate_figshare_article(article: dict[str, Any]) -> dict[str, Any]:
    """Validate the live lightweight metadata against the pinned v3 contract."""
    errors: list[str] = []
    if article.get("id") != FIGSHARE_ARTICLE_ID:
        errors.append("figshare_article_id_mismatch")
    if article.get("version") != FIGSHARE_VERSION:
        errors.append("figshare_version_mismatch")
    if str(article.get("doi") or "") != SOURCE_DOI:
        errors.append("figshare_doi_mismatch")
    if article.get("is_public") is not True:
        errors.append("figshare_article_not_public")
    license_record = article.get("license") or {}
    if license_record.get("name") != SOURCE_LICENSE:
        errors.append("figshare_license_mismatch")
    if str(license_record.get("url") or "").rstrip("/") != SOURCE_LICENSE_URL.rstrip("/"):
        errors.append("figshare_license_url_mismatch")
    files = [item for item in article.get("files") or [] if item.get("id") == FIGSHARE_FILE_ID]
    if len(files) != 1:
        errors.append("figshare_pinned_file_missing_or_ambiguous")
        file_record: dict[str, Any] = {}
    else:
        file_record = files[0]
        expected = {
            "name": ARCHIVE_FILENAME,
            "size": EXPECTED_ARCHIVE_BYTES,
            "computed_md5": EXPECTED_ARCHIVE_MD5,
        }
        for field, value in expected.items():
            if file_record.get(field) != value:
                errors.append(f"figshare_file_{field}_mismatch")
        if str(file_record.get("download_url") or "") != ARCHIVE_DOWNLOAD_URL:
            errors.append("figshare_download_url_mismatch")
    return {
        "source_contract_verified": not errors,
        "gate_errors": errors,
        "article_id": article.get("id"),
        "version": article.get("version"),
        "doi": article.get("doi"),
        "license": license_record.get("name"),
        "license_url": license_record.get("url"),
        "file_id": file_record.get("id"),
        "file_name": file_record.get("name"),
        "file_size": file_record.get("size"),
        "file_md5": file_record.get("computed_md5"),
        "download_url": file_record.get("download_url"),
    }


def fetch_figshare_article() -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 - fixed official HTTPS metadata endpoint
        FIGSHARE_API_URL,
        headers={"Accept": "application/json", "User-Agent": "FireViewer-Pointing/2.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        if response.geturl().split(":", 1)[0].casefold() != "https":
            raise ValueError("Figshare metadata redirected away from HTTPS")
        payload = response.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("Figshare metadata response exceeds bounded size")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Figshare metadata response is not an object")
    return value


def download_archive(destination: Path) -> dict[str, Any]:
    """Stream the pinned archive into ephemeral cloud storage and verify it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    downloaded = 0
    request = urllib.request.Request(  # noqa: S310 - immutable Figshare file id
        ARCHIVE_DOWNLOAD_URL,
        headers={"Accept-Encoding": "identity", "User-Agent": "FireViewer-Pointing/2.0"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=180) as response,  # noqa: S310
            partial.open("wb") as output,
        ):
            if response.geturl().split(":", 1)[0].casefold() != "https":
                raise ValueError("Figshare archive redirected away from HTTPS")
            while chunk := response.read(4 * 1024 * 1024):
                downloaded += len(chunk)
                if downloaded > EXPECTED_ARCHIVE_BYTES:
                    raise ValueError("IJmond archive exceeds the pinned byte count")
                md5.update(chunk)
                sha256.update(chunk)
                output.write(chunk)
        if downloaded != EXPECTED_ARCHIVE_BYTES:
            raise ValueError(
                f"IJmond archive byte-count mismatch: {downloaded} != {EXPECTED_ARCHIVE_BYTES}"
            )
        if md5.hexdigest() != EXPECTED_ARCHIVE_MD5:
            raise ValueError(f"IJmond archive MD5 mismatch: {md5.hexdigest()}")
        if sha256.hexdigest() != EXPECTED_ARCHIVE_SHA256:
            raise ValueError(f"IJmond archive SHA-256 mismatch: {sha256.hexdigest()}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "bytes": downloaded,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "figshare_file_id": FIGSHARE_FILE_ID,
        "download_url": ARCHIVE_DOWNLOAD_URL,
        "pinned_md5_verified": True,
        "pinned_sha256_verified": True,
        "pinned_size_verified": True,
    }


def _normalized_member_name(name: str) -> str:
    if "\x00" in name:
        raise ValueError("unsafe tar member contains a NUL byte")
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"unsafe tar member path: {name}")
    if not normalized.parts or normalized.as_posix() in {"", "."}:
        raise ValueError("unsafe empty tar member path")
    if ":" in normalized.parts[0]:
        raise ValueError(f"unsafe drive-qualified tar member: {name}")
    return normalized.as_posix()


def _validated_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    total_bytes = 0
    for member in archive:
        if len(members) >= MAX_MEMBER_COUNT:
            raise ValueError("IJmond tar member count exceeds safety limit")
        normalized = _normalized_member_name(member.name)
        key = normalized.casefold()
        if key in members:
            raise ValueError(f"duplicate tar member path: {normalized}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise ValueError(f"unsupported tar member type: {normalized}")
        if member.size < 0 or member.size > MAX_SINGLE_MEMBER_BYTES:
            raise ValueError(f"unsafe tar member size: {normalized}:{member.size}")
        if member.isfile():
            total_bytes += member.size
            if total_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("IJmond tar uncompressed byte count exceeds safety limit")
        members[key] = member
    return members


def _member_payload(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"unreadable tar member: {member.name}")
    with source:
        payload = source.read()
    if len(payload) != member.size:
        raise ValueError(f"short tar member read: {member.name}")
    return payload


def _is_cropped(path: str) -> bool:
    return "cropped" in {part.casefold() for part in PurePosixPath(path).parts}


def _is_mask(path: str) -> bool:
    parts = {part.casefold() for part in PurePosixPath(path).parts[:-1]}
    stem_tokens = set(re.findall(r"[a-z0-9]+", PurePosixPath(path).stem.casefold()))
    return bool(parts & {"mask", "masks", "raw_masks", "raw-masks"}) or "mask" in stem_tokens


def _mask_key(path: str) -> str:
    stem = PurePosixPath(path).stem.casefold()
    stem = re.sub(r"^(?:raw[_ -]?)?mask(?:ed)?[_ -]?", "", stem)
    stem = re.sub(r"[_ -]?(?:raw[_ -]?)?mask(?:ed)?$", "", stem)
    return re.sub(r"[^a-z0-9]+", "", stem)


def _coco_polygon_count(annotations: list[dict[str, Any]]) -> tuple[int, int]:
    polygons = 0
    unsupported = 0
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            unsupported += 1
            continue
        if segmentation and all(isinstance(value, (int, float)) for value in segmentation):
            polygons += 1
            continue
        for polygon in segmentation:
            if isinstance(polygon, list) and len(polygon) >= 6:
                polygons += 1
            else:
                unsupported += 1
    return polygons, unsupported


def _find_raw_coco(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
) -> tuple[str, dict[str, Any], bytes]:
    candidates: list[tuple[int, int, str, dict[str, Any], bytes]] = []
    for member in members.values():
        name = _normalized_member_name(member.name)
        if not member.isfile() or _is_cropped(name) or not name.casefold().endswith(".json"):
            continue
        if member.size > MAX_JSON_BYTES:
            continue
        try:
            payload = _member_payload(archive, member)
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        images = value.get("images")
        annotations = value.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            continue
        polygons, _ = _coco_polygon_count(annotations)
        candidates.append((len(images), polygons, name, value, payload))
    if not candidates:
        raise ValueError("no non-cropped MS COCO annotation document found")
    candidates.sort(
        key=lambda row: (
            row[0] == EXPECTED_RAW_IMAGES and row[1] == EXPECTED_RAW_POLYGONS,
            row[0],
            row[1],
            row[2],
        ),
        reverse=True,
    )
    _, _, name, value, payload = candidates[0]
    return name, value, payload


def _resolve_image_member(
    file_name: str,
    members: dict[str, tarfile.TarInfo],
) -> tarfile.TarInfo | None:
    normalized = _normalized_member_name(file_name)
    candidates = []
    for member in members.values():
        name = _normalized_member_name(member.name)
        if (
            member.isfile()
            and not _is_cropped(name)
            and not _is_mask(name)
            and PurePosixPath(name).suffix.casefold() in IMAGE_SUFFIXES
        ):
            candidates.append(member)
    exact = [member for member in candidates if _normalized_member_name(member.name) == normalized]
    if len(exact) == 1:
        return exact[0]
    suffix = [
        member
        for member in candidates
        if _normalized_member_name(member.name).casefold().endswith(normalized.casefold())
    ]
    if len(suffix) == 1:
        return suffix[0]
    basename = PurePosixPath(normalized).name.casefold()
    by_name = [
        member for member in candidates if PurePosixPath(member.name).name.casefold() == basename
    ]
    return by_name[0] if len(by_name) == 1 else None


def _resolve_mask_member(
    image_member: tarfile.TarInfo,
    members: dict[str, tarfile.TarInfo],
) -> tarfile.TarInfo | None:
    image_key = _mask_key(image_member.name)
    candidates = []
    for member in members.values():
        name = _normalized_member_name(member.name)
        if (
            member.isfile()
            and not _is_cropped(name)
            and _is_mask(name)
            and PurePosixPath(name).suffix.casefold() in IMAGE_SUFFIXES
            and _mask_key(name) == image_key
        ):
            candidates.append(member)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    image_parts = PurePosixPath(_normalized_member_name(image_member.name)).parts
    ranked: list[tuple[int, str, tarfile.TarInfo]] = []
    for member in candidates:
        mask_parts = PurePosixPath(_normalized_member_name(member.name)).parts
        shared_positions = sum(
            left.casefold() == right.casefold()
            for left, right in zip(image_parts, mask_parts, strict=False)
        )
        ranked.append((-shared_positions, _normalized_member_name(member.name), member))
    ranked.sort(key=lambda item: (item[0], item[1]))
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][2]


def _camera(value: str) -> str | None:
    tokens = re.findall(r"[a-z]+|\d+", value.casefold())
    matches: list[str] = []
    for camera in CAMERAS:
        expected = re.findall(r"[a-z]+|\d+", camera)
        if any(tokens[index : index + len(expected)] == expected for index in range(len(tokens))):
            matches.append(camera)
    return matches[0] if len(matches) == 1 else None


def _resolve_camera(value: str, coco_row: dict[str, Any]) -> tuple[str | None, str | None]:
    filename_camera = _camera(value)
    metadata_values = [
        str(coco_row[key]).strip()
        for key in ("camera", "camera_name", "camera_id")
        if coco_row.get(key) not in (None, "")
    ]
    metadata_cameras = [_camera(item) for item in metadata_values]
    if metadata_values and any(item is None for item in metadata_cameras):
        return None, "camera_metadata_invalid"
    resolved_metadata = {item for item in metadata_cameras if item is not None}
    if len(resolved_metadata) > 1:
        return None, "camera_metadata_fields_disagree"
    metadata_camera = next(iter(resolved_metadata), None)
    if filename_camera and metadata_camera and filename_camera != metadata_camera:
        return None, "camera_filename_metadata_mismatch"
    resolved = metadata_camera or filename_camera
    return (resolved, None) if resolved else (None, "camera_unresolved")


def _parse_timestamp(value: str) -> datetime | None:
    captured = value.strip()
    if not captured:
        return None
    if re.search(r"[tT ]\d{2}:\d{2}", captured):
        try:
            parsed = datetime.fromisoformat(captured.replace("Z", "+00:00"))
            return (
                parsed.astimezone(UTC_COMPAT)
                if parsed.tzinfo
                else parsed.replace(tzinfo=UTC_COMPAT)
            )
        except ValueError:
            pass
    for pattern in _DATE_TIME_PATTERNS:
        match = pattern.search(captured)
        if match is None:
            continue
        parts = {key: int(number) for key, number in match.groupdict(default="0").items()}
        try:
            return datetime(
                parts["year"],
                parts["month"],
                parts["day"],
                parts.get("hour", 0),
                parts.get("minute", 0),
                parts.get("second", 0),
                tzinfo=UTC_COMPAT,
            )
        except ValueError:
            continue
    return None


def _classify_coco_date_captured(coco_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a dataset-constant COCO date as export metadata, fail closed."""
    raw_values = [
        str(row.get("date_captured") or "").strip()
        for row in coco_rows
    ]
    filename_values = [str(row.get("file_name") or "") for row in coco_rows]
    parsed_metadata = [_parse_timestamp(value) if value else None for value in raw_values]
    parsed_filenames = [_parse_timestamp(value) if value else None for value in filename_values]
    distinct_raw = sorted({value for value in raw_values if value})
    distinct_metadata = sorted(
        {value.isoformat() for value in parsed_metadata if value is not None}
    )
    distinct_filenames = sorted(
        {value.isoformat() for value in parsed_filenames if value is not None}
    )
    rows = len(coco_rows)
    complete_metadata = rows > 0 and all(raw_values)
    valid_metadata = complete_metadata and all(value is not None for value in parsed_metadata)
    complete_filename_timestamps = rows > 0 and all(
        value is not None for value in parsed_filenames
    )
    export_metadata_proven = bool(
        rows > 1
        and valid_metadata
        and len(distinct_raw) == 1
        and len(distinct_metadata) == 1
        and complete_filename_timestamps
        and len(distinct_filenames) > 1
    )
    return {
        "schema_version": 1,
        "field": "date_captured",
        "rows": rows,
        "rows_with_value": sum(bool(value) for value in raw_values),
        "rows_with_valid_value": sum(value is not None for value in parsed_metadata),
        "distinct_raw_values": len(distinct_raw),
        "distinct_parsed_values": len(distinct_metadata),
        "constant_value": distinct_raw[0] if len(distinct_raw) == 1 else None,
        "filename_timestamp_valid_rows": sum(
            value is not None for value in parsed_filenames
        ),
        "filename_timestamp_distinct_values": len(distinct_filenames),
        "classification": (
            "dataset_constant_export_metadata"
            if export_metadata_proven
            else "capture_metadata_not_ignorable"
        ),
        "ignored_for_captured_at": export_metadata_proven,
        "captured_at_basis": (
            "filename_timestamp_with_explicit_timestamp_fields_still_enforced"
            if export_metadata_proven
            else "filename_and_all_present_capture_metadata_must_agree"
        ),
        "fail_closed": True,
    }


def _resolve_timestamp(
    value: str,
    coco_row: dict[str, Any],
    *,
    ignore_date_captured: bool = False,
) -> tuple[datetime | None, str | None]:
    filename_timestamp = _parse_timestamp(value)
    metadata_fields = ["timestamp", "captured_at"]
    if not ignore_date_captured:
        metadata_fields.insert(0, "date_captured")
    metadata_values = [
        str(coco_row[key]).strip()
        for key in metadata_fields
        if coco_row.get(key) not in (None, "")
    ]
    metadata_timestamps = [_parse_timestamp(item) for item in metadata_values]
    if metadata_values and any(item is None for item in metadata_timestamps):
        return None, "timestamp_metadata_invalid_or_missing_clock"
    resolved_metadata = {item for item in metadata_timestamps if item is not None}
    if len(resolved_metadata) > 1:
        return None, "timestamp_metadata_fields_disagree"
    metadata_timestamp = next(iter(resolved_metadata), None)
    if (
        filename_timestamp
        and metadata_timestamp
        and filename_timestamp != metadata_timestamp
    ):
        return None, "timestamp_filename_metadata_mismatch"
    resolved = metadata_timestamp or filename_timestamp
    return (resolved, None) if resolved else (None, "timestamp_unresolved_or_missing_clock")


def _timestamp(value: str, coco_row: dict[str, Any]) -> datetime | None:
    """Return only an unambiguous minute-resolved timestamp."""
    return _resolve_timestamp(value, coco_row)[0]


def _perceptual_hash(image: Image.Image) -> str:
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    transformed = fftpack.dct(fftpack.dct(pixels, axis=0), axis=1)
    low = transformed[:8, :8]
    bits = (low > np.median(low)).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _metrics(image: Image.Image) -> dict[str, float]:
    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    stats = ImageStat.Stat(gray)
    edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
    return {
        "brightness": float(stats.mean[0] / 255.0),
        "contrast": float(stats.stddev[0] / 255.0),
        "edge_energy": float(edges.mean[0] / 255.0),
    }


def _visible_source_point(
    image: Image.Image,
    smoke_mask: np.ndarray,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    """Return a point only for connected smoke with stable source edges below it."""
    failures: list[str] = []
    diagnostics: dict[str, Any] = {}
    foreground = int(np.count_nonzero(smoke_mask))
    if foreground == 0:
        return None, ["negative_image_has_no_smoke_point"], diagnostics
    labels, component_count = ndimage.label(
        smoke_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels.reshape(-1))[1:]
    if component_count == 0 or sizes.size == 0:
        return None, ["empty_connected_component_set"], diagnostics
    dominant_label = int(np.argmax(sizes)) + 1
    dominant_size = int(sizes[dominant_label - 1])
    dominant_ratio = dominant_size / foreground
    material_threshold = max(16, math.ceil(foreground * 0.02))
    material_components = int(np.count_nonzero(sizes >= material_threshold))
    diagnostics.update(
        {
            "smoke_component_count": int(component_count),
            "material_smoke_component_count": material_components,
            "dominant_smoke_component_fraction": dominant_ratio,
        }
    )
    if material_components != 1 or dominant_ratio < 0.98:
        failures.append("smoke_not_single_connected_dominant_component")
    component = labels == dominant_label
    ys, xs = np.nonzero(component)
    height, width = component.shape
    bottom = int(ys.max())
    bottom_x = xs[ys == bottom]
    if bottom_x.size == 0:
        failures.append("smoke_base_not_resolved")
        return None, failures, diagnostics
    base_x = round(float(np.median(bottom_x)))
    base_span = int(bottom_x.max() - bottom_x.min() + 1)
    diagnostics.update(
        {
            "base_x_px": base_x,
            "base_y_px": bottom,
            "base_span_px": base_span,
            "base_span_fraction": base_span / width,
        }
    )
    if bottom >= height - max(3, round(height * 0.04)):
        failures.append("smoke_base_truncated_by_frame_bottom")
    if base_x < round(width * 0.04) or base_x > round(width * 0.96):
        failures.append("smoke_base_too_close_to_frame_side")
    if base_span > max(8, round(width * 0.08)):
        failures.append("smoke_base_too_wide_for_unique_emission_source")

    source_depth = max(8, round(height * 0.08))
    stop = min(height, bottom + 1 + source_depth)
    if stop - (bottom + 1) < source_depth:
        failures.append("insufficient_visible_area_below_smoke_base")
    half_width = max(5, round(width * 0.035))
    left = max(0, base_x - half_width)
    right = min(width, base_x + half_width + 1)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    source_region = gray[bottom + 1 : stop, left:right]
    mask_region = smoke_mask[bottom + 1 : stop, left:right]
    if source_region.shape[0] < 4 or source_region.shape[1] < 7:
        failures.append("visible_source_corridor_too_small")
        return None, sorted(set(failures)), diagnostics
    mask_occupancy = float(np.count_nonzero(mask_region) / mask_region.size)
    diagnostics["source_corridor_smoke_fraction"] = mask_occupancy
    if mask_occupancy > 0.02:
        failures.append("smoke_continues_through_source_corridor")

    gradients = np.abs(np.diff(source_region, axis=1))
    center = max(1, min(gradients.shape[1] - 1, base_x - left))
    left_gradients = gradients[:, :center]
    right_gradients = gradients[:, center:]
    if left_gradients.shape[1] == 0 or right_gradients.shape[1] == 0:
        failures.append("source_vertical_edges_not_bracketed")
    else:
        left_values = left_gradients.max(axis=1)
        right_values = right_gradients.max(axis=1)
        left_positions = left_gradients.argmax(axis=1)
        right_positions = right_gradients.argmax(axis=1) + center
        supported = (left_values >= 0.12) & (right_values >= 0.12)
        support_fraction = float(np.mean(supported))
        diagnostics["source_vertical_edge_support_fraction"] = support_fraction
        if support_fraction < 0.70:
            failures.append("source_vertical_edges_not_visible")
        elif np.count_nonzero(supported) >= 4:
            left_spread = float(np.std(left_positions[supported]))
            right_spread = float(np.std(right_positions[supported]))
            diagnostics["source_left_edge_spread_px"] = left_spread
            diagnostics["source_right_edge_spread_px"] = right_spread
            allowed_spread = max(1.5, width * 0.006)
            if left_spread > allowed_spread or right_spread > allowed_spread:
                failures.append("source_vertical_edges_not_stable")
            separations = right_positions[supported] - left_positions[supported]
            if float(np.median(separations)) < 2.0:
                failures.append("source_vertical_edges_have_no_physical_width")

    above = component[max(0, bottom - max(4, round(height * 0.03))) : bottom + 1, left:right]
    connected_above_fraction = float(np.count_nonzero(above) / above.size)
    diagnostics["smoke_connection_above_base_fraction"] = connected_above_fraction
    if connected_above_fraction < 0.08:
        failures.append("smoke_not_connected_above_candidate_source")
    failures = sorted(set(failures))
    if failures:
        return None, failures, diagnostics
    return (
        {
            "kind": "smoke_column_base",
            "x": round(base_x / max(1, width - 1), 8),
            "y": round(bottom / max(1, height - 1), 8),
            "origin": "human_revised_mask_connected_visible_source_gate_v1",
        },
        [],
        diagnostics,
    )


def _load_exclusion_index(
    root: Path,
    name: str,
    *,
    expected_receipt_sha256: str,
    expected_rows: int,
) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
    expected_files = {root / "hash-index.jsonl", root / "receipt.json"}
    manifests = [root / "hash-index.jsonl"] if (root / "hash-index.jsonl").is_file() else []
    receipts = [root / "receipt.json"] if (root / "receipt.json").is_file() else []
    unexpected_files = [path for path in files if path not in expected_files]
    raw_media = [path for path in unexpected_files if path.suffix.casefold() in IMAGE_SUFFIXES]
    errors: list[str] = []
    if len(manifests) != 1:
        errors.append(f"{name}_exclusion_index_count:{len(manifests)}:1")
    if len(receipts) != 1:
        errors.append(f"{name}_exclusion_receipt_count:{len(receipts)}:1")
    if unexpected_files:
        errors.append(f"{name}_exclusion_index_unexpected_files:{len(unexpected_files)}")
    if not _HEX64.fullmatch(expected_receipt_sha256):
        errors.append(f"{name}_expected_receipt_sha256_invalid")
    if expected_rows <= 0:
        errors.append(f"{name}_expected_rows_invalid:{expected_rows}")

    receipt: dict[str, Any] = {}
    receipt_sha256: str | None = None
    if len(receipts) == 1:
        receipt_path = receipts[0]
        receipt_sha256 = _sha256_file(receipt_path)
        if receipt_sha256 != expected_receipt_sha256:
            errors.append(f"{name}_exclusion_receipt_sha256_mismatch")
        try:
            loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("receipt is not an object")
            receipt = loaded
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{name}_exclusion_receipt_invalid:{type(exc).__name__}:{exc}")

    manifest = manifests[0] if len(manifests) == 1 else None
    if manifest is not None and receipt:
        receipt_contract = {
            "schema_version": receipt.get("schema_version") == 1,
            "partition": receipt.get("partition") == name,
            "index_filename": receipt.get("index_filename") == manifest.name,
            "index_rows": int(receipt.get("index_rows", -1)) == expected_rows,
            "index_bytes": int(receipt.get("index_bytes", -1)) == manifest.stat().st_size,
            "index_sha256": receipt.get("index_sha256") == _sha256_file(manifest),
            "incomplete_rows": int(receipt.get("incomplete_rows", -1)) == 0,
            "media_files_output": int(receipt.get("media_files_output", -1)) == 0,
            "source_gate_passed": receipt.get("source_gate_passed") is True,
            "gate_errors": receipt.get("gate_errors") == [],
            "publication_allowed": receipt.get("publication_allowed") is False,
        }
        errors.extend(
            f"{name}_exclusion_receipt_contract:{field}"
            for field, passed in receipt_contract.items()
            if not passed
        )

    sha_rows: dict[str, str] = {}
    phash_rows: list[tuple[str, str]] = []
    indexed_rows = 0
    incomplete_rows = 0
    seen_sample_ids: set[str] = set()
    seen_sha256: set[str] = set()
    for manifest in manifests[:1]:
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{name}_invalid_json:{manifest.name}:{line_number}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{name}_non_object_row:{manifest.name}:{line_number}")
                incomplete_rows += 1
                continue
            sample_id = str(row.get("sample_id") or f"{manifest.name}:{line_number}")
            digest = str(
                row.get("image_sha256") or row.get("source_image_sha256") or row.get("sha256") or ""
            ).casefold()
            signature = str(row.get("phash") or row.get("perceptual_hash") or "").casefold()
            flipped = str(row.get("phash64_flipped") or "").casefold()
            if (
                not _HEX64.fullmatch(digest)
                or not _HEX16.fullmatch(signature)
                or sample_id in seen_sample_ids
                or digest in seen_sha256
                or row.get("index_partition") not in {None, name}
                or (name == "benchmark" and not _HEX16.fullmatch(flipped))
            ):
                incomplete_rows += 1
                continue
            indexed_rows += 1
            sha_rows[digest] = sample_id
            phash_rows.append((signature, sample_id))
            if name == "benchmark":
                phash_rows.append((flipped, f"{sample_id}:horizontal-flip"))
            seen_sample_ids.add(sample_id)
            seen_sha256.add(digest)
    if indexed_rows == 0:
        errors.append(f"{name}_exclusion_index_has_no_complete_sha256_phash_rows")
    if indexed_rows != expected_rows:
        errors.append(f"{name}_exclusion_index_rows:{indexed_rows}:{expected_rows}")
    if incomplete_rows:
        errors.append(f"{name}_exclusion_index_incomplete_rows:{incomplete_rows}")
    return {
        "name": name,
        "manifests": [str(path.relative_to(root)).replace("\\", "/") for path in manifests],
        "sha": sha_rows,
        "phash": phash_rows,
        "indexed_rows": indexed_rows,
        "phash_signatures": len(phash_rows),
        "incomplete_rows": incomplete_rows,
        "raw_media_files": len(raw_media),
        "unexpected_files": [
            str(path.relative_to(root)).replace("\\", "/") for path in unexpected_files
        ],
        "receipt_sha256": receipt_sha256,
        "receipt": receipt,
        "gate_errors": errors,
    }


def _nearest_phash(
    signature: str,
    rows: list[tuple[str, str]],
) -> tuple[int | None, str | None]:
    if not rows:
        return None, None
    distance, sample_id = min((_hamming(signature, value), key) for value, key in rows)
    return distance, sample_id


def _group_camera_episodes(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("camera") and row.get("captured_at"):
            grouped[str(row["camera"])].append(row)
    for camera, camera_rows in grouped.items():
        camera_rows.sort(key=lambda row: (str(row["captured_at"]), str(row["sample_id"])))
        episode = 0
        previous: datetime | None = None
        for row in camera_rows:
            current = datetime.fromisoformat(str(row["captured_at"]).replace("Z", "+00:00"))
            if previous is None or (current - previous).total_seconds() > EPISODE_GAP_SECONDS:
                episode += 1
            group = f"ijmond:{camera}:episode-{episode:04d}"
            row["source_group"] = group
            row["split_group"] = group
            previous = current


def _assign_group_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for row in rows:
        group = str(row.get("split_group") or "")
        if not group or not row.get("camera") or not row.get("captured_at"):
            errors.append(f"split_candidate_missing_group_metadata:{row.get('sample_id')}")
            continue
        groups[group].append(row)
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            min(str(row["captured_at"]) for row in item[1]),
            item[0],
        ),
    )
    total = sum(len(group_rows) for _, group_rows in ordered)
    if len(ordered) < 3:
        errors.append(f"chronological_episode_groups_insufficient:{len(ordered)}:3")
    if total != len(rows):
        errors.append(f"split_candidate_accounting_mismatch:{total}:{len(rows)}")

    best: tuple[tuple[float, float, int, int], int, int] | None = None
    prefix = [0]
    for _, group_rows in ordered:
        prefix.append(prefix[-1] + len(group_rows))
    for train_end in range(1, len(ordered) - 1):
        for validation_end in range(train_end + 1, len(ordered)):
            counts = {
                "train": prefix[train_end],
                "validation": prefix[validation_end] - prefix[train_end],
                "test": total - prefix[validation_end],
            }
            deviations = {
                name: abs(counts[name] / total - target)
                for name, target in SPLIT_TARGET_RATIOS.items()
            }
            objective = (
                sum(value * value for value in deviations.values()),
                max(deviations.values()),
                train_end,
                validation_end,
            )
            candidate = (objective, train_end, validation_end)
            if best is None or candidate < best:
                best = candidate

    assignments: dict[str, str] = {}
    boundaries: dict[str, str | None] = {
        "train_last_group": None,
        "validation_last_group": None,
    }
    if best is not None:
        _, train_end, validation_end = best
        boundaries = {
            "train_last_group": ordered[train_end - 1][0],
            "validation_last_group": ordered[validation_end - 1][0],
        }
        for index, (group, group_rows) in enumerate(ordered):
            split = (
                "train"
                if index < train_end
                else "validation"
                if index < validation_end
                else "test"
            )
            assignments[group] = split
            for row in group_rows:
                row["split"] = split
                row["final_split"] = split

    counts = Counter(assignments.get(str(row.get("split_group")), "") for row in rows)
    observed_counts = {name: int(counts.get(name, 0)) for name in SPLIT_TARGET_RATIOS}
    observed_ratios = {
        name: observed_counts[name] / total if total else 0.0
        for name in SPLIT_TARGET_RATIOS
    }
    deviations = {
        name: abs(observed_ratios[name] - target)
        for name, target in SPLIT_TARGET_RATIOS.items()
    }
    empty_splits = [name for name, count in observed_counts.items() if count == 0]
    if empty_splits:
        errors.append(f"chronological_episode_splits_empty:{','.join(empty_splits)}")
    outside_tolerance = [
        name for name, deviation in deviations.items() if deviation > SPLIT_RATIO_TOLERANCE
    ]
    if outside_tolerance:
        errors.append(
            "chronological_episode_split_ratio_outside_tolerance:"
            + ",".join(outside_tolerance)
        )

    group_rows_receipt = []
    for group, group_rows in ordered:
        cameras = sorted({str(row["camera"]) for row in group_rows})
        if len(cameras) != 1:
            errors.append(f"episode_group_camera_mismatch:{group}")
        group_rows_receipt.append(
            {
                "split_group": group,
                "camera": cameras[0] if len(cameras) == 1 else None,
                "rows": len(group_rows),
                "first_captured_at": min(str(row["captured_at"]) for row in group_rows),
                "last_captured_at": max(str(row["captured_at"]) for row in group_rows),
                "split": assignments.get(group),
            }
        )
    errors = sorted(set(errors))
    return {
        "schema_version": 1,
        "strategy": "optimize_two_chronological_group_boundaries_whole_episode_v1",
        "target_ratios": SPLIT_TARGET_RATIOS,
        "absolute_ratio_tolerance": SPLIT_RATIO_TOLERANCE,
        "locally_admissible_rows": len(rows),
        "episode_groups": len(ordered),
        "boundaries": boundaries,
        "observed_counts": observed_counts,
        "observed_ratios": observed_ratios,
        "absolute_ratio_deviations": deviations,
        "groups": group_rows_receipt,
        "gate_errors": errors,
        "passed": not errors,
    }


def _allocate_proportional_quotas(counts: dict[str, int], budget: int) -> dict[str, int]:
    keys = sorted(key for key, count in counts.items() if count > 0)
    budget = min(max(0, budget), sum(counts[key] for key in keys))
    quotas = {key: 0 for key in keys}
    if budget >= len(keys):
        for key in keys:
            quotas[key] = 1
        budget -= len(keys)
    while budget:
        selected: str | None = None
        for key in keys:
            if quotas[key] >= counts[key]:
                continue
            if selected is None:
                selected = key
                continue
            left = counts[key] * (quotas[selected] + 1)
            right = counts[selected] * (quotas[key] + 1)
            if left > right:
                selected = key
        if selected is None:
            break
        quotas[selected] += 1
        budget -= 1
    return quotas


def _center_out(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (str(row["captured_at"]), str(row["sample_id"])))
    middle = (len(ordered) - 1) / 2
    return [
        row
        for _, row in sorted(
            enumerate(ordered), key=lambda item: (abs(item[0] - middle), item[0])
        )
    ]


def _select_across_episodes(rows: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episodes[str(row["split_group"])].append(row)
    ordered_episodes = sorted(
        episodes,
        key=lambda group: (
            min(str(row["captured_at"]) for row in episodes[group]),
            group,
        ),
    )
    queues = {group: _center_out(episodes[group]) for group in ordered_episodes}
    selected: list[dict[str, Any]] = []
    if 0 < quota < len(ordered_episodes):
        spread = [
            ordered_episodes[
                min(
                    len(ordered_episodes) - 1,
                    ((2 * index + 1) * len(ordered_episodes)) // (2 * quota),
                )
            ]
            for index in range(quota)
        ]
        return [queues[group].pop(0) for group in spread]
    while len(selected) < quota:
        progressed = False
        for group in ordered_episodes:
            if not queues[group]:
                continue
            selected.append(queues[group].pop(0))
            progressed = True
            if len(selected) == quota:
                break
        if not progressed:
            break
    return selected


def _sample_ids_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        f"{row['sample_id']}\n" for row in sorted(rows, key=lambda row: row["sample_id"])
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _apply_point_cap(rows: list[dict[str, Any]], cap: int) -> dict[str, Any]:
    if cap < 0:
        raise ValueError("professional source point cap must be non-negative")
    candidates = [
        row
        for row in rows
        if not row["exclusion_reasons"] and row.get("point_supervised") is True
    ]
    errors = []
    missing_strata = [
        str(row["sample_id"])
        for row in candidates
        if not row.get("split") or not row.get("camera") or not row.get("split_group")
    ]
    if missing_strata:
        errors.append(f"point_cap_candidates_missing_strata:{len(missing_strata)}")

    selection_budget = min(cap, len(candidates))
    selected: list[dict[str, Any]] = []
    eligible_candidates = [row for row in candidates if row["sample_id"] not in missing_strata]
    split_counts = Counter(str(row["split"]) for row in eligible_candidates)
    split_quotas = _allocate_proportional_quotas(dict(split_counts), selection_budget)
    camera_quotas: dict[str, dict[str, int]] = {}
    for split in sorted(split_quotas):
        split_rows = [row for row in eligible_candidates if row["split"] == split]
        camera_counts = Counter(str(row["camera"]) for row in split_rows)
        camera_quotas[split] = _allocate_proportional_quotas(
            dict(camera_counts), split_quotas[split]
        )
        for camera in sorted(camera_quotas[split]):
            stratum = [row for row in split_rows if row["camera"] == camera]
            selected.extend(_select_across_episodes(stratum, camera_quotas[split][camera]))
    if len(selected) != selection_budget:
        errors.append(f"point_cap_selection_count:{len(selected)}:{selection_budget}")

    selected_ids = {str(row["sample_id"]) for row in selected}
    demoted = [row for row in candidates if str(row["sample_id"]) not in selected_ids]
    for row in rows:
        if row not in candidates:
            row["point_cap_status"] = "not_eligible_point_candidate"
        elif str(row["sample_id"]) in selected_ids:
            row["point_cap_status"] = "selected"
        else:
            row["point_cap_status"] = "demoted_to_abstention"
            row["point_supervised"] = False
            row["anchor_points"] = []
            row["point_derivation"] = "none_professional_source_point_cap"
            row["point_gate_errors"] = sorted(
                set([*row["point_gate_errors"], "professional_source_point_cap"])
            )
            row["visual_abstention_reason"] = "professional_source_point_cap"

    selected = [row for row in candidates if str(row["sample_id"]) in selected_ids]
    selected_by_split = Counter(str(row["split"]) for row in selected)
    selected_by_camera = Counter(str(row["camera"]) for row in selected)
    selected_by_episode = Counter(str(row["split_group"]) for row in selected)
    errors = sorted(set(errors))
    return {
        "schema_version": 1,
        "strategy": (
            "proportional_split_then_camera_quota_with_deterministic_"
            "episode_round_robin_and_center_out_rows_v1"
        ),
        "cap": cap,
        "candidate_point_rows": len(candidates),
        "selection_budget": selection_budget,
        "selected_point_rows": len(selected),
        "demoted_to_abstention_rows": len(demoted),
        "split_candidate_counts": dict(sorted(split_counts.items())),
        "split_quotas": split_quotas,
        "camera_quotas_by_split": camera_quotas,
        "selected_by_split": dict(sorted(selected_by_split.items())),
        "selected_by_camera": dict(sorted(selected_by_camera.items())),
        "selected_by_episode": dict(sorted(selected_by_episode.items())),
        "candidate_sample_ids_sha256": _sample_ids_sha256(candidates),
        "selected_sample_ids_sha256": _sample_ids_sha256(selected),
        "demoted_sample_ids_sha256": _sample_ids_sha256(demoted),
        "selected_sample_ids": sorted(selected_ids),
        "demoted_sample_ids": sorted(str(row["sample_id"]) for row in demoted),
        "gate_errors": errors,
        "passed": not errors,
    }


def _deduplicate_within_source(rows: list[dict[str, Any]]) -> dict[str, int]:
    kept_sha: dict[str, str] = {}
    kept_phash: list[tuple[str, str]] = []
    exact = 0
    perceptual = 0
    for row in sorted(rows, key=lambda value: (str(value.get("captured_at")), value["sample_id"])):
        if row["exclusion_reasons"]:
            continue
        digest = str(row["image_sha256"])
        signature = str(row["phash"])
        if digest in kept_sha:
            row["exclusion_reasons"].append("within_source_exact_sha_duplicate")
            row["within_source_duplicate_of"] = kept_sha[digest]
            exact += 1
            continue
        distance, nearest = _nearest_phash(signature, kept_phash)
        if distance is not None and distance <= MAX_PHASH_DISTANCE:
            row["exclusion_reasons"].append("within_source_phash_duplicate")
            row["within_source_duplicate_of"] = nearest
            row["within_source_nearest_phash_distance"] = distance
            perceptual += 1
            continue
        kept_sha[digest] = str(row["sample_id"])
        kept_phash.append((signature, str(row["sample_id"])))
    return {"exact_rows_excluded": exact, "phash_rows_excluded": perceptual}


def audit_ijmond_archive(
    *,
    archive_path: Path,
    detection_index_root: Path,
    pointing_index_root: Path,
    benchmark_index_root: Path,
    output_dir: Path,
    output_s3_prefix: str | None,
    source_contract_receipt: dict[str, Any],
    expected_index_receipts: dict[str, dict[str, Any]],
    expected_archive_bytes: int = EXPECTED_ARCHIVE_BYTES,
    expected_archive_md5: str = EXPECTED_ARCHIVE_MD5,
    expected_archive_sha256: str = EXPECTED_ARCHIVE_SHA256,
    expected_raw_images: int = EXPECTED_RAW_IMAGES,
    expected_raw_positive_images: int = EXPECTED_RAW_POSITIVE_IMAGES,
    expected_raw_polygons: int = EXPECTED_RAW_POLYGONS,
    professional_source_point_cap: int = PROFESSIONAL_SOURCE_POINT_CAP,
) -> dict[str, Any]:
    """Audit a staged archive without publishing or touching corpus registries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_receipt = {
        "bytes": archive_path.stat().st_size,
        "md5": _md5_file(archive_path),
        "sha256": _sha256_file(archive_path),
        "expected_bytes": expected_archive_bytes,
        "expected_md5": expected_archive_md5,
        "expected_sha256": expected_archive_sha256,
    }
    gate_errors = list(source_contract_receipt.get("gate_errors") or [])
    if source_contract_receipt.get("source_contract_verified") is not True:
        gate_errors.append("figshare_source_contract_not_verified")
    if archive_receipt["bytes"] != expected_archive_bytes:
        gate_errors.append(
            f"archive_byte_count_mismatch:{archive_receipt['bytes']}:{expected_archive_bytes}"
        )
    if archive_receipt["md5"] != expected_archive_md5:
        gate_errors.append("archive_md5_mismatch")
    if not _HEX64.fullmatch(expected_archive_sha256):
        gate_errors.append("expected_archive_sha256_invalid")
    elif archive_receipt["sha256"] != expected_archive_sha256:
        gate_errors.append("archive_sha256_mismatch")

    if set(expected_index_receipts) != {"detection", "pointing", "benchmark"}:
        raise ValueError("exactly three pinned exclusion-index receipts are required")
    index_roots = {
        "detection": detection_index_root,
        "pointing": pointing_index_root,
        "benchmark": benchmark_index_root,
    }
    indexes = {
        name: _load_exclusion_index(
            index_roots[name],
            name,
            expected_receipt_sha256=str(expected_index_receipts[name]["receipt_sha256"]),
            expected_rows=int(expected_index_receipts[name]["rows"]),
        )
        for name in ("detection", "pointing", "benchmark")
    }
    for index in indexes.values():
        gate_errors.extend(index["gate_errors"])

    candidates: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    pairing_errors: list[str] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = _validated_members(archive)
        ordered_members = sorted(
            members.values(), key=lambda value: _normalized_member_name(value.name)
        )
        for member in ordered_members:
            name = _normalized_member_name(member.name)
            inventory.append(
                {
                    "path": name,
                    "type": "file" if member.isfile() else "directory",
                    "bytes": member.size,
                    "cropped": _is_cropped(name),
                    "raw_image_candidate": bool(
                        member.isfile()
                        and not _is_cropped(name)
                        and not _is_mask(name)
                        and PurePosixPath(name).suffix.casefold() in IMAGE_SUFFIXES
                    ),
                    "raw_mask_candidate": bool(
                        member.isfile()
                        and not _is_cropped(name)
                        and _is_mask(name)
                        and PurePosixPath(name).suffix.casefold() in IMAGE_SUFFIXES
                    ),
                }
            )
        try:
            coco_name, coco, coco_payload = _find_raw_coco(archive, members)
        except ValueError as exc:
            gate_errors.append(f"raw_coco_inventory_error:{exc}")
            coco_name, coco, coco_payload = "", {"images": [], "annotations": []}, b""
        coco_images = [row for row in coco.get("images") or [] if isinstance(row, dict)]
        coco_annotations = [row for row in coco.get("annotations") or [] if isinstance(row, dict)]
        date_captured_receipt = _classify_coco_date_captured(coco_images)
        polygon_count, unsupported_polygons = _coco_polygon_count(coco_annotations)
        annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco_annotations:
            annotations_by_image[str(annotation.get("image_id"))].append(annotation)

        for coco_row in coco_images:
            file_name = str(coco_row.get("file_name") or "")
            image_id = str(coco_row.get("id") or file_name)
            image_member = _resolve_image_member(file_name, members) if file_name else None
            if image_member is None:
                pairing_errors.append(f"{image_id}:raw_image_unresolved:{file_name}")
                continue
            mask_member = _resolve_mask_member(image_member, members)
            if mask_member is None:
                pairing_errors.append(f"{image_id}:raw_mask_unresolved:{file_name}")
                continue
            reasons: list[str] = []
            try:
                image_payload = _member_payload(archive, image_member)
                mask_payload = _member_payload(archive, mask_member)
                with Image.open(io.BytesIO(image_payload)) as opened:
                    opened.load()
                    source_mode = opened.mode
                    source_size = opened.size
                    image = opened.convert("RGB")
                with Image.open(io.BytesIO(mask_payload)) as opened_mask:
                    opened_mask.load()
                    mask_size = opened_mask.size
                    raw_mask = np.asarray(opened_mask)
            except Exception as exc:
                pairing_errors.append(f"{image_id}:decode:{type(exc).__name__}:{exc}")
                continue
            if source_mode != "RGB":
                reasons.append("source_image_not_rgb")
            if source_size != mask_size:
                reasons.append("image_mask_dimension_mismatch")
            if raw_mask.ndim == 3:
                channels_agree = all(
                    np.array_equal(raw_mask[..., 0], raw_mask[..., index])
                    for index in range(1, raw_mask.shape[2])
                )
                if channels_agree:
                    raw_mask = raw_mask[..., 0]
                else:
                    reasons.append("mask_channels_disagree")
            if raw_mask.ndim != 2:
                reasons.append("mask_not_scalar")
                raw_mask = np.zeros((source_size[1], source_size[0]), dtype=np.uint8)
            mask_values = sorted(int(value) for value in np.unique(raw_mask))
            if not set(mask_values).issubset(MASK_VALUES):
                reasons.append("unexpected_mask_values")
            smoke_mask = raw_mask > 0
            mask_positive = bool(np.any(smoke_mask))
            coco_positive = bool(annotations_by_image.get(image_id))
            if mask_positive != coco_positive:
                reasons.append("coco_mask_presence_disagreement")
            camera, camera_error = _resolve_camera(file_name, coco_row)
            captured_at, timestamp_error = _resolve_timestamp(
                file_name,
                coco_row,
                ignore_date_captured=bool(date_captured_receipt["ignored_for_captured_at"]),
            )
            if camera_error:
                reasons.append(camera_error)
            if timestamp_error:
                reasons.append(timestamp_error)
            image_sha = _sha256_bytes(image_payload)
            mask_sha = _sha256_bytes(mask_payload)
            phash = _perceptual_hash(image)
            point, point_failures, point_diagnostics = _visible_source_point(image, smoke_mask)
            anchor_points = [point] if point is not None else []
            sample_id = f"ijmond:{camera or 'unknown'}:{image_id}"
            external_matches: dict[str, dict[str, Any]] = {}
            for index_name, index in indexes.items():
                exact = index["sha"].get(image_sha)
                distance, nearest = _nearest_phash(phash, index["phash"])
                external_matches[index_name] = {
                    "exact_sha_sample_id": exact,
                    "nearest_phash_sample_id": nearest,
                    "nearest_phash_distance": distance,
                }
                if exact is not None:
                    reasons.append(f"{index_name}_exact_sha_overlap")
                if distance is not None and distance <= MAX_PHASH_DISTANCE:
                    reasons.append(f"{index_name}_phash_overlap")
            candidates.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "source_id": SOURCE_ID,
                    "source_family": SOURCE_FAMILY,
                    "source_revision": SOURCE_DOI,
                    "source_record_url": SOURCE_RECORD_URL,
                    "source_paper_url": SOURCE_PAPER_URL,
                    "source_archive_file_id": FIGSHARE_FILE_ID,
                    "source_archive_sha256": archive_receipt["sha256"],
                    "source_archive_member": _normalized_member_name(image_member.name),
                    "mask_archive_member": _normalized_member_name(mask_member.name),
                    "source_coco_member": coco_name,
                    "source_coco_image_id": image_id,
                    "camera": camera,
                    "captured_at": captured_at.isoformat() if captured_at else None,
                    "captured_at_basis": date_captured_receipt["captured_at_basis"],
                    "source_date_captured": coco_row.get("date_captured"),
                    "source_date_captured_classification": date_captured_receipt[
                        "classification"
                    ],
                    "source_group": None,
                    "split_group": None,
                    "split": None,
                    "final_split": None,
                    "image_sha256": image_sha,
                    "source_image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "phash": phash,
                    "width": source_size[0],
                    "height": source_size[1],
                    "source_mode": source_mode,
                    "mask_values": mask_values,
                    "mask_positive": mask_positive,
                    "mask_nonzero_fraction": float(np.count_nonzero(smoke_mask) / smoke_mask.size),
                    "mask_semantics": "background_0_low_opacity_smoke_155_high_opacity_smoke_255",
                    "mask_quality": "source_human_revised_all_masks_checked_and_edited",
                    "annotation_strength": "strong_human_revised",
                    "annotation_provenance": (
                        "roboflow_sam_prompted_polygon_manually_refined_jointly_and_all_masks_"
                        "manually_checked_edited_by_source_author"
                    ),
                    "segmentation_supervised": True,
                    "presence_supervised": True,
                    "presence_supervised_classes": ["smoke_visible"],
                    "presence_targets": {
                        "flame_visible": None,
                        "smoke_visible": mask_positive,
                    },
                    "presence_provenance": "source_human_revised_smoke_mask_only",
                    "abstention_supervised": True,
                    "point_supervised": bool(anchor_points),
                    "anchor_points": anchor_points,
                    "point_derivation": (
                        "human_revised_mask_connected_visible_source_gate_v1"
                        if anchor_points
                        else "none_fail_closed"
                    ),
                    "point_gate_errors": point_failures,
                    "point_gate_diagnostics": point_diagnostics,
                    "visual_abstention_reason": (
                        None if anchor_points else ";".join(point_failures) or "no_point_proof"
                    ),
                    "external_exclusion_matches": external_matches,
                    "license": SOURCE_LICENSE,
                    "license_url": SOURCE_LICENSE_URL,
                    "redistribution_allowed": True,
                    "reviews_admitted": False,
                    "publication_allowed": False,
                    "benchmark_hash_exclusion_only": True,
                    "benchmark_payload_used": False,
                    "validation_profile": "fireviewer_pointing_ijmond_strict_v1",
                    "exclusion_reasons": sorted(set(reasons)),
                    "strict_keep": False,
                    "training_eligible": False,
                    "sample_validation_status": "pending_source_gate",
                    **_metrics(image),
                }
            )

    deduplication = _deduplicate_within_source(candidates)
    local_valid = [row for row in candidates if not row["exclusion_reasons"]]
    _group_camera_episodes(local_valid)
    for row in local_valid:
        if not row.get("split_group"):
            row["exclusion_reasons"].append("camera_episode_group_unresolved")
    local_valid = [row for row in candidates if not row["exclusion_reasons"]]
    split_receipt = _assign_group_splits(local_valid)
    point_cap_receipt = _apply_point_cap(candidates, professional_source_point_cap)
    for row in candidates:
        row["exclusion_reasons"] = sorted(set(row["exclusion_reasons"]))

    positive_masks = sum(bool(row["mask_positive"]) for row in candidates)
    coco_positive_images = sum(
        bool(annotations_by_image.get(str(row.get("id")))) for row in coco_images
    )
    coco_mask_presence_disagreements = sum(
        "coco_mask_presence_disagreement" in row["exclusion_reasons"] for row in candidates
    )
    if len(coco_images) != expected_raw_images:
        gate_errors.append(f"raw_coco_image_count:{len(coco_images)}:{expected_raw_images}")
    if polygon_count != expected_raw_polygons:
        gate_errors.append(f"raw_coco_polygon_count:{polygon_count}:{expected_raw_polygons}")
    if unsupported_polygons:
        gate_errors.append(f"unsupported_coco_segmentations:{unsupported_polygons}")
    if len(candidates) != expected_raw_images:
        gate_errors.append(f"paired_raw_image_mask_count:{len(candidates)}:{expected_raw_images}")
    if coco_positive_images != expected_raw_positive_images:
        gate_errors.append(
            f"positive_raw_coco_image_count:{coco_positive_images}:"
            f"{expected_raw_positive_images}"
        )
    if pairing_errors:
        gate_errors.append(f"raw_pairing_or_decode_errors:{len(pairing_errors)}")
    if not local_valid:
        gate_errors.append("no_locally_valid_independent_rows")
    gate_errors.extend(split_receipt["gate_errors"])
    gate_errors.extend(point_cap_receipt["gate_errors"])
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in local_valid:
        if row.get("split_group") and row.get("split"):
            group_splits[str(row["split_group"])].add(str(row["split"]))
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking_groups:
        gate_errors.append(f"camera_episode_split_leakage:{len(leaking_groups)}")
    gate_errors = sorted(set(gate_errors))
    source_gate_passed = not gate_errors

    for row in candidates:
        keep = source_gate_passed and not row["exclusion_reasons"]
        row["strict_keep"] = keep
        row["training_eligible"] = keep
        row["sample_validation_status"] = (
            "strict_automated_validated_point"
            if keep and row["point_supervised"]
            else "strict_automated_validated_auxiliary"
            if keep
            else "excluded_source_gate_failed"
            if not source_gate_passed
            else "excluded_unvalidated"
        )
        row["corpus_role"] = (
            "pointing_segmentation_presence_abstention"
            if keep and row["point_supervised"]
            else "segmentation_presence_abstention_auxiliary"
            if keep
            else "excluded"
        )
    validated = [row for row in candidates if row["training_eligible"]]
    point_rows = [row for row in validated if row["point_supervised"]]
    point_cap_receipt["effective_training_point_rows"] = len(point_rows)
    point_cap_receipt["source_gate_passed"] = source_gate_passed

    inventory_path = output_dir / "ijmond_archive_inventory.jsonl"
    split_receipt_path = output_dir / "ijmond_split_receipt.json"
    point_cap_receipt_path = output_dir / "ijmond_point_cap_receipt.json"
    date_captured_receipt_path = output_dir / "ijmond_date_captured_receipt.json"
    _write_jsonl(inventory_path, inventory)
    _write_jsonl(output_dir / "ijmond_automatic_dispositions.jsonl", candidates)
    _write_jsonl(output_dir / "ijmond_strict_validated_manifest.jsonl", validated)
    _write_jsonl(output_dir / "ijmond_strict_point_manifest.jsonl", point_rows)
    split_receipt_path.write_text(
        json.dumps(split_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    point_cap_receipt_path.write_text(
        json.dumps(point_cap_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    date_captured_receipt_path.write_text(
        json.dumps(date_captured_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_family": SOURCE_FAMILY,
        "source_revision": SOURCE_DOI,
        "source_record_url": SOURCE_RECORD_URL,
        "source_paper_url": SOURCE_PAPER_URL,
        "declared_license": SOURCE_LICENSE,
        "license_url": SOURCE_LICENSE_URL,
        "figshare_contract_receipt": source_contract_receipt,
        "archive_receipt": archive_receipt,
        "archive_inventory_rows": len(inventory),
        "archive_inventory_sha256": _sha256_file(inventory_path),
        "source_coco_member": coco_name,
        "source_coco_sha256": _sha256_bytes(coco_payload) if coco_payload else None,
        "raw_coco_images": len(coco_images),
        "raw_coco_annotations": len(coco_annotations),
        "raw_coco_polygons": polygon_count,
        "paired_raw_rgb_masks": len(candidates),
        "positive_raw_masks": positive_masks,
        "positive_raw_coco_images": coco_positive_images,
        "coco_mask_presence_disagreements": coco_mask_presence_disagreements,
        "coco_mask_presence_disagreement_policy": (
            "exclude_conflicting_row_only_and_preserve_source_gate_when_"
            "published_source_contract_counts_still_match"
        ),
        "date_captured_receipt": date_captured_receipt_path.name,
        "date_captured_receipt_sha256": _sha256_file(date_captured_receipt_path),
        "date_captured_classification": date_captured_receipt["classification"],
        "date_captured_ignored_for_captured_at": date_captured_receipt[
            "ignored_for_captured_at"
        ],
        "captured_at_from_filename_rows": (
            len(candidates) if date_captured_receipt["ignored_for_captured_at"] else 0
        ),
        "cropped_assets_admitted": 0,
        "rgb_only": True,
        "pairing_or_decode_errors": pairing_errors,
        "camera_counts": dict(
            sorted(Counter(str(row.get("camera")) for row in candidates).items())
        ),
        "camera_episode_groups": len(group_splits),
        "episode_gap_seconds": EPISODE_GAP_SECONDS,
        "split_strategy": split_receipt["strategy"],
        "split_target_ratios": SPLIT_TARGET_RATIOS,
        "split_absolute_ratio_tolerance": SPLIT_RATIO_TOLERANCE,
        "split_receipt": split_receipt_path.name,
        "split_receipt_sha256": _sha256_file(split_receipt_path),
        "split_observed_ratios": split_receipt["observed_ratios"],
        "validated_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in validated).items())
        ),
        "camera_episode_split_leakage": leaking_groups,
        "external_exclusion_indexes": {
            name: {
                "manifests": index["manifests"],
                "indexed_rows": index["indexed_rows"],
                "phash_signatures": index["phash_signatures"],
                "incomplete_rows": index["incomplete_rows"],
                "raw_media_files": index["raw_media_files"],
                "unexpected_files": index["unexpected_files"],
                "receipt_sha256": index["receipt_sha256"],
                "index_sha256": index["receipt"].get("index_sha256"),
            }
            for name, index in indexes.items()
        },
        "deduplication": deduplication,
        "exclusions_by_reason": dict(
            sorted(
                Counter(reason for row in candidates for reason in row["exclusion_reasons"]).items()
            )
        ),
        "point_abstentions_by_reason": dict(
            sorted(
                Counter(reason for row in validated for reason in row["point_gate_errors"]).items()
            )
        ),
        "strict_automated_validated_rows": len(validated),
        "segmentation_supervised_rows": len(validated),
        "presence_supervised_rows": len(validated),
        "presence_supervised_classes": ["smoke_visible"],
        "abstention_supervised_rows": len(validated),
        "smoke_column_base_points": len(point_rows),
        "smoke_column_base_candidates_before_cap": point_cap_receipt[
            "candidate_point_rows"
        ],
        "point_cap_demoted_to_abstention_rows": point_cap_receipt[
            "demoted_to_abstention_rows"
        ],
        "source_raw_positive_ceiling": expected_raw_positive_images,
        "professional_source_point_cap": professional_source_point_cap,
        "strict_defensible_point_ceiling": len(point_rows),
        "point_cap_receipt": point_cap_receipt_path.name,
        "point_cap_receipt_sha256": _sha256_file(point_cap_receipt_path),
        "source_gate_passed": source_gate_passed,
        "gate_errors": gate_errors,
        "reviews_admitted": False,
        "publication_allowed": False,
        "detection_corpus_used_for_training": False,
        "pointing_corpus_used_for_training": False,
        "independent_benchmark_used_for_training": False,
        "benchmark_hash_exclusion_only": True,
        "strict_payload_artifacts_materialized": 0,
        "audit_report_s3_prefix": output_s3_prefix,
        "next_action": (
            "inspect_report_then_explicitly_authorize_composition"
            if source_gate_passed
            else "repair_gate_errors_without_admitting_or_publishing_rows"
        ),
    }
    (output_dir / "ijmond_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-index-root", type=Path, required=True)
    parser.add_argument("--pointing-index-root", type=Path, required=True)
    parser.add_argument("--benchmark-index-root", type=Path, required=True)
    parser.add_argument("--detection-index-receipt-sha256", required=True)
    parser.add_argument("--pointing-index-receipt-sha256", required=True)
    parser.add_argument("--benchmark-index-receipt-sha256", required=True)
    parser.add_argument("--detection-index-rows", type=int, required=True)
    parser.add_argument("--pointing-index-rows", type=int, required=True)
    parser.add_argument("--benchmark-index-rows", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-s3-prefix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    contract = validate_figshare_article(fetch_figshare_article())
    if not contract["source_contract_verified"]:
        raise ValueError(f"IJmond Figshare contract failed: {contract['gate_errors']}")
    archive_path = args.work_dir / ARCHIVE_FILENAME
    download_receipt = download_archive(archive_path)
    report = audit_ijmond_archive(
        archive_path=archive_path,
        detection_index_root=args.detection_index_root,
        pointing_index_root=args.pointing_index_root,
        benchmark_index_root=args.benchmark_index_root,
        output_dir=args.output_dir,
        output_s3_prefix=args.output_s3_prefix,
        source_contract_receipt=contract,
        expected_index_receipts={
            "detection": {
                "receipt_sha256": args.detection_index_receipt_sha256,
                "rows": args.detection_index_rows,
            },
            "pointing": {
                "receipt_sha256": args.pointing_index_receipt_sha256,
                "rows": args.pointing_index_rows,
            },
            "benchmark": {
                "receipt_sha256": args.benchmark_index_receipt_sha256,
                "rows": args.benchmark_index_rows,
            },
        },
    )
    report["download_receipt"] = download_receipt
    (args.output_dir / "ijmond_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive_path.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["source_gate_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
