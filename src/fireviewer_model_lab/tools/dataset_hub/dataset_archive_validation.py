"""Validation helpers for legacy source archives used by train bundles.

This module is not a publication entrypoint. FireWarning publishes one ZIP per
training objective through ``finalize_train_bundle.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from PIL import Image

BUFFER_SIZE = 4 * 1024 * 1024
REQUIRED_MANIFEST_FIELDS = {
    "annotations",
    "corpus_role",
    "height",
    "image_relpath",
    "sample_id",
    "sha256",
    "split",
    "split_group",
    "width",
}


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def _safe_relative_path(raw_path: str, *, prefix: str | None = None) -> PurePosixPath:
    normalized = raw_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {raw_path}")
    if prefix is not None:
        if path.parts[0] != prefix or len(path.parts) == 1:
            raise ValueError(f"Archive entry is outside {prefix}/: {raw_path}")
        path = PurePosixPath(*path.parts[1:])
    if not path.parts or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"Invalid relative path: {raw_path}")
    return path


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> int:
    copied = 0
    while True:
        chunk = source.read(BUFFER_SIZE)
        if not chunk:
            return copied
        destination.write(chunk)
        copied += len(chunk)


def load_archive_manifest(path: Path, dataset_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != dataset_id:
        raise ValueError(
            f"Archive manifest dataset_id mismatch: {payload.get('dataset_id')!r} != {dataset_id!r}"
        )
    if not payload.get("shards"):
        raise ValueError("Archive manifest contains no shards")
    return payload


def validate_and_extract_shards(
    *, archive_manifest: dict[str, Any], manifest_dir: Path, dataset_root: Path
) -> dict[str, Any]:
    seen_paths: set[str] = set()
    extracted_bytes = 0
    extracted_files = 0
    shard_reports: list[dict[str, Any]] = []

    for shard in archive_manifest["shards"]:
        shard_path = manifest_dir / Path(str(shard["path"])).name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        actual_size = shard_path.stat().st_size
        if actual_size != int(shard["size_bytes"]):
            raise ValueError(f"Shard size mismatch: {shard_path}")
        actual_sha256 = sha256_file(shard_path)
        if actual_sha256 != str(shard["sha256"]):
            raise ValueError(f"Shard SHA-256 mismatch: {shard_path}")

        shard_files = 0
        shard_bytes = 0
        with tarfile.open(shard_path, mode="r:") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsupported tar member type: {member.name}")
                relative = _safe_relative_path(member.name, prefix="payload")
                relative_string = relative.as_posix()
                if relative_string in seen_paths:
                    raise ValueError(f"Duplicate payload path across shards: {relative_string}")
                seen_paths.add(relative_string)
                target = dataset_root.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unable to read tar member: {member.name}")
                with source, target.open("wb") as destination:
                    copied = _copy_stream(source, destination)
                if copied != member.size:
                    raise ValueError(f"Extracted size mismatch: {member.name}")
                os.chmod(target, 0o644)
                shard_files += 1
                shard_bytes += copied

        if shard_files != int(shard["file_count"]):
            raise ValueError(f"Shard file count mismatch: {shard_path}")
        if shard_bytes != int(shard["source_bytes"]):
            raise ValueError(f"Shard source byte count mismatch: {shard_path}")
        extracted_files += shard_files
        extracted_bytes += shard_bytes
        shard_reports.append(
            {
                "name": shard_path.name,
                "sha256": actual_sha256,
                "size_bytes": actual_size,
                "file_count": shard_files,
                "source_bytes": shard_bytes,
            }
        )
        print(
            "train-bundle source shard verified"
            f" name={shard_path.name} files={shard_files} bytes={shard_bytes}",
            flush=True,
        )

    if extracted_files != int(archive_manifest["file_count"]):
        raise ValueError("Dataset file count does not match archive manifest")
    if extracted_bytes != int(archive_manifest["source_bytes"]):
        raise ValueError("Dataset byte count does not match archive manifest")
    return {
        "file_count": extracted_files,
        "source_bytes": extracted_bytes,
        "shards": shard_reports,
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def validate_optional_location(location: Any, *, line_number: int) -> bool:
    """Validate coordinates while allowing explicit non-coordinate geographic context.

    A record may carry a massif/event reference without an exact point. In that
    case both coordinate values must be null. A partial pair is ambiguous and is
    therefore rejected.
    """
    if not isinstance(location, dict):
        raise ValueError(f"Invalid location object at manifest line {line_number}")
    latitude_value = location.get("latitude")
    longitude_value = location.get("longitude")
    if latitude_value is None and longitude_value is None:
        return False
    if latitude_value is None or longitude_value is None:
        raise ValueError(f"Incomplete coordinate pair at manifest line {line_number}")
    try:
        latitude = float(latitude_value)
        longitude = float(longitude_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid coordinate value at manifest line {line_number}") from error
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError(f"Non-finite location at manifest line {line_number}")
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError(f"Invalid location at manifest line {line_number}")
    return True


def validate_firewarning_dataset(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FireWarning dataset manifest missing: {manifest_path}")

    sample_ids: set[str] = set()
    image_digests: set[str] = set()
    manifest_images: set[str] = set()
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    total_image_bytes = 0
    coordinate_location_rows = 0
    context_only_location_rows = 0
    rows = 0

    for line_number, record in _iter_jsonl(manifest_path):
        missing = REQUIRED_MANIFEST_FIELDS - set(record)
        if missing:
            raise ValueError(f"Missing fields at manifest line {line_number}: {sorted(missing)}")
        rows += 1
        if rows % 10_000 == 0:
            print(f"train-bundle image validation rows={rows}", flush=True)
        sample_id = str(record["sample_id"])
        digest = str(record["sha256"]).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Invalid SHA-256 at manifest line {line_number}")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id at manifest line {line_number}: {sample_id}")
        if digest in image_digests:
            raise ValueError(f"Duplicate image SHA-256 at manifest line {line_number}: {digest}")
        sample_ids.add(sample_id)
        image_digests.add(digest)

        relative = _safe_relative_path(str(record["image_relpath"]))
        relative_string = relative.as_posix()
        if relative_string in manifest_images:
            raise ValueError(
                f"Duplicate image path at manifest line {line_number}: {relative_string}"
            )
        manifest_images.add(relative_string)
        image_path = dataset_root.joinpath(*relative.parts)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if image_path.stem.lower() != digest:
            raise ValueError(f"Content-addressed filename mismatch: {relative_string}")
        actual_digest = sha256_file(image_path)
        if actual_digest != digest:
            raise ValueError(f"Image SHA-256 mismatch: {relative_string}")
        total_image_bytes += image_path.stat().st_size
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
        if width != int(record["width"]) or height != int(record["height"]):
            raise ValueError(f"Image dimensions mismatch: {relative_string}")

        split = str(record["split"])
        split_group = str(record["split_group"])
        split_groups[split_group].add(split)
        split_counts[split] += 1
        role_counts[str(record["corpus_role"])] += 1
        source_counts[str(record.get("source_id", "unknown"))] += 1

        location = record.get("location")
        if location is not None:
            if validate_optional_location(location, line_number=line_number):
                coordinate_location_rows += 1
            else:
                context_only_location_rows += 1

        for annotation in record["annotations"]:
            x, y, box_width, box_height = (float(value) for value in annotation["bbox_xywh"])
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Invalid annotation box at manifest line {line_number}")
            if x + box_width > width + 1e-3 or y + box_height > height + 1e-3:
                raise ValueError(f"Annotation box exceeds image at manifest line {line_number}")

    leaking_groups = sorted(group for group, splits in split_groups.items() if len(splits) > 1)
    if leaking_groups:
        raise ValueError(f"Split leakage detected in {len(leaking_groups)} groups")
    on_disk_images = set()
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(dataset_root)
        if relative.parts and relative.parts[0] == "images":
            on_disk_images.add(relative.as_posix())
    missing_from_manifest = sorted(on_disk_images - manifest_images)
    missing_on_disk = sorted(manifest_images - on_disk_images)
    if missing_from_manifest or missing_on_disk:
        raise ValueError(
            "Image inventory mismatch: "
            f"unreferenced={len(missing_from_manifest)} missing={len(missing_on_disk)}"
        )

    return {
        "manifest_sha256": sha256_file(manifest_path),
        "rows": rows,
        "unique_sample_ids": len(sample_ids),
        "unique_image_sha256": len(image_digests),
        "image_files": len(on_disk_images),
        "image_bytes": total_image_bytes,
        "split_counts": dict(sorted(split_counts.items())),
        "split_group_counts": dict(
            sorted(Counter(next(iter(splits)) for splits in split_groups.values()).items())
        ),
        "split_leakage_groups": 0,
        "role_counts": dict(sorted(role_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "files_verified": True,
        "images_decoded": True,
        "coordinate_location_rows": coordinate_location_rows,
        "context_only_location_rows": context_only_location_rows,
    }


def iter_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file()), key=lambda path: path.as_posix()
    )


def payload_checksums(dataset_root: Path) -> tuple[str, dict[str, str]]:
    checksums: dict[str, str] = {}
    for path in iter_files(dataset_root):
        checksums[path.relative_to(dataset_root).as_posix()] = sha256_file(path)
    content = "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items())
    return content, checksums


def write_generated_metadata(
    *,
    dataset_root: Path,
    dataset_id: str,
    archive_report: dict[str, Any],
    data_report: dict[str, Any],
) -> dict[str, bytes]:
    acquisition_report_path = dataset_root / "acquisition-report.json"
    acquisition_report = (
        json.loads(acquisition_report_path.read_text(encoding="utf-8"))
        if acquisition_report_path.is_file()
        else {}
    )
    package_manifest = {
        "schema_version": 1,
        "package_format": "firewarning-dataset-zip-v1",
        "dataset_id": dataset_id,
        "root_directory": dataset_id.split("/")[-1],
        "source_repository": acquisition_report.get("source_repository"),
        "source_id": acquisition_report.get("source_id"),
        "source_license": acquisition_report.get("source_license"),
        "archive_validation": archive_report,
        "dataset_validation": data_report,
    }
    readme = (
        f"# {dataset_id}\n\n"
        "FireWarning dataset package. Extract this ZIP into an empty directory.\n\n"
        "- `manifest.jsonl` is the machine-readable sample inventory.\n"
        "- `PAYLOAD_CHECKSUMS.sha256` covers every original dataset file.\n"
        "- `VALIDATION_REPORT.json` records the validation performed before publication.\n"
        "- Source license and provenance remain dataset-specific; see `PACKAGE_MANIFEST.json`.\n"
    ).encode()
    return {
        "PACKAGE_MANIFEST.json": (
            json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "README.md": readme,
        "VALIDATION_REPORT.json": (
            json.dumps(data_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }


def _zip_info(name: str, *, compressed: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def build_zip(
    *, dataset_root: Path, dataset_slug: str, output_path: Path, generated: dict[str, bytes]
) -> dict[str, str]:
    checksum_text, payload_digests = payload_checksums(dataset_root)
    generated = dict(generated)
    generated["PAYLOAD_CHECKSUMS.sha256"] = checksum_text.encode("utf-8")
    expected: dict[str, str] = {}
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, mode="w", allowZip64=True) as archive:
        for path in iter_files(dataset_root):
            relative = path.relative_to(dataset_root).as_posix()
            entry_name = f"{dataset_slug}/{relative}"
            info = _zip_info(
                entry_name, compressed=path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
            )
            with (
                path.open("rb") as source,
                archive.open(info, mode="w", force_zip64=True) as destination,
            ):
                _copy_stream(source, destination)
            expected[entry_name] = payload_digests[relative]
        for relative, content in sorted(generated.items()):
            entry_name = f"{dataset_slug}/{relative}"
            archive.writestr(_zip_info(entry_name, compressed=True), content)
            expected[entry_name] = hashlib.sha256(content).hexdigest()
    os.replace(temporary, output_path)
    return expected


def validate_zip(path: Path, expected: dict[str, str], dataset_slug: str) -> dict[str, Any]:
    seen: set[str] = set()
    with zipfile.ZipFile(path, mode="r", allowZip64=True) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC validation failed: {corrupt}")
        infos = archive.infolist()
        for info in infos:
            relative = _safe_relative_path(info.filename)
            if relative.parts[0] != dataset_slug:
                raise ValueError(f"ZIP entry outside dataset root: {info.filename}")
            if info.filename in seen:
                raise ValueError(f"Duplicate ZIP entry: {info.filename}")
            seen.add(info.filename)
            with archive.open(info, "r") as stream:
                actual_digest = sha256_stream(stream)
            if expected.get(info.filename) != actual_digest:
                raise ValueError(f"ZIP entry SHA-256 mismatch: {info.filename}")
    if seen != set(expected):
        raise ValueError("ZIP entry inventory mismatch")
    return {
        "zip_sha256": sha256_file(path),
        "zip_size_bytes": path.stat().st_size,
        "entry_count": len(seen),
        "crc_verified": True,
        "entry_sha256_verified": True,
        "single_dataset_root": dataset_slug,
    }
