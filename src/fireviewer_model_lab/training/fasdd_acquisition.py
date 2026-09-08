from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from fireviewer_model_lab.training.corpus_pipeline import (
    _inspect_image,
    deterministic_split,
    normalized_identifier,
    sha256_bytes,
    validate_manifest,
)

USER_AGENT = "FireWarningCorpusBuilder/0.1 (https://github.com/unicornwhodev/fireviewer)"
REPORT_INTERVAL_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024
FASDD_LICENSE_REFERENCE = "https://creativecommons.org/licenses/by-sa/4.0/"
EXPECTED_ROWS = {"CV": 95_314, "RS": 2_223, "UAV": 25_097}
COCO_PATH = re.compile(
    r"annotations/COCO[^/]*/Annotations/(?P<split>train|val|test)\.json",
    re.IGNORECASE,
)
SEQUENCE_NUMBER = re.compile(r"(?P<prefix>.*?)(?P<number>\d+)$")
SOURCE_CATEGORY_MAP = {
    "smoke": (0, "smoke_visible"),
    "fire": (1, "flame_visible"),
}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
VISUAL_FINGERPRINT_VERSION = "rgb32q5-v1"


@dataclass(frozen=True)
class ArchiveSpec:
    lot: str
    filename: str
    file_id: str
    expected_bytes: int
    expected_md5: str

    @property
    def url(self) -> str:
        return f"https://china.scidb.cn/download?fileId={self.file_id}"


ARCHIVES = {
    spec.lot: spec
    for spec in (
        ArchiveSpec(
            lot="CV",
            filename="FASDD_CV.zip",
            file_id="85bad3972ed60e0a20a790032c0d85fb",
            expected_bytes=12_326_081_298,
            expected_md5="da5e1410e13113fc9d605567aea54508",
        ),
        ArchiveSpec(
            lot="RS",
            filename="FASDD_RS.zip",
            file_id="0fefc7486bf648a0ce754e093f3a56f2",
            expected_bytes=4_707_516_376,
            expected_md5="42d07b005f99f6903f5c1610fb1a5bbb",
        ),
        ArchiveSpec(
            lot="UAV",
            filename="FASDD_UAV.zip",
            file_id="9456106d26c5fc6b74143c3707115d39",
            expected_bytes=14_888_184_890,
            expected_md5="c5ea9651ca672fc128c6f17d0797c2f2",
        ),
    )
}


def _ensure_within(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path escaped dataset root: {resolved_candidate}") from exc
    return resolved_candidate


def safe_zip_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if (
        not normalized
        or member.is_absolute()
        or ".." in member.parts
        or (member.parts and member.parts[0].endswith(":"))
    ):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return member


def logical_zip_member(name: str) -> PurePosixPath:
    member = safe_zip_member(name)
    if len(member.parts) > 1 and member.parts[0].casefold().startswith("fasdd_"):
        return PurePosixPath(*member.parts[1:])
    return member


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, spec: ArchiveSpec) -> None:
    actual_bytes = path.stat().st_size
    if actual_bytes != spec.expected_bytes:
        raise ValueError(
            f"{spec.filename} has {actual_bytes} bytes, expected {spec.expected_bytes}"
        )
    actual_md5 = md5_file(path)
    if actual_md5 != spec.expected_md5:
        raise ValueError(f"{spec.filename} MD5 is {actual_md5}, expected {spec.expected_md5}")


def download_archive(root: Path, spec: ArchiveSpec) -> Path:
    staging = _ensure_within(root, root / "_staging" / "fasdd")
    staging.mkdir(parents=True, exist_ok=True)
    destination = _ensure_within(root, staging / spec.filename)
    partial = _ensure_within(root, destination.with_suffix(destination.suffix + ".part"))

    if destination.exists():
        verify_archive(destination, spec)
        print(f"verified existing {destination}", flush=True)
        return destination

    start = partial.stat().st_size if partial.exists() else 0
    if start > spec.expected_bytes:
        raise ValueError(
            f"Partial archive is larger than expected: {start} > {spec.expected_bytes}"
        )
    if start == spec.expected_bytes:
        verify_archive(partial, spec)
        partial.replace(destination)
        print(f"verified completed partial {destination}", flush=True)
        return destination

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
    timeout = httpx.Timeout(connect=30.0, read=180.0, write=180.0, pool=30.0)
    started_at = time.monotonic()
    next_report = start + REPORT_INTERVAL_BYTES

    with (
        httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client,
        client.stream("GET", spec.url) as response,
    ):
        if start and response.status_code == 200:
            print("server ignored Range; restarting this archive", flush=True)
            start = 0
            next_report = REPORT_INTERVAL_BYTES
        elif start and response.status_code != 206:
            response.raise_for_status()
            raise RuntimeError(f"Expected HTTP 206 for resume, got {response.status_code}")
        elif not start and response.status_code not in {200, 206}:
            response.raise_for_status()
        if start:
            content_range = response.headers.get("content-range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")

        mode = "ab" if start else "wb"
        downloaded = start
        with partial.open(mode) as handle:
            for chunk in response.iter_bytes(CHUNK_BYTES):
                remaining = spec.expected_bytes - downloaded
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    elapsed = max(time.monotonic() - started_at, 0.001)
                    transferred = downloaded - start
                    rate_mib = transferred / elapsed / (1024 * 1024)
                    percent = downloaded / spec.expected_bytes * 100
                    print(
                        f"{spec.lot}: {downloaded}/{spec.expected_bytes} "
                        f"({percent:.1f}%) at {rate_mib:.1f} MiB/s",
                        flush=True,
                    )
                    next_report = downloaded + REPORT_INTERVAL_BYTES
                if downloaded == spec.expected_bytes:
                    break
            handle.flush()
            os.fsync(handle.fileno())

    verify_archive(partial, spec)
    partial.replace(destination)
    print(f"download verified: {destination}", flush=True)
    return destination


def inspect_archive(path: Path) -> dict[str, Any]:
    suffixes: Counter[str] = Counter()
    prefixes: Counter[str] = Counter()
    member_samples: list[str] = []
    uncompressed_bytes = 0
    compressed_bytes = 0
    directory_count = 0

    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member = logical_zip_member(info.filename)
            if info.flag_bits & 0x1:
                raise ValueError(f"Encrypted ZIP member is not supported: {info.filename!r}")
            if info.is_dir():
                directory_count += 1
                continue
            if len(member_samples) < 200:
                member_samples.append(member.as_posix())
            suffixes[member.suffix.casefold() or "<none>"] += 1
            prefix_depth = 3 if member.parts[0].casefold() == "annotations" else 1
            prefixes["/".join(member.parts[: min(prefix_depth, len(member.parts))])] += 1
            uncompressed_bytes += info.file_size
            compressed_bytes += info.compress_size

    return {
        "archive": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "member_count": sum(suffixes.values()),
        "directory_count": directory_count,
        "uncompressed_bytes": uncompressed_bytes,
        "compressed_member_bytes": compressed_bytes,
        "suffix_counts": dict(sorted(suffixes.items())),
        "prefix_counts": dict(prefixes.most_common(200)),
        "member_samples": member_samples,
    }


def fasdd_split_group(lot: str, filename: str) -> str:
    stem = PurePosixPath(filename).stem
    match = SEQUENCE_NUMBER.fullmatch(stem)
    if match is None:
        return f"fasdd-v9:{lot.casefold()}:{normalized_identifier(stem)}"
    prefix = normalized_identifier(match.group("prefix"))
    sequence_block = int(match.group("number")) // 25
    return f"fasdd-v9:{lot.casefold()}:{prefix}:{sequence_block:06d}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid manifest object at line {line_number}: {path}")
            records.append(record)
    return records


def _manifest_digests(paths: Iterable[Path]) -> set[str]:
    digests: set[str] = set()
    for path in paths:
        for record in _load_jsonl(path):
            digest = str(record["sha256"])
            if digest in digests:
                raise ValueError(f"Duplicate digest in exclusion manifests: {digest}")
            digests.add(digest)
    return digests


def _source_image_payload(archive: zipfile.ZipFile, member_name: str) -> bytes:
    from PIL import Image

    payload = archive.read(member_name)
    suffix = PurePosixPath(member_name).suffix.casefold()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return payload
    if suffix not in {".tif", ".tiff", ".bmp"}:
        raise ValueError(f"Unsupported FASDD image format: {member_name}")
    Image.MAX_IMAGE_PIXELS = 120_000_000
    with Image.open(io.BytesIO(payload)) as source:
        rgb = source.convert("RGB")
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=92, subsampling=0)
    return output.getvalue()


def _coco_annotations(
    annotations: list[dict[str, Any]],
    categories: dict[int, str],
    *,
    width: int,
    height: int,
    lot: str,
    source_name: str,
    quality_fixes: Counter[str],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for annotation in annotations:
        category_name = categories[int(annotation["category_id"])].casefold()
        if category_name not in SOURCE_CATEGORY_MAP:
            raise ValueError(f"Unexpected FASDD category: {category_name!r}")
        class_id, class_name = SOURCE_CATEGORY_MAP[category_name]
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        if x < 0 or y < 0:
            raise ValueError(f"Invalid FASDD bbox in lot {lot}: {annotation['bbox']!r}")
        if box_width <= 0 or box_height <= 0:
            quality_fixes["dropped_non_positive_bbox"] += 1
            continue
        overflow_x = x + box_width - width
        overflow_y = y + box_height - height
        if overflow_x > 1.000_001 or overflow_y > 1.000_001:
            raise ValueError(
                f"FASDD bbox exceeds image {source_name!r} in lot {lot}: "
                f"bbox={annotation['bbox']!r}, dimensions=({width}, {height})"
            )
        if overflow_x > 0 or overflow_y > 0:
            box_width = min(box_width, width - x)
            box_height = min(box_height, height - y)
            quality_fixes["clamped_one_pixel_bbox"] += 1
        converted.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "bbox_xywh": [x, y, box_width, box_height],
                "visibility": "unknown",
                "occlusion": "unknown",
                "origin": f"FASDD-V9-COCO:{lot}",
                "annotated_at": None,
                "annotator_id": "FASDD-authors",
                "validation_status": "source_provided",
            }
        )
    return converted


def _normalized_coco_image_annotations(
    source_image: dict[str, Any],
    source_annotations: list[dict[str, Any]],
    categories: dict[int, str],
    *,
    width: int,
    height: int,
    lot: str,
    filename: str,
    quality_fixes: Counter[str],
) -> list[dict[str, Any]]:
    if width != int(source_image["width"]) or height != int(source_image["height"]):
        if source_annotations:
            raise ValueError(
                f"COCO dimensions do not match annotated image bytes: {filename}; "
                f"declared=({source_image['width']}, {source_image['height']}), "
                f"actual=({width}, {height})"
            )
        quality_fixes["accepted_negative_dimension_mismatch"] += 1
    return _coco_annotations(
        source_annotations,
        categories,
        width=width,
        height=height,
        lot=lot,
        source_name=filename,
        quality_fixes=quality_fixes,
    )


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def visual_fingerprint(payload: bytes) -> str:
    """Return a collision-resistant fingerprint of a quantized RGB thumbnail."""

    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        thumbnail = image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
        quantized = bytes(channel & 0xF8 for channel in thumbnail.tobytes())
    digest = hashlib.sha256(quantized, usedforsecurity=False).hexdigest()
    return f"{VISUAL_FINGERPRINT_VERSION}:{digest}"


def ensure_visual_fingerprints(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, int]:
    """Populate the strong visual fingerprint on legacy or resumed records."""

    cache_path = _ensure_within(
        output_dir,
        output_dir / "visual-fingerprints.cache.jsonl",
    )
    cached: dict[str, str] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid visual fingerprint cache line {line_number}"
                    ) from exc
                cached[str(entry["sha256"])] = str(entry["visual_fingerprint"])

    cache_hits = 0
    missing: list[dict[str, Any]] = []
    for record in records:
        if record.get("visual_fingerprint"):
            continue
        fingerprint = cached.get(str(record["sha256"]))
        if fingerprint:
            record["visual_fingerprint"] = fingerprint
            cache_hits += 1
        else:
            missing.append(record)

    def calculate(record: dict[str, Any]) -> tuple[str, str]:
        image_path = _ensure_within(
            output_dir,
            output_dir / str(record["image_relpath"]),
        )
        return str(record["sha256"]), visual_fingerprint(image_path.read_bytes())

    added = 0
    workers = min(8, os.cpu_count() or 4)
    with (
        cache_path.open("a", encoding="utf-8", newline="\n") as cache,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        for record, (digest, fingerprint) in zip(
            missing,
            executor.map(calculate, missing),
            strict=True,
        ):
            record["visual_fingerprint"] = fingerprint
            cache.write(
                json.dumps(
                    {"sha256": digest, "visual_fingerprint": fingerprint},
                    sort_keys=True,
                )
                + "\n"
            )
            added += 1
            if added % 1_000 == 0:
                cache.flush()
                print(
                    f"visual fingerprints: {added}/{len(missing)} computed with {workers} workers",
                    flush=True,
                )
        cache.flush()
        os.fsync(cache.fileno())
    return {"computed": added, "cache_hits": cache_hits}


def reconcile_fasdd_visual_splits(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Balance sequence components without joining pHash collision chains.

    A duplicate link requires an exact match on pHash, dimensions and the stronger
    quantized RGB thumbnail fingerprint. Sequence blocks remain indivisible.
    """

    parents: dict[str, str] = {}

    def find(group: str) -> str:
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    previous_links = 0
    first_by_visual_key: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    visual_duplicate_links = 0
    for record in records:
        group = str(record["split_group"])
        find(group)
        if record.get("near_duplicate_of"):
            previous_links += 1
        record["near_duplicate_of"] = None
        fingerprint = str(record.get("visual_fingerprint") or "")
        if not fingerprint:
            raise ValueError("FASDD record is missing visual_fingerprint")
        visual_key = (
            str(record["phash"]),
            int(record["width"]),
            int(record["height"]),
            fingerprint,
        )
        first = first_by_visual_key.get(visual_key)
        if first is None:
            first_by_visual_key[visual_key] = record
            continue
        record["near_duplicate_of"] = str(first["sha256"])
        union(group, str(first["split_group"]))
        visual_duplicate_links += 1

    component_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        component_records[find(str(record["split_group"]))].append(record)

    targets = {split: len(records) * ratio for split, ratio in SPLIT_RATIOS.items()}
    assigned_counts = {split: 0 for split in SPLIT_RATIOS}
    component_splits: dict[str, str] = {}
    split_order = tuple(SPLIT_RATIOS)
    for root, component in sorted(
        component_records.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        assigned_split = min(
            split_order,
            key=lambda split: (
                assigned_counts[split] / targets[split],
                split_order.index(split),
            ),
        )
        component_splits[root] = assigned_split
        assigned_counts[assigned_split] += len(component)

    reassigned_rows = 0
    for record in records:
        assigned_split = component_splits[find(str(record["split_group"]))]
        if str(record["split"]) != assigned_split:
            reassigned_rows += 1
            record["split"] = assigned_split

    digest_records = {str(record["sha256"]): record for record in records}
    cross_split_after = sum(
        1
        for record in records
        if record.get("near_duplicate_of")
        and str(record["split"]) != str(digest_records[str(record["near_duplicate_of"])]["split"])
    )
    if cross_split_after:
        raise ValueError("FASDD visual split reconciliation did not converge")

    return {
        "previous_near_duplicate_links_cleared": previous_links,
        "visual_fingerprint_duplicate_links": visual_duplicate_links,
        "cross_split_visual_duplicates_after": cross_split_after,
        "component_count": len(component_records),
        "merged_group_components": sum(
            len({str(record["split_group"]) for record in component}) > 1
            for component in component_records.values()
        ),
        "largest_component_rows": max(
            (len(component) for component in component_records.values()),
            default=0,
        ),
        "reassigned_rows": reassigned_rows,
        "target_split_rows": {split: round(target) for split, target in targets.items()},
        "split_row_counts": assigned_counts,
    }


def curate_archive(
    root: Path,
    archive_path: Path,
    spec: ArchiveSpec,
    *,
    expected_rows: int,
    exclusion_manifests: Iterable[Path] = (),
    delete_archive: bool = False,
    verify_source: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    archive_path = _ensure_within(root, archive_path)
    if verify_source:
        verify_archive(archive_path, spec)
    output_dir = _ensure_within(root, root / "corpus" / "fasdd")
    output_dir.mkdir(parents=True, exist_ok=True)
    lots_dir = _ensure_within(root, output_dir / "lots")
    lots_dir.mkdir(parents=True, exist_ok=True)
    partial_path = _ensure_within(root, output_dir / "manifest.partial.jsonl")
    final_path = _ensure_within(root, output_dir / "manifest.jsonl")
    if final_path.exists():
        raise FileExistsError(f"Final FASDD manifest already exists: {final_path}")

    records = _load_jsonl(partial_path)
    source_records = {str(record["source_record_id"]): record for record in records}
    seen_digests = {str(record["sha256"]) for record in records}
    excluded_digests = _manifest_digests(exclusion_manifests)
    source_rows = 0
    appended_rows = 0
    resumed_rows = 0
    duplicate_rows = 0
    class_counts: Counter[str] = Counter()
    negative_rows = 0
    source_json_sha256: dict[str, str] = {}
    source_quality_fixes: Counter[str] = Counter()

    with zipfile.ZipFile(archive_path) as archive:
        image_members: dict[str, str] = {}
        for info in archive.infolist():
            member = logical_zip_member(info.filename)
            if info.is_dir() or not member.parts or member.parts[0].casefold() != "images":
                continue
            basename = member.name
            if basename in image_members:
                raise ValueError(f"Duplicate FASDD image basename: {basename}")
            image_members[basename] = info.filename

        coco_members: list[tuple[str, str]] = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            logical_name = logical_zip_member(info.filename).as_posix()
            match = COCO_PATH.fullmatch(logical_name)
            if match is not None:
                coco_members.append((match.group("split").casefold(), info.filename))
        coco_members.sort()
        if {split for split, _name in coco_members} != {"test", "train", "val"}:
            raise ValueError(f"Lot {spec.lot} does not contain the three expected COCO splits")

        seen_source_images: set[str] = set()
        for source_split, coco_member in coco_members:
            raw_json = archive.read(coco_member)
            source_json_sha256[source_split] = sha256_bytes(raw_json)
            document = json.loads(raw_json)
            categories = {
                int(category["id"]): str(category["name"]) for category in document["categories"]
            }
            annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
            for annotation in document["annotations"]:
                annotations_by_image[int(annotation["image_id"])].append(annotation)

            for source_image in document["images"]:
                source_rows += 1
                filename = PurePosixPath(str(source_image["file_name"])).name
                if filename in seen_source_images:
                    raise ValueError(f"FASDD image appears in multiple splits: {filename}")
                seen_source_images.add(filename)
                source_record_id = f"{spec.lot}:{source_split}:{filename}"
                source_annotations = annotations_by_image[int(source_image["id"])]
                existing_record = source_records.get(source_record_id)
                if existing_record is not None:
                    _normalized_coco_image_annotations(
                        source_image,
                        source_annotations,
                        categories,
                        width=int(existing_record["width"]),
                        height=int(existing_record["height"]),
                        lot=spec.lot,
                        filename=filename,
                        quality_fixes=source_quality_fixes,
                    )
                    resumed_rows += 1
                    continue
                member_name = image_members.get(filename)
                if member_name is None:
                    raise ValueError(f"Missing FASDD image member: {filename}")
                payload = _source_image_payload(archive, member_name)
                width, height, extension, perceptual_hash = _inspect_image(payload)
                annotations = _normalized_coco_image_annotations(
                    source_image,
                    source_annotations,
                    categories,
                    width=width,
                    height=height,
                    lot=spec.lot,
                    filename=filename,
                    quality_fixes=source_quality_fixes,
                )
                digest = sha256_bytes(payload)
                if digest in seen_digests or digest in excluded_digests:
                    duplicate_rows += 1
                    continue
                split_group = fasdd_split_group(spec.lot, filename)
                split = deterministic_split(split_group)

                destination = _ensure_within(
                    root,
                    output_dir / "images" / digest[:2] / f"{digest}.{extension}",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and sha256_bytes(destination.read_bytes()) != digest:
                    raise ValueError(f"Existing FASDD corpus image has a bad digest: {destination}")
                if not destination.exists():
                    destination.write_bytes(payload)

                candidate_classes = sorted(
                    {str(annotation["class_name"]) for annotation in annotations}
                )
                record = {
                    "sample_id": f"fasdd-v9-{spec.lot.casefold()}-{digest[:24]}",
                    "source_id": "fasdd_v9",
                    "source_record_id": source_record_id,
                    "corpus_role": "detector_training",
                    "image_relpath": destination.relative_to(output_dir).as_posix(),
                    "sha256": digest,
                    "phash": perceptual_hash,
                    "visual_fingerprint": visual_fingerprint(payload),
                    "near_duplicate_of": None,
                    "width": width,
                    "height": height,
                    "event_id": f"fasdd-v9-{spec.lot.casefold()}",
                    "sequence_id": split_group,
                    "split_group": split_group,
                    "split": split,
                    "license": "CC-BY-SA-4.0",
                    "consent_basis": {
                        "kind": "source_license",
                        "reference": FASDD_LICENSE_REFERENCE,
                    },
                    "sample_validation_status": "source_provided",
                    "candidate_classes": candidate_classes,
                    "annotations": annotations,
                    "negative_tags": [] if annotations else ["no_target_visible"],
                    "location": None,
                }
                records.append(record)
                source_records[source_record_id] = record
                seen_digests.add(digest)
                appended_rows += 1
                if annotations:
                    class_counts.update(candidate_classes)
                else:
                    negative_rows += 1
                with partial_path.open("a", encoding="utf-8", newline="\n") as manifest:
                    manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    manifest.flush()
                if appended_rows % 1_000 == 0:
                    print(
                        f"{spec.lot}: curated {appended_rows}/{expected_rows} new rows",
                        flush=True,
                    )

    if source_rows != expected_rows:
        raise ValueError(f"Lot {spec.lot} has {source_rows} COCO images, expected {expected_rows}")
    validation = validate_manifest(partial_path, output_dir=output_dir, verify_files=True)
    report = {
        "source_id": "fasdd_v9",
        "lot": spec.lot,
        "archive_filename": spec.filename,
        "archive_bytes": archive_path.stat().st_size,
        "archive_md5": md5_file(archive_path),
        "source_rows": source_rows,
        "appended_rows": appended_rows,
        "resumed_rows": resumed_rows,
        "exact_duplicate_rows_skipped": duplicate_rows,
        "source_annotation_json_sha256": source_json_sha256,
        "new_annotation_class_presence": dict(sorted(class_counts.items())),
        "new_negative_rows": negative_rows,
        "source_quality_fixes": dict(sorted(source_quality_fixes.items())),
        "discarded_representations": ["YOLO", "VOC", "TDML"],
        "validation": validation,
    }
    report_path = _ensure_within(root, lots_dir / f"{spec.lot}.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if delete_archive:
        archive_path.unlink()
        structure_report = archive_path.with_suffix(".structure.json")
        if structure_report.exists():
            structure_report.unlink()
    return report


def finalize_corpus(root: Path) -> dict[str, Any]:
    root = root.resolve()
    output_dir = _ensure_within(root, root / "corpus" / "fasdd")
    partial_path = _ensure_within(root, output_dir / "manifest.partial.jsonl")
    final_path = _ensure_within(root, output_dir / "manifest.jsonl")
    lots_dir = _ensure_within(root, output_dir / "lots")
    missing_lots = sorted(lot for lot in ARCHIVES if not (lots_dir / f"{lot}.json").exists())
    if missing_lots:
        raise ValueError(f"Cannot finalize FASDD corpus; missing lots: {missing_lots}")
    source_path = partial_path if partial_path.exists() else final_path
    if not source_path.exists():
        raise FileNotFoundError("FASDD manifest is missing")
    records = _load_jsonl(source_path)
    fingerprint_materialization = ensure_visual_fingerprints(
        records,
        output_dir=output_dir,
    )
    reconciliation = reconcile_fasdd_visual_splits(records)
    candidate_path = _ensure_within(root, output_dir / "manifest.finalizing.jsonl")
    _write_manifest(candidate_path, records)
    try:
        validation = validate_manifest(
            candidate_path,
            output_dir=output_dir,
            verify_files=True,
        )
        candidate_path.replace(final_path)
    finally:
        if candidate_path.exists():
            candidate_path.unlink()
    report = {
        "source_id": "fasdd_v9",
        "rows": len(records),
        "visual_fingerprint_version": VISUAL_FINGERPRINT_VERSION,
        "visual_fingerprint_materialization": fingerprint_materialization,
        "selected_lots": sorted(ARCHIVES),
        "excluded_archives": {
            "FASDD.zip": "duplicates the selected domain archives",
            "FASDD_RS_RAW.zip": "raw remote-sensing source is not detector-ready RGB",
            "FASDD_RS_SWIR.zip": "different spectral domain from the RGB deployment input",
        },
        "reconciliation": reconciliation,
        "validation": validation,
    }
    (output_dir / "quality-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if partial_path.exists():
        partial_path.unlink()
    fingerprint_cache = output_dir / "visual-fingerprints.cache.jsonl"
    if fingerprint_cache.exists():
        fingerprint_cache.unlink()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire verified, selected FASDD archives")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--root", type=Path, required=True)
    download.add_argument("--lot", choices=sorted(ARCHIVES), required=True)

    inspect_command = subparsers.add_parser("inspect")
    inspect_command.add_argument("--archive", type=Path, required=True)
    inspect_command.add_argument("--output", type=Path)

    registry = subparsers.add_parser("registry")
    registry.add_argument("--output", type=Path)

    curate = subparsers.add_parser("curate")
    curate.add_argument("--root", type=Path, required=True)
    curate.add_argument("--lot", choices=sorted(ARCHIVES), required=True)
    curate.add_argument("--archive", type=Path, required=True)
    curate.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    curate.add_argument("--delete-archive", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download":
        download_archive(args.root, ARCHIVES[args.lot])
        return
    if args.command == "inspect":
        report = inspect_archive(args.archive)
    elif args.command == "curate":
        spec = ARCHIVES[args.lot]
        report = curate_archive(
            args.root,
            args.archive,
            spec,
            expected_rows=EXPECTED_ROWS[args.lot],
            exclusion_manifests=args.exclude_manifest,
            delete_archive=args.delete_archive,
        )
    elif args.command == "finalize":
        report = finalize_corpus(args.root)
    else:
        report = {
            "selected_archives": [asdict(ARCHIVES[lot]) for lot in sorted(ARCHIVES)],
            "excluded_archives": {
                "FASDD.zip": "duplicates the selected domain archives",
                "FASDD_RS_RAW.zip": "raw remote-sensing source is not detector-ready RGB",
                "FASDD_RS_SWIR.zip": "different spectral domain from the RGB deployment input",
            },
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output := getattr(args, "output", None):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()

