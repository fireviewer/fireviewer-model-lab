"""Strict Roboflow Universe native-polygon ingestion for the pointing corpus.

The input is a pinned JSONL export of public Roboflow *Raw Data* pages.  This
program downloads only the declared original payloads inside the SageMaker
runtime, verifies their hashes and dimensions, rasterizes published polygons,
applies the three immutable exclusion indexes, groups events/near-duplicates,
and creates a fresh leak-free split.

It deliberately never consumes boxes, augmented views, pseudo-labels or human
review.  A polygon that does not prove one unambiguous, off-border physical
base becomes a segmentation abstention and never enters the point manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import fftpack

ALLOWED_LABELS = {"fire": "fire_base", "smoke": "smoke_column_base"}
ALLOWED_SPLITS = ("train", "validation", "test")
SPLIT_TARGETS = {"train": 0.70, "validation": 0.10, "test": 0.20}
ALLOWED_LICENSES = frozenset({"CC BY 4.0", "CC0 1.0", "Public Domain"})
MAX_MANIFEST_BYTES = 64 * 1024**2
MAX_IMAGE_BYTES = 64 * 1024**2
MAX_PIXELS = 100_000_000
MAX_PHASH_DISTANCE = 6
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PROVENANCE = re.compile(
    r"(?:pseudo|sam(?:$|[_ -])|segment.anything|teacher|synthetic|bbox|box.derived|human.review)",
    re.IGNORECASE,
)


class RecordRejected(ValueError):
    """A row-level, fail-closed source rejection."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_hash(image: Image.Image) -> str:
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
    )
    transformed = fftpack.dct(fftpack.dct(pixels, axis=0), axis=1)
    low = transformed[:8, :8]
    bits = (low > np.median(low)).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _nearest_phash(
    signature: str, rows: list[tuple[str, str]]
) -> tuple[int | None, str | None]:
    if not rows:
        return None, None
    return min(
        ((int(signature, 16) ^ int(value, 16)).bit_count(), sample_id)
        for value, sample_id in rows
    )


def _load_exclusion_index(
    root: Path,
    name: str,
    *,
    expected_receipt_sha256: str,
    expected_rows: int,
) -> dict[str, Any]:
    """Load an immutable hash-only index and reject incomplete contracts."""
    expected_files = {root / "hash-index.jsonl", root / "receipt.json"}
    files = (
        set(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else set()
    )
    index_path, receipt_path = root / "hash-index.jsonl", root / "receipt.json"
    errors: list[str] = []
    if files != expected_files:
        errors.append(f"{name}_exclusion_index_file_set_mismatch")
    if not HEX64.fullmatch(expected_receipt_sha256.casefold()):
        errors.append(f"{name}_expected_receipt_sha256_invalid")
    if expected_rows <= 0:
        errors.append(f"{name}_expected_rows_invalid")
    receipt: dict[str, Any] = {}
    receipt_sha: str | None = None
    if receipt_path.is_file():
        receipt_sha = _sha256_file(receipt_path)
        if receipt_sha != expected_receipt_sha256.casefold():
            errors.append(f"{name}_exclusion_receipt_sha256_mismatch")
        try:
            loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("receipt is not an object")
            receipt = loaded
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{name}_exclusion_receipt_invalid:{type(exc).__name__}")
    else:
        errors.append(f"{name}_exclusion_receipt_missing")
    if index_path.is_file() and receipt:
        checks = {
            "schema_version": receipt.get("schema_version") == 1,
            "partition": receipt.get("partition") == name,
            "index_filename": receipt.get("index_filename") == index_path.name,
            "index_rows": receipt.get("index_rows") == expected_rows,
            "index_bytes": receipt.get("index_bytes") == index_path.stat().st_size,
            "index_sha256": receipt.get("index_sha256") == _sha256_file(index_path),
            "incomplete_rows": receipt.get("incomplete_rows") == 0,
            "media_files_output": receipt.get("media_files_output") == 0,
            "source_gate_passed": receipt.get("source_gate_passed") is True,
            "gate_errors": receipt.get("gate_errors") == [],
            "publication_allowed": receipt.get("publication_allowed") is False,
        }
        errors.extend(
            f"{name}_exclusion_receipt_contract:{field}"
            for field, passed in checks.items()
            if not passed
        )
    sha_rows: dict[str, str] = {}
    phash_rows: list[tuple[str, str]] = []
    incomplete = 0
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    if index_path.is_file():
        for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{name}_exclusion_index_invalid_json:{line_number}")
                incomplete += 1
                continue
            if not isinstance(row, dict):
                incomplete += 1
                continue
            sample_id = str(row.get("sample_id") or "")
            digest = str(
                row.get("image_sha256") or row.get("source_image_sha256") or ""
            ).lower()
            signature = str(
                row.get("phash") or row.get("perceptual_hash") or ""
            ).lower()
            flipped = str(row.get("phash64_flipped") or "").lower()
            valid = (
                bool(sample_id)
                and sample_id not in seen_ids
                and digest not in seen_digests
                and HEX64.fullmatch(digest) is not None
                and re.fullmatch(r"[0-9a-f]{16}", signature) is not None
                and row.get("index_partition") in (None, name)
                and (
                    name != "benchmark"
                    or re.fullmatch(r"[0-9a-f]{16}", flipped) is not None
                )
            )
            if not valid:
                incomplete += 1
                continue
            seen_ids.add(sample_id)
            seen_digests.add(digest)
            sha_rows[digest] = sample_id
            phash_rows.append((signature, sample_id))
            if name == "benchmark":
                phash_rows.append((flipped, f"{sample_id}:horizontal-flip"))
    indexed_rows = len(seen_ids)
    if indexed_rows != expected_rows:
        errors.append(f"{name}_exclusion_index_rows:{indexed_rows}:{expected_rows}")
    if incomplete:
        errors.append(f"{name}_exclusion_index_incomplete_rows:{incomplete}")
    return {
        "sha": sha_rows,
        "phash": phash_rows,
        "indexed_rows": indexed_rows,
        "receipt_sha256": receipt_sha,
        "receipt": receipt,
        "gate_errors": sorted(set(errors)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _require_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "audit output must be empty; immutable receipts cannot be overwritten"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_manifest(path: Path, expected_sha256: str) -> tuple[list[dict[str, Any]], str]:
    if not HEX64.fullmatch(expected_sha256.casefold()):
        raise ValueError("expected manifest SHA-256 is invalid")
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("input manifest is missing or exceeds the bounded size")
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected_sha256.casefold():
        raise ValueError(f"input manifest SHA-256 mismatch: {actual}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"input manifest line {line_number} is not an object")
        value["_input_line"] = line_number
        rows.append(value)
    if not rows:
        raise ValueError("input manifest is empty")
    return rows, actual


def _https_url(value: Any, *, host: str, purpose: str) -> urllib.parse.SplitResult:
    if not isinstance(value, str):
        raise RecordRejected(f"{purpose}_missing")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold() != host
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise RecordRejected(f"{purpose}_not_pinned_https_{host}")
    return parsed


def _is_empty_preprocessing(value: Any) -> bool:
    return value in (None, "", "none", "original", [], {})


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordRejected(f"{field}_not_numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecordRejected(f"{field}_not_finite")
    return parsed


def _dimensions(source: dict[str, Any], annotation: dict[str, Any]) -> tuple[int, int]:
    values = []
    for record, prefix in ((source, "source"), (annotation, "annotation")):
        width = record.get("width")
        height = record.get("height")
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 1
            or height <= 1
            or width * height > MAX_PIXELS
        ):
            raise RecordRejected(f"{prefix}_dimensions_invalid")
        values.append((width, height))
    if values[0] != values[1]:
        raise RecordRejected("source_annotation_dimensions_disagree")
    return values[0]


def _point(value: Any, width: int, height: int) -> tuple[float, float]:
    if isinstance(value, dict):
        x, y = _number(value.get("x"), "polygon_x"), _number(value.get("y"), "polygon_y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = _number(value[0], "polygon_x"), _number(value[1], "polygon_y")
    else:
        raise RecordRejected("polygon_point_invalid")
    if not (0 <= x < width and 0 <= y < height):
        raise RecordRejected("polygon_point_out_of_bounds")
    return x, y


def _one_polygon(value: Any, width: int, height: int) -> list[tuple[float, float]]:
    if isinstance(value, dict):
        value = value.get("points")
    if not isinstance(value, list):
        raise RecordRejected("polygon_invalid")
    if value and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        if len(value) % 2:
            raise RecordRejected("polygon_flat_coordinate_count_odd")
        value = list(zip(value[::2], value[1::2], strict=True))
    points = [_point(item, width, height) for item in value]
    if len(points) < 3 or len(set(points)) < 3:
        raise RecordRejected("polygon_has_fewer_than_three_unique_points")
    area = abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )
    if area < 4.0:
        raise RecordRejected("polygon_area_too_small")
    return points


def _polygons(
    value: Any, width: int, height: int, *, expected_label: str
) -> list[list[tuple[float, float]]]:
    if not isinstance(value, list) or not value:
        raise RecordRejected("native_polygon_missing")
    first = value[0]
    coordinate_pair = (
        isinstance(first, (list, tuple))
        and len(first) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in first)
    )
    if (
        (isinstance(first, dict) and "x" in first)
        or isinstance(first, (int, float))
        or coordinate_pair
    ):
        candidates = [value]
    else:
        candidates = value
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("label") is not None:
            polygon_label = str(candidate["label"]).strip().casefold()
            if polygon_label != expected_label:
                raise RecordRejected("polygon_label_disagrees_with_annotation_label")
    parsed = [_one_polygon(candidate, width, height) for candidate in candidates]
    if not parsed:
        raise RecordRejected("native_polygon_missing")
    return parsed


def _validate_record(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source")
    annotation = row.get("annotation")
    if not isinstance(source, dict) or not isinstance(annotation, dict):
        raise RecordRejected("source_or_annotation_object_missing")
    source_id = str(source.get("id") or "").strip()
    owner = str(source.get("owner") or "").strip()
    name = str(source.get("name") or "").strip()
    if not source_id or not owner or not name:
        raise RecordRejected("source_identity_incomplete")
    if str(source.get("status") or "").strip().casefold() not in {"original", "raw-original"}:
        raise RecordRejected("source_view_not_original")
    if not _is_empty_preprocessing(source.get("preprocessing")):
        raise RecordRejected("source_preprocessing_forbidden")
    page = _https_url(row.get("page_url"), host="universe.roboflow.com", purpose="page_url")
    original = _https_url(
        row.get("original_url"), host="source.roboflow.com", purpose="original_url"
    )
    path_parts = [urllib.parse.unquote(part) for part in original.path.split("/") if part]
    if len(path_parts) != 3 or path_parts[:2] != [owner, source_id]:
        raise RecordRejected("original_url_identity_mismatch")
    if not path_parts[2].casefold().startswith("original.") or original.query or original.fragment:
        raise RecordRejected("original_url_is_not_raw_original")
    if source_id not in urllib.parse.unquote(page.path):
        raise RecordRejected("page_url_source_id_mismatch")
    provenance_text = json.dumps(annotation, ensure_ascii=False, sort_keys=True)
    if FORBIDDEN_PROVENANCE.search(provenance_text):
        raise RecordRejected("forbidden_annotation_provenance")
    label = str(annotation.get("label") or "").strip().casefold()
    if label not in ALLOWED_LABELS:
        raise RecordRejected("annotation_label_not_fire_or_smoke")
    width, height = _dimensions(source, annotation)
    polygons = _polygons(
        annotation.get("polygons"), width, height, expected_label=label
    )
    # Boxes may be present as Roboflow-computed display metadata.  They are
    # intentionally neither validated nor copied; polygons are the sole signal.
    return {
        "line": row["_input_line"],
        "page_url": row["page_url"],
        "original_url": row["original_url"],
        "source_id": source_id,
        "owner": owner,
        "name": name,
        "upstream_split": source.get("split"),
        "event_id": str(source.get("event_id") or source.get("sequence_id") or "").strip(),
        "label": label,
        "width": width,
        "height": height,
        "polygons": polygons,
        "boxes_ignored": bool(annotation.get("boxes")),
    }


def _download_original(url: str) -> bytes:
    pinned = _https_url(url, host="source.roboflow.com", purpose="original_url")
    request = urllib.request.Request(  # noqa: S310 - URL was restricted by _validate_record
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "FireViewer-Pointing/2.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - validated URL
        final = _https_url(
            response.geturl(),
            host="source.roboflow.com",
            purpose="redirected_original_url",
        )
        if (final.path, final.query, final.fragment) != (
            pinned.path,
            pinned.query,
            pinned.fragment,
        ):
            raise RecordRejected("original_payload_redirect_identity_mismatch")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_IMAGE_BYTES:
            raise RecordRejected("original_payload_too_large")
        payload = response.read(MAX_IMAGE_BYTES + 1)
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise RecordRejected("original_payload_empty_or_too_large")
    return payload


def _decode_image(payload: bytes, expected: tuple[int, int]) -> tuple[Image.Image, str]:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            image = opened.convert("RGB")
            source_format = str(opened.format or "").casefold()
    except Exception as exc:
        raise RecordRejected(f"original_payload_decode_failed:{type(exc).__name__}") from exc
    if image.size != expected:
        raise RecordRejected("decoded_dimensions_disagree")
    suffixes = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}
    if source_format not in suffixes:
        raise RecordRejected("original_payload_format_forbidden")
    return image, suffixes[source_format]


def _rasterize(polygons: list[list[tuple[float, float]]], size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return mask


def _derive_physical_base(
    polygons: list[list[tuple[float, float]]],
    *,
    label: str,
    width: int,
    height: int,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    """Accept only one compact lower neck, never an arbitrary mask bottom."""
    candidates: list[tuple[float, float, dict[str, float]]] = []
    failures: list[str] = []
    for polygon in polygons:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        box_width, box_height = right - left, bottom - top
        border = max(1.0, min(width, height) * 0.005)
        if left <= border or right >= width - 1 - border or bottom >= height - 1 - border:
            failures.append("polygon_touches_image_border")
            continue
        if box_width < 4 or box_height < 6:
            failures.append("polygon_extent_too_small")
            continue
        band_height = max(2.0, box_height * 0.08)
        bottom_points = [(x, y) for x, y in polygon if y >= bottom - band_height]
        if len(bottom_points) < 2:
            failures.append("lower_support_not_observed")
            continue
        support_width = max(x for x, _ in bottom_points) - min(x for x, _ in bottom_points)
        ratio = support_width / box_width
        maximum = 0.60 if label == "fire" else 0.35
        if ratio <= 0.02 or ratio > maximum:
            failures.append("lower_support_not_unique_compact_neck")
            continue
        base_x = sum(x for x, _ in bottom_points) / len(bottom_points)
        if not (left + box_width * 0.10 <= base_x <= right - box_width * 0.10):
            failures.append("lower_support_not_centrally_supported")
            continue
        candidates.append(
            (
                base_x,
                bottom,
                {
                    "polygon_bbox_width": round(box_width, 4),
                    "polygon_bbox_height": round(box_height, 4),
                    "lower_support_width_ratio": round(ratio, 6),
                },
            )
        )
    if len(candidates) != 1:
        failures.append("physical_base_candidate_count_not_one")
        return None, sorted(set(failures)), {"candidate_count": len(candidates)}
    x, y, diagnostics = candidates[0]
    return (
        {
            "kind": ALLOWED_LABELS[label],
            "x": round(x / (width - 1), 8),
            "y": round(y / (height - 1), 8),
            "origin": "published_native_polygon_unique_lower_neck_gate_v1",
        },
        [],
        {"candidate_count": 1, **diagnostics},
    )


def _external_matches(
    digest: str, signature: str, indexes: dict[str, dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    reasons: list[str] = []
    matches: list[dict[str, Any]] = []
    for name, index in indexes.items():
        if digest in index["sha"]:
            reasons.append(f"{name}_exact_sha_overlap")
            matches.append({"partition": name, "kind": "sha256", "sample_id": index["sha"][digest]})
        distance, sample_id = _nearest_phash(signature, index["phash"])
        if distance is not None and distance <= MAX_PHASH_DISTANCE:
            reasons.append(f"{name}_phash_overlap")
            matches.append(
                {"partition": name, "kind": "phash", "sample_id": sample_id, "distance": distance}
            )
    return sorted(set(reasons)), matches


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _group_and_deduplicate(rows: list[dict[str, Any]]) -> dict[str, int]:
    dsu = _DisjointSet(len(rows))
    by_event: dict[str, list[int]] = defaultdict(list)
    by_sha: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["event_id"]:
            by_event[f"{row['source_owner']}:{row['event_id']}"].append(index)
        by_sha[row["image_sha256"]].append(index)
    for members in [*by_event.values(), *by_sha.values()]:
        for other in members[1:]:
            dsu.union(members[0], other)
    for left in range(len(rows)):
        for right in range(left):
            if (
                int(rows[left]["phash"], 16) ^ int(rows[right]["phash"], 16)
            ).bit_count() <= MAX_PHASH_DISTANCE:
                dsu.union(left, right)
    exact_excluded = 0
    for members in by_sha.values():
        ordered = sorted(members, key=lambda index: rows[index]["sample_id"])
        for index in ordered[1:]:
            rows[index]["exclusion_reasons"].append("within_source_exact_sha_duplicate")
            rows[index]["within_source_duplicate_of"] = rows[ordered[0]]["sample_id"]
            exact_excluded += 1
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[dsu.find(index)].append(index)
    for members in groups.values():
        identity = "\n".join(sorted(rows[index]["sample_id"] for index in members)).encode()
        group = f"roboflow:event-phash:{_sha256_bytes(identity)[:20]}"
        for index in members:
            rows[index]["source_group"] = group
            rows[index]["split_group"] = group
    return {"exact_rows_excluded": exact_excluded, "event_phash_groups": len(groups)}


def _assign_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if not row["exclusion_reasons"]]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[row["split_group"]].append(row)
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode()).hexdigest()),
    )
    target_counts = {name: len(eligible) * ratio for name, ratio in SPLIT_TARGETS.items()}
    counts = Counter()
    assignments: dict[str, str] = {}
    for index, (group, members) in enumerate(ordered):
        remaining = len(ordered) - index
        empty = [name for name in ALLOWED_SPLITS if counts[name] == 0]
        choices = empty if remaining <= len(empty) else list(ALLOWED_SPLITS)
        split = min(
            choices,
            key=lambda name: (
                (counts[name] + len(members) - target_counts[name]) / max(1.0, target_counts[name]),
                counts[name] / max(1.0, target_counts[name]),
                ALLOWED_SPLITS.index(name),
            ),
        )
        assignments[group] = split
        counts[split] += len(members)
        for row in members:
            row["split"] = split
            row["final_split"] = split
    leaks = [
        group
        for group, members in groups.items()
        if len({row.get("split") for row in members}) != 1
    ]
    errors = []
    if eligible and len(groups) < 3:
        errors.append(f"independent_split_groups_insufficient:{len(groups)}:3")
    if eligible and any(counts[name] == 0 for name in ALLOWED_SPLITS):
        errors.append("independent_split_has_empty_partition")
    if leaks:
        errors.append(f"split_group_leakage:{len(leaks)}")
    return {
        "schema_version": 1,
        "strategy": "ignore_upstream_split_greedy_whole_event_phash_groups_v1",
        "target_ratios": SPLIT_TARGETS,
        "eligible_rows": len(eligible),
        "groups": len(groups),
        "counts": {name: counts[name] for name in ALLOWED_SPLITS},
        "assignments": dict(sorted(assignments.items())),
        "split_group_leaks": leaks,
        "gate_errors": errors,
        "passed": not errors,
    }


def audit_roboflow_manifest(
    *,
    input_manifest: Path,
    expected_manifest_sha256: str,
    detection_index_root: Path,
    pointing_index_root: Path,
    benchmark_index_root: Path,
    expected_index_receipts: dict[str, dict[str, Any]],
    output_dir: Path,
    source_id: str,
    source_family: str,
    source_revision: str,
    source_license: str,
    output_s3_prefix: str | None = None,
    fetcher: Callable[[str], bytes] = _download_original,
) -> dict[str, Any]:
    if source_license not in ALLOWED_LICENSES:
        raise ValueError(f"source licence is not allowlisted: {source_license}")
    if not all(value.strip() for value in (source_id, source_family, source_revision)):
        raise ValueError("immutable source identity is incomplete")
    if set(expected_index_receipts) != {"detection", "pointing", "benchmark"}:
        raise ValueError("exactly three pinned exclusion-index receipts are required")
    _require_empty_output(output_dir)
    input_rows, manifest_sha256 = _read_manifest(input_manifest, expected_manifest_sha256)
    indexes = {
        name: _load_exclusion_index(
            root,
            name,
            expected_receipt_sha256=str(expected_index_receipts[name]["receipt_sha256"]),
            expected_rows=int(expected_index_receipts[name]["rows"]),
        )
        for name, root in {
            "detection": detection_index_root,
            "pointing": pointing_index_root,
            "benchmark": benchmark_index_root,
        }.items()
    }
    gate_errors = sorted({error for index in indexes.values() for error in index["gate_errors"]})
    payload_root = output_dir / "strict-payload"
    image_root, mask_root = payload_root / "images", payload_root / "masks"
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    dispositions: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for raw in input_rows:
        base = {"schema_version": 1, "input_line": raw["_input_line"]}
        try:
            record = _validate_record(raw)
            if record["source_id"] in seen_source_ids:
                raise RecordRejected("source_id_duplicate_in_manifest")
            seen_source_ids.add(record["source_id"])
            payload = fetcher(record["original_url"])
            image, suffix = _decode_image(payload, (record["width"], record["height"]))
            digest = _sha256_bytes(payload)
            signature = _perceptual_hash(image)
            sample_id = f"{source_id}:{record['owner']}:{record['source_id']}"
            mask = _rasterize(record["polygons"], image.size)
            if mask.getbbox() is None:
                raise RecordRejected("rasterized_native_polygon_is_empty")
            mask_buffer = io.BytesIO()
            mask.save(mask_buffer, format="PNG", optimize=False)
            mask_payload = mask_buffer.getvalue()
            mask_digest = _sha256_bytes(mask_payload)
            image_rel = f"strict-payload/images/{digest}{suffix}"
            mask_rel = f"strict-payload/masks/{digest}-{mask_digest[:16]}.png"
            image_path, mask_path = output_dir / image_rel, output_dir / mask_rel
            if not image_path.exists():
                image_path.write_bytes(payload)
            mask_path.write_bytes(mask_payload)
            reasons, external_matches = _external_matches(digest, signature, indexes)
            point, point_errors, diagnostics = _derive_physical_base(
                record["polygons"],
                label=record["label"],
                width=record["width"],
                height=record["height"],
            )
            row = {
                "schema_version": 1,
                "sample_id": sample_id,
                "source_id": source_id,
                "source_family": source_family,
                "source_revision": source_revision,
                "source_record_id": record["source_id"],
                "source_owner": record["owner"],
                "source_name": record["name"],
                "source_page_url": record["page_url"],
                "source_original_url": record["original_url"],
                "upstream_split_ignored": record["upstream_split"],
                "event_id": record["event_id"],
                "validation_profile": "roboflow_native_polygon_strict_v1",
                "sample_validation_status": "strict_automated_validated",
                "annotation_provenance": "published_native_polygon",
                "annotation_strength": "published_native_polygon_strong",
                "mask_quality": "published_native_polygon_rasterization",
                "mask_semantics": record["label"],
                "media_license": source_license,
                "mask_license": source_license,
                "redistribution_allowed": True,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "source_image_sha256": digest,
                "image_sha256": digest,
                "provided_mask_sha256": mask_digest,
                "phash": signature,
                "width": record["width"],
                "height": record["height"],
                "label": record["label"],
                "presence_targets": {
                    "fire_visible": record["label"] == "fire",
                    "smoke_visible": record["label"] == "smoke",
                },
                "anchor_points": [point] if point else [],
                "point_supervised": point is not None,
                "mask_to_point_conversion": (
                    "published_native_polygon_unique_lower_neck_gate_v1" if point else "none"
                ),
                "point_gate_errors": point_errors,
                "point_geometry_diagnostics": diagnostics,
                "visual_abstention_reason": None if point else "physical_base_not_provable",
                "boxes_ignored_not_converted": record["boxes_ignored"],
                "detection_corpus_used": False,
                "independent_benchmark_used": False,
                "benchmark_hash_exclusion_only": True,
                "reviews_admitted": False,
                "human_reviewed": False,
                "pseudo_labels_admitted": False,
                "bbox_conversion_used": False,
                "exclusion_reasons": reasons,
                "external_exclusion_matches": external_matches,
                "training_eligible": not reasons,
                "strict_keep": False,
                "publication_allowed": False,
            }
            accepted.append(row)
            dispositions.append(row)
        except RecordRejected as exc:
            dispositions.append(
                {
                    **base,
                    "sample_id": None,
                    "training_eligible": False,
                    "strict_keep": False,
                    "point_supervised": False,
                    "exclusion_reasons": [str(exc)],
                    "publication_allowed": False,
                }
            )
    dedup_receipt = _group_and_deduplicate(accepted)
    split_receipt = _assign_splits(accepted)
    gate_errors.extend(split_receipt["gate_errors"])
    structurally_eligible = [row for row in accepted if not row["exclusion_reasons"]]
    point_candidates = [row for row in structurally_eligible if row["point_supervised"]]
    if not structurally_eligible:
        gate_errors.append("no_structurally_valid_native_polygon_rows")
    elif not point_candidates:
        gate_errors.append("no_strict_physical_base_points")
    source_gate_passed = not gate_errors
    for row in accepted:
        row["exclusion_reasons"] = sorted(set(row["exclusion_reasons"]))
        row["training_eligible"] = source_gate_passed and not row["exclusion_reasons"]
        row["strict_keep"] = row["training_eligible"] and row["point_supervised"]
        row["corpus_role"] = (
            "point_supervision"
            if row["strict_keep"]
            else "segmentation_presence_abstention_auxiliary"
            if row["training_eligible"]
            else "excluded"
        )
    strict_rows = sorted(
        (row for row in accepted if row["strict_keep"]), key=lambda row: row["sample_id"]
    )
    abstentions = sorted(
        (
            row
            for row in accepted
            if row["training_eligible"] and not row["point_supervised"]
        ),
        key=lambda row: row["sample_id"],
    )
    dispositions_path = output_dir / "roboflow_automatic_dispositions.jsonl"
    strict_path = output_dir / "roboflow_strict_validated_manifest.jsonl"
    abstention_path = output_dir / "roboflow_segmentation_abstentions.jsonl"
    split_path = output_dir / "roboflow_split_receipt.json"
    _write_jsonl(dispositions_path, dispositions)
    _write_jsonl(strict_path, strict_rows)
    _write_jsonl(abstention_path, abstentions)
    _write_json(split_path, split_receipt)
    artifacts = {}
    for path in (dispositions_path, strict_path, abstention_path, split_path):
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    summary = {
        "schema_version": 1,
        "source_id": source_id,
        "source_family": source_family,
        "source_revision": source_revision,
        "source_license": source_license,
        "input_manifest_sha256": manifest_sha256,
        "input_rows": len(input_rows),
        "decoded_native_polygon_rows": len(accepted),
        "rejected_before_payload_admission": len(input_rows) - len(accepted),
        "strict_point_rows": len(strict_rows),
        "segmentation_abstention_rows": len(abstentions),
        "rows_by_label": dict(sorted(Counter(row["label"] for row in accepted).items())),
        "points_by_kind": dict(
            sorted(
                Counter(
                    point["kind"] for row in strict_rows for point in row["anchor_points"]
                ).items()
            )
        ),
        "exclusions_by_reason": dict(
            sorted(
                Counter(
                    reason for row in dispositions for reason in row["exclusion_reasons"]
                ).items()
            )
        ),
        "deduplication": dedup_receipt,
        "split": split_receipt,
        "external_exclusion_indexes": {
            name: {
                "rows": index["indexed_rows"],
                "receipt_sha256": index["receipt_sha256"],
                "index_sha256": index["receipt"].get("index_sha256"),
            }
            for name, index in indexes.items()
        },
        "payload_download_location": "sagemaker_runtime_only",
        "boxes_consumed": 0,
        "pseudo_labels_admitted": 0,
        "human_reviews_admitted": 0,
        "source_gate_passed": source_gate_passed,
        "gate_errors": sorted(set(gate_errors)),
        "publication_allowed": False,
        "output_s3_prefix": output_s3_prefix,
        "artifacts": artifacts,
        "receipt": "ROBoflow_NATIVE_POLYGON_AUDIT_RECEIPT.json",
    }
    summary_path = output_dir / "roboflow_summary.json"
    _write_json(summary_path, summary)
    artifacts[summary_path.name] = {
        "bytes": summary_path.stat().st_size,
        "sha256": _sha256_file(summary_path),
    }
    receipt = {
        "schema_version": 1,
        "receipt_type": "fireviewer_roboflow_native_polygon_audit_v1",
        "source_contract": {
            "source_id": source_id,
            "source_family": source_family,
            "source_revision": source_revision,
            "source_license": source_license,
            "input_manifest_sha256": manifest_sha256,
        },
        "artifacts": dict(sorted(artifacts.items())),
        "source_gate_passed": source_gate_passed,
        "strict_point_rows": len(strict_rows),
        "publication_allowed": False,
        "policy": {
            "original_views_only": True,
            "native_polygons_only": True,
            "human_annotation_work": False,
            "pseudo_labels": False,
            "bbox_to_point_or_mask": False,
            "independent_benchmark_payload_access": False,
        },
    }
    receipt_path = output_dir / "ROBoflow_NATIVE_POLYGON_AUDIT_RECEIPT.json"
    _write_json(receipt_path, receipt)
    return {**summary, "receipt_sha256": _sha256_file(receipt_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--detection-index-root", type=Path, required=True)
    parser.add_argument("--pointing-index-root", type=Path, required=True)
    parser.add_argument("--benchmark-index-root", type=Path, required=True)
    for name in ("detection", "pointing", "benchmark"):
        parser.add_argument(f"--{name}-index-receipt-sha256", required=True)
        parser.add_argument(f"--{name}-index-rows", type=int, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-family", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-s3-prefix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = audit_roboflow_manifest(
        input_manifest=args.input_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        detection_index_root=args.detection_index_root,
        pointing_index_root=args.pointing_index_root,
        benchmark_index_root=args.benchmark_index_root,
        expected_index_receipts={
            name: {
                "receipt_sha256": getattr(args, f"{name}_index_receipt_sha256"),
                "rows": getattr(args, f"{name}_index_rows"),
            }
            for name in ("detection", "pointing", "benchmark")
        },
        output_dir=args.output_dir,
        source_id=args.source_id,
        source_family=args.source_family,
        source_revision=args.source_revision,
        source_license=args.source_license,
        output_s3_prefix=args.output_s3_prefix,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
