"""Assemble immutable strict pointing sources behind fail-closed publication gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageStat

POINT_KINDS = frozenset({"fire_base", "smoke_column_base"})
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
ALLOWED_VIEWPOINTS = frozenset({"ground", "ground_or_oblique", "oblique_ground"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DHASH_PATTERN = re.compile(r"^[0-9a-f]{16}$")
STRICT_MANIFEST_SUFFIX = "_strict_validated_manifest.jsonl"
HARD_FORBIDDEN_REFERENCES = frozenset(
    {
        "benchmark",
        "fire-smoke-detection-corpus-v1",
        "detection",
        "benchdata",
        "fireviewer_bench",
        "independent-benchmark",
        "benchmark-independent",
        "firebench",
    }
)
FORBIDDEN_SEMANTIC_MARKERS = (
    "weak",
    "teacher",
    "box_bottom",
    "bounding_box",
    "box-derived",
    "box_derived",
    "box-to-point",
    "box_to_point",
    "bbox",
    "yolo_box",
    "top_down",
    "top-down",
    "top down",
    "hotspot",
)
REFERENCE_FIELDS = (
    "source_id",
    "source_family",
    "source_repository",
    "dataset_family",
    "corpus_id",
    "image_relpath",
    "mask_relpath",
    "source_image_relpath",
    "image_s3_uri",
    "mask_s3_uri",
    "source_image_s3_uri",
    "purpose",
    "role",
    "benchmark_role",
)
QUALITY_GATE_OPERATORS = {
    "unique_source_images_min": ">=",
    "strict_automated_validated_point_images_min": ">=",
    "fire_base_points_min": ">=",
    "smoke_column_base_points_min": ">=",
    "explicit_negative_images_min": ">=",
    "source_families_min": ">=",
    "largest_source_share_max": "<=",
    "top_three_source_share_max": "<=",
    "low_light_positive_images_min": ">=",
    "small_or_faint_positive_images_min": ">=",
    "validation_images_min": ">=",
    "test_images_min": ">=",
    "materialized_augmentation_fraction_max": "<=",
    "unknown_semantics_max": "<=",
    "invalid_geometry_max": "<=",
    "exact_cross_split_duplicates_max": "<=",
    "split_group_leaks_max": "<=",
    "unknown_or_incompatible_rights_max": "<=",
}
REQUIRED_CONTRACT_FIELDS = frozenset(
    {
        "manifest_name",
        "manifest_sha256",
        "source_id",
        "source_family",
        "source_revision",
        "validation_profile",
        "viewpoint",
        "allowed_licenses",
        "required_license_fields",
        "allowed_annotation_provenances",
        "allowed_positive_annotation_strengths",
        "allowed_negative_annotation_strengths",
        "allowed_mask_to_point_conversions",
        "allowed_point_origins",
        "canonical_image_hash_field",
        "artifacts",
        "redistribution_allowed",
        "reviews_admitted",
        "quarantines_admitted",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_sha256(contract: dict[str, Any]) -> str:
    public_contract = {key: value for key, value in contract.items() if not key.startswith("_")}
    payload = json.dumps(
        public_contract, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"strict manifest is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return _sha256(path)


def _difference_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.tobytes())
    bits = 0
    for y_value in range(8):
        offset = y_value * 9
        for x_value in range(8):
            bits = (bits << 1) | int(pixels[offset + x_value] > pixels[offset + x_value + 1])
    return f"{bits:016x}"


def _string_set(value: Any, *, field: str, allow_empty: bool = False) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"source contract {field} must be a list of non-empty strings")
    result = frozenset(value)
    if not allow_empty and not result:
        raise ValueError(f"source contract {field} must not be empty")
    if len(result) != len(value):
        raise ValueError(f"source contract {field} contains duplicates")
    return result


def _contains_forbidden_semantics(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in FORBIDDEN_SEMANTIC_MARKERS)


def _safe_relative_path(value: Any, *, field: str) -> PurePosixPath:
    raw = str(value or "")
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or path.is_absolute()
        or ".." in path.parts
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"unsafe or missing relative path in {field}: {raw!r}")
    return path


def _validate_artifact_specs(contract: dict[str, Any]) -> list[dict[str, str]]:
    raw_specs = contract["artifacts"]
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("source contract artifacts must be a non-empty list")
    specs: list[dict[str, str]] = []
    roles: set[str] = set()
    path_fields: set[str] = set()
    sha_fields: set[str] = set()
    for raw in raw_specs:
        if not isinstance(raw, dict):
            raise ValueError("source contract artifact entry must be an object")
        spec = {key: str(raw.get(key) or "") for key in ("role", "path_field", "sha256_field")}
        if any(not value for value in spec.values()):
            raise ValueError("source contract artifact entry is incomplete")
        if spec["role"] in roles or spec["path_field"] in path_fields:
            raise ValueError("source contract artifact roles and path fields must be unique")
        roles.add(spec["role"])
        path_fields.add(spec["path_field"])
        sha_fields.add(spec["sha256_field"])
        specs.append(spec)
    if not {"image", "mask"}.issubset(roles):
        raise ValueError("source contract must verify at least image and mask artifacts")
    if str(contract["canonical_image_hash_field"]) not in sha_fields:
        raise ValueError("canonical_image_hash_field is not verified by an artifact contract")
    return specs


def _validate_contract(
    raw: Any,
    *,
    campaign_profile: str,
    forbidden_references: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("strict source contract must be an object")
    missing = REQUIRED_CONTRACT_FIELDS - set(raw)
    if missing:
        raise ValueError(f"strict source contract missing fields: {sorted(missing)}")
    contract = dict(raw)
    manifest_name = str(contract["manifest_name"])
    if (
        "/" in manifest_name
        or "\\" in manifest_name
        or Path(manifest_name).name != manifest_name
        or not manifest_name.endswith(STRICT_MANIFEST_SUFFIX)
    ):
        raise ValueError(f"invalid strict manifest contract name: {manifest_name!r}")
    manifest_sha256 = str(contract["manifest_sha256"]).lower()
    if not SHA256_PATTERN.fullmatch(manifest_sha256):
        raise ValueError(f"invalid manifest sha256 for {manifest_name}")
    contract["manifest_sha256"] = manifest_sha256
    for field in (
        "source_id",
        "source_family",
        "source_revision",
        "validation_profile",
        "viewpoint",
        "canonical_image_hash_field",
    ):
        if not isinstance(contract[field], str) or not contract[field]:
            raise ValueError(f"source contract {manifest_name} has invalid {field}")
    if contract["validation_profile"] != campaign_profile:
        raise ValueError(f"source contract profile mismatch for {manifest_name}")
    if contract["viewpoint"] not in ALLOWED_VIEWPOINTS:
        raise ValueError(f"source contract has non-ground viewpoint: {manifest_name}")
    for field, expected in (
        ("redistribution_allowed", True),
        ("reviews_admitted", False),
        ("quarantines_admitted", False),
    ):
        if contract[field] is not expected:
            raise ValueError(f"source contract {manifest_name} violates {field}")

    allowed_licenses = _string_set(contract["allowed_licenses"], field="allowed_licenses")
    required_license_fields = _string_set(
        contract["required_license_fields"], field="required_license_fields"
    )
    allowed_provenances = _string_set(
        contract["allowed_annotation_provenances"],
        field="allowed_annotation_provenances",
    )
    positive_strengths = _string_set(
        contract["allowed_positive_annotation_strengths"],
        field="allowed_positive_annotation_strengths",
        allow_empty=True,
    )
    negative_strengths = _string_set(
        contract["allowed_negative_annotation_strengths"],
        field="allowed_negative_annotation_strengths",
        allow_empty=True,
    )
    allowed_conversions = _string_set(
        contract["allowed_mask_to_point_conversions"],
        field="allowed_mask_to_point_conversions",
    )
    allowed_origins = _string_set(
        contract["allowed_point_origins"], field="allowed_point_origins", allow_empty=True
    )
    if not positive_strengths and not negative_strengths:
        raise ValueError(f"source contract {manifest_name} admits no annotation strength")
    if positive_strengths & negative_strengths:
        raise ValueError(f"source contract {manifest_name} has ambiguous annotation strengths")
    if negative_strengths - {"negative"}:
        raise ValueError(f"source contract {manifest_name} admits non-explicit negatives")
    semantic_contract_values = (
        allowed_provenances
        | positive_strengths
        | negative_strengths
        | allowed_conversions
        | allowed_origins
    )
    if any(_contains_forbidden_semantics(value) for value in semantic_contract_values):
        raise ValueError(
            f"source contract {manifest_name} admits forbidden weak/box/top-down semantics"
        )
    serialized_identity = " ".join(
        str(contract[field]).lower() for field in ("source_id", "source_family")
    )
    if any(marker in serialized_identity for marker in forbidden_references):
        raise ValueError(f"source contract {manifest_name} references a forbidden corpus")
    contract["_allowed_licenses"] = allowed_licenses
    contract["_required_license_fields"] = required_license_fields
    contract["_allowed_provenances"] = allowed_provenances
    contract["_positive_strengths"] = positive_strengths
    contract["_negative_strengths"] = negative_strengths
    contract["_allowed_conversions"] = allowed_conversions
    contract["_allowed_origins"] = allowed_origins
    contract["_artifact_specs"] = _validate_artifact_specs(contract)
    payload_root = _safe_relative_path(contract.get("payload_root", "."), field="payload_root")
    contract["_payload_root"] = payload_root
    return contract


def _load_contracts(registry: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], frozenset[str]]:
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported pointing corpus registry schema")
    automation = registry.get("strict_automation")
    if not isinstance(automation, dict):
        raise ValueError("pointing registry has no strict_automation object")
    for field in (
        "reviews_admitted",
        "quarantines_admitted",
        "unknown_rights_admitted",
        "missing_point_annotations_admitted",
        "cross_split_overlaps_admitted",
        "ambiguous_semantics_admitted",
    ):
        if automation.get(field) is not False:
            raise ValueError(f"pointing registry must set {field}=false")
    profile = automation.get("validation_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("pointing registry has no validation profile")

    boundary = registry.get("corpus_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("pointing registry has no corpus boundary")
    if boundary.get("box_to_point_policy") != "never_promote_box_bottom_center_to_ground_truth":
        raise ValueError("pointing registry does not forbid box-to-point promotion")
    forbidden = set(HARD_FORBIDDEN_REFERENCES)
    raw_forbidden = boundary.get("forbidden_references", [])
    if not isinstance(raw_forbidden, list):
        raise ValueError("corpus forbidden_references must be a list")
    forbidden.update(str(value).lower() for value in raw_forbidden if str(value))
    raw_families = boundary.get("forbidden_source_families", [])
    if not isinstance(raw_families, list):
        raise ValueError("corpus forbidden_source_families must be a list")
    forbidden.update(str(value).lower() for value in raw_families if str(value))
    forbidden_references = frozenset(forbidden)

    raw_contracts = automation.get("strict_source_contracts")
    if not isinstance(raw_contracts, list) or not raw_contracts:
        raise ValueError("pointing registry requires strict_source_contracts")
    contracts: dict[str, dict[str, Any]] = {}
    for raw_contract in raw_contracts:
        contract = _validate_contract(
            raw_contract,
            campaign_profile=profile,
            forbidden_references=forbidden_references,
        )
        name = str(contract["manifest_name"])
        if name in contracts:
            raise ValueError(f"duplicate strict source contract: {name}")
        contracts[name] = contract
    required_names = automation.get("required_strict_manifest_names")
    if not isinstance(required_names, list) or any(
        not isinstance(name, str) for name in required_names
    ):
        raise ValueError("registry required_strict_manifest_names must be a string list")
    if len(required_names) != len(set(required_names)):
        raise ValueError("registry contains duplicate required strict manifest names")
    if set(required_names) != set(contracts):
        raise ValueError("required strict manifests and strict source contracts differ")
    return contracts, forbidden_references


def _find_manifests(input_root: Path, contracts: dict[str, dict[str, Any]]) -> dict[str, Path]:
    discovered: dict[str, list[Path]] = defaultdict(list)
    for path in input_root.rglob(f"*{STRICT_MANIFEST_SUFFIX}"):
        discovered[path.name].append(path)
    unexpected = sorted(set(discovered) - set(contracts))
    if unexpected:
        raise ValueError(f"uncontracted strict source manifests found: {unexpected}")
    missing = sorted(set(contracts) - set(discovered))
    if missing:
        raise FileNotFoundError(f"strict source manifests missing: {missing}")
    duplicates = sorted(name for name, paths in discovered.items() if len(paths) != 1)
    if duplicates:
        raise ValueError(f"strict source manifest basename is ambiguous: {duplicates}")
    return {name: paths[0] for name, paths in discovered.items()}


def _source_mount_root(input_root: Path, manifest_path: Path) -> Path:
    relative = manifest_path.relative_to(input_root)
    return input_root if len(relative.parts) == 1 else input_root / relative.parts[0]


def _resolve_payload(
    *,
    input_root: Path,
    manifest_path: Path,
    contract: dict[str, Any],
    row: dict[str, Any],
    path_field: str,
) -> Path:
    relative = _safe_relative_path(row.get(path_field), field=path_field)
    mount_root = _source_mount_root(input_root, manifest_path).resolve()
    payload_root = (mount_root / Path(*contract["_payload_root"].parts)).resolve()
    if payload_root != mount_root and mount_root not in payload_root.parents:
        raise ValueError(f"payload_root escapes source mount for {manifest_path.name}")
    payload = (payload_root / Path(*relative.parts)).resolve()
    if payload != payload_root and payload_root not in payload.parents:
        raise ValueError(f"payload path escapes source mount: {relative}")
    if not payload.is_file():
        raise ValueError(f"mounted payload is missing: {relative}")
    return payload


def _primary_provenance(row: dict[str, Any]) -> str:
    for field in ("annotation_provenance", "provided_mask_role", "mask_quality"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _row_reference_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(field) or "").lower() for field in REFERENCE_FIELDS)


def _validate_row(
    row: dict[str, Any],
    *,
    line_number: int,
    input_root: Path,
    manifest_path: Path,
    contract: dict[str, Any],
    forbidden_references: frozenset[str],
) -> dict[str, Any]:
    context = f"{manifest_path.name}:{line_number}"

    def fail(message: str) -> None:
        raise ValueError(f"{context}: {message}")

    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        fail("sample_id is missing")
    if row.get("schema_version") != 1:
        fail("unsupported row schema_version")
    for field in ("source_id", "source_family", "source_revision", "validation_profile"):
        if row.get(field) != contract[field]:
            fail(f"{field} does not match its immutable source contract")
    if row.get("sample_validation_status") != "strict_automated_validated":
        fail("sample_validation_status is not strict_automated_validated")
    if row.get("training_eligible") is not True or row.get("strict_keep") is not True:
        fail("strict row is not training eligible")
    if row.get("reviews_admitted", contract["reviews_admitted"]) is not False:
        fail("review-derived row is forbidden")
    if row.get("redistribution_allowed", contract["redistribution_allowed"]) is not True:
        fail("redistribution is not explicitly allowed")
    exclusion_reasons = row.get("exclusion_reasons")
    if not isinstance(exclusion_reasons, list) or exclusion_reasons:
        fail("strict row must have an explicit empty exclusion_reasons list")
    for field in ("corpus_disposition", "admission_status", "review_status"):
        value = str(row.get(field) or "").lower()
        if any(marker in value for marker in ("quarantine", "pending", "excluded", "review")):
            fail(f"{field} is not eligible for strict training")
    if row.get("quarantine") is True or row.get("human_reviewed") is True or row.get("review_id"):
        fail("quarantine or human-review metadata is forbidden")

    split = row.get("split")
    if split not in ALLOWED_SPLITS:
        fail(f"unsupported split: {split!r}")
    if row.get("final_split", split) != split:
        fail("final_split disagrees with split")
    split_group = row.get("split_group")
    if not isinstance(split_group, str) or not split_group.strip():
        fail("split_group is missing")

    reference_text = _row_reference_text(row)
    if any(marker in reference_text for marker in forbidden_references):
        fail("row references a detection or independent-benchmark corpus")
    if row.get("detection_corpus_used") is True or row.get("independent_benchmark_used") is True:
        fail("row declares forbidden corpus use")
    if row.get("ground_view") is False:
        fail("non-ground view is forbidden")
    for field in ("objects", "boxes", "bbox", "bboxes"):
        if row.get(field):
            fail(f"box annotation field {field} is forbidden")

    allowed_licenses: frozenset[str] = contract["_allowed_licenses"]
    for field in contract["_required_license_fields"]:
        value = row.get(field)
        if not isinstance(value, str) or value not in allowed_licenses:
            fail(f"missing or incompatible required licence field: {field}")
    for field in ("license", "media_license", "mask_license"):
        value = row.get(field)
        if value is not None and value not in allowed_licenses:
            fail(f"incompatible licence value in {field}")

    provenance = _primary_provenance(row)
    if provenance not in contract["_allowed_provenances"]:
        fail(f"annotation provenance is missing or forbidden: {provenance!r}")
    strength = row.get("annotation_strength")
    positive_strengths: frozenset[str] = contract["_positive_strengths"]
    negative_strengths: frozenset[str] = contract["_negative_strengths"]
    if strength not in positive_strengths | negative_strengths:
        fail(f"annotation strength is not contracted: {strength!r}")
    semantic_values = [
        str(strength),
        provenance,
        str(row.get("mask_quality") or ""),
        str(row.get("mask_semantics") or ""),
        str(row.get("viewpoint") or ""),
        str(row.get("view_geometry") or ""),
    ]
    if any(_contains_forbidden_semantics(value) for value in semantic_values):
        fail("weak, teacher, box-derived, or top-down semantics are forbidden")

    specs: list[dict[str, str]] = contract["_artifact_specs"]
    artifacts: dict[str, Path] = {}
    artifact_digests: dict[str, str] = {}
    for spec in specs:
        expected = str(row.get(spec["sha256_field"]) or "").lower()
        if not SHA256_PATTERN.fullmatch(expected):
            fail(f"invalid or missing {spec['sha256_field']}")
        try:
            payload = _resolve_payload(
                input_root=input_root,
                manifest_path=manifest_path,
                contract=contract,
                row=row,
                path_field=spec["path_field"],
            )
        except ValueError as exc:
            fail(str(exc))
        actual = _sha256(payload)
        if actual != expected:
            fail(f"payload checksum mismatch for {spec['path_field']}")
        artifacts[spec["role"]] = payload
        artifact_digests[spec["sha256_field"]] = actual
    canonical_field = str(contract["canonical_image_hash_field"])
    canonical_sha = str(row.get(canonical_field) or "").lower()
    if canonical_sha != artifact_digests.get(canonical_field):
        fail("canonical source image hash is not payload-verified")

    try:
        with Image.open(artifacts["image"]) as opened:
            opened.load()
            image = opened.convert("RGB")
        with Image.open(artifacts["mask"]) as opened_mask:
            opened_mask.load()
            mask = opened_mask.convert("L")
    except Exception as exc:
        fail(f"payload decode failed: {type(exc).__name__}: {exc}")
    width = row.get("width")
    height = row.get("height")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or (width, height) != image.size
    ):
        fail("declared image dimensions do not match decoded payload")
    if mask.size != image.size:
        fail("image and mask dimensions differ")
    dhash = str(row.get("dhash") or "").lower()
    if not DHASH_PATTERN.fullmatch(dhash) or dhash != _difference_hash(image):
        fail("dhash is missing, invalid, or does not match the image payload")

    points = row.get("anchor_points")
    if not isinstance(points, list):
        fail("anchor_points must be a list")
    point_kinds: set[str] = set()
    for point in points:
        if not isinstance(point, dict):
            fail("anchor point is not an object")
        kind = point.get("kind")
        origin = point.get("origin")
        if kind not in POINT_KINDS:
            fail(f"unknown point semantics: {kind!r}")
        if kind in point_kinds:
            fail(f"duplicate point kind in one row: {kind}")
        point_kinds.add(str(kind))
        if origin not in contract["_allowed_origins"]:
            fail(f"point origin is missing or forbidden: {origin!r}")
        if _contains_forbidden_semantics(str(origin)):
            fail("point origin is weak, box-derived, or top-down")
        raw_x = point.get("x")
        raw_y = point.get("y")
        if isinstance(raw_x, bool) or isinstance(raw_y, bool):
            fail("boolean point coordinates are forbidden")
        try:
            x = float(raw_x)
            y = float(raw_y)
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid point coordinates: {exc}")
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
        ):
            fail("point coordinates are outside normalized image bounds")

    if "visual_abstention_reason" not in row or row["visual_abstention_reason"] is not None:
        fail("abstentions cannot enter the strict pointing corpus")
    conversion = str(row.get("mask_to_point_conversion") or "none")
    if conversion not in contract["_allowed_conversions"]:
        fail(f"mask-to-point conversion is missing or forbidden: {conversion!r}")
    if _contains_forbidden_semantics(conversion):
        fail("box-derived or top-down point conversion is forbidden")

    mask_histogram = mask.histogram()
    mask_pixels = mask.width * mask.height
    foreground_fraction = float(sum(mask_histogram[1:]) / mask_pixels)
    is_negative = strength in negative_strengths
    explicit_negative_evidence = (
        row.get("negative") is True
        or bool(row.get("negative_evidence"))
        or row.get("source_annotations_exactly_empty") is True
        or row.get("provided_mask_signal") is False
    )
    if is_negative:
        if points:
            fail("negative row also contains positive anchor points")
        if foreground_fraction != 0.0:
            fail("negative row has a non-empty verified mask")
        if not explicit_negative_evidence:
            fail("negative row lacks explicit source evidence")
    else:
        if not points:
            fail("positive row has no anchor point")
        if foreground_fraction == 0.0:
            fail("positive row has an empty verified mask")
        if row.get("negative") is True:
            fail("positive row is also marked negative")
        if conversion == "none":
            fail("positive row has no contracted mask-to-point conversion")

    try:
        sample_weight = float(row.get("sample_weight", 1.0))
    except (TypeError, ValueError) as exc:
        fail(f"sample_weight is invalid: {exc}")
    if not math.isfinite(sample_weight) or sample_weight <= 0.0:
        fail("sample_weight must be finite and positive")
    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    brightness = float(ImageStat.Stat(gray).mean[0] / 255.0)
    return {
        "sample_id": sample_id,
        "source_id": contract["source_id"],
        "source_family": contract["source_family"],
        "canonical_sha256": canonical_sha,
        "dhash": dhash,
        "split": split,
        "split_group": split_group,
        "point_kinds": point_kinds,
        "is_negative": is_negative,
        "brightness": brightness,
        "foreground_fraction": foreground_fraction,
        "artifacts_verified": len(specs),
    }


def _first_near_duplicate_pair(
    metadata: list[dict[str, Any]], *, radius: int = 4
) -> dict[str, Any] | None:
    """Find one <= radius 64-bit dHash pair using pigeonhole partitions."""

    part_count = radius + 1
    base_width, wide_parts = divmod(64, part_count)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    values = [int(str(item["dhash"]), 16) for item in metadata]
    for index, value in enumerate(values):
        keys: list[tuple[int, int]] = []
        offset = 0
        for part in range(part_count):
            width = base_width + int(part < wide_parts)
            keys.append((part, (value >> offset) & ((1 << width) - 1)))
            offset += width
        candidates: set[int] = set()
        for key in keys:
            candidates.update(buckets[key])
        for other_index in sorted(candidates):
            distance = (value ^ values[other_index]).bit_count()
            if distance <= radius:
                return {
                    "left": metadata[other_index]["sample_id"],
                    "left_split": metadata[other_index]["split"],
                    "right": metadata[index]["sample_id"],
                    "right_split": metadata[index]["split"],
                    "dhash_distance": distance,
                }
        for key in keys:
            buckets[key].append(index)
    return None


def _validate_quality_policy(registry: dict[str, Any]) -> dict[str, int | float]:
    quality = registry.get("quality_gates")
    if not isinstance(quality, dict):
        raise ValueError("pointing registry has no quality_gates object")
    missing = set(QUALITY_GATE_OPERATORS) - set(quality)
    if missing:
        raise ValueError(f"pointing registry is missing quality gates: {sorted(missing)}")
    validated: dict[str, int | float] = {}
    for name in QUALITY_GATE_OPERATORS:
        value = quality[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"quality gate {name} is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"quality gate {name} is invalid")
        if ("fraction" in name or "share" in name) and numeric > 1.0:
            raise ValueError(f"quality gate {name} must be between zero and one")
        validated[name] = value
    return validated


def _gate(*, name: str, actual: int | float, target: int | float) -> dict[str, Any]:
    operator = QUALITY_GATE_OPERATORS[name]
    passed = actual >= target if operator == ">=" else actual <= target
    return {
        "name": name,
        "actual": actual,
        "target": target,
        "operator": operator,
        "passed": passed,
    }


def assemble(*, input_root: Path, registry_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "candidate_combined_manifest.jsonl",
        "publish_ready_manifest.jsonl",
        "strict_combined_manifest.jsonl",
        "PUBLISH_READY_RECEIPT.json",
        "SATISFACTION_STATUS.json",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("pointing registry root must be an object")
    contracts, forbidden_references = _load_contracts(registry)
    quality = _validate_quality_policy(registry)
    manifest_paths = _find_manifests(input_root, contracts)

    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for manifest_name in sorted(contracts):
        contract = contracts[manifest_name]
        path = manifest_paths[manifest_name]
        actual_manifest_sha = _sha256(path)
        if actual_manifest_sha != contract["manifest_sha256"]:
            raise ValueError(f"strict manifest sha256 mismatch: {manifest_name}")
        source_rows = _read_jsonl(path)
        source_metadata: list[dict[str, Any]] = []
        for line_number, row in enumerate(source_rows, 1):
            source_metadata.append(
                _validate_row(
                    row,
                    line_number=line_number,
                    input_root=input_root,
                    manifest_path=path,
                    contract=contract,
                    forbidden_references=forbidden_references,
                )
            )
            rows.append(dict(row))
        metadata.extend(source_metadata)
        source_receipts.append(
            {
                "manifest_name": manifest_name,
                "manifest_sha256": actual_manifest_sha,
                "contract_sha256": _contract_sha256(contract),
                "rows": len(source_rows),
                "source_id": contract["source_id"],
                "source_family": contract["source_family"],
                "source_revision": contract["source_revision"],
                "validation_profile": contract["validation_profile"],
                "artifacts_verified": sum(item["artifacts_verified"] for item in source_metadata),
            }
        )

    sample_counts = Counter(str(item["sample_id"]) for item in metadata)
    duplicate_sample_ids = sorted(key for key, count in sample_counts.items() if count > 1)
    if duplicate_sample_ids:
        raise ValueError(f"duplicate strict sample ids: {duplicate_sample_ids[:10]}")
    sha_counts = Counter(str(item["canonical_sha256"]) for item in metadata)
    duplicate_source_hashes = sorted(key for key, count in sha_counts.items() if count > 1)
    if duplicate_source_hashes:
        raise ValueError(f"duplicate canonical source images: {duplicate_source_hashes[:10]}")
    near_pair = _first_near_duplicate_pair(metadata)
    if near_pair:
        raise ValueError(
            "strict sources contain global perceptual duplicates, including within-split pairs: "
            f"{near_pair}"
        )

    group_splits: dict[str, set[str]] = defaultdict(set)
    for item in metadata:
        group_splits[str(item["split_group"])].add(str(item["split"]))
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking_groups:
        raise ValueError(f"strict sources contain split-group leakage: {leaking_groups[:10]}")

    source_counts = Counter(str(item["source_id"]) for item in metadata)
    source_family_counts = Counter(str(item["source_family"]) for item in metadata)
    split_counts = Counter(str(item["split"]) for item in metadata)
    total = len(rows)
    family_shares = sorted((count / total for count in source_family_counts.values()), reverse=True)
    largest_source_share = family_shares[0] if family_shares else 0.0
    top_three_source_share = sum(family_shares[:3])
    positive_metadata = [item for item in metadata if not item["is_negative"]]
    negative_rows = sum(bool(item["is_negative"]) for item in metadata)
    fire_points = sum("fire_base" in item["point_kinds"] for item in metadata)
    smoke_points = sum("smoke_column_base" in item["point_kinds"] for item in metadata)
    low_light_positive = sum(item["brightness"] <= 0.25 for item in positive_metadata)
    small_or_faint_positive = sum(item["foreground_fraction"] <= 0.02 for item in positive_metadata)
    augmented = sum(str(row.get("variant") or "clean") != "clean" for row in rows)
    augmentation_fraction = augmented / total if total else 0.0
    actuals: dict[str, int | float] = {
        "unique_source_images_min": len(sha_counts),
        "strict_automated_validated_point_images_min": total,
        "fire_base_points_min": fire_points,
        "smoke_column_base_points_min": smoke_points,
        "explicit_negative_images_min": negative_rows,
        "source_families_min": len(source_family_counts),
        "largest_source_share_max": largest_source_share,
        "top_three_source_share_max": top_three_source_share,
        "low_light_positive_images_min": low_light_positive,
        "small_or_faint_positive_images_min": small_or_faint_positive,
        "validation_images_min": split_counts["validation"],
        "test_images_min": split_counts["test"],
        "materialized_augmentation_fraction_max": augmentation_fraction,
        "unknown_semantics_max": 0,
        "invalid_geometry_max": 0,
        "exact_cross_split_duplicates_max": 0,
        "split_group_leaks_max": 0,
        "unknown_or_incompatible_rights_max": 0,
    }
    gates = [
        _gate(name=name, actual=actuals[name], target=quality[name])
        for name in QUALITY_GATE_OPERATORS
    ]
    publication_allowed = all(gate["passed"] for gate in gates)

    paired = sorted(zip(rows, metadata, strict=True), key=lambda pair: str(pair[0]["sample_id"]))
    rows = [row for row, _item in paired]
    candidate_path = output_dir / "candidate_combined_manifest.jsonl"
    candidate_sha = _write_jsonl(candidate_path, rows)
    publish_path = output_dir / "publish_ready_manifest.jsonl"
    legacy_strict_path = output_dir / "strict_combined_manifest.jsonl"
    receipt_path = output_dir / "PUBLISH_READY_RECEIPT.json"
    publish_sha: str | None = None
    if publication_allowed:
        publish_sha = _write_jsonl(publish_path, rows)
        legacy_sha = _write_jsonl(legacy_strict_path, rows)
        if publish_sha != candidate_sha or legacy_sha != publish_sha:
            raise RuntimeError("candidate and publish-ready manifests are not byte-identical")
        receipt = {
            "schema_version": 1,
            "campaign_id": registry["campaign_id"],
            "validation_profile": registry["strict_automation"]["validation_profile"],
            "publication_allowed": True,
            "manifest_name": publish_path.name,
            "manifest_sha256": publish_sha,
            "rows": total,
            "source_manifest_receipts": source_receipts,
        }
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        for stale in (publish_path, legacy_strict_path, receipt_path):
            stale.unlink(missing_ok=True)

    report = {
        "schema_version": 2,
        "campaign_id": registry["campaign_id"],
        "validation_profile": registry["strict_automation"]["validation_profile"],
        "source_receipts": source_receipts,
        "strict_rows": total,
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "fire_base_points": fire_points,
        "smoke_column_base_points": smoke_points,
        "explicit_negative_images": negative_rows,
        "payload_artifacts_verified": sum(int(item["artifacts_verified"]) for item in metadata),
        "global_near_duplicate_pairs": [],
        "split_group_leakage": [],
        "reviews_admitted": False,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "candidate_manifest": candidate_path.name,
        "candidate_manifest_sha256": candidate_sha,
        "publish_ready_manifest": publish_path.name if publication_allowed else None,
        "publish_ready_manifest_sha256": publish_sha,
        "publish_ready_receipt": receipt_path.name if publication_allowed else None,
        "quality_gates": gates,
        "blocking_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "publication_allowed": publication_allowed,
        "next_action": (
            "publish_replacement" if publication_allowed else "acquire_next_sources_and_repeat"
        ),
    }
    (output_dir / "SATISFACTION_STATUS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = assemble(
        input_root=args.input_root,
        registry_path=args.registry,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
