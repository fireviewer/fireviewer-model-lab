"""Build a strict, self-contained KIT flame auxiliary set without fake points."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat

SOURCE_ID = "kit-industrial-burner-flames"
SOURCE_RECORD_URL = "https://publikationen.bibliothek.kit.edu/1000159497"
SOURCE_PAPER_URL = "https://publikationen.bibliothek.kit.edu/1000159876/150918207"
SOURCE_DOI = "10.35097/1452"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_ARCHIVE_SHA256 = "bce34b091b4d200f4c9804269d4e1f3b7f4b9bd3c97c0e296332a18c717a0a54"
ARCHIVE_FILENAME = "kit-industrial-burner-flames.tar"
VALID_MASK_VALUES = frozenset({0, 1, 255, 65535})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _normalized_member_name(name: str) -> str:
    if "\x00" in name:
        raise ValueError("unsafe tar member contains a NUL byte")
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"unsafe tar member path: {name}")
    return normalized.as_posix()


def _validated_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    total_bytes = 0
    for member in archive:
        name = _normalized_member_name(member.name)
        key = name.casefold()
        if key in members:
            raise ValueError(f"duplicate tar member path: {name}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise ValueError(f"unsupported tar member type: {name}")
        if member.size < 0 or member.size > 16 * 1024**3:
            raise ValueError(f"unsafe tar member size: {name}:{member.size}")
        if member.isfile():
            total_bytes += member.size
            if total_bytes > 32 * 1024**3:
                raise ValueError("tar uncompressed byte count exceeds limit")
        members[key] = member
    return members


def _payload(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"unreadable tar member: {member.name}")
    with source:
        value = source.read()
    if len(value) != member.size:
        raise ValueError(f"short tar member read: {member.name}")
    return value


def _kit_locations(
    members: dict[str, tarfile.TarInfo],
) -> dict[str, dict[str, dict[str, tuple[str, tarfile.TarInfo]]]]:
    locations: dict[str, dict[str, dict[str, tuple[str, tarfile.TarInfo]]]] = {
        "DataA": defaultdict(dict),
        "DataB": defaultdict(dict),
    }
    prefix = "unnamed entity/dataset/"
    for key, member in members.items():
        if not member.isfile() or not key.startswith(prefix) or not key.endswith(".tif"):
            continue
        parts = PurePosixPath(_normalized_member_name(member.name)).parts
        if len(parts) != 6:
            continue
        _, dataset_root, dataset, source_split, kind, filename = parts
        if dataset_root != "Dataset" or dataset not in locations:
            continue
        if source_split not in {"train", "test"} or kind not in {"images", "masks"}:
            continue
        identifier = filename.removesuffix(".tif").removesuffix(".tif")
        if not identifier.isdigit():
            continue
        if kind in locations[dataset][identifier]:
            raise ValueError(f"duplicate KIT pair member: {dataset}:{identifier}:{kind}")
        locations[dataset][identifier][kind] = (source_split, member)
    return locations


def _decode_image(payload: bytes) -> tuple[Image.Image, tuple[int, int]]:
    with Image.open(io.BytesIO(payload)) as opened:
        opened.load()
        size = opened.size
        image = opened.convert("L")
    return image, size


def _decode_mask(payload: bytes) -> tuple[np.ndarray, list[int], tuple[int, int]]:
    with Image.open(io.BytesIO(payload)) as opened:
        opened.load()
        size = opened.size
        raw = np.asarray(opened)
    if raw.ndim == 3:
        channels_agree = all(
            np.array_equal(raw[..., 0], raw[..., index]) for index in range(1, raw.shape[2])
        )
        if not channels_agree:
            raise ValueError("KIT mask channels disagree")
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"KIT mask has unsupported shape: {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"KIT mask has unsupported dtype: {raw.dtype}")
    values = sorted(int(value) for value in np.unique(raw))
    return raw > 0, values, size


def _difference_hash(image: Image.Image) -> str:
    resized = image.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.tobytes())
    bits = 0
    for y_value in range(8):
        offset = y_value * 9
        for x_value in range(8):
            bits = (bits << 1) | int(pixels[offset + x_value] > pixels[offset + x_value + 1])
    return f"{bits:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _metrics(image: Image.Image) -> dict[str, float]:
    resized = image.resize((256, 256), Image.Resampling.BILINEAR)
    stats = ImageStat.Stat(resized)
    edges = ImageStat.Stat(resized.filter(ImageFilter.FIND_EDGES))
    return {
        "brightness": float(stats.mean[0] / 255.0),
        "contrast": float(stats.stddev[0] / 255.0),
        "edge_energy": float(edges.mean[0] / 255.0),
    }


def _load_baseline(root: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    manifests = list(root.rglob("strict_combined_manifest.jsonl"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"expected one strict combined manifest, found {len(manifests)}")
    sha_rows: dict[str, str] = {}
    dhash_rows: list[tuple[str, str]] = []
    for line in manifests[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        digest = str(row.get("image_sha256") or row.get("source_image_sha256") or "")
        signature = str(row.get("dhash") or "")
        if len(digest) == 64:
            sha_rows[digest] = sample_id
        if len(signature) == 16:
            dhash_rows.append((signature, sample_id))
    return sha_rows, dhash_rows


def _nearest(signature: str, rows: list[tuple[str, str]]) -> tuple[int | None, str | None]:
    if not rows:
        return None, None
    distance, sample_id = min((_hamming(signature, value), key) for value, key in rows)
    return distance, sample_id


def _cross_split_near_exclusions(
    rows: list[dict[str, Any]], maximum_distance: int = 4
) -> tuple[list[dict[str, Any]], set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = _hamming(str(left["dhash"]), str(right["dhash"]))
            if distance > maximum_distance:
                continue
            left_id, right_id = str(left["sample_id"]), str(right["sample_id"])
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)
            pairs.append(
                {
                    "left": left_id,
                    "left_split": left["split"],
                    "right": right_id,
                    "right_split": right["split"],
                    "dhash_distance": distance,
                }
            )
    priority = {"test": 0, "validation": 1, "train": 2}
    by_id = {str(row["sample_id"]): row for row in rows}
    excluded: set[str] = set()
    visited: set[str] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            sample_id = stack.pop()
            if sample_id in component:
                continue
            component.add(sample_id)
            stack.extend(adjacency[sample_id])
        visited.update(component)
        keep = min(
            component,
            key=lambda sample_id: (
                priority[str(by_id[sample_id]["split"])],
                sample_id,
            ),
        )
        excluded.update(component - {keep})
    return pairs, excluded


def audit_kit(
    *,
    inventory_root: Path,
    baseline_root: Path,
    output_dir: Path,
    output_s3_prefix: str | None = None,
    expected_archive_sha256: str = SOURCE_ARCHIVE_SHA256,
    expected_train_rows: int = 160,
    expected_test_rows: int = 40,
    validation_rows: int = 32,
    expected_image_size: int = 552,
) -> dict[str, Any]:
    archives = list(inventory_root.rglob(ARCHIVE_FILENAME))
    if len(archives) != 1:
        raise FileNotFoundError(f"expected one KIT archive, found {len(archives)}")
    archive_path = archives[0]
    archive_sha = _sha256_file(archive_path)
    if archive_sha != expected_archive_sha256:
        raise ValueError(f"KIT archive SHA-256 mismatch: {archive_sha}")
    baseline_sha, baseline_dhash = _load_baseline(baseline_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = _validated_members(archive)
        locations = _kit_locations(members)
        expected_identifiers = set(locations["DataB"])
        source_errors: list[str] = []
        if set(locations["DataA"]) != expected_identifiers:
            source_errors.append("data_a_b_identifier_sets_differ")
        for dataset, expected_counts in (
            ("DataA", {"train": expected_test_rows, "test": expected_train_rows}),
            ("DataB", {"train": expected_train_rows, "test": expected_test_rows}),
        ):
            counts = Counter(
                pair.get("images", ("missing", None))[0]
                for pair in locations[dataset].values()
            )
            if counts != Counter(expected_counts):
                source_errors.append(f"{dataset.lower()}_split_counts:{dict(sorted(counts.items()))}")
            incomplete = [
                identifier
                for identifier, pair in locations[dataset].items()
                if set(pair) != {"images", "masks"}
                or pair["images"][0] != pair["masks"][0]
            ]
            if incomplete:
                source_errors.append(f"{dataset.lower()}_incomplete_pairs:{len(incomplete)}")

        data_a_hashes: dict[str, tuple[str, str]] = {}
        for identifier, pair in locations["DataA"].items():
            if set(pair) != {"images", "masks"}:
                continue
            data_a_hashes[identifier] = (
                _sha256_bytes(_payload(archive, pair["images"][1])),
                _sha256_bytes(_payload(archive, pair["masks"][1])),
            )

        train_identifiers = [
            identifier
            for identifier, pair in locations["DataB"].items()
            if pair.get("images", (None, None))[0] == "train"
        ]
        validation_identifiers = set(
            sorted(
                train_identifiers,
                key=lambda value: hashlib.sha256(
                    f"{SOURCE_ID}:{value}:validation".encode()
                ).hexdigest(),
            )[:validation_rows]
        )
        candidates: list[dict[str, Any]] = []
        payloads: dict[str, tuple[bytes, bytes]] = {}
        duplicate_copy_matches = 0
        for identifier in sorted(expected_identifiers):
            pair = locations["DataB"][identifier]
            reasons: list[str] = []
            if set(pair) != {"images", "masks"}:
                continue
            source_split = pair["images"][0]
            image_payload = _payload(archive, pair["images"][1])
            mask_payload = _payload(archive, pair["masks"][1])
            image_sha = _sha256_bytes(image_payload)
            mask_sha = _sha256_bytes(mask_payload)
            if data_a_hashes.get(identifier) == (image_sha, mask_sha):
                duplicate_copy_matches += 1
            else:
                reasons.append("data_a_b_payload_mismatch")
            try:
                image, image_size = _decode_image(image_payload)
                mask, mask_values, mask_size = _decode_mask(mask_payload)
            except Exception as exc:
                reasons.append(f"decode_error:{type(exc).__name__}")
                image = Image.new("L", (1, 1))
                image_size = (1, 1)
                mask = np.zeros((1, 1), dtype=np.bool_)
                mask_values = []
                mask_size = (1, 1)
            if image_size != mask_size:
                reasons.append("image_mask_dimension_mismatch")
            if image_size != (expected_image_size, expected_image_size):
                reasons.append("unexpected_image_dimensions")
            if not set(mask_values).issubset(VALID_MASK_VALUES):
                reasons.append("non_binary_mask_values")
            foreground_fraction = float(np.count_nonzero(mask) / mask.size)
            if not math.isfinite(foreground_fraction) or not 0.001 <= foreground_fraction <= 0.95:
                reasons.append("implausible_mask_foreground_fraction")
            dhash = _difference_hash(image)
            baseline_exact = baseline_sha.get(image_sha)
            baseline_distance, baseline_nearest = _nearest(dhash, baseline_dhash)
            if baseline_exact is not None:
                reasons.append("baseline_exact_sha_overlap")
            if baseline_distance is not None and baseline_distance <= 4:
                reasons.append("baseline_perceptual_overlap")
            split = (
                "test"
                if source_split == "test"
                else "validation"
                if identifier in validation_identifiers
                else "train"
            )
            sample_id = f"kit-industrial-burner-{identifier}"
            payloads[sample_id] = (image_payload, mask_payload)
            candidates.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "source_id": SOURCE_ID,
                    "source_revision": SOURCE_ARCHIVE_SHA256,
                    "source_record_url": SOURCE_RECORD_URL,
                    "source_paper_url": SOURCE_PAPER_URL,
                    "source_doi": SOURCE_DOI,
                    "license": SOURCE_LICENSE,
                    "redistribution_allowed": True,
                    "source_dataset_copy": "DataB",
                    "source_split": source_split,
                    "split": split,
                    "split_group": f"{SOURCE_ID}:{image_sha}",
                    "image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "dhash": dhash,
                    "image_width": image_size[0],
                    "image_height": image_size[1],
                    "mask_values": mask_values,
                    "mask_nonzero_fraction": foreground_fraction,
                    "mask_quality": "human_relabelled_source_mask",
                    "mask_semantics": "industrial_burner_flame",
                    "annotation_strength": "strong",
                    "annotation_provenance": "human_relabelled_by_source_authors",
                    "sample_validation_status": "pending_strict_automated_validation",
                    "segmentation_supervised": True,
                    "point_supervised": False,
                    "presence_supervised": True,
                    "abstention_supervised": True,
                    "presence_targets": {
                        "flame_visible": True,
                        "smoke_visible": False,
                    },
                    "presence_provenance": "human_relabelled_flame_mask",
                    "anchor_points": [],
                    "point_derivation": "none",
                    "mask_to_point_conversion": "forbidden_industrial_non_ground_geometry",
                    "visual_abstention_reason": (
                        "industrial_burner_has_no_wildfire_ground_contact_target"
                    ),
                    "corpus_role": "segmentation_presence_abstention_auxiliary_only",
                    "ground_point_eligible": False,
                    "sample_weight": 0.5,
                    "baseline_exact_sha_match": baseline_exact,
                    "baseline_nearest_dhash_distance": baseline_distance,
                    "baseline_nearest_sample_id": baseline_nearest,
                    "validation_profile": "fireviewer_pointing_strict_automated_v2",
                    "reviews_admitted": False,
                    "strict_keep": False,
                    "training_eligible": False,
                    "exclusion_reasons": sorted(set(reasons)),
                    **_metrics(image),
                }
            )

    exact_groups: dict[str, list[str]] = defaultdict(list)
    dhash_groups: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        exact_groups[str(row["image_sha256"])].append(str(row["sample_id"]))
        dhash_groups[str(row["dhash"])].append(str(row["sample_id"]))
    exact_groups = {key: sorted(value) for key, value in exact_groups.items() if len(value) > 1}
    dhash_groups = {key: sorted(value) for key, value in dhash_groups.items() if len(value) > 1}
    near_pairs, near_excluded = _cross_split_near_exclusions(candidates)
    priority = {"test": 0, "validation": 1, "train": 2}
    by_id = {str(row["sample_id"]): row for row in candidates}
    for groups, reason in (
        (exact_groups, "within_source_exact_sha_duplicate"),
        (dhash_groups, "within_source_identical_dhash_neighbor"),
    ):
        for sample_ids in groups.values():
            keep = min(sample_ids, key=lambda value: (priority[str(by_id[value]["split"])], value))
            for sample_id in set(sample_ids) - {keep}:
                by_id[sample_id]["exclusion_reasons"].append(reason)
    for sample_id in near_excluded:
        by_id[sample_id]["exclusion_reasons"].append(
            "within_source_near_cross_split_duplicate"
        )
    for row in candidates:
        row["exclusion_reasons"] = sorted(set(row["exclusion_reasons"]))

    initially_validated = [row for row in candidates if not row["exclusion_reasons"]]
    initial_splits = Counter(str(row["split"]) for row in initially_validated)
    gate_errors = list(source_errors)
    if duplicate_copy_matches != expected_train_rows + expected_test_rows:
        gate_errors.append(f"data_a_b_exact_copy_matches:{duplicate_copy_matches}")
    if len(initially_validated) < max(3, (expected_train_rows + expected_test_rows) // 2):
        gate_errors.append(f"insufficient_validated_rows:{len(initially_validated)}")
    missing_splits = sorted({"train", "validation", "test"} - set(initial_splits))
    if missing_splits:
        gate_errors.append(f"missing_validated_splits:{missing_splits}")
    source_gate_passed = not gate_errors
    validated = initially_validated if source_gate_passed else []
    validated_ids = {str(row["sample_id"]) for row in validated}
    for row in candidates:
        keep = str(row["sample_id"]) in validated_ids
        row["strict_keep"] = keep
        row["training_eligible"] = keep
        row["sample_validation_status"] = (
            "strict_automated_validated" if keep else "excluded_unvalidated"
        )
        row["corpus_disposition"] = (
            "eligible_multitask_auxiliary" if keep else "excluded_unvalidated"
        )

    materialized = 0
    for row in validated:
        sample_id = str(row["sample_id"])
        image_payload, mask_payload = payloads[sample_id]
        for kind, payload, digest, field, uri_field in (
            ("images", image_payload, row["image_sha256"], "image_relpath", "image_s3_uri"),
            ("masks", mask_payload, row["mask_sha256"], "mask_relpath", "mask_s3_uri"),
        ):
            relative = f"strict-payload/{kind}/{digest}.tif"
            destination = output_dir / Path(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            if _sha256_file(destination) != digest:
                raise ValueError(f"materialized KIT payload hash mismatch: {sample_id}:{kind}")
            row[field] = relative
            if output_s3_prefix:
                row[uri_field] = f"{output_s3_prefix.rstrip('/')}/{relative}"
            materialized += 1

    _write_jsonl(output_dir / "kit_automatic_dispositions.jsonl", candidates)
    _write_jsonl(output_dir / "kit_strict_validated_manifest.jsonl", validated)
    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_ARCHIVE_SHA256,
        "source_record_url": SOURCE_RECORD_URL,
        "source_paper_url": SOURCE_PAPER_URL,
        "source_doi": SOURCE_DOI,
        "declared_license": SOURCE_LICENSE,
        "archive_sha256_verified": True,
        "data_a_b_exact_copy_matches": duplicate_copy_matches,
        "rows_evaluated": len(candidates),
        "strict_automated_validated_rows": len(validated),
        "validated_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in validated).items())
        ),
        "source_gate_passed": source_gate_passed,
        "gate_errors": gate_errors,
        "strict_payload_artifacts_materialized": materialized,
        "point_supervised_rows": 0,
        "segmentation_supervised_rows": len(validated),
        "presence_supervised_rows": len(validated),
        "abstention_supervised_rows": len(validated),
        "exclusions_by_reason": dict(
            sorted(
                Counter(
                    reason for row in candidates for reason in row["exclusion_reasons"]
                ).items()
            )
        ),
        "baseline_images_compared": len(baseline_sha),
        "within_source_exact_sha_groups": len(exact_groups),
        "within_source_identical_dhash_groups": len(dhash_groups),
        "near_cross_split_pairs": near_pairs,
        "near_cross_split_rows_excluded": len(near_excluded),
        "reviews_admitted": False,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "publication_allowed": False,
        "training_eligible_rows": len(validated),
        "corpus_role": "segmentation_presence_abstention_auxiliary_only",
        "next_action": "merge_only_if_global_multitask_composition_gate_passes",
    }
    (output_dir / "kit_strict_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-s3-prefix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_kit(
        inventory_root=args.inventory_root,
        baseline_root=args.baseline_root,
        output_dir=args.output_dir,
        output_s3_prefix=args.output_s3_prefix,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
