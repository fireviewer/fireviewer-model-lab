from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CLASS_NAMES = {
    0: "smoke_visible",
    1: "flame_visible",
    2: "firefighting_aircraft_visible",
    3: "fire_response_vehicle_visible",
}
PYRO_SOURCE_ID = "pyro_sdis_a1e553e"
PYRO_REPOSITORY = "pyronear/pyro-sdis"
PYRO_REVISION = "a1e553ec4d806f71fc6db744cc22bc3469487382"
PYRO_EXPECTED_ROWS = 33_636
SPLIT_SEED = "firewarning-corpus-v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload, usedforsecurity=False).hexdigest()


def deterministic_split(split_group: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{split_group}".encode(), usedforsecurity=False).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "validation"
    return "test"


def normalized_identifier(value: object) -> str:
    text = str(value).strip().lower()
    normalized = "".join(character if character.isalnum() else "-" for character in text)
    return "-".join(part for part in normalized.split("-") if part)


def parse_yolo_annotations(
    raw_annotations: object,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    text = str(raw_annotations or "").strip()
    if not text:
        return []

    tokens_by_box: list[list[str]] = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) == 5:
            tokens_by_box.append(tokens)
        elif len(tokens) > 5 and len(tokens) % 5 == 0:
            tokens_by_box.extend(tokens[index : index + 5] for index in range(0, len(tokens), 5))
        elif tokens:
            raise ValueError(f"Malformed YOLO annotation with {len(tokens)} fields")

    annotations: list[dict[str, Any]] = []
    for tokens in tokens_by_box:
        source_class = int(tokens[0])
        if source_class != 1:
            raise ValueError(f"Unexpected Pyro-SDIS class id: {source_class}")
        x_center, y_center, box_width, box_height = (float(value) for value in tokens[1:])
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, box_width, box_height)):
            raise ValueError("YOLO coordinates must be normalized to [0, 1]")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError("YOLO boxes must have a positive area")

        x_min = x_center - (box_width / 2.0)
        y_min = y_center - (box_height / 2.0)
        if x_min < -1e-5 or y_min < -1e-5:
            raise ValueError("YOLO box starts outside the image")
        if x_center + (box_width / 2.0) > 1.00001:
            raise ValueError("YOLO box ends outside the image width")
        if y_center + (box_height / 2.0) > 1.00001:
            raise ValueError("YOLO box ends outside the image height")

        annotations.append(
            {
                "class_id": 0,
                "class_name": CLASS_NAMES[0],
                "bbox_xywh": [
                    max(0.0, x_min) * width,
                    max(0.0, y_min) * height,
                    box_width * width,
                    box_height * height,
                ],
                "visibility": "unknown",
                "occlusion": "unknown",
                "origin": "pyronear/pyro-sdis:annotations",
                "annotated_at": None,
                "annotator_id": "pyronear-volunteers",
                "validation_status": "source_provided",
            }
        )
    return annotations


@dataclass
class BKNode:
    value: int
    sha256: str
    split: str
    children: dict[int, BKNode] = field(default_factory=dict)

    def add(self, value: int, sha256: str, split: str) -> None:
        distance = (self.value ^ value).bit_count()
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = BKNode(value=value, sha256=sha256, split=split)
        else:
            child.add(value, sha256, split)

    def find(self, value: int, maximum_distance: int) -> tuple[str, str] | None:
        distance = (self.value ^ value).bit_count()
        if distance <= maximum_distance:
            return self.sha256, self.split
        lower = max(0, distance - maximum_distance)
        upper = distance + maximum_distance
        for edge, child in self.children.items():
            if lower <= edge <= upper:
                match = child.find(value, maximum_distance)
                if match is not None:
                    return match
        return None


def _image_payload(raw_image: object, cache_root: Path) -> bytes:
    if not isinstance(raw_image, dict):
        raise ValueError("Decoded image metadata must be a mapping")
    payload = raw_image.get("bytes")
    if isinstance(payload, bytes):
        return payload
    raw_path = raw_image.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("Image has neither embedded bytes nor a local path")
    image_path = Path(raw_path).resolve()
    resolved_cache = cache_root.resolve()
    if resolved_cache not in image_path.parents:
        raise ValueError("Dataset image path escaped the configured Hugging Face cache")
    return image_path.read_bytes()


def _inspect_image(payload: bytes) -> tuple[int, int, str, str]:
    import imagehash
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 120_000_000
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
        image_format = str(image.format).upper()
        perceptual_hash = str(imagehash.phash(image.convert("RGB")))
    extension_by_format = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
    extension = extension_by_format.get(image_format)
    if extension is None:
        raise ValueError(f"Unsupported image format: {image_format}")
    if width < 32 or height < 32:
        raise ValueError("Image is too small for the corpus")
    return width, height, extension, perceptual_hash


def _load_partial_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid partial manifest line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Partial manifest line {line_number} is not an object")
            records.append(record)
    return records


def reconcile_near_duplicate_splits(records: list[dict[str, Any]]) -> dict[str, int]:
    """Keep every camera group and near-duplicate component in exactly one split."""

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

    digest_records = {str(record["sha256"]): record for record in records}
    for record in records:
        find(str(record["split_group"]))

    missing_references = 0
    cross_split_before = 0
    for record in records:
        reference = record.get("near_duplicate_of")
        if not reference:
            continue
        referenced = digest_records.get(str(reference))
        if referenced is None:
            missing_references += 1
            continue
        if str(record["split"]) != str(referenced["split"]):
            cross_split_before += 1
        union(str(record["split_group"]), str(referenced["split_group"]))

    component_groups: dict[str, set[str]] = defaultdict(set)
    for group in parents:
        component_groups[find(group)].add(group)
    component_splits: dict[str, str] = {}
    for root, groups in component_groups.items():
        component_key = (
            next(iter(groups)) if len(groups) == 1 else "near:" + "|".join(sorted(groups))
        )
        component_splits[root] = deterministic_split(component_key)

    reassigned_rows = 0
    for record in records:
        assigned = component_splits[find(str(record["split_group"]))]
        if str(record["split"]) != assigned:
            reassigned_rows += 1
            record["split"] = assigned

    cross_split_after = 0
    for record in records:
        reference = record.get("near_duplicate_of")
        referenced = digest_records.get(str(reference)) if reference else None
        if referenced is not None and str(record["split"]) != str(referenced["split"]):
            cross_split_after += 1
    if cross_split_after:
        raise ValueError("Near-duplicate split reconciliation did not converge")
    return {
        "cross_split_near_duplicates_before": cross_split_before,
        "cross_split_near_duplicates_after": cross_split_after,
        "missing_near_duplicate_references": missing_references,
        "merged_group_components": sum(len(groups) > 1 for groups in component_groups.values()),
        "reassigned_rows": reassigned_rows,
    }


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def ingest_pyro_sdis(
    output_dir: Path,
    cache_dir: Path,
    *,
    max_rows: int = 0,
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    from datasets import Image as DatasetImage
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "manifest.partial.jsonl"
    final_path = output_dir / "manifest.jsonl"
    if final_path.exists():
        raise FileExistsError(f"Final manifest already exists: {final_path}")

    existing_records = _load_partial_manifest(partial_path)
    seen_source_records = {str(record["source_record_id"]) for record in existing_records}
    sha_to_record = {str(record["sha256"]): record for record in existing_records}
    tree: BKNode | None = None
    for record in existing_records:
        value = int(str(record["phash"]), 16)
        if tree is None:
            tree = BKNode(value, str(record["sha256"]), str(record["split"]))
        else:
            tree.add(value, str(record["sha256"]), str(record["split"]))

    print(
        f"Loading {PYRO_REPOSITORY}@{PYRO_REVISION} into {cache_dir}",
        flush=True,
    )
    dataset = load_dataset(
        PYRO_REPOSITORY,
        revision=PYRO_REVISION,
        cache_dir=str(cache_dir),
    )
    source_row_count = sum(len(split) for split in dataset.values())
    if source_row_count != PYRO_EXPECTED_ROWS:
        raise ValueError(
            f"Pinned Pyro-SDIS revision has {source_row_count} rows, expected {PYRO_EXPECTED_ROWS}"
        )
    print(f"Pinned source verified: {source_row_count} rows", flush=True)

    exact_duplicates = 0
    near_duplicates = sum(1 for record in existing_records if record.get("near_duplicate_of"))
    cross_split_near_duplicates = 0
    processed_source_rows = 0
    stop_requested = False
    mode = "a" if partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8", newline="\n") as manifest:
        for original_split in sorted(dataset):
            split = dataset[original_split].cast_column("image", DatasetImage(decode=False))
            for row in split:
                if max_rows > 0 and processed_source_rows >= max_rows:
                    stop_requested = True
                    break
                source_record_id = str(row["image_name"])
                if source_record_id in seen_source_records:
                    continue
                processed_source_rows += 1
                payload = _image_payload(row["image"], cache_dir)
                digest = sha256_bytes(payload)
                if digest in sha_to_record:
                    exact_duplicates += 1
                    seen_source_records.add(source_record_id)
                    continue

                width, height, extension, perceptual_hash = _inspect_image(payload)
                partner = normalized_identifier(row["partner"])
                camera = normalized_identifier(row["camera"])
                captured_at_literal = str(row["date"]).strip() or None
                date_group = (captured_at_literal or "date-unknown")[:10]
                split_group = f"{PYRO_SOURCE_ID}:{partner}:{camera}"
                assigned_split = deterministic_split(split_group)
                sequence_id = f"{split_group}:{date_group}"
                phash_value = int(perceptual_hash, 16)
                near_duplicate = tree.find(phash_value, 6) if tree is not None else None
                if near_duplicate is not None:
                    near_duplicates += 1
                    if near_duplicate[1] != assigned_split:
                        cross_split_near_duplicates += 1
                annotations = parse_yolo_annotations(row["annotations"], width=width, height=height)

                relative_path = Path("images") / digest[:2] / f"{digest}.{extension}"
                destination = output_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if sha256_bytes(destination.read_bytes()) != digest:
                        raise ValueError(f"Existing corpus file has a bad digest: {destination}")
                else:
                    destination.write_bytes(payload)

                record: dict[str, Any] = {
                    "sample_id": f"{PYRO_SOURCE_ID}:{digest[:24]}",
                    "source_id": PYRO_SOURCE_ID,
                    "source_record_id": source_record_id,
                    "corpus_role": "detector_training",
                    "image_relpath": relative_path.as_posix(),
                    "sha256": digest,
                    "phash": perceptual_hash,
                    "near_duplicate_of": near_duplicate[0] if near_duplicate else None,
                    "width": width,
                    "height": height,
                    "event_id": sequence_id,
                    "sequence_id": sequence_id,
                    "split_group": split_group,
                    "captured_at_literal": captured_at_literal,
                    "split": assigned_split,
                    "license": "Apache-2.0",
                    "consent_basis": {
                        "kind": "source_license",
                        "reference": f"{PYRO_REPOSITORY}@{PYRO_REVISION}",
                    },
                    "sample_validation_status": "source_provided",
                    "candidate_classes": [],
                    "annotations": annotations,
                    "negative_tags": [] if annotations else ["no_target_visible"],
                    "location": None,
                }
                manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                manifest.flush()
                existing_records.append(record)
                seen_source_records.add(source_record_id)
                sha_to_record[digest] = record
                if tree is None:
                    tree = BKNode(phash_value, digest, assigned_split)
                else:
                    tree.add(phash_value, digest, assigned_split)
                if processed_source_rows % 250 == 0:
                    print(
                        f"Processed {processed_source_rows} source rows; "
                        f"persisted {len(existing_records)} unique images",
                        flush=True,
                    )
            if stop_requested:
                break

    reconciliation = reconcile_near_duplicate_splits(existing_records)
    _write_manifest(partial_path, existing_records)
    partial_path.replace(final_path)
    report = validate_manifest(final_path, output_dir=output_dir, verify_files=False)
    report.update(
        {
            "source_rows": source_row_count,
            "processed_source_rows": processed_source_rows,
            "exact_duplicates_removed": exact_duplicates,
            "near_duplicates_flagged": near_duplicates,
            "cross_split_near_duplicates_flagged": cross_split_near_duplicates,
            "split_reconciliation": reconciliation,
        }
    )
    report_path = output_dir / "quality-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def validate_manifest(
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    base_dir = output_dir or manifest_path.parent
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    seen_sample_ids: set[str] = set()
    seen_sha256: set[str] = set()
    negative_count = 0
    geo_pair_count = 0
    training_rows = 0
    role_counts: Counter[str] = Counter()
    digest_splits: dict[str, str] = {}
    near_duplicate_links: list[tuple[str, str]] = []
    rows = 0

    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            sample_id = str(record["sample_id"])
            digest = str(record["sha256"])
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id at line {line_number}: {sample_id}")
            if digest in seen_sha256:
                raise ValueError(f"Duplicate image SHA-256 at line {line_number}: {digest}")
            seen_sample_ids.add(sample_id)
            seen_sha256.add(digest)

            split = str(record["split"])
            digest_splits[digest] = split
            near_duplicate = record.get("near_duplicate_of")
            if near_duplicate:
                near_duplicate_links.append((str(near_duplicate), split))
            split_counts[split] += 1
            role = str(record["corpus_role"])
            role_counts[role] += 1
            if role == "detector_training":
                training_rows += 1
            if record.get("location") is not None:
                geo_pair_count += 1
            split_group = str(record["split_group"])
            group_splits[split_group].add(split)
            annotations = record["annotations"]
            if role == "detector_training" and not annotations:
                negative_count += 1
            for annotation in annotations:
                class_id = int(annotation["class_id"])
                class_name = str(annotation["class_name"])
                if CLASS_NAMES.get(class_id) != class_name:
                    raise ValueError(f"Class id/name mismatch at line {line_number}")
                x, y, width, height = (float(value) for value in annotation["bbox_xywh"])
                if width <= 0.0 or height <= 0.0:
                    raise ValueError(f"Non-positive box at line {line_number}")
                if x < 0.0 or y < 0.0:
                    raise ValueError(f"Negative box origin at line {line_number}")
                if x + width > int(record["width"]) + 1e-3:
                    raise ValueError(f"Box exceeds image width at line {line_number}")
                if y + height > int(record["height"]) + 1e-3:
                    raise ValueError(f"Box exceeds image height at line {line_number}")
                if role in {"detector_training", "detector_critical_test"}:
                    class_counts[class_name] += 1

            if verify_files:
                image_path = (base_dir / str(record["image_relpath"])).resolve()
                if base_dir.resolve() not in image_path.parents:
                    raise ValueError(f"Image escaped corpus root at line {line_number}")
                if sha256_bytes(image_path.read_bytes()) != digest:
                    raise ValueError(f"Image digest mismatch at line {line_number}")
                if rows % 10_000 == 0:
                    print(
                        f"manifest file verification rows={rows} path={manifest_path}",
                        file=sys.stderr,
                        flush=True,
                    )

    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking_groups:
        raise ValueError(f"Split leakage detected in {len(leaking_groups)} groups")
    missing_classes = sorted(set(CLASS_NAMES.values()) - set(class_counts))
    missing_near_duplicate_references = sum(
        reference not in digest_splits for reference, _split in near_duplicate_links
    )
    cross_split_near_duplicates = sum(
        reference in digest_splits and digest_splits[reference] != split
        for reference, split in near_duplicate_links
    )
    manifest_digest = sha256_bytes(manifest_path.read_bytes())
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_digest,
        "rows": rows,
        "training_rows": training_rows,
        "negative_rows": negative_count,
        "geo_pair_rows": geo_pair_count,
        "role_counts": dict(sorted(role_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_group_counts": dict(
            sorted(Counter(next(iter(splits)) for splits in group_splits.values()).items())
        ),
        "annotation_counts": dict(sorted(class_counts.items())),
        "missing_target_classes": missing_classes,
        "split_leakage_groups": 0,
        "cross_split_near_duplicates": cross_split_near_duplicates,
        "missing_near_duplicate_references": missing_near_duplicate_references,
        "four_class_training_ready": (
            not missing_classes
            and negative_count > 0
            and cross_split_near_duplicates == 0
            and missing_near_duplicate_references == 0
        ),
        "files_verified": verify_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate the FireWarning corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-pyro-sdis")
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--cache", type=Path, required=True)
    ingest.add_argument("--max-rows", type=int, default=0)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--verify-files", action="store_true")
    reconcile = subparsers.add_parser("reconcile-splits")
    reconcile.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "ingest-pyro-sdis":
        if args.max_rows < 0:
            raise ValueError("--max-rows cannot be negative")
        report = ingest_pyro_sdis(args.output, args.cache, max_rows=args.max_rows)
    elif args.command == "validate":
        report = validate_manifest(args.manifest, verify_files=args.verify_files)
    else:
        records = _load_partial_manifest(args.manifest)
        reconciliation = reconcile_near_duplicate_splits(records)
        temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
        _write_manifest(temporary, records)
        temporary.replace(args.manifest)
        report = validate_manifest(args.manifest)
        report["split_reconciliation"] = reconciliation
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
