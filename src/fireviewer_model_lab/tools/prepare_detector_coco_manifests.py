"""Adapt the frozen FireViewer Parquet corpus to its materialized COCO images.

The Hub corpus stores image bytes inside Parquet files while the historical
detector trainer consumes file-backed manifests. RF-DETR already materializes
the exact same rows as JPEG files. This tool joins both representations by the
deterministic shard/row identity embedded in each JPEG filename, preserves the
source digest as ``source_sha256``, and writes manifests whose primary digest
matches the JPEG bytes used for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOURCE_NAMES = ("fasdd", "pyro-sdis", "alarmod", "boreal")
COCO_CLASSES = {
    0: (1, "flame_visible"),
    1: (0, "smoke_visible"),
}
SPLITS = (
    ("train", "train"),
    ("validation", "valid"),
    ("test", "test"),
)
ROW_FILE_PATTERN = re.compile(r"^(?P<index>\d+)_row-(?P<row>\d+)_(?P<digest>[0-9a-f]{12})\.[^.]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _parquet_rows(dataset_root: Path, parquet_split: str) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("pyarrow is required to adapt the detector corpus") from exc

    parquet_dir = dataset_root / "data" / parquet_split
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet shards found under {parquet_dir}")
    columns = ("source_name", "sha256", "width", "height", "original_record_json")
    for parquet_path in files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=columns, batch_size=2048):
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                row = {column: values[column][index] for column in columns}
                row["_source_shard"] = parquet_path.relative_to(dataset_root).as_posix()
                yield row


def _coco_payload(
    coco_split_dir: Path,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    annotations = _json_object(coco_split_dir / "_annotations.coco.json")
    images = annotations.get("images")
    if not isinstance(images, list):
        raise ValueError(f"COCO images list is missing under {coco_split_dir}")
    raw_annotations = annotations.get("annotations")
    if not isinstance(raw_annotations, list):
        raise ValueError(f"COCO annotations list is missing under {coco_split_dir}")
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in raw_annotations:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    return images, annotations_by_image


def _adapt_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted: list[dict[str, Any]] = []
    for annotation in annotations:
        coco_class_id = int(annotation["category_id"])
        if coco_class_id not in COCO_CLASSES:
            raise ValueError(f"Unsupported COCO class id: {coco_class_id}")
        class_id, class_name = COCO_CLASSES[coco_class_id]
        bbox = [float(value) for value in annotation["bbox"]]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"Invalid COCO bbox: {bbox}")
        adapted.append(
            {
                "annotated_at": None,
                "annotator_id": "FireViewer COCO normalization",
                "bbox_xywh": bbox,
                "class_id": class_id,
                "class_name": class_name,
                "occlusion": "unknown",
                "origin": "fireviewer-rfdetr-coco-v1",
                "validation_status": "source_provided",
                "visibility": "unknown",
            }
        )
    return adapted


def _validate_join(
    *,
    split: str,
    row_index: int,
    parquet_row: dict[str, Any],
    image: dict[str, Any],
    image_path: Path,
) -> None:
    file_name = str(image.get("file_name", ""))
    match = ROW_FILE_PATTERN.match(file_name)
    if not match or int(match.group("index")) != row_index or int(match.group("row")) != row_index:
        raise ValueError(f"COCO row ordering drift at index {row_index}: {file_name}")
    if int(image.get("id", -1)) != row_index + 1:
        raise ValueError(f"COCO image id drift at index {row_index}: {image.get('id')}")
    identity = f"{split}|{parquet_row['_source_shard']}|{row_index}|row-{row_index}".encode()
    expected_digest = hashlib.sha1(identity, usedforsecurity=False).hexdigest()[:12]
    if match.group("digest") != expected_digest:
        raise ValueError(f"COCO shard/row identity drift at index {row_index}: {file_name}")
    if int(image.get("width", -1)) < 1 or int(image.get("height", -1)) < 1:
        raise ValueError(f"COCO dimensions are invalid at index {row_index}: {file_name}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Materialized COCO image is missing: {image_path}")


def prepare_manifests(dataset_root: Path, output_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    coco_root = (dataset_root / "_rfdetr_coco").resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root != coco_root:
        raise ValueError("Output root must be the frozen _rfdetr_coco directory")

    pending_paths = {
        source: output_root / f"manifest.{source}.pending.jsonl" for source in SOURCE_NAMES
    }
    for path in pending_paths.values():
        path.unlink(missing_ok=True)
    handles = {
        source: path.open("w", encoding="utf-8", newline="\n")
        for source, path in pending_paths.items()
    }
    source_to_materialized: dict[str, str] = {}
    materialized_seen: set[str] = set()
    retained_by_source: dict[str, set[str]] = defaultdict(set)
    input_source_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    try:
        for parquet_split, coco_split in SPLITS:
            images, annotations_by_image = _coco_payload(coco_root / coco_split)
            rows = _parquet_rows(dataset_root, parquet_split)
            row_count = 0
            for row_index, (parquet_row, image) in enumerate(zip(rows, images, strict=True)):
                file_name = str(image["file_name"])
                image_path = coco_root / coco_split / file_name
                _validate_join(
                    split=coco_split,
                    row_index=row_index,
                    parquet_row=parquet_row,
                    image=image,
                    image_path=image_path,
                )
                record = json.loads(str(parquet_row["original_record_json"]))
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Original record is not an object at {parquet_split}:{row_index}"
                    )
                source = str(parquet_row["source_name"])
                if source not in handles:
                    raise ValueError(
                        f"Unexpected source_name at {parquet_split}:{row_index}: {source}"
                    )
                input_source_counts[source] += 1
                source_sha256 = str(parquet_row["sha256"])
                if str(record.get("sha256")) != source_sha256:
                    raise ValueError(f"Source digest drift at {parquet_split}:{row_index}")
                materialized_sha256 = _sha256(image_path)
                if source_sha256 in source_to_materialized:
                    raise ValueError(f"Duplicate source digest: {source_sha256}")
                source_to_materialized[source_sha256] = materialized_sha256
                row_count += 1
                if materialized_sha256 in materialized_seen:
                    duplicate_counts[source] += 1
                    continue
                materialized_seen.add(materialized_sha256)
                retained_by_source[source].add(materialized_sha256)
                record_annotations = _adapt_annotations(
                    annotations_by_image.get(int(image["id"]), [])
                )
                record["source_sha256"] = source_sha256
                record["source_image_relpath"] = str(record["image_relpath"])
                record["sha256"] = materialized_sha256
                record["image_relpath"] = (Path(coco_split) / file_name).as_posix()
                record["width"] = int(image["width"])
                record["height"] = int(image["height"])
                record["annotations"] = record_annotations
                record["negative"] = not record_annotations
                if record_annotations:
                    record["negative_tags"] = []
                handles[source].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                source_counts[source] += 1
                split_counts[str(record["split"])] += 1
            if row_count != len(images):
                raise ValueError(
                    f"Parquet/COCO row count drift for {parquet_split}: "
                    f"parquet={row_count}, coco={len(images)}"
                )
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in pending_paths.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles.values():
            handle.close()

    manifest_reports: dict[str, dict[str, Any]] = {}
    for source, pending_path in pending_paths.items():
        final_path = output_root / f"manifest.{source}.jsonl"
        final_temporary = output_root / f"manifest.{source}.jsonl.tmp"
        rows = 0
        with (
            pending_path.open(encoding="utf-8") as source_handle,
            final_temporary.open("w", encoding="utf-8", newline="\n") as output_handle,
        ):
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                reference = record.get("near_duplicate_of")
                if reference:
                    mapped = source_to_materialized.get(str(reference))
                    if mapped is None:
                        raise ValueError(
                            f"Missing near-duplicate source digest in {pending_path}:{line_number}"
                        )
                    record["source_near_duplicate_of"] = str(reference)
                    record["near_duplicate_of"] = (
                        mapped
                        if mapped != record["sha256"] and mapped in retained_by_source[source]
                        else None
                    )
                output_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
        final_temporary.replace(final_path)
        pending_path.unlink()
        manifest_reports[source] = {
            "path": str(final_path),
            "rows": rows,
            "sha256": _sha256(final_path),
        }

    report = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "coco_root": str(coco_root),
        "conversion_complete_sha256": _sha256(coco_root / "_conversion_complete.json"),
        "rows": sum(source_counts.values()),
        "input_rows": sum(input_source_counts.values()),
        "input_source_counts": dict(sorted(input_source_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "duplicate_materialized_images_removed": sum(duplicate_counts.values()),
        "duplicate_counts": dict(sorted(duplicate_counts.items())),
        "manifests": manifest_reports,
        "join_contract": "deterministic_parquet_shard_and_split_row_to_frozen_coco_image",
        "digest_contract": "sha256=unique_materialized_jpeg; source_sha256=parquet_source_bytes",
    }
    report_path = output_root / "adapted-manifests-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root or args.dataset_root / "_rfdetr_coco"
    report = prepare_manifests(args.dataset_root, output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
