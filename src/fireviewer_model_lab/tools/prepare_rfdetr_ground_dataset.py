#!/usr/bin/env python3
"""Materialize the ground-view subset of the FireViewer detection corpus.

Only FASDD-CV and Pyro-SDIS rows are retained. Remote Parquet row-group
statistics are used to skip UAV and remote-sensing payloads without first
downloading the complete detection repository.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, HfFileSystem

DATASET_ID = "fireviewer/fire-smoke-detection-corpus-v1"
DATASET_REVISION = "38efeda8e6b2638855cfb1789f2618a3c40f8cea"
GROUND_EVENT = "fasdd-v9-cv"
GROUND_SOURCE = "pyro_sdis_a1e553e"
CLASSES = ("flame_visible", "smoke_visible")
CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}
SPLITS = {"train": "train", "validation": "valid", "test": "test"}
EXPECTED = {
    "train": {"images": 88374, "annotations": 107691},
    "valid": {"images": 19593, "annotations": 24960},
    "test": {"images": 19791, "annotations": 24734},
}
SAFE_EXTENSION = re.compile(r"^\.(?:jpe?g|png|webp)$", re.IGNORECASE)


def is_ground_row(row: dict[str, Any]) -> bool:
    return row.get("event_id") == GROUND_EVENT or row.get("source_id") == GROUND_SOURCE


def _range_contains(statistics: Any, value: str) -> bool:
    if statistics is None or not statistics.has_min_max:
        return True
    minimum = statistics.min.decode() if isinstance(statistics.min, bytes) else statistics.min
    maximum = statistics.max.decode() if isinstance(statistics.max, bytes) else statistics.max
    return str(minimum) <= value <= str(maximum)


def _selected_row_groups(parquet: Any) -> list[int]:
    names = [parquet.metadata.schema.column(i).name for i in range(parquet.metadata.num_columns)]
    source_index = names.index("source_id")
    event_index = names.index("event_id")
    selected: list[int] = []
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        source_stats = group.column(source_index).statistics
        event_stats = group.column(event_index).statistics
        if _range_contains(source_stats, GROUND_SOURCE) or _range_contains(
            event_stats, GROUND_EVENT
        ):
            selected.append(index)
    return selected


def _extension(image: dict[str, Any]) -> str:
    suffix = Path(str(image.get("path") or "")).suffix.lower()
    return suffix if SAFE_EXTENSION.match(suffix) else ".jpg"


def _materialize_shard(
    *,
    repo_id: str,
    revision: str,
    remote_path: str,
    split_dir: Path,
    receipt_dir: Path,
    token: str,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    stem = Path(remote_path).stem
    done = receipt_dir / f"{stem}.done.json"
    records = receipt_dir / f"{stem}.records.jsonl"
    if done.is_file() and records.is_file():
        return json.loads(done.read_text(encoding="utf-8"))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    temporary = records.with_suffix(".jsonl.partial")
    temporary.unlink(missing_ok=True)
    filesystem = HfFileSystem(token=token)
    columns = (
        "image",
        "source_id",
        "event_id",
        "sample_id",
        "width",
        "height",
        "annotations_json",
    )
    images = 0
    annotations = 0
    bytes_written = 0
    hf_path = f"datasets/{repo_id}/{remote_path}"
    with (
        filesystem.open(hf_path, "rb", revision=revision) as remote,
        temporary.open("w", encoding="utf-8", newline="\n") as receipt,
    ):
        parquet = pq.ParquetFile(remote)
        selected = _selected_row_groups(parquet)
        for group_index in selected:
            table = parquet.read_row_group(group_index, columns=list(columns))
            values = table.to_pylist()
            for row_index, row in enumerate(values):
                if not is_ground_row(row):
                    continue
                image = row["image"]
                payload = image.get("bytes") if isinstance(image, dict) else None
                if not isinstance(payload, bytes) or not payload:
                    raise ValueError(
                        f"Missing image bytes in {remote_path}:{group_index}:{row_index}"
                    )
                name = f"{stem}-rg{group_index:04d}-row{row_index:04d}{_extension(image)}"
                destination = split_dir / name
                pending = destination.with_suffix(destination.suffix + ".partial")
                pending.write_bytes(payload)
                pending.replace(destination)
                raw_annotations = json.loads(str(row["annotations_json"]))
                converted: list[dict[str, Any]] = []
                for annotation in raw_annotations:
                    class_name = str(annotation.get("class_name"))
                    if class_name not in CLASS_IDS:
                        continue
                    bbox = [float(value) for value in annotation["bbox_xywh"]]
                    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    converted.append(
                        {
                            "bbox": bbox,
                            "category_id": CLASS_IDS[class_name],
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    )
                receipt.write(
                    json.dumps(
                        {
                            "file_name": name,
                            "width": int(row["width"]),
                            "height": int(row["height"]),
                            "sample_id": str(row["sample_id"]),
                            "annotations": converted,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                images += 1
                annotations += len(converted)
                bytes_written += len(payload)
    temporary.replace(records)
    result = {
        "remote_path": remote_path,
        "images": images,
        "annotations": annotations,
        "bytes": bytes_written,
    }
    done.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _materialize_viewer_tail(
    *,
    split_dir: Path,
    receipt_dir: Path,
    manifest: Path,
    manifest_offset: int = 1427,
    length: int = 4110,
) -> dict[str, Any]:
    """Rebuild the Pyro-SDIS receipt for the final test Parquet shard.

    Windows range reads stopped after writing the first 4,096 images from this
    shard, before flushing their receipt. The published Pyro-SDIS manifest
    supplies the same source rows in order, so those existing files can be
    registered without downloading them again. The final fourteen files were
    materialized by an earlier Dataset Viewer retry.
    """

    remote_path = "data/test/test-00003.parquet"
    stem = Path(remote_path).stem
    done = receipt_dir / f"{stem}.done.json"
    records = receipt_dir / f"{stem}.records.jsonl"
    if done.is_file() and records.is_file():
        result = json.loads(done.read_text(encoding="utf-8"))
        if result.get("images") == length:
            return result
    receipt_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    test_rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "test":
                    test_rows.append(row)
    rows = test_rows[manifest_offset : manifest_offset + length]
    if len(test_rows) != 5537 or len(rows) != length:
        raise ValueError("Pyro-SDIS test manifest count drift")

    range_files = sorted(split_dir.glob(f"{stem}-rg*"))
    range_count = length - 14
    if len(range_files) != range_count:
        raise ValueError(
            f"Expected {range_count} existing range-read images, found {len(range_files)}"
        )
    viewer_files = sorted(split_dir.glob(f"{stem}-viewer-row*"))
    if len(viewer_files) != 14:
        raise ValueError(f"Expected 14 Viewer tail images, found {len(viewer_files)}")
    local_files = [*range_files, *viewer_files]
    temporary = records.with_suffix(".jsonl.partial")
    temporary.unlink(missing_ok=True)
    annotations = 0
    bytes_written = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as receipt:
        for row, destination in zip(rows, local_files, strict=True):
            if not destination.is_file() or destination.stat().st_size == 0:
                raise FileNotFoundError(destination)
            name = destination.name
            converted: list[dict[str, Any]] = []
            for annotation in row["annotations"]:
                class_name = str(annotation.get("class_name"))
                if class_name not in CLASS_IDS:
                    continue
                bbox = [float(value) for value in annotation["bbox_xywh"]]
                if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                converted.append(
                    {
                        "bbox": bbox,
                        "category_id": CLASS_IDS[class_name],
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
            receipt.write(
                json.dumps(
                    {
                        "file_name": name,
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "sample_id": str(row["sample_id"]),
                        "annotations": converted,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            annotations += len(converted)
            bytes_written += destination.stat().st_size
    temporary.replace(records)
    result = {
        "remote_path": remote_path,
        "images": len(rows),
        "annotations": annotations,
        "bytes": bytes_written,
        "transport": "range-read-images-plus-published-manifest",
    }
    done.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _write_filtered_manifests(source_root: Path, output: Path) -> None:
    targets = (
        (
            source_root / "fasdd/manifest.jsonl",
            output / "manifests/fasdd-cv/manifest.jsonl",
            GROUND_EVENT,
        ),
        (
            source_root / "pyro-sdis/manifest.jsonl",
            output / "manifests/pyro-sdis/manifest.jsonl",
            None,
        ),
    )
    for source, destination, event in targets:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".jsonl.partial")
        rows = 0
        with (
            source.open(encoding="utf-8") as reader,
            temporary.open("w", encoding="utf-8", newline="\n") as writer,
        ):
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                if event is not None and row.get("event_id") != event:
                    continue
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows += 1
        if rows == 0:
            raise ValueError(f"Filtered manifest is empty: {source}")
        temporary.replace(destination)


def _assemble_split(output: Path, split: str) -> dict[str, int]:
    split_dir = output / "_rfdetr_coco" / split
    receipt_dir = output / "_materialization" / split
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for records in sorted(receipt_dir.glob("*.records.jsonl")):
        with records.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                image_id = len(images) + 1
                images.append(
                    {
                        "id": image_id,
                        "file_name": row["file_name"],
                        "width": row["width"],
                        "height": row["height"],
                    }
                )
                for annotation in row["annotations"]:
                    annotations.append({"id": annotation_id, "image_id": image_id, **annotation})
                    annotation_id += 1
    expected = EXPECTED[split]
    if len(images) != expected["images"] or len(annotations) != expected["annotations"]:
        raise ValueError(
            f"Ground subset count drift in {split}: "
            f"images={len(images)} annotations={len(annotations)} expected={expected}"
        )
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name, "supercategory": "wildfire"}
            for index, name in enumerate(CLASSES)
        ],
    }
    annotations_path = split_dir / "_annotations.coco.json"
    annotations_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations)}


def prepare(*, output: Path, manifests: Path, token_file: Path, workers: int) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    token = token_file.read_text(encoding="utf-8-sig").strip()
    if not token:
        raise ValueError("Hugging Face token file is empty")
    _write_filtered_manifests(manifests, output)
    info = HfApi(token=token).dataset_info(DATASET_ID, revision=DATASET_REVISION)
    remote_files = [item.rfilename for item in info.siblings or []]
    jobs: list[tuple[str, str, Path, Path]] = []
    for remote_split, local_split in SPLITS.items():
        prefix = f"data/{remote_split}/"
        for remote_path in sorted(
            path for path in remote_files if path.startswith(prefix) and path.endswith(".parquet")
        ):
            jobs.append(
                (
                    remote_path,
                    local_split,
                    output / "_rfdetr_coco" / local_split,
                    output / "_materialization" / local_split,
                )
            )
    if not jobs:
        raise RuntimeError("No remote Parquet shards found")
    receipts: list[dict[str, Any]] = [
        _materialize_viewer_tail(
            split_dir=output / "_rfdetr_coco" / "test",
            receipt_dir=output / "_materialization" / "test",
            manifest=manifests / "pyro-sdis" / "manifest.jsonl",
        )
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _materialize_shard,
                repo_id=DATASET_ID,
                revision=DATASET_REVISION,
                remote_path=remote_path,
                split_dir=split_dir,
                receipt_dir=receipt_dir,
                token=token,
            ): (remote_path, local_split)
            for remote_path, local_split, split_dir, receipt_dir in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            receipts.append(result)
            print(json.dumps({"stage": "shard_complete", **result}), flush=True)
    split_reports = {split: _assemble_split(output, split) for split in EXPECTED}
    completion = {
        "schema_version": 1,
        "dataset_profile": "ground-only",
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "selection": ["fasdd-v9-cv", "pyro_sdis_a1e553e"],
        "excluded": [
            "fasdd-v9-uav",
            "fasdd-v9-rs",
            "alarmod_forest_fire",
            "boreal-forest-fire-detection-v1",
        ],
        "classes": list(CLASSES),
        "splits": split_reports,
        "max_samples_per_split": None,
        "remote_parquet_shards": len(jobs),
        "downloaded_payload_bytes": sum(item["bytes"] for item in receipts),
    }
    coco_root = output / "_rfdetr_coco"
    (coco_root / "_conversion_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    (output / "corpus-audit.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    (output / "publication-manifest.json").write_text(
        json.dumps(
            {
                "dataset_profile": "ground-only",
                "upstream_dataset": DATASET_ID,
                "upstream_revision": DATASET_REVISION,
                "redistribution": "inherits_per_source_licenses",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    result = prepare(
        output=args.output,
        manifests=args.manifests,
        token_file=args.token_file,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
