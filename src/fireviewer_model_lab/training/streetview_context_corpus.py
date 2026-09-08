from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fireviewer_model_lab.training.corpus_pipeline import (
    _inspect_image,
    deterministic_split,
    sha256_bytes,
    validate_manifest,
)

SOURCE_ID = "streetview_global_context_v1"
SOURCE_REPOSITORY = "Reubencf/streetview-global"
SOURCE_LICENSE = "CC-BY-SA-4.0"
SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
SOURCE_DATASET_URL = "https://huggingface.co/datasets/Reubencf/streetview-global"
CONTEXT_SETTINGS = frozenset({"rural", "forest", "mountain"})
FIELD_TERMS = re.compile(r"\b(field|farmland|farm|agricultur(?:al|e)|crop|pasture)\b", re.I)


def context_labels(row: dict[str, Any]) -> tuple[str, ...]:
    """Return source-provided weak scene labels without promoting them to truth."""

    labels: set[str] = set()
    setting = str(row.get("setting") or "").strip().casefold()
    if setting in CONTEXT_SETTINGS:
        labels.add(setting)
    description = str(row.get("scene_description") or "")
    if FIELD_TERMS.search(description):
        labels.add("field")
    return tuple(sorted(labels))


def selected_rows(
    rows: Iterable[dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], tuple[str, ...]]]:
    for row in rows:
        labels = context_labels(row)
        if labels:
            yield row, labels


def _source_image_bytes(row: dict[str, Any]) -> bytes:
    image = row.get("image")
    if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
        raise ValueError(f"StreetView Global row {row.get('id')!r} has no embedded image bytes")
    return image["bytes"]


def _location(row: dict[str, Any]) -> dict[str, float] | None:
    latitude, longitude = row.get("latitude"), row.get("longitude")
    if latitude is None or longitude is None:
        return None
    return {"latitude": float(latitude), "longitude": float(longitude)}


def curate_streetview_context(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Materialize weak scene-context images from the acquired source parquet files.

    These labels are not detector labels and are deliberately kept out of the
    smoke/flame model manifests. They may be used for contextual robustness
    studies only after human review.
    """

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    parquet_files = sorted((source_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No StreetView Global parquet files under {source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to merge into a non-empty corpus: {output_dir}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "pyarrow is required only to materialize the acquired StreetView parquet source"
        ) from exc

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    label_counts: Counter[str] = Counter()
    rejected = 0

    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=64):
            for row, labels in selected_rows(batch.to_pylist()):
                try:
                    payload = _source_image_bytes(row)
                    width, height, extension, perceptual_hash = _inspect_image(payload)
                except (OSError, ValueError):
                    rejected += 1
                    continue
                digest = sha256_bytes(payload)
                if digest in seen_digests:
                    continue
                seen_digests.add(digest)
                relative_path = Path("images") / f"{digest}.{extension}"
                (output_dir / relative_path).write_bytes(payload)
                source_record_id = str(row.get("id") or digest)
                sequence = str(row.get("sequence") or source_record_id)
                records.append(
                    {
                        "sample_id": f"{SOURCE_ID}:{digest[:24]}",
                        "source_id": SOURCE_ID,
                        "source_record_id": source_record_id,
                        "corpus_role": "context_weak_supervision",
                        "split": deterministic_split(f"{SOURCE_ID}:{sequence}"),
                        "split_group": f"{SOURCE_ID}:{sequence}",
                        "image_relpath": relative_path.as_posix(),
                        "sha256": digest,
                        "perceptual_hash": perceptual_hash,
                        "width": width,
                        "height": height,
                        "annotations": [],
                        "location": _location(row),
                        "scene_context_labels": list(labels),
                        "context_label_quality": "weak_source_metadata",
                        "source_asset": {
                            "dataset": SOURCE_REPOSITORY,
                            "dataset_url": SOURCE_DATASET_URL,
                            "license": SOURCE_LICENSE,
                            "license_url": SOURCE_LICENSE_URL,
                            "attribution_required": True,
                            "provider": "Mapillary",
                            "captured_at": row.get("captured_at"),
                            "compass_angle": row.get("compass_angle"),
                            "region": row.get("region"),
                            "scene_description": row.get("scene_description"),
                        },
                    }
                )
                label_counts.update(labels)

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    validation = validate_manifest(manifest_path, output_dir=output_dir, verify_files=True)
    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_license": SOURCE_LICENSE,
        "rows": len(records),
        "weak_context_label_counts": dict(sorted(label_counts.items())),
        "rejected_images": rejected,
        "validation": validation,
        "training_policy": {
            "eligible_for_detector_training": False,
            "eligible_for_spatial_ground_truth": False,
            "requires_human_review_before_promotion": True,
        },
    }
    (output_dir / "acquisition-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate weak StreetView Global scene-context examples"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = curate_streetview_context(args.source, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
