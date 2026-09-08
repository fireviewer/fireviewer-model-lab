from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_ID = "nasa_hls_burn_scars_v1"
SOURCE_REPOSITORY = "nasa-impact/hls_burn_scars"
SOURCE_REVISION = "1864285e25010d346a842e4f068b1a1d4248ed6d"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_URL = "https://huggingface.co/datasets/nasa-impact/hls_burn_scars"
TARGET_CLASS = "burned_area_binary"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paired_scene_files(split_dir: Path) -> list[tuple[Path, Path]]:
    scenes = sorted(split_dir.glob("*_merged.tif"))
    pairs: list[tuple[Path, Path]] = []
    for scene in scenes:
        mask = scene.with_name(scene.name.replace("_merged.tif", ".mask.tif"))
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask for scene {scene.name}")
        pairs.append((scene, mask))
    masks = {path.name for path in split_dir.glob("*.mask.tif")}
    expected_masks = {mask.name for _scene, mask in pairs}
    orphan_masks = sorted(masks - expected_masks)
    if orphan_masks:
        raise ValueError(f"Masks without matching scenes: {orphan_masks[:5]}")
    return pairs


def _inspect_pair(scene_path: Path, mask_path: Path) -> dict[str, Any]:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "rasterio is required only to validate the acquired HLS burn-scar corpus"
        ) from exc

    with rasterio.open(scene_path) as scene, rasterio.open(mask_path) as mask:
        if scene.width != 512 or scene.height != 512 or scene.count != 6:
            raise ValueError(f"Unexpected HLS scene shape for {scene_path.name}")
        if mask.width != 512 or mask.height != 512 or mask.count != 1:
            raise ValueError(f"Unexpected HLS mask shape for {mask_path.name}")
        if scene.transform != mask.transform or scene.crs != mask.crs:
            raise ValueError(f"Scene/mask georeference mismatch for {scene_path.name}")
        values = mask.read(1, masked=False)
        invalid_values = set(values.flatten().tolist()) - {-1, 0, 1}
        if invalid_values:
            raise ValueError(
                f"Unexpected mask values for {mask_path.name}: {sorted(invalid_values)}"
            )
        return {
            "width": scene.width,
            "height": scene.height,
            "crs": scene.crs.to_string() if scene.crs else None,
            "bounds": [
                scene.bounds.left,
                scene.bounds.bottom,
                scene.bounds.right,
                scene.bounds.top,
            ],
            "burned_pixels": int((values == 1).sum()),
            "not_burned_pixels": int((values == 0).sum()),
            "nodata_pixels": int((values == -1).sum()),
        }


def build_hls_manifest(source_root: Path, output_dir: Path) -> dict[str, Any]:
    """Write a segmentation-only manifest without mixing it into RT-DETR classes."""

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to merge into non-empty output: {output_dir}")
    split_sources = {"train": source_root / "training", "validation": source_root / "validation"}
    for directory in split_sources.values():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    output_dir.mkdir(parents=True, exist_ok=False)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    for split, split_dir in split_sources.items():
        for scene_path, mask_path in paired_scene_files(split_dir):
            metadata = _inspect_pair(scene_path, mask_path)
            scene_sha256 = sha256_file(scene_path)
            mask_sha256 = sha256_file(mask_path)
            image_relative = Path("images") / f"{scene_sha256}.tif"
            mask_relative = Path("masks") / f"{mask_sha256}.tif"
            image_destination = output_dir / image_relative
            mask_destination = output_dir / mask_relative
            # Hard links keep the five-gigabyte source only once on the same D: volume.
            os.link(scene_path, image_destination)
            os.link(mask_path, mask_destination)
            if sha256_file(image_destination) != scene_sha256:
                raise ValueError(f"Promoted scene hash mismatch for {scene_path.name}")
            if sha256_file(mask_destination) != mask_sha256:
                raise ValueError(f"Promoted mask hash mismatch for {mask_path.name}")
            records.append(
                {
                    "sample_id": f"{SOURCE_ID}:{scene_sha256[:24]}",
                    "source_id": SOURCE_ID,
                    "source_record_id": scene_path.stem,
                    "corpus_role": "burned_area_segmentation_training",
                    "split": split,
                    "split_group": f"{SOURCE_ID}:{scene_path.stem}",
                    "image_relpath": image_relative.as_posix(),
                    "mask_relpath": mask_relative.as_posix(),
                    "source_scene_path": scene_path.as_posix(),
                    "source_mask_path": mask_path.as_posix(),
                    "sha256": scene_sha256,
                    "mask_sha256": mask_sha256,
                    "mask_class": TARGET_CLASS,
                    "mask_values": {"burned": 1, "not_burned": 0, "ignore": -1},
                    "annotations": [],
                    "raster": metadata,
                    "source_asset": {
                        "dataset": SOURCE_REPOSITORY,
                        "dataset_url": SOURCE_URL,
                        "revision": SOURCE_REVISION,
                        "license": SOURCE_LICENSE,
                        "attribution_required": True,
                        "sensor": "Harmonized Landsat and Sentinel-2",
                        "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
                    },
                }
            )
            split_counts[split] += 1

    if not records:
        raise ValueError("No HLS scene/mask pairs found")
    if split_counts["train"] == 0 or split_counts["validation"] == 0:
        raise ValueError("HLS corpus must retain its supplied train and validation splits")

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "rows": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "target": {
            "kind": "binary_segmentation",
            "class_name": TARGET_CLASS,
            "detector_class_set_changed": False,
        },
        "training_policy": {
            "eligible_for_rtdetr_training": False,
            "eligible_for_burned_area_segmentation": True,
            "eligible_for_active_fire_boundary_inference": False,
        },
    }
    (output_dir / "acquisition-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def validate_hls_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.jsonl"
    split_counts: Counter[str] = Counter()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        image_path = (output_dir / str(record["image_relpath"])).resolve()
        mask_path = (output_dir / str(record["mask_relpath"])).resolve()
        if output_dir not in image_path.parents or output_dir not in mask_path.parents:
            raise ValueError(f"Manifest path escapes corpus at line {line_number}")
        if sha256_file(image_path) != record["sha256"]:
            raise ValueError(f"Image hash mismatch at line {line_number}")
        if sha256_file(mask_path) != record["mask_sha256"]:
            raise ValueError(f"Mask hash mismatch at line {line_number}")
        _inspect_pair(image_path, mask_path)
        split_counts[str(record["split"])] += 1
    if split_counts != Counter({"train": 540, "validation": 264}):
        raise ValueError(f"Unexpected HLS split counts: {dict(split_counts)}")
    return {"rows": sum(split_counts.values()), "split_counts": dict(sorted(split_counts.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a NASA HLS burn-scar corpus")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        report = validate_hls_manifest(args.output)
    else:
        report = build_hls_manifest(args.source, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
