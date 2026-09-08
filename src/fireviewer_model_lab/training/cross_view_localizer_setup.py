"""Acquire the licensed coarse cross-view bootstrap without downloading the full corpus.

The resulting corpus trains only the first, coarse localization stage.  It cannot validate
dense RoMa correspondences and it is never accepted as the rural/mountain promotion gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fireviewer_model_lab.training.spatial_training_setup import (
    SetupError,
    _deny_operational_path,
    _require_production_license,
    _sha256_file,
    _write_json,
)

DATASET_REPO_ID = "pcvlab/justzoomin"
DATASET_REPO_REVISION = "349f392b3620e7cea2596b8ade47b726786f9a1d"
DATASET_LICENSE = "CC-BY-SA-4.0"
CORPUS_ID = "cross-view-coarse-localizer-v0.1.0"
GROUND_IMAGE_SUFFIX = "_undistorted.jpg"
REQUIRED_SATELLITE_LEVELS = (-8, -6, -4, -2)


@dataclass(frozen=True)
class AcquisitionArtifact:
    relative_path: str
    size: int
    sha256: str | None = None
    extract: bool = False


SELECTED_ARTIFACTS = (
    AcquisitionArtifact("README.md", 7_917),
    AcquisitionArtifact("metadata/large_area_train_map.csv", 17_954_291),
    AcquisitionArtifact("metadata/large_area_val_map.csv", 1_994_447),
    AcquisitionArtifact("satellite/layout.yaml", 179),
    AcquisitionArtifact(
        "archives/streetview_images_000.tar",
        5_438_074_880,
        "54dba44c2be943cb59efa53b1aaf0c7a133242c3334ee1dd10b6a70f45973195",
        True,
    ),
    AcquisitionArtifact(
        "archives/satellite_level_m2_000.tar",
        1_221_888_000,
        "1c3ea4d67567d0db9645d38eafff4ae5e9274ea5fe477ca029a0d4326d95a662",
        True,
    ),
    AcquisitionArtifact(
        "archives/satellite_level_m4_000.tar",
        86_394_880,
        "b81af490daf9debc6ad526fac9fe45b09b05dacf06a8cf8b4890c9008acb820a",
        True,
    ),
    AcquisitionArtifact(
        "archives/satellite_level_m6_000.tar",
        5_990_400,
        "db8565712592c154b70fff5ba1aadf1192e3b52013f1f77b35f25017ae744967",
        True,
    ),
    AcquisitionArtifact(
        "archives/satellite_level_m8_000.tar",
        389_120,
        "9da77b3a48ed8b2ae659120708ce10e170757ec96503f9653f81644b024e4c4d",
        True,
    ),
)


def _source_root(dataset_root: Path) -> Path:
    root = dataset_root.resolve() / "sources" / "justzoomin-selective"
    _deny_operational_path(root)
    return root


def _artifact_state(source_root: Path, artifact: AcquisitionArtifact) -> dict[str, Any]:
    path = source_root / "repository" / artifact.relative_path
    present = path.is_file()
    actual_size = path.stat().st_size if present else 0
    valid_size = present and actual_size == artifact.size
    return {
        "relative_path": artifact.relative_path,
        "expected_bytes": artifact.size,
        "present_bytes": actual_size,
        "present": present,
        "valid_size": valid_size,
        "extract": artifact.extract,
    }


def plan_acquisition(dataset_root: Path) -> dict[str, Any]:
    """Return the exact selective payload without mutating the dataset."""

    _require_production_license(DATASET_LICENSE, source_id="justzoomin_selective_v1")
    source_root = _source_root(dataset_root)
    files = [_artifact_state(source_root, artifact) for artifact in SELECTED_ARTIFACTS]
    planned_bytes = sum(artifact.size for artifact in SELECTED_ARTIFACTS)
    remaining_bytes = sum(item["expected_bytes"] for item in files if not item["valid_size"])
    return {
        "schema_version": 1,
        "source_id": "justzoomin_selective_v1",
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REPO_REVISION,
        "declared_license": DATASET_LICENSE,
        "planned_payload_bytes": planned_bytes,
        "remaining_download_bytes": remaining_bytes,
        "full_dataset_downloaded": False,
        "selected_artifacts": files,
        "training_role": "coarse_cross_view_localization_only",
        "dense_matching_training_role": False,
        "production_promotion_gate": False,
    }


def _validate_download(path: Path, artifact: AcquisitionArtifact) -> str:
    if not path.is_file() or path.stat().st_size != artifact.size:
        actual = path.stat().st_size if path.is_file() else None
        raise SetupError(
            f"JustZoomIn artifact size mismatch: {artifact.relative_path} "
            f"expected={artifact.size} actual={actual}"
        )
    actual_sha256 = _sha256_file(path)
    if artifact.sha256 is not None and actual_sha256 != artifact.sha256:
        raise SetupError(f"JustZoomIn artifact SHA-256 mismatch: {artifact.relative_path}")
    return actual_sha256


def _safe_extract_tar(archive: Path, destination: Path) -> dict[str, Any]:
    """Extract regular files only and reject traversal, links and device entries."""

    archive_sha256 = _sha256_file(archive)
    marker = destination / ".firewarning-extracted" / f"{archive.name}.{archive_sha256}.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))

    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    extracted_files = 0
    extracted_bytes = 0
    with tarfile.open(archive, "r") as handle:
        for member in handle:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise SetupError(f"unsafe archive path: {member.name}")
            target = (destination / relative).resolve()
            if target != resolved_destination and resolved_destination not in target.parents:
                raise SetupError(f"archive path escapes extraction root: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise SetupError(f"unsupported archive entry: {member.name}")
            source = handle.extractfile(member)
            if source is None:
                raise SetupError(f"archive file cannot be read: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            try:
                with source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.replace(partial, target)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
            extracted_files += 1
            extracted_bytes += target.stat().st_size

    report = {
        "archive": archive.name,
        "archive_sha256": archive_sha256,
        "extracted_files": extracted_files,
        "extracted_bytes": extracted_bytes,
    }
    _write_json(marker, report)
    return report


def acquire(dataset_root: Path) -> dict[str, Any]:
    """Download and extract only the pinned coarse-localizer subset."""

    source_root = _source_root(dataset_root)
    repository_root = source_root / "repository"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - explicit runtime dependency
        raise SetupError("huggingface-hub is required for source acquisition") from exc

    downloaded: list[dict[str, Any]] = []
    extraction: list[dict[str, Any]] = []
    for artifact in SELECTED_ARTIFACTS:
        path = Path(
            hf_hub_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                revision=DATASET_REPO_REVISION,
                filename=artifact.relative_path,
                local_dir=repository_root,
            )
        )
        actual_sha256 = _validate_download(path, artifact)
        downloaded.append(
            {
                "relative_path": artifact.relative_path,
                "bytes": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
        if artifact.extract:
            extraction.append(_safe_extract_tar(path, source_root / "extracted"))

    layout_source = repository_root / "satellite" / "layout.yaml"
    layout_destination = source_root / "extracted" / "satellite" / "layout.yaml"
    layout_destination.parent.mkdir(parents=True, exist_ok=True)
    layout_temporary = layout_destination.with_suffix(".yaml.partial")
    shutil.copyfile(layout_source, layout_temporary)
    os.replace(layout_temporary, layout_destination)

    report = plan_acquisition(dataset_root)
    report.update(
        {
            "acquisition_complete": report["remaining_download_bytes"] == 0,
            "downloaded_artifacts": downloaded,
            "extraction": extraction,
            "attribution": [
                "Street-view imagery derived from Mapillary, CC BY-SA 4.0.",
                (
                    "Aerial orthophotography derived from Open Data DC / Government of "
                    "the District of Columbia, CC BY 4.0."
                ),
            ],
        }
    )
    _write_json(source_root / "firewarning-acquisition-report.json", report)
    return report


def _parse_action_sequence(value: str) -> tuple[int, int, int, int]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise SetupError(f"invalid cross-view action sequence: {value}") from exc
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise SetupError(f"cross-view action sequence must contain four steps: {value}")
    actions = tuple(int(item) for item in parsed)
    if any(action < 0 or action > 15 for action in actions):
        raise SetupError(f"cross-view action outside 4x4 grid: {value}")
    return actions[0], actions[1], actions[2], actions[3]


def _write_manifest(
    metadata_path: Path,
    dataset_root: Path,
    ground_root: Path,
    available_image_ids: set[str],
    output_path: Path,
    *,
    split: str,
) -> tuple[int, set[str]]:
    rows = 0
    image_ids: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with (
        metadata_path.open(encoding="utf-8", newline="") as source,
        temporary.open("w", encoding="utf-8", newline="\n") as destination,
    ):
        for source_row in csv.DictReader(source):
            image_id = str(source_row["image_id"])
            if image_id not in available_image_ids:
                continue
            image_path = ground_root / f"{image_id}{GROUND_IMAGE_SUFFIX}"
            if not image_path.is_file():
                raise SetupError(f"indexed JustZoomIn image is missing: {image_path}")
            actions = _parse_action_sequence(str(source_row["sequence"]))
            row = {
                "schema_version": "1.0",
                "family": "coarse_cross_view_localization",
                "sample_id": f"justzoomin:{image_id}",
                "source_id": "justzoomin_selective_v1",
                "source_revision": DATASET_REPO_REVISION,
                "source_view_relpath": image_path.relative_to(dataset_root).as_posix(),
                "source_view_sha256": _sha256_file(image_path),
                "latitude": float(source_row["latitude"]),
                "longitude": float(source_row["longitude"]),
                "action_sequence": list(actions),
                "satellite_levels": list(REQUIRED_SATELLITE_LEVELS),
                "split": split,
                "split_basis": "upstream_bootstrap_split",
                "license": DATASET_LICENSE,
                "training_membership": True,
                "critical_test_membership": False,
                "production_promotion_gate": False,
                "operational_incident": False,
            }
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            image_ids.add(image_id)
            rows += 1
    os.replace(temporary, output_path)
    return rows, image_ids


def prepare(
    dataset_root: Path,
    *,
    minimum_train_rows: int = 10_000,
    minimum_validation_rows: int = 1_000,
) -> dict[str, Any]:
    """Build reference-only manifests after the selected archives were extracted."""

    source_root = _source_root(dataset_root)
    repository_root = source_root / "repository"
    extracted_root = source_root / "extracted"
    ground_root = extracted_root / "streetview" / "images"
    for level in REQUIRED_SATELLITE_LEVELS:
        if not (extracted_root / "satellite" / str(level)).is_dir():
            raise SetupError(f"missing selected satellite level: {level}")
    if not ground_root.is_dir():
        raise SetupError("selected JustZoomIn ground archive is not extracted")
    available_image_ids = {
        path.name.removesuffix(GROUND_IMAGE_SUFFIX)
        for path in ground_root.iterdir()
        if path.is_file() and path.name.endswith(GROUND_IMAGE_SUFFIX)
    }
    if not available_image_ids:
        raise SetupError("selected JustZoomIn ground archive contains no usable images")

    output_root = dataset_root.resolve() / "corpus" / CORPUS_ID
    train_rows, train_ids = _write_manifest(
        repository_root / "metadata" / "large_area_train_map.csv",
        dataset_root.resolve(),
        ground_root,
        available_image_ids,
        output_root / "train.jsonl",
        split="train",
    )
    validation_rows, validation_ids = _write_manifest(
        repository_root / "metadata" / "large_area_val_map.csv",
        dataset_root.resolve(),
        ground_root,
        available_image_ids,
        output_root / "validation.jsonl",
        split="validation",
    )
    overlap = train_ids & validation_ids
    gates = {
        "minimum_train_rows": train_rows >= minimum_train_rows,
        "minimum_validation_rows": validation_rows >= minimum_validation_rows,
        "no_image_id_split_overlap": not overlap,
        "required_satellite_levels_present": True,
        "production_license_allowed": DATASET_LICENSE == "CC-BY-SA-4.0",
    }
    report = {
        "schema_version": 1,
        "corpus_id": CORPUS_ID,
        "source_id": "justzoomin_selective_v1",
        "source_revision": DATASET_REPO_REVISION,
        "rows": {"train": train_rows, "validation": validation_rows},
        "gates": gates,
        "bootstrap_training_ready": all(gates.values()),
        "production_training_ready": False,
        "production_promotion_gate": False,
        "blockers_before_production_promotion": [
            "bootstrap geography is limited to Washington DC",
            "France rural and mountain domain adaptation is not acquired",
            "independent double-validated geographic critical test is missing",
        ],
        "training_launched": False,
    }
    _write_json(output_root / "build-report.json", report)
    if not report["bootstrap_training_ready"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise SetupError(f"coarse localizer bootstrap is incomplete: {failed}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "acquire", "prepare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "plan":
        report = plan_acquisition(args.dataset_root)
    elif args.command == "acquire":
        report = acquire(args.dataset_root)
    elif args.command == "prepare":
        report = prepare(args.dataset_root)
    else:  # pragma: no cover - argparse rejects unknown commands
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
