from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from dataset_archive_validation import validate_firewarning_dataset
from finalize_train_bundle import (
    _normalize_boreal_detection,
    _validate_prithvi_materialized_dataset,
)


@dataclass(frozen=True)
class BundleContract:
    train_id: str
    filename: str
    size_bytes: int
    sha256: str
    prefixes: tuple[str, ...]


DFINE_BUNDLE = BundleContract(
    train_id="wildfire-smoke-detection-v1",
    filename="wildfire-smoke-detection-v1.zip",
    size_bytes=13_518_198_597,
    sha256="27abdbe3d3703d9c6fa67bfde136088845726313e148644eb0e948f9a6211e13",
    prefixes=("sources/boreal-forest-fire-detection-v1",),
)
PRITHVI_BUNDLE = BundleContract(
    train_id="burned-area-segmentation-v1",
    filename="burned-area-segmentation-v1.zip",
    size_bytes=27_926_097_197,
    sha256="85c2f17248528ebbd5aa8395e72435ba5a12626bb5a53f5730109b11ea5dde36",
    prefixes=("materialized",),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(path: Path, contract: BundleContract) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = path.stat().st_size
    if observed_size != contract.size_bytes:
        raise ValueError(
            f"Bundle size mismatch for {path.name}: {observed_size} != {contract.size_bytes}"
        )
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != contract.sha256:
        raise ValueError(f"Bundle SHA-256 mismatch for {path.name}: {observed_sha256}")
    return {
        "filename": path.name,
        "size_bytes": observed_size,
        "sha256": observed_sha256,
    }


def _safe_archive_relative(name: str, contract: BundleContract) -> Path | None:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP entry: {name}")
    if not relative.parts or relative.parts[0] != contract.train_id:
        raise ValueError(f"ZIP entry outside the expected train root: {name}")
    payload = PurePosixPath(*relative.parts[1:])
    if not payload.parts:
        return None
    payload_text = payload.as_posix()
    if not any(
        payload_text == prefix or payload_text.startswith(prefix + "/")
        for prefix in contract.prefixes
    ):
        return None
    return Path(*payload.parts)


def extract_required_payloads(
    archive_path: Path,
    destination: Path,
    contract: BundleContract,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(
            f"Interrupted staging directory must be reviewed before retry: {staging}"
        )
    staging.mkdir(parents=True)
    extracted_files = 0
    extracted_bytes = 0
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path, mode="r", allowZip64=True) as archive:
        for info in archive.infolist():
            target_relative = _safe_archive_relative(info.filename, contract)
            if target_relative is None or info.is_dir():
                continue
            normalized = target_relative.as_posix()
            if normalized in seen:
                raise ValueError(f"Duplicate ZIP entry selected for extraction: {normalized}")
            seen.add(normalized)
            target = staging / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, mode="r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if target.stat().st_size != info.file_size:
                raise ValueError(f"Extracted size mismatch: {info.filename}")
            extracted_files += 1
            extracted_bytes += info.file_size
    if not extracted_files:
        raise ValueError(f"No required payload was found in {archive_path}")
    os.replace(staging, destination)
    return {
        "extracted_files": extracted_files,
        "extracted_bytes": extracted_bytes,
        "prefixes": list(contract.prefixes),
    }


def prepare_dfine(bundle_dir: Path, destination: Path) -> dict[str, Any]:
    contract = DFINE_BUNDLE
    archive = bundle_dir / contract.filename
    bundle = verify_bundle(archive, contract)
    existing_report = destination / "preparation-report.json"
    if existing_report.is_file():
        report = json.loads(existing_report.read_text(encoding="utf-8"))
        if report.get("bundle") != bundle:
            raise ValueError("Existing D-FINE preparation does not match the public bundle")
        boreal = destination / contract.prefixes[0]
        report["validation"] = validate_firewarning_dataset(boreal)
        report["reused_existing_preparation"] = True
        return report
    extraction = extract_required_payloads(archive, destination, contract)
    boreal = destination / contract.prefixes[0]
    normalization = _normalize_boreal_detection(boreal)
    validation = validate_firewarning_dataset(boreal)
    report = {
        "schema_version": 1,
        "campaign": "dfine-fire-smoke-v1",
        "bundle": bundle,
        "extraction": extraction,
        "normalization": normalization,
        "validation": validation,
        "training_manifest": str((boreal / "manifest.jsonl").resolve()),
    }
    (destination / "preparation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def prepare_prithvi(bundle_dir: Path, destination: Path) -> dict[str, Any]:
    contract = PRITHVI_BUNDLE
    archive = bundle_dir / contract.filename
    bundle = verify_bundle(archive, contract)
    materialized = destination / "materialized"
    existing_report = destination / "preparation-report.json"
    if existing_report.is_file():
        report = json.loads(existing_report.read_text(encoding="utf-8"))
        if report.get("bundle") != bundle:
            raise ValueError("Existing Prithvi preparation does not match the public bundle")
        report["validation"] = _validate_prithvi_materialized_dataset(
            materialized,
            expected_samples=32_534,
            expected_source_counts={"hls": 804, "eo4": 31_730},
        )
        report["reused_existing_preparation"] = True
        return report
    extraction = extract_required_payloads(archive, destination, contract)
    validation = _validate_prithvi_materialized_dataset(
        materialized,
        expected_samples=32_534,
        expected_source_counts={"hls": 804, "eo4": 31_730},
    )
    report = {
        "schema_version": 1,
        "campaign": "prithvi-burnscars-v1",
        "bundle": bundle,
        "extraction": extraction,
        "training_dataset_root": str(materialized.resolve()),
        "validation": validation,
    }
    (destination / "preparation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract verified public payloads for the frozen D-FINE and Prithvi campaigns"
    )
    parser.add_argument("campaign", choices=("dfine", "prithvi"))
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    if args.campaign == "dfine":
        report = prepare_dfine(args.bundle_dir.resolve(), args.destination.resolve())
    else:
        report = prepare_prithvi(args.bundle_dir.resolve(), args.destination.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
