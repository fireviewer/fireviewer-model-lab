"""Build the two independent FireWarning spatial-training corpus families.

The fire-pointing manifest references existing media by hash and path.  It never copies the
155k source images.  The cross-view manifest is built from the historical, public
AerialExtreMatch-Localization set and never accepts a live FireWarning spatial package.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

DATASET_REPO_ID = "Xecades/AerialExtreMatch-Localization"
DATASET_REPO_REVISION = "b70225c2fd468976d5f9fc9bf435645da4184492"
PRODUCTION_LICENSE_ALLOWLIST = frozenset(
    {
        "Apache-2.0",
        "CC0-1.0",
        "CC-BY-2.0",
        "CC-BY-2.5",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-2.0",
        "CC-BY-SA-2.5",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "Licence-Ouverte-2.0",
        "MIT",
        "PDM-1.0",
        "Public-Domain",
        "FireViewer-Synthetic-1.0",
    }
)
DENIED_LICENSE_TOKENS = ("-NC-", "NONCOMMERCIAL", "NON-COMMERCIAL")
ODM_DOMAIN_SOURCES = (
    {
        "key": "seneca",
        "source_id": "odm_seneca_rural",
        "domain": "rural_farmland",
        "repository": "https://github.com/OpenDroneMap/odm_data_seneca",
        "revision": "9220acfdca75cc5e2dd6d27fe48550c5ae0877f2",
        "license": "CC0-1.0",
        "license_filename": "license.txt",
        "license_marker": "CC0 1.0 Universal",
        "hub_api": "https://hub.dronedb.app/orgs/odm/ds/seneca",
        "orthophoto_path": "odm_orthophoto/odm_orthophoto.tif",
        "orthophoto_sha256": ("f718f3f1496b961dfe04e33c252921f4e8f499aea041bba1c43ac3bb65ebe177"),
        "selection_limit": 64,
    },
    {
        "key": "sance",
        "source_id": "odm_sance_mountain",
        "domain": "mountain_pass",
        "repository": "https://github.com/merkato/odm_sance",
        "revision": "29d0ffd933321dd64aad6bfd108ffede5271b9b8",
        "license": "CC-BY-SA-4.0",
        "license_filename": "LICENSE.md",
        "license_marker": "CC-BY-SA 4.0",
        "hub_api": "https://hub.dronedb.app/orgs/odm/ds/sance",
        "orthophoto_path": "odm_orthophoto/odm_orthophoto.tif",
        "orthophoto_sha256": ("430298d2ac77b7b3fb14f523c5f18e724a37f1979fbcf7333aff568a0bea2c2f"),
        "selection_limit": 64,
        # These edge views cannot yield a crop that keeps both the camera and the
        # optical-axis target inside at least 95% valid orthophoto pixels.
        "selection_exclusions": ("images/dji_0031.jpg", "images/dji_0156.jpg"),
    },
)
ALLOWED_ACQUISITION_HOSTS = frozenset({"hub.dronedb.app", "raw.githubusercontent.com"})
POINTING_CORPUS_ID = "fire-pointing-v0.1.0"
REGISTRATION_CORPUS_ID = "cross-view-registration-v0.1.0"
POINTING_CRITICAL_CORPUS_ID = "fire-pointing-critical-v0.1.0"
REGISTRATION_CRITICAL_CORPUS_ID = "cross-view-registration-critical-v0.1.0"
DENIED_OPERATIONAL_TOKENS = (
    "fireviewer-operational-reference",
    "operational-reference-a",
    "operational-reference-a",
)

POINTING_SOURCES = (
    ("fasdd_v9", "corpus/fasdd/manifest.jsonl"),
    ("pyro_sdis_v0_1_0", "corpus/pyro-sdis-v0.1.0/manifest.jsonl"),
    ("wikimedia_candidates_v0_1_0", "corpus/wikimedia-candidates-v0.1.0/manifest.jsonl"),
)

ANCHORS = {
    "flame_visible": "fire_base",
    "smoke_visible": "smoke_column_base",
    "fire_response_vehicle_visible": "fire_response_vehicle",
    "firefighting_aircraft_visible": "firefighting_aircraft",
}


class SetupError(RuntimeError):
    """Raised when source data violates the frozen corpus contract."""


@dataclass(frozen=True)
class CameraRecord:
    name: str
    quaternion_wxyz: tuple[float, float, float, float]
    translation_w2c: tuple[float, float, float]
    center_xyz: tuple[float, float, float]
    intrinsic: tuple[float, float, float, float, int, int]


@dataclass(frozen=True)
class RayIntersection:
    xyz: tuple[float, float, float]
    surface_z: float
    vertical_residual_m: float
    raster_row: float
    raster_column: float
    distance: float


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_within(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise SetupError(f"path escapes corpus root: {relative}")
    return candidate


def _deny_operational_path(path: Path | str) -> None:
    normalized = str(path).replace("\\", "/").lower()
    for token in DENIED_OPERATIONAL_TOKENS:
        if token in normalized:
            raise SetupError(f"operational incident source denied for training: {token}")


def _require_production_license(license_id: str, *, source_id: str) -> None:
    normalized = license_id.strip()
    upper = normalized.upper()
    if any(token in upper for token in DENIED_LICENSE_TOKENS):
        raise SetupError(
            f"non-commercial license denied for production training: {source_id}={normalized}"
        )
    if normalized not in PRODUCTION_LICENSE_ALLOWLIST:
        raise SetupError(
            f"license is not allowlisted for free production/open-source use: "
            f"{source_id}={normalized}"
        )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SetupError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SetupError(f"manifest row is not an object at {path}:{line_number}")
            yield value


def _normalized_bottom_center(
    annotation: dict[str, Any], width: int, height: int
) -> tuple[float, float]:
    bbox = annotation.get("bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise SetupError("pointing source annotation is missing bbox_xywh")
    x, y, box_width, box_height = (float(item) for item in bbox)
    if width <= 0 or height <= 0 or box_width < 0 or box_height < 0:
        raise SetupError("invalid image or bounding-box dimensions")
    point_x = min(1.0, max(0.0, (x + box_width / 2.0) / width))
    point_y = min(1.0, max(0.0, (y + box_height) / height))
    return round(point_x, 8), round(point_y, 8)


def _pointing_targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    width = int(row["width"])
    height = int(row["height"])
    targets: list[dict[str, Any]] = []
    for annotation_index, annotation in enumerate(row.get("annotations", [])):
        class_name = annotation.get("class_name")
        if class_name not in ANCHORS:
            continue
        targets.append(
            {
                "target_id": f"bbox-{annotation_index}",
                "semantic_anchor": ANCHORS[class_name],
                "source_pixel_normalized": list(
                    _normalized_bottom_center(annotation, width, height)
                ),
                "label_origin": "bbox_bottom_center_weak",
                "validation_status": "candidate_unreviewed",
            }
        )
    return targets


def build_pointing(dataset_root: Path, *, verify_files: bool = False) -> dict[str, Any]:
    """Build a reference-only pointing manifest from the three existing media corpora."""

    dataset_root = dataset_root.resolve()
    output_root = dataset_root / "corpus" / POINTING_CORPUS_ID
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    partial_path = output_root / "manifest.partial.jsonl"

    rows = 0
    missing_files = 0
    verified_files = 0
    status_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    trainable_split_counts: Counter[str] = Counter()

    with partial_path.open("w", encoding="utf-8", newline="\n") as output:
        for source_name, manifest_relative in POINTING_SOURCES:
            source_manifest = _resolve_within(dataset_root, manifest_relative)
            if not source_manifest.is_file():
                raise SetupError(f"missing pointing source manifest: {source_manifest}")
            source_directory = source_manifest.parent
            for source_row in _iter_jsonl(source_manifest):
                _require_production_license(
                    str(source_row["license"]), source_id=str(source_row["source_id"])
                )
                image_relative = str(source_row["image_relpath"])
                source_image = _resolve_within(source_directory, image_relative)
                if not source_image.is_file():
                    missing_files += 1
                    continue
                if verify_files:
                    actual_sha256 = _sha256_file(source_image)
                    if actual_sha256 != source_row["sha256"]:
                        raise SetupError(f"source image SHA-256 mismatch: {source_image}")
                    verified_files += 1

                targets = _pointing_targets(source_row)
                if targets:
                    status = "point_candidate"
                    eligibility = "weak_supervision_only_until_point_validation"
                elif "no_target_visible" in source_row.get("negative_tags", []):
                    status = "non_fire_negative_candidate"
                    eligibility = "excluded_until_abstention_review"
                else:
                    status = "annotation_candidate"
                    eligibility = "requires_point_annotation"

                dataset_relative_image = source_image.relative_to(dataset_root).as_posix()
                pointing_row = {
                    "schema_version": "1.0",
                    "family": "fire_pointing",
                    "sample_id": f"pointing:{source_name}:{source_row['sample_id']}",
                    "source": {
                        "source_id": source_row["source_id"],
                        "source_record_id": source_row["source_record_id"],
                        "source_manifest_relpath": manifest_relative,
                        "image_relpath": dataset_relative_image,
                        "sha256": source_row["sha256"],
                        "width": source_row["width"],
                        "height": source_row["height"],
                        "license": source_row["license"],
                        "consent_basis": source_row["consent_basis"],
                    },
                    "event_id": source_row["event_id"],
                    "sequence_id": source_row["sequence_id"],
                    "split_group": source_row["split_group"],
                    "proposed_split": source_row["split"],
                    "pointing_status": status,
                    "training_eligibility": eligibility,
                    "targets": targets,
                    "human_validation": {
                        "required": True,
                        "minimum_validators": 2,
                        "completed_validators": 0,
                    },
                }
                output.write(json.dumps(pointing_row, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
                status_counts[status] += 1
                split_counts[str(source_row["split"])] += 1
                source_counts[source_name] += 1
                for target in targets:
                    target_counts[str(target["semantic_anchor"])] += 1
                if targets:
                    trainable_split_counts[str(source_row["split"])] += 1

    if missing_files:
        raise SetupError(f"{missing_files} pointing source images are missing")
    os.replace(partial_path, manifest_path)

    report = {
        "schema_version": 1,
        "family": "fire_pointing",
        "corpus_id": POINTING_CORPUS_ID,
        "manifest_relpath": manifest_path.relative_to(dataset_root).as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "rows": rows,
        "media_files_copied": 0,
        "verified_source_files": verified_files,
        "verification_mode": "sha256" if verify_files else "existence_and_manifest_hash_reference",
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "weak_target_counts": dict(sorted(target_counts.items())),
        "weakly_trainable_split_counts": dict(sorted(trainable_split_counts.items())),
        "gates": {
            "setup_ready": rows > 0,
            "smoke_ready": trainable_split_counts["train"] >= 8
            and trainable_split_counts["validation"] >= 4,
            "bootstrap_training_ready": trainable_split_counts["train"] >= 1_000
            and trainable_split_counts["validation"] >= 100,
            "training_ready": trainable_split_counts["train"] >= 1_000
            and trainable_split_counts["validation"] >= 100,
            "production_training_ready": False,
            "deployment_ready": False,
        },
        "blockers": [
            (
                "bbox bottom-centres permit weak-supervision bootstrap but are not "
                "validated point labels"
            ),
            "non-fire negatives are not valid insufficient-geometry examples until human review",
            "no double-validated fire-pointing critical test exists",
        ],
    }
    _write_json(output_root / "build-report.json", report)
    return report


def download_registration_source(dataset_root: Path) -> dict[str, Any]:
    """Download only the useful 1.5 GB subset; the 11.7 GB HQ references are excluded."""

    dataset_root = dataset_root.resolve()
    _require_production_license("MIT", source_id="aerialextrematch_localization")
    source_root = dataset_root / "sources" / "aerialextrematch-localization"
    _deny_operational_path(source_root)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit at runtime
        raise SetupError("huggingface-hub is required for source acquisition") from exc

    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=DATASET_REPO_REVISION,
        local_dir=source_root,
        allow_patterns=[
            "README.md",
            "gt_pose.txt",
            "pose.txt",
            "intrinsic.txt",
            "rgb/*",
            "LQref/*",
        ],
        ignore_patterns=["HQref/*"],
    )

    expected = (
        "README.md",
        "gt_pose.txt",
        "pose.txt",
        "intrinsic.txt",
        "LQref/DOM.tif",
        "LQref/DOM.prj",
        "LQref/DOM.tfw",
        "LQref/DSM.tif",
        "LQref/DSM.prj",
        "LQref/DSM.tfw",
    )
    missing = [relative for relative in expected if not (source_root / relative).is_file()]
    image_count = sum(1 for _ in (source_root / "rgb").glob("*.JPG"))
    if missing or image_count != 264:
        raise SetupError(f"incomplete registration source: missing={missing}, images={image_count}")

    payload_files = [path for path in source_root.rglob("*") if path.is_file()]
    payload_bytes = sum(path.stat().st_size for path in payload_files)
    report = {
        "source_id": "aerialextrematch_localization",
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REPO_REVISION,
        "license_declared_by_dataset_card": "MIT",
        "image_count": image_count,
        "payload_bytes": payload_bytes,
        "hq_reference_downloaded": False,
        "source_root_relpath": source_root.relative_to(dataset_root).as_posix(),
    }
    _write_json(source_root / "firewarning-acquisition-report.json", report)
    return report


def _download_https_file(
    client: Any,
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_ACQUISITION_HOSTS:
        raise SetupError(f"acquisition URL denied: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual_sha256 = _sha256_file(destination)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise SetupError(f"existing acquisition hash mismatch: {destination}")
        return {
            "bytes": destination.stat().st_size,
            "sha256": actual_sha256,
            "downloaded": False,
        }

    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with client.stream("GET", url) as response, partial.open("wb") as output:
            response.raise_for_status()
            for chunk in response.iter_bytes(1024 * 1024):
                output.write(chunk)
        actual_sha256 = _sha256_file(partial)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise SetupError(f"downloaded acquisition hash mismatch: {destination}")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "bytes": destination.stat().st_size,
        "sha256": actual_sha256,
        "downloaded": True,
    }


def _dronedb_listing(client: Any, hub_api: str, path: str) -> list[dict[str, Any]]:
    parsed = urlparse(hub_api)
    if parsed.scheme != "https" or parsed.hostname != "hub.dronedb.app":
        raise SetupError(f"DroneDB API URL denied: {hub_api}")
    response = client.post(f"{hub_api}/list", files={"path": (None, path)})
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise SetupError(f"unexpected DroneDB listing for {hub_api}:{path}")
    return payload


def _evenly_spaced_selection(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(entries, key=lambda item: str(item["path"]))
    if limit <= 0 or len(ordered) < limit:
        raise SetupError(f"requested {limit} records but only {len(ordered)} are available")
    if len(ordered) == limit:
        return ordered
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(indices)) != limit:
        raise SetupError("deterministic source selection produced duplicate indices")
    return [ordered[index] for index in indices]


def download_odm_domains(dataset_root: Path) -> dict[str, Any]:
    """Download two compact, production-compatible ODM cross-view domains."""

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit at runtime
        raise SetupError("install spatial-training dependencies before ODM acquisition") from exc

    dataset_root = dataset_root.resolve()
    source_root = dataset_root / "sources" / "odm-cross-view"
    _deny_operational_path(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    source_reports: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    total_bytes = 0
    downloaded_bytes = 0

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(300.0, connect=30.0),
        headers={"User-Agent": "FireWarningCorpus/0.1"},
    ) as client:
        for definition in ODM_DOMAIN_SOURCES:
            source_id = str(definition["source_id"])
            license_id = str(definition["license"])
            _require_production_license(license_id, source_id=source_id)
            key = str(definition["key"])
            source_directory = source_root / key
            hub_api = str(definition["hub_api"])
            revision = str(definition["revision"])
            license_filename = str(definition["license_filename"])
            repository = str(definition["repository"])
            repository_path = repository.removeprefix("https://github.com/")
            license_url = (
                f"https://raw.githubusercontent.com/{repository_path}/{revision}/"
                f"{quote(license_filename)}"
            )
            license_result = _download_https_file(
                client,
                license_url,
                source_directory / license_filename,
            )
            license_text = (source_directory / license_filename).read_text(encoding="utf-8")
            if str(definition["license_marker"]) not in license_text:
                raise SetupError(f"unexpected license text for {source_id}")

            root_entries = _dronedb_listing(client, hub_api, ".")
            root_by_path = {str(item["path"]): item for item in root_entries}
            metadata_files: dict[str, dict[str, Any]] = {}
            for metadata_path in ("README.md", "cameras.json", "images.json"):
                metadata_entry = root_by_path.get(metadata_path)
                if metadata_entry is None:
                    raise SetupError(f"missing DroneDB metadata: {source_id}/{metadata_path}")
                metadata_url = f"{hub_api}/download/{quote(metadata_path, safe='/')}?inline=1"
                metadata_result = _download_https_file(
                    client,
                    metadata_url,
                    source_directory / metadata_path,
                    expected_sha256=str(metadata_entry["hash"]),
                )
                metadata_files[metadata_path] = metadata_result

            image_entries = _dronedb_listing(client, hub_api, "images")
            eligible = [
                item
                for item in image_entries
                if item.get("hash")
                and item.get("point_geom")
                and item.get("polygon_geom")
                and isinstance(item.get("properties"), dict)
                and item["properties"].get("width")
                and item["properties"].get("height")
                and item["properties"].get("cameraPitch") is not None
            ]
            selected = _evenly_spaced_selection(eligible, int(definition["selection_limit"]))
            selection_exclusions = {
                str(path) for path in definition.get("selection_exclusions", ())
            }
            selected = [item for item in selected if str(item["path"]) not in selection_exclusions]

            orthophoto_entries = _dronedb_listing(client, hub_api, "odm_orthophoto")
            orthophoto_path = str(definition["orthophoto_path"])
            orthophoto_entry = next(
                (item for item in orthophoto_entries if item.get("path") == orthophoto_path),
                None,
            )
            if orthophoto_entry is None:
                raise SetupError(f"missing production orthophoto for {source_id}")
            expected_orthophoto_sha256 = str(definition["orthophoto_sha256"])
            if orthophoto_entry.get("hash") != expected_orthophoto_sha256:
                raise SetupError(f"DroneDB orthophoto snapshot changed for {source_id}")
            planned_large_bytes = 0
            if not (source_directory / orthophoto_path).is_file():
                planned_large_bytes += int(orthophoto_entry["size"])
            planned_large_bytes += sum(
                int(entry["size"])
                for entry in selected
                if not (source_directory / str(entry["path"])).is_file()
            )
            free_bytes = shutil.disk_usage(dataset_root).free
            reserve_bytes = 5 * 1024**3
            if free_bytes - planned_large_bytes < reserve_bytes:
                raise SetupError(
                    f"insufficient free space for {source_id}: need {planned_large_bytes} bytes "
                    f"while preserving a {reserve_bytes}-byte reserve"
                )
            orthophoto_result = _download_https_file(
                client,
                f"{hub_api}/download/{quote(orthophoto_path, safe='/')}?inline=1",
                source_directory / orthophoto_path,
                expected_sha256=expected_orthophoto_sha256,
            )

            source_records: list[dict[str, Any]] = []
            for index, entry in enumerate(selected, start=1):
                remote_path = str(entry["path"])
                local_path = source_directory / remote_path
                image_result = _download_https_file(
                    client,
                    f"{hub_api}/download/{quote(remote_path, safe='/')}?inline=1",
                    local_path,
                    expected_sha256=str(entry["hash"]),
                )
                record = {
                    "schema_version": 1,
                    "source_id": source_id,
                    "source_revision": revision,
                    "domain": definition["domain"],
                    "license": license_id,
                    "repository": repository,
                    "image": {
                        "relpath": local_path.relative_to(dataset_root).as_posix(),
                        "sha256": image_result["sha256"],
                        "bytes": image_result["bytes"],
                    },
                    "orthophoto": {
                        "relpath": (source_directory / orthophoto_path)
                        .relative_to(dataset_root)
                        .as_posix(),
                        "sha256": orthophoto_result["sha256"],
                    },
                    "dronedb_entry": entry,
                }
                source_records.append(record)
                all_records.append(record)
                if index % 16 == 0 or index == len(selected):
                    print(f"{source_id}: {index}/{len(selected)} images", flush=True)

            manifest_path = source_directory / "acquisition-manifest.jsonl"
            partial_path = manifest_path.with_suffix(".partial.jsonl")
            with partial_path.open("w", encoding="utf-8", newline="\n") as output:
                for record in source_records:
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(partial_path, manifest_path)

            retained_results = [license_result, orthophoto_result, *metadata_files.values()]
            retained_results.extend(
                {
                    "bytes": int(record["image"]["bytes"]),
                    "downloaded": False,
                }
                for record in source_records
            )
            source_bytes = sum(int(result["bytes"]) for result in retained_results)
            source_reports.append(
                {
                    "source_id": source_id,
                    "domain": definition["domain"],
                    "revision": revision,
                    "license": license_id,
                    "commercial_use_allowed": True,
                    "selection_count": len(source_records),
                    "available_eligible_images": len(eligible),
                    "payload_bytes": source_bytes,
                    "manifest_relpath": manifest_path.relative_to(dataset_root).as_posix(),
                    "manifest_sha256": _sha256_file(manifest_path),
                    "orthophoto_sha256": orthophoto_result["sha256"],
                    "attribution_required": license_id != "CC0-1.0",
                    "share_alike_required": license_id.endswith("BY-SA-4.0"),
                }
            )
            total_bytes += source_bytes

    for path in source_root.rglob("*"):
        if path.is_file() and path.name.endswith(".partial"):
            raise SetupError(f"partial acquisition residue: {path}")
    downloaded_bytes = sum(path.stat().st_size for path in source_root.rglob("*") if path.is_file())
    report = {
        "schema_version": 1,
        "source_family": "odm_cross_view_domains",
        "license_policy": "production_open_source_free_use_allowlist",
        "sources": source_reports,
        "sample_count": len(all_records),
        "payload_bytes_accounted": total_bytes,
        "payload_bytes_on_disk": downloaded_bytes,
        "modalities_retained": [
            "selected_raw_uav_images",
            "processed_orthophoto",
            "camera_metadata",
            "geographic_image_footprints",
            "license_and_readme",
        ],
        "modalities_excluded": [
            "duplicate_original_orthophoto",
            "render_orthophoto",
            "meshes",
            "point_clouds",
            "textures",
            "reports",
        ],
        "storage_policy": "local_only_no_docker_image_no_docker_volume",
    }
    _write_json(source_root / "firewarning-acquisition-report.json", report)
    return report


def _quaternion_rotation(quaternion: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = quaternion
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _camera_center(
    quaternion: tuple[float, float, float, float], translation: tuple[float, float, float]
) -> tuple[float, float, float]:
    rotation = _quaternion_rotation(quaternion)
    return tuple(
        -sum(rotation[row][column] * translation[row] for row in range(3)) for column in range(3)
    )  # type: ignore[return-value]


def _parse_intrinsics(path: Path) -> dict[str, tuple[float, float, float, float, int, int]]:
    intrinsics: dict[str, tuple[float, float, float, float, int, int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 8 or parts[1] != "PINHOLE":
            raise SetupError(f"invalid intrinsic row at {path}:{line_number}")
        intrinsics[parts[0]] = (
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
            float(parts[7]),
            int(parts[2]),
            int(parts[3]),
        )
    return intrinsics


def _parse_cameras(source_root: Path) -> list[CameraRecord]:
    intrinsics = _parse_intrinsics(source_root / "intrinsic.txt")
    cameras: list[CameraRecord] = []
    for line_number, line in enumerate(
        (source_root / "gt_pose.txt").read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if len(parts) != 8:
            raise SetupError(f"invalid pose row at gt_pose.txt:{line_number}")
        name = parts[0]
        if name not in intrinsics:
            raise SetupError(f"pose has no intrinsic: {name}")
        quaternion = tuple(float(value) for value in parts[1:5])
        translation = tuple(float(value) for value in parts[5:8])
        if not math.isclose(sum(value * value for value in quaternion), 1.0, abs_tol=1e-5):
            raise SetupError(f"non-unit pose quaternion: {name}")
        cameras.append(
            CameraRecord(
                name=name,
                quaternion_wxyz=quaternion,  # type: ignore[arg-type]
                translation_w2c=translation,  # type: ignore[arg-type]
                center_xyz=_camera_center(quaternion, translation),  # type: ignore[arg-type]
                intrinsic=intrinsics[name],
            )
        )
    return cameras


def _camera_world_ray(
    camera: CameraRecord, pixel_x: float, pixel_y: float
) -> tuple[float, float, float]:
    """Return a normalized world-space ray for a pixel and a world-to-camera pose."""

    fx, fy, cx, cy, _, _ = camera.intrinsic
    direction_camera = ((pixel_x - cx) / fx, (pixel_y - cy) / fy, 1.0)
    rotation = _quaternion_rotation(camera.quaternion_wxyz)
    direction_world = tuple(
        sum(rotation[row][column] * direction_camera[row] for row in range(3))
        for column in range(3)
    )
    norm = math.sqrt(sum(value * value for value in direction_world))
    if norm <= 1e-12:
        raise SetupError(f"degenerate camera ray: {camera.name}")
    return tuple(value / norm for value in direction_world)  # type: ignore[return-value]


def _project_world_point(
    camera: CameraRecord, point_xyz: tuple[float, float, float]
) -> tuple[float, float]:
    rotation = _quaternion_rotation(camera.quaternion_wxyz)
    camera_xyz = tuple(
        sum(rotation[row][column] * point_xyz[column] for column in range(3))
        + camera.translation_w2c[row]
        for row in range(3)
    )
    if camera_xyz[2] <= 1e-9:
        raise SetupError(f"world point projects behind camera: {camera.name}")
    fx, fy, cx, cy, _, _ = camera.intrinsic
    return (
        fx * camera_xyz[0] / camera_xyz[2] + cx,
        fy * camera_xyz[1] / camera_xyz[2] + cy,
    )


def _dsm_value(dsm: Any, x: float, y: float) -> float | None:
    if not (dsm.bounds.left <= x <= dsm.bounds.right and dsm.bounds.bottom <= y <= dsm.bounds.top):
        return None
    from rasterio.windows import Window

    row_float, column_float = dsm.index(x, y, op=float)
    center_row = float(row_float) - 0.5
    center_column = float(column_float) - 0.5
    row0 = math.floor(center_row)
    column0 = math.floor(center_column)
    if row0 < 0 or column0 < 0 or row0 + 1 >= dsm.height or column0 + 1 >= dsm.width:
        return None
    values = dsm.read(
        1,
        window=Window(column0, row0, 2, 2),
        masked=True,
    )
    if values.count() != 4:
        return None
    row_weight = center_row - row0
    column_weight = center_column - column0
    top = float(values[0, 0]) * (1.0 - column_weight) + float(values[0, 1]) * column_weight
    bottom = float(values[1, 0]) * (1.0 - column_weight) + float(values[1, 1]) * column_weight
    value = top * (1.0 - row_weight) + bottom * row_weight
    return value if math.isfinite(value) else None


def _intersect_ray_with_dsm(
    dsm: Any,
    origin_xyz: tuple[float, float, float],
    direction_xyz: tuple[float, float, float],
    *,
    max_distance: float = 2_000.0,
) -> RayIntersection | None:
    """Intersect a descending ray with a DSM using a bracket then bisection."""

    if direction_xyz[2] >= -1e-9:
        return None

    def height_delta(distance: float) -> tuple[float, float, float, float] | None:
        x = origin_xyz[0] + distance * direction_xyz[0]
        y = origin_xyz[1] + distance * direction_xyz[1]
        z = origin_xyz[2] + distance * direction_xyz[2]
        surface = _dsm_value(dsm, x, y)
        if surface is None:
            return None
        return z - surface, x, y, surface

    start = height_delta(0.0)
    if start is None or start[0] < -0.25:
        return None
    lower = 0.0
    upper = 1.0
    upper_value = height_delta(upper)
    while upper <= max_distance and (upper_value is None or upper_value[0] > 0.0):
        if upper_value is None:
            return None
        lower = upper
        upper *= 2.0
        upper_value = height_delta(upper)
    if upper > max_distance or upper_value is None:
        return None

    for _ in range(48):
        middle = (lower + upper) / 2.0
        middle_value = height_delta(middle)
        if middle_value is None:
            return None
        if middle_value[0] > 0.0:
            lower = middle
        else:
            upper = middle
    final = height_delta((lower + upper) / 2.0)
    if final is None:
        return None
    distance = (lower + upper) / 2.0
    ray_z = origin_xyz[2] + distance * direction_xyz[2]
    row, column = dsm.index(final[1], final[2], op=float)
    return RayIntersection(
        xyz=(final[1], final[2], ray_z),
        surface_z=final[3],
        vertical_residual_m=abs(ray_z - final[3]),
        raster_row=float(row),
        raster_column=float(column),
        distance=distance,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise SetupError("cannot compute a percentile from an empty series")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _parse_pose_file(
    path: Path,
) -> dict[str, tuple[tuple[float, float, float, float], tuple[float, float, float]]]:
    poses: dict[str, tuple[tuple[float, float, float, float], tuple[float, float, float]]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 8:
            raise SetupError(f"invalid pose row at {path}:{line_number}")
        poses[parts[0]] = (
            tuple(float(value) for value in parts[1:5]),  # type: ignore[arg-type]
            tuple(float(value) for value in parts[5:8]),  # type: ignore[arg-type]
        )
    return poses


def audit_registration_poses(
    dataset_root: Path, *, crop_source_pixels: int = 1536
) -> dict[str, Any]:
    """Numerically audit the AerialExtreMatch poses and their DSM intersections."""

    try:
        import numpy as np
        import rasterio
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit at runtime
        raise SetupError("install spatial-training dependencies before auditing poses") from exc

    dataset_root = dataset_root.resolve()
    source_root = dataset_root / "sources" / "aerialextrematch-localization"
    _deny_operational_path(source_root)
    cameras = _parse_cameras(source_root)
    prior_poses = _parse_pose_file(source_root / "pose.txt")
    dsm_path = source_root / "LQref" / "DSM.tif"
    if not dsm_path.is_file():
        raise SetupError(f"missing DSM for pose audit: {dsm_path}")

    quaternion_errors: list[float] = []
    determinant_errors: list[float] = []
    orthonormal_errors: list[float] = []
    center_identity_errors: list[float] = []
    camera_heights: list[float] = []
    optical_axis_z: list[float] = []
    axis_offsets: list[float] = []
    reprojection_errors: list[float] = []
    surface_residuals: list[float] = []
    prior_translation_errors: list[float] = []
    prior_rotation_errors: list[float] = []
    centers_inside = 0
    principal_intersections = 0
    intersections_inside_crop = 0
    half_crop = crop_source_pixels / 2.0

    with rasterio.open(dsm_path) as dsm:
        dsm_horizontal_resolution_m = max(abs(float(dsm.transform.a)), abs(float(dsm.transform.e)))
        surface_residual_tolerance_m = max(1.0, 2.0 * dsm_horizontal_resolution_m)
        for camera in cameras:
            rotation = np.asarray(_quaternion_rotation(camera.quaternion_wxyz), dtype=float)
            quaternion_errors.append(
                abs(sum(value * value for value in camera.quaternion_wxyz) - 1.0)
            )
            determinant_errors.append(abs(float(np.linalg.det(rotation)) - 1.0))
            orthonormal_errors.append(float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))))
            recomputed_center = _camera_center(camera.quaternion_wxyz, camera.translation_w2c)
            center_identity_errors.append(math.dist(camera.center_xyz, recomputed_center))

            x, y, z = camera.center_xyz
            row, column = dsm.index(x, y, op=float)
            if 0 <= row < dsm.height and 0 <= column < dsm.width:
                centers_inside += 1
            surface = _dsm_value(dsm, x, y)
            if surface is None:
                continue
            camera_heights.append(z - surface)
            _, _, cx, cy, _, _ = camera.intrinsic
            axis = _camera_world_ray(camera, cx, cy)
            optical_axis_z.append(axis[2])
            intersection = _intersect_ray_with_dsm(dsm, camera.center_xyz, axis)
            if intersection is None:
                continue
            principal_intersections += 1
            axis_offsets.append(math.dist((x, y), intersection.xyz[:2]))
            surface_residuals.append(intersection.vertical_residual_m)
            if (
                abs(intersection.raster_column - float(column)) <= half_crop
                and abs(intersection.raster_row - float(row)) <= half_crop
            ):
                intersections_inside_crop += 1
            projected_x, projected_y = _project_world_point(camera, intersection.xyz)
            reprojection_errors.append(math.hypot(projected_x - cx, projected_y - cy))

            if camera.name in prior_poses:
                prior_quaternion, prior_translation = prior_poses[camera.name]
                prior_center = _camera_center(prior_quaternion, prior_translation)
                prior_translation_errors.append(math.dist(camera.center_xyz, prior_center))
                prior_rotation = np.asarray(_quaternion_rotation(prior_quaternion), dtype=float)
                relative = prior_rotation @ rotation.T
                cosine = min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) / 2.0))
                prior_rotation_errors.append(math.degrees(math.acos(cosine)))

    count = len(cameras)
    gates = {
        "all_camera_centers_inside_dsm": centers_inside == count,
        "all_principal_axes_descend": len(optical_axis_z) == count and max(optical_axis_z) < 0.0,
        "all_principal_axes_intersect_dsm": principal_intersections == count,
        "all_intersections_inside_training_crop": intersections_inside_crop == count,
        "rotation_matrices_valid": max(determinant_errors) <= 1e-9
        and max(orthonormal_errors) <= 1e-9,
        "camera_center_identity_valid": max(center_identity_errors) <= 1e-8,
        "principal_reprojection_valid": len(reprojection_errors) == count
        and max(reprojection_errors) <= 1e-4,
        "surface_intersection_residual_valid": len(surface_residuals) == count
        and max(surface_residuals) <= surface_residual_tolerance_m,
        "camera_height_plausible": len(camera_heights) == count
        and min(camera_heights) >= 5.0
        and max(camera_heights) <= 500.0,
    }
    report = {
        "schema_version": 1,
        "source_id": "aerialextrematch_localization",
        "source_revision": DATASET_REPO_REVISION,
        "camera_count": count,
        "dsm_relpath": dsm_path.relative_to(dataset_root).as_posix(),
        "metrics": {
            "quaternion_unit_max_abs_error": max(quaternion_errors),
            "rotation_determinant_max_abs_error": max(determinant_errors),
            "rotation_orthonormal_max_abs_error": max(orthonormal_errors),
            "camera_center_identity_max_error_m": max(center_identity_errors),
            "camera_height_agl_m": {
                "min": min(camera_heights),
                "median": _percentile(camera_heights, 50),
                "p95": _percentile(camera_heights, 95),
                "max": max(camera_heights),
            },
            "principal_axis_world_z": {
                "min": min(optical_axis_z),
                "median": _percentile(optical_axis_z, 50),
                "max": max(optical_axis_z),
            },
            "axis_ground_horizontal_offset_m": {
                "min": min(axis_offsets),
                "median": _percentile(axis_offsets, 50),
                "p95": _percentile(axis_offsets, 95),
                "max": max(axis_offsets),
            },
            "principal_intersection_reprojection_max_px": max(reprojection_errors),
            "surface_intersection_vertical_residual_m": {
                "median": _percentile(surface_residuals, 50),
                "p95": _percentile(surface_residuals, 95),
                "max": max(surface_residuals),
                "tolerance": surface_residual_tolerance_m,
            },
            "dsm_horizontal_resolution_m": dsm_horizontal_resolution_m,
            "prior_pose_translation_error_m": {
                "median": _percentile(prior_translation_errors, 50),
                "p95": _percentile(prior_translation_errors, 95),
                "max": max(prior_translation_errors),
            },
            "prior_pose_rotation_error_deg": {
                "median": _percentile(prior_rotation_errors, 50),
                "p95": _percentile(prior_rotation_errors, 95),
                "max": max(prior_rotation_errors),
            },
        },
        "counts": {
            "camera_centers_inside_dsm": centers_inside,
            "principal_axis_dsm_intersections": principal_intersections,
            "principal_intersections_inside_training_crop": intersections_inside_crop,
        },
        "gates": gates,
        "pose_audit_passed": all(gates.values()),
        "interpretation": (
            "pose.txt is a noisy localization prior; gt_pose.txt is the audited ground truth"
        ),
    }
    _write_json(source_root / "pose-audit-report.json", report)
    return report


def _spatial_group(camera: CameraRecord) -> str:
    x, y, _ = camera.center_xyz
    return f"epsg4547:{math.floor(x / 100)}:{math.floor(y / 100)}"


def _spatial_split(cameras: list[CameraRecord]) -> dict[str, str]:
    """Hold out west/east 100 m cells; a cell can never cross two splits."""

    groups: dict[str, list[CameraRecord]] = defaultdict(list)
    for camera in cameras:
        groups[_spatial_group(camera)].append(camera)
    ordered_groups = sorted(
        groups,
        key=lambda group: sum(item.center_xyz[0] for item in groups[group]) / len(groups[group]),
    )
    west_group_count = max(1, math.floor(len(ordered_groups) * 0.20))
    east_group_start = min(len(ordered_groups) - 1, math.ceil(len(ordered_groups) * 0.80))
    validation_groups = set(ordered_groups[:west_group_count])
    test_groups = set(ordered_groups[east_group_start:])
    result: dict[str, str] = {}
    for camera in cameras:
        group = _spatial_group(camera)
        if group in validation_groups:
            result[camera.name] = "validation"
        elif group in test_groups:
            result[camera.name] = "test"
        else:
            result[camera.name] = "train"
    return result


def _content_addressed_jpeg(image: Any, output_root: Path) -> tuple[str, str, bool]:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    payload = buffer.getvalue()
    sha256 = _sha256_bytes(payload)
    relative = Path("map-crops") / sha256[:2] / f"{sha256}.jpg"
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != sha256:
            raise SetupError(f"content-address collision: {destination}")
        created = False
    else:
        destination.write_bytes(payload)
        created = True
    return relative.as_posix(), sha256, created


def _footprint_points(entry: dict[str, Any]) -> list[tuple[float, float, float]]:
    try:
        raw_points = entry["polygon_geom"]["geometry"]["coordinates"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise SetupError("ODM image footprint is missing") from exc
    points = [
        (float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0)
        for point in raw_points
    ]
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise SetupError("ODM image footprint has fewer than three unique points")
    return points


def _deterministic_crop_offset(sample_id: str, crop_pixels: int) -> tuple[float, float]:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    unit_x = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    unit_y = int.from_bytes(digest[4:8], "big") / (2**32 - 1)
    return (
        (unit_x - 0.5) * 0.30 * crop_pixels,
        (unit_y - 0.5) * 0.30 * crop_pixels,
    )


def _build_odm_rows(
    dataset_root: Path,
    output_root: Path,
    *,
    output_pixels: int,
    verify_files: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    try:
        import numpy as np
        import rasterio
        from PIL import Image
        from pyproj import Transformer
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit at runtime
        raise SetupError("install spatial-training dependencies before building ODM pairs") from exc

    source_root = dataset_root / "sources" / "odm-cross-view"
    records: list[dict[str, Any]] = []
    for definition in ODM_DOMAIN_SOURCES:
        manifest_path = source_root / str(definition["key"]) / "acquisition-manifest.jsonl"
        if not manifest_path.is_file():
            raise SetupError(f"missing ODM acquisition manifest: {manifest_path}")
        records.extend(_iter_jsonl(manifest_path))
    if not records:
        raise SetupError("ODM cross-view acquisition manifests are empty")

    rows: list[dict[str, Any]] = []
    created_map_crops: list[Path] = []
    source_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    valid_fractions: list[float] = []
    target_offsets: list[float] = []
    source_image_bytes = 0
    orthophoto_cache: dict[str, tuple[Any, Any, str]] = {}
    verified_orthophotos: set[Path] = set()

    try:
        for record in records:
            source_id = str(record["source_id"])
            license_id = str(record["license"])
            _require_production_license(license_id, source_id=source_id)
            image_record = record["image"]
            image_path = _resolve_within(dataset_root, str(image_record["relpath"]))
            orthophoto_record = record["orthophoto"]
            orthophoto_path = _resolve_within(dataset_root, str(orthophoto_record["relpath"]))
            if not image_path.is_file() or not orthophoto_path.is_file():
                raise SetupError(f"missing ODM pair media for {source_id}")
            if verify_files and _sha256_file(image_path) != image_record["sha256"]:
                raise SetupError(f"ODM source image SHA-256 mismatch: {image_path}")
            if verify_files and orthophoto_path not in verified_orthophotos:
                if _sha256_file(orthophoto_path) != orthophoto_record["sha256"]:
                    raise SetupError(f"ODM orthophoto SHA-256 mismatch: {orthophoto_path}")
                verified_orthophotos.add(orthophoto_path)
            source_image_bytes += image_path.stat().st_size

            cache_key = str(orthophoto_path)
            if cache_key not in orthophoto_cache:
                dataset = rasterio.open(orthophoto_path)
                if dataset.crs is None:
                    dataset.close()
                    raise SetupError(f"ODM orthophoto has no CRS: {orthophoto_path}")
                transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
                orthophoto_cache[cache_key] = (
                    dataset,
                    transformer,
                    str(orthophoto_record["sha256"]),
                )
            orthophoto, transformer, orthophoto_sha256 = orthophoto_cache[cache_key]

            entry = record["dronedb_entry"]
            properties = entry["properties"]
            footprint_wgs84 = _footprint_points(entry)
            target_longitude = sum(point[0] for point in footprint_wgs84) / len(footprint_wgs84)
            target_latitude = sum(point[1] for point in footprint_wgs84) / len(footprint_wgs84)
            target_altitude = sum(point[2] for point in footprint_wgs84) / len(footprint_wgs84)
            footprint_xy = [transformer.transform(point[0], point[1]) for point in footprint_wgs84]
            target_x, target_y = transformer.transform(target_longitude, target_latitude)
            camera_coordinates = entry["point_geom"]["geometry"]["coordinates"]
            camera_longitude = float(camera_coordinates[0])
            camera_latitude = float(camera_coordinates[1])
            camera_altitude = float(camera_coordinates[2])
            camera_x, camera_y = transformer.transform(camera_longitude, camera_latitude)

            footprint_span_m = max(
                max(point[0] for point in footprint_xy) - min(point[0] for point in footprint_xy),
                max(point[1] for point in footprint_xy) - min(point[1] for point in footprint_xy),
            )
            pixel_resolution_m = max(
                abs(float(orthophoto.transform.a)), abs(float(orthophoto.transform.e))
            )
            crop_width_m = max(64.0, footprint_span_m * 2.0)
            crop_source_pixels = math.ceil(crop_width_m / pixel_resolution_m)
            crop_source_pixels = min(
                crop_source_pixels, orthophoto.width - 2, orthophoto.height - 2
            )
            sample_id = f"{source_id}:{Path(str(entry['path'])).stem.lower()}"
            deterministic_offset = _deterministic_crop_offset(sample_id, crop_source_pixels)
            target_row, target_column = orthophoto.index(target_x, target_y, op=float)
            camera_row, camera_column = orthophoto.index(camera_x, camera_y, op=float)
            offset_scale = 0.20 * crop_source_pixels
            candidate_offsets = [
                deterministic_offset,
                (0.0, 0.0),
                *[
                    (column_step * offset_scale, row_step * offset_scale)
                    for row_step in (-1, 0, 1)
                    for column_step in (-1, 0, 1)
                    if row_step != 0 or column_step != 0
                ],
            ]
            window: Any | None = None
            mask: Any | None = None
            valid_fraction = -1.0
            for offset_column, offset_row in candidate_offsets:
                column_off = min(
                    max(
                        float(target_column) + offset_column - crop_source_pixels / 2.0,
                        0.0,
                    ),
                    orthophoto.width - crop_source_pixels,
                )
                row_off = min(
                    max(
                        float(target_row) + offset_row - crop_source_pixels / 2.0,
                        0.0,
                    ),
                    orthophoto.height - crop_source_pixels,
                )
                candidate = Window(
                    column_off,
                    row_off,
                    crop_source_pixels,
                    crop_source_pixels,
                )
                coordinates = (
                    (float(target_column), float(target_row)),
                    (float(camera_column), float(camera_row)),
                )
                if not all(
                    candidate.col_off <= column <= candidate.col_off + crop_source_pixels
                    and candidate.row_off <= row <= candidate.row_off + crop_source_pixels
                    for column, row in coordinates
                ):
                    continue
                candidate_mask = orthophoto.read_masks(
                    1,
                    window=candidate,
                    out_shape=(output_pixels, output_pixels),
                    resampling=Resampling.nearest,
                )
                candidate_valid_fraction = float(np.count_nonzero(candidate_mask)) / (
                    output_pixels * output_pixels
                )
                if candidate_valid_fraction > valid_fraction:
                    valid_fraction = candidate_valid_fraction
                    window = candidate
                    mask = candidate_mask
            if window is None or mask is None or valid_fraction < 0.95:
                rejected_counts[f"{source_id}:insufficient_map_coverage"] += 1
                continue
            raster = orthophoto.read(
                indexes=(1, 2, 3),
                window=window,
                out_shape=(3, output_pixels, output_pixels),
                resampling=Resampling.bilinear,
            )
            valid_fractions.append(valid_fraction)
            if raster.dtype != np.uint8:
                raster = np.clip(raster, 0, 255).astype(np.uint8)
            rgb = np.moveaxis(raster, 0, -1)
            map_relpath, map_sha256, created = _content_addressed_jpeg(
                Image.fromarray(rgb), output_root
            )
            if created:
                created_map_crops.append(output_root / map_relpath)

            target_map_x = (float(target_column) - float(window.col_off)) / crop_source_pixels
            target_map_y = (float(target_row) - float(window.row_off)) / crop_source_pixels
            camera_map_x = (float(camera_column) - float(window.col_off)) / crop_source_pixels
            camera_map_y = (float(camera_row) - float(window.row_off)) / crop_source_pixels
            if not (
                0.0 <= target_map_x <= 1.0
                and 0.0 <= target_map_y <= 1.0
                and 0.0 <= camera_map_x <= 1.0
                and 0.0 <= camera_map_y <= 1.0
            ):
                raise SetupError(f"ODM target or camera lies outside map crop: {sample_id}")
            target_offsets.append(math.hypot(target_map_x - 0.5, target_map_y - 0.5))

            with Image.open(image_path) as image:
                image_width, image_height = image.size
            declared_width = int(properties["width"])
            declared_height = int(properties["height"])
            if (image_width, image_height) != (declared_width, declared_height):
                raise SetupError(f"ODM image dimensions changed: {image_path}")
            focal_length = float(properties["focalLength"])
            sensor_width = float(properties["sensorWidth"])
            sensor_height = float(properties["sensorHeight"])
            intrinsics = {
                "fx": focal_length / sensor_width * image_width,
                "fy": focal_length / sensor_height * image_height,
                "cx": image_width / 2.0,
                "cy": image_height / 2.0,
                "origin": "exif_focal_and_sensor_dimensions",
            }
            rows.append(
                {
                    "schema_version": "1.0",
                    "family": "cross_view_registration",
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "source_revision": record["source_revision"],
                    "source_view": {
                        "image_relpath": image_path.relative_to(dataset_root).as_posix(),
                        "sha256": image_record["sha256"],
                        "width": image_width,
                        "height": image_height,
                        "camera_model": "BROWN_EXIF_APPROXIMATION",
                        "intrinsics": intrinsics,
                    },
                    "map_view": {
                        "image_relpath": (output_root / map_relpath)
                        .relative_to(dataset_root)
                        .as_posix(),
                        "sha256": map_sha256,
                        "width": output_pixels,
                        "height": output_pixels,
                        "source_window_pixels": crop_source_pixels,
                        "camera_center_pixel_normalized": [
                            round(camera_map_x, 8),
                            round(camera_map_y, 8),
                        ],
                        "optical_axis_ground_pixel_normalized": [
                            round(target_map_x, 8),
                            round(target_map_y, 8),
                        ],
                        "reference_dom_relpath": orthophoto_path.relative_to(
                            dataset_root
                        ).as_posix(),
                        "reference_dom_sha256": orthophoto_sha256,
                    },
                    "ground_truth": {
                        "crs": str(orthophoto.crs),
                        "camera_position_wgs84": [
                            camera_longitude,
                            camera_latitude,
                            camera_altitude,
                        ],
                        "camera_position_xy": [camera_x, camera_y],
                        "optical_axis_ground_wgs84": [
                            target_longitude,
                            target_latitude,
                            target_altitude,
                        ],
                        "optical_axis_ground_xy": [target_x, target_y],
                        "camera_orientation_deg": {
                            "yaw": float(properties["cameraYaw"]),
                            "pitch": float(properties["cameraPitch"]),
                            "roll": float(properties["cameraRoll"]),
                        },
                        "footprint_wgs84": [list(point) for point in footprint_wgs84],
                        "target_derivation": "centroid_of_dronedb_ground_footprint",
                    },
                    "domain": {
                        "label": record["domain"],
                        "site_id": str(record["source_id"]),
                    },
                    "split": "train",
                    "split_group": f"odm_site:{record['source_id']}",
                    "validation_status": "source_geotag_footprint_derived",
                    "license": license_id,
                    "consent_basis": {
                        "kind": "source_license",
                        "reference": record["repository"],
                        "commercial_use_allowed": True,
                    },
                    "operational_incident": False,
                }
            )
            source_counts[source_id] += 1
            domain_counts[str(record["domain"])] += 1
            license_counts[license_id] += 1
    except Exception:
        for created_path in created_map_crops:
            created_path.unlink(missing_ok=True)
        raise
    finally:
        for dataset, _, _ in orthophoto_cache.values():
            dataset.close()

    gates = {
        "minimum_rows_per_domain_present": all(
            source_counts[str(definition["source_id"])] >= 56 for definition in ODM_DOMAIN_SOURCES
        ),
        "rural_and_mountain_domains_present": {
            "rural_farmland",
            "mountain_pass",
        }.issubset(domain_counts),
        "all_map_crops_valid": len(valid_fractions) == len(rows) and min(valid_fractions) >= 0.95,
        "all_targets_offset_inside_crop": len(target_offsets) == len(rows)
        and max(target_offsets) < 0.40,
        "production_licenses_only": all(
            license_id in PRODUCTION_LICENSE_ALLOWLIST for license_id in license_counts
        ),
    }
    report = {
        "schema_version": 1,
        "source_family": "odm_cross_view_domains",
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "rejected_counts": dict(sorted(rejected_counts.items())),
        "source_image_bytes": source_image_bytes,
        "map_crop_valid_fraction_min": min(valid_fractions),
        "target_offset_normalized": {
            "median": _percentile(target_offsets, 50),
            "max": max(target_offsets),
        },
        "gates": gates,
        "build_passed": all(gates.values()),
        "pose_scope": (
            "geotag, EXIF orientation and source-derived ground footprint; no full bundle-adjusted "
            "6DoF pose is claimed"
        ),
    }
    if not report["build_passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise SetupError(f"ODM cross-view build failed: {failed}")
    return rows, report, created_map_crops


def build_registration(
    dataset_root: Path,
    *,
    crop_source_pixels: int = 1536,
    output_pixels: int = 768,
    verify_source_images: bool = True,
) -> dict[str, Any]:
    """Build real UAV/map pairs from a historical source unrelated to live incidents."""

    dataset_root = dataset_root.resolve()
    _require_production_license("MIT", source_id="aerialextrematch_localization")
    source_root = dataset_root / "sources" / "aerialextrematch-localization"
    output_root = dataset_root / "corpus" / REGISTRATION_CORPUS_ID
    _deny_operational_path(source_root)
    _deny_operational_path(output_root)
    if crop_source_pixels < output_pixels or output_pixels < 128:
        raise SetupError("map crop dimensions are invalid")

    try:
        import numpy as np
        import rasterio
        from PIL import Image
        from pyproj import Transformer
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit at runtime
        message = "install the spatial-training dependencies before building registration"
        raise SetupError(message) from exc

    cameras = _parse_cameras(source_root)
    if len(cameras) != 264:
        raise SetupError(f"expected 264 camera records, found {len(cameras)}")
    pose_audit = audit_registration_poses(dataset_root, crop_source_pixels=crop_source_pixels)
    if not pose_audit["pose_audit_passed"]:
        failed = [name for name, passed in pose_audit["gates"].items() if not passed]
        raise SetupError(f"registration pose audit failed: {failed}")
    splits = _spatial_split(cameras)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    partial_path = output_root / "manifest.partial.jsonl"
    dom_path = source_root / "LQref" / "DOM.tif"
    dsm_path = source_root / "LQref" / "DSM.tif"
    if not dom_path.is_file() or not dsm_path.is_file():
        raise SetupError("registration DOM/DSM source is incomplete")

    split_counts: Counter[str] = Counter()
    spatial_groups: dict[str, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    source_bytes = 0
    map_bytes_before = sum(path.stat().st_size for path in output_root.rglob("*.jpg"))
    dom_sha256 = _sha256_file(dom_path)
    created_map_crops: list[Path] = []
    odm_rows, odm_report, odm_created_map_crops = _build_odm_rows(
        dataset_root,
        output_root,
        output_pixels=output_pixels,
        verify_files=verify_source_images,
    )
    created_map_crops.extend(odm_created_map_crops)
    source_bytes += int(odm_report["source_image_bytes"])

    try:
        with rasterio.open(dom_path) as dom, rasterio.open(dsm_path) as dsm:
            if dom.crs is None or dsm.crs is None or dom.crs != dsm.crs:
                raise SetupError("DOM and DSM must share a declared CRS")
            if dom.width != dsm.width or dom.height != dsm.height:
                raise SetupError("DOM and DSM dimensions differ")
            transformer = Transformer.from_crs(dom.crs, "EPSG:4326", always_xy=True)
            with partial_path.open("w", encoding="utf-8", newline="\n") as manifest:
                for camera in cameras:
                    source_image = source_root / "rgb" / camera.name
                    if not source_image.is_file():
                        raise SetupError(f"missing UAV image: {source_image}")
                    source_sha256 = (
                        _sha256_file(source_image) if verify_source_images else "not-verified"
                    )
                    source_bytes += source_image.stat().st_size

                    x, y, z_camera = camera.center_xyz
                    row_float, column_float = dom.index(x, y, op=float)
                    if not (0 <= row_float < dom.height and 0 <= column_float < dom.width):
                        raise SetupError(f"camera centre outside reference map: {camera.name}")
                    half = crop_source_pixels / 2.0
                    window = Window(
                        column_float - half,
                        row_float - half,
                        crop_source_pixels,
                        crop_source_pixels,
                    )
                    raster = dom.read(
                        indexes=(1, 2, 3),
                        window=window,
                        out_shape=(3, output_pixels, output_pixels),
                        boundless=True,
                        fill_value=0,
                        resampling=Resampling.bilinear,
                    )
                    mask = dom.read_masks(
                        1,
                        window=window,
                        out_shape=(output_pixels, output_pixels),
                        boundless=True,
                        resampling=Resampling.nearest,
                    )
                    if raster.dtype != np.uint8:
                        raster = np.clip(raster, 0, 255).astype(np.uint8)
                    rgb = np.moveaxis(raster, 0, -1)
                    valid_fraction = float(np.count_nonzero(mask)) / (output_pixels * output_pixels)
                    if valid_fraction < 0.90:
                        raise SetupError(f"map crop has insufficient coverage: {camera.name}")
                    map_relpath, map_sha256, created = _content_addressed_jpeg(
                        Image.fromarray(rgb), output_root
                    )
                    if created:
                        created_map_crops.append(output_root / map_relpath)

                    dsm_value = _dsm_value(dsm, x, y)
                    if dsm_value is None:
                        raise SetupError(f"invalid DSM below camera centre: {camera.name}")
                    _, _, principal_x, principal_y, _, _ = camera.intrinsic
                    optical_axis = _camera_world_ray(camera, principal_x, principal_y)
                    axis_intersection = _intersect_ray_with_dsm(
                        dsm, camera.center_xyz, optical_axis
                    )
                    if axis_intersection is None:
                        raise SetupError(
                            f"principal optical axis does not intersect DSM: {camera.name}"
                        )
                    axis_map_x = (
                        axis_intersection.raster_column - float(window.col_off)
                    ) / crop_source_pixels
                    axis_map_y = (
                        axis_intersection.raster_row - float(window.row_off)
                    ) / crop_source_pixels
                    if not (0.0 <= axis_map_x <= 1.0 and 0.0 <= axis_map_y <= 1.0):
                        raise SetupError(
                            f"principal optical-axis target outside map crop: {camera.name}"
                        )
                    longitude, latitude = transformer.transform(x, y)
                    split = splits[camera.name]
                    group = _spatial_group(camera)
                    spatial_groups[group].add(split)
                    fx, fy, cx, cy, width, height = camera.intrinsic
                    row = {
                        "schema_version": "1.0",
                        "family": "cross_view_registration",
                        "sample_id": f"aerialextrematch:{Path(camera.name).stem}",
                        "source_id": "aerialextrematch_localization",
                        "source_revision": DATASET_REPO_REVISION,
                        "source_view": {
                            "image_relpath": source_image.relative_to(dataset_root).as_posix(),
                            "sha256": source_sha256,
                            "width": width,
                            "height": height,
                            "camera_model": "PINHOLE",
                            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                        },
                        "map_view": {
                            "image_relpath": (output_root / map_relpath)
                            .relative_to(dataset_root)
                            .as_posix(),
                            "sha256": map_sha256,
                            "width": output_pixels,
                            "height": output_pixels,
                            "source_window_pixels": crop_source_pixels,
                            "camera_center_pixel_normalized": [0.5, 0.5],
                            "optical_axis_ground_pixel_normalized": [
                                round(axis_map_x, 8),
                                round(axis_map_y, 8),
                            ],
                            "reference_dom_relpath": dom_path.relative_to(dataset_root).as_posix(),
                            "reference_dom_sha256": dom_sha256,
                        },
                        "ground_truth": {
                            "crs": str(dom.crs),
                            "camera_center_xyz": [round(x, 6), round(y, 6), round(z_camera, 6)],
                            "surface_z_at_camera_xy": round(dsm_value, 6),
                            "camera_height_agl_m": round(z_camera - dsm_value, 6),
                            "optical_axis_ground_xyz": [
                                round(axis_intersection.xyz[0], 6),
                                round(axis_intersection.xyz[1], 6),
                                round(axis_intersection.surface_z, 6),
                            ],
                            "optical_axis_ray_z": round(axis_intersection.xyz[2], 6),
                            "surface_intersection_vertical_residual_m": round(
                                axis_intersection.vertical_residual_m, 6
                            ),
                            "axis_horizontal_offset_m": round(
                                math.dist((x, y), axis_intersection.xyz[:2]), 6
                            ),
                            "wgs84_derived": {
                                "longitude": round(float(longitude), 9),
                                "latitude": round(float(latitude), 9),
                            },
                            "quaternion_wxyz_world_to_camera": list(camera.quaternion_wxyz),
                            "translation_xyz_world_to_camera": list(camera.translation_w2c),
                        },
                        "split": split,
                        "split_group": group,
                        "validation_status": "source_pose_provided",
                        "license": "MIT",
                        "consent_basis": {
                            "kind": "source_license",
                            "reference": (
                                "https://huggingface.co/datasets/"
                                "Xecades/AerialExtreMatch-Localization"
                            ),
                        },
                        "operational_incident": False,
                    }
                    manifest.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    split_counts[split] += 1
                    source_counts["aerialextrematch_localization"] += 1
                for row in odm_rows:
                    manifest.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    split = str(row["split"])
                    group = str(row["split_group"])
                    split_counts[split] += 1
                    spatial_groups[group].add(split)
                    source_counts[str(row["source_id"])] += 1
    except Exception:
        partial_path.unlink(missing_ok=True)
        for created_path in created_map_crops:
            created_path.unlink(missing_ok=True)
        raise

    leaking_groups = sorted(group for group, values in spatial_groups.items() if len(values) > 1)
    if leaking_groups:
        raise SetupError(f"spatial split-group leakage detected: {leaking_groups[:5]}")
    os.replace(partial_path, manifest_path)
    map_bytes_after = sum(path.stat().st_size for path in output_root.rglob("*.jpg"))
    report = {
        "schema_version": 1,
        "family": "cross_view_registration",
        "corpus_id": REGISTRATION_CORPUS_ID,
        "source_ids": list(sorted(source_counts)),
        "source_counts": dict(sorted(source_counts.items())),
        "manifest_relpath": manifest_path.relative_to(dataset_root).as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "rows": len(cameras) + len(odm_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "spatial_group_count": len(spatial_groups),
        "split_group_leaks": 0,
        "pose_audit_relpath": (source_root / "pose-audit-report.json")
        .relative_to(dataset_root)
        .as_posix(),
        "pose_audit_passed": True,
        "odm_domain_build": odm_report,
        "source_image_bytes": source_bytes,
        "derived_map_crop_bytes": map_bytes_after,
        "derived_map_crop_bytes_added": max(0, map_bytes_after - map_bytes_before),
        "excluded_live_incident_tokens": list(DENIED_OPERATIONAL_TOKENS),
        "gates": {
            "setup_ready": True,
            "smoke_ready": split_counts["train"] >= 8 and split_counts["validation"] >= 4,
            "bootstrap_training_ready": True,
            "training_ready": True,
            "production_training_ready": False,
            "deployment_ready": False,
        },
        "blockers": [
            "no double-validated cross-view critical test exists",
            "no independent French-massif production validation has been completed",
        ],
    }
    _write_json(output_root / "build-report.json", report)
    return report


def _stable_record_selection(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["sample_id"]))
    if len(ordered) < limit:
        raise SetupError(f"requested {limit} critical records but only {len(ordered)} exist")
    if len(ordered) == limit:
        return ordered
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(indices)) != limit:
        raise SetupError("critical-lot spacing produced duplicate indices")
    return [ordered[index] for index in indices]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.jsonl")
    count = 0
    with partial.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(partial, path)
    return count


def build_critical_lots(dataset_root: Path) -> dict[str, Any]:
    """Build two disjoint human-review lots without claiming either is validated."""

    dataset_root = dataset_root.resolve()
    pointing_manifest = dataset_root / "corpus" / POINTING_CORPUS_ID / "manifest.jsonl"
    registration_manifest = dataset_root / "corpus" / REGISTRATION_CORPUS_ID / "manifest.jsonl"
    if not pointing_manifest.is_file() or not registration_manifest.is_file():
        raise SetupError("both spatial manifests are required before critical-lot build")

    pointing_group_splits: dict[str, set[str]] = defaultdict(set)
    pointing_candidate_pool: list[dict[str, Any]] = []
    for row in _iter_jsonl(pointing_manifest):
        pointing_group_splits[str(row["split_group"])].add(str(row["proposed_split"]))
        if row["proposed_split"] == "test" and row["pointing_status"] == "point_candidate":
            pointing_candidate_pool.append(row)
    pointing_candidates = [
        row
        for row in pointing_candidate_pool
        if pointing_group_splits[str(row["split_group"])] == {"test"}
    ]
    pointing_selected = _stable_record_selection(pointing_candidates, limit=64)
    pointing_output: list[dict[str, Any]] = []
    for row in pointing_selected:
        output_row = dict(row)
        output_row["critical_lot"] = {
            "lot_id": POINTING_CRITICAL_CORPUS_ID,
            "source_split": "test",
            "excluded_from_training": True,
            "validation_status": "awaiting_double_validation",
            "minimum_validators": 2,
            "validators": [],
        }
        output_row["targets"] = [
            {
                **target,
                "critical_validation": {
                    "status": "awaiting_double_validation",
                    "minimum_validators": 2,
                    "validators": [],
                },
            }
            for target in row["targets"]
        ]
        pointing_output.append(output_row)

    registration_rows = list(_iter_jsonl(registration_manifest))
    registration_group_splits: dict[str, set[str]] = defaultdict(set)
    for row in registration_rows:
        registration_group_splits[str(row["split_group"])].add(str(row["split"]))
    registration_candidates = [
        row
        for row in registration_rows
        if row["split"] == "test" and registration_group_splits[str(row["split_group"])] == {"test"}
    ]
    registration_selected = _stable_record_selection(registration_candidates, limit=32)
    registration_output = [
        {
            **row,
            "critical_lot": {
                "lot_id": REGISTRATION_CRITICAL_CORPUS_ID,
                "source_split": "test",
                "excluded_from_training": True,
                "validation_status": "awaiting_double_validation",
                "minimum_validators": 2,
                "validators": [],
            },
        }
        for row in registration_selected
    ]

    pointing_root = dataset_root / "corpus" / POINTING_CRITICAL_CORPUS_ID
    registration_root = dataset_root / "corpus" / REGISTRATION_CRITICAL_CORPUS_ID
    pointing_path = pointing_root / "manifest.jsonl"
    registration_path = registration_root / "manifest.jsonl"
    pointing_count = _write_jsonl(pointing_path, pointing_output)
    registration_count = _write_jsonl(registration_path, registration_output)
    reports = {
        "fire_pointing": {
            "corpus_id": POINTING_CRITICAL_CORPUS_ID,
            "rows": pointing_count,
            "candidate_pool_rows": len(pointing_candidates),
            "manifest_relpath": pointing_path.relative_to(dataset_root).as_posix(),
            "manifest_sha256": _sha256_file(pointing_path),
            "source_split": "test",
            "split_group_leaks": 0,
            "double_validation_complete": False,
            "deployment_ready": False,
        },
        "cross_view_registration": {
            "corpus_id": REGISTRATION_CRITICAL_CORPUS_ID,
            "rows": registration_count,
            "candidate_pool_rows": len(registration_candidates),
            "manifest_relpath": registration_path.relative_to(dataset_root).as_posix(),
            "manifest_sha256": _sha256_file(registration_path),
            "source_split": "test",
            "split_group_leaks": 0,
            "double_validation_complete": False,
            "deployment_ready": False,
        },
    }
    _write_json(pointing_root / "build-report.json", reports["fire_pointing"])
    _write_json(
        registration_root / "build-report.json",
        reports["cross_view_registration"],
    )
    report = {
        "schema_version": 1,
        "lots": reports,
        "training_membership": False,
        "human_validation_required": True,
        "minimum_validators_per_example": 2,
        "double_validation_complete": False,
    }
    _write_json(dataset_root / "critical-lots-report.json", report)
    return report


def preflight(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    pointing_path = dataset_root / "corpus" / POINTING_CORPUS_ID / "build-report.json"
    registration_path = dataset_root / "corpus" / REGISTRATION_CORPUS_ID / "build-report.json"
    if not pointing_path.is_file() or not registration_path.is_file():
        raise SetupError("both corpus build reports are required")
    pointing = json.loads(pointing_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "families": {
            "fire_pointing": pointing["gates"],
            "cross_view_registration": registration["gates"],
        },
        "gates": {
            "setup_ready": bool(
                pointing["gates"]["setup_ready"] and registration["gates"]["setup_ready"]
            ),
            "smoke_ready": bool(
                pointing["gates"]["smoke_ready"] and registration["gates"]["smoke_ready"]
            ),
            "bootstrap_training_ready": bool(
                pointing["gates"]["bootstrap_training_ready"]
                and registration["gates"]["bootstrap_training_ready"]
            ),
            "training_ready": bool(
                pointing["gates"]["training_ready"] and registration["gates"]["training_ready"]
            ),
            "production_training_ready": bool(
                pointing["gates"]["production_training_ready"]
                and registration["gates"]["production_training_ready"]
            ),
            "deployment_ready": bool(
                pointing["gates"]["deployment_ready"] and registration["gates"]["deployment_ready"]
            ),
        },
        "operational_training_denylist": list(DENIED_OPERATIONAL_TOKENS),
        "training_launched": False,
    }
    _write_json(dataset_root / "spatial-training-preflight.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pointing = subparsers.add_parser("build-pointing")
    pointing.add_argument("--dataset-root", type=Path, required=True)
    pointing.add_argument("--verify-files", action="store_true")

    download = subparsers.add_parser("download-registration")
    download.add_argument("--dataset-root", type=Path, required=True)

    download_odm = subparsers.add_parser("download-odm-domains")
    download_odm.add_argument("--dataset-root", type=Path, required=True)

    audit = subparsers.add_parser("audit-registration-poses")
    audit.add_argument("--dataset-root", type=Path, required=True)
    audit.add_argument("--crop-source-pixels", type=int, default=1536)

    registration = subparsers.add_parser("build-registration")
    registration.add_argument("--dataset-root", type=Path, required=True)
    registration.add_argument("--crop-source-pixels", type=int, default=1536)
    registration.add_argument("--output-pixels", type=int, default=768)
    registration.add_argument("--skip-source-image-sha256", action="store_true")

    critical = subparsers.add_parser("build-critical-lots")
    critical.add_argument("--dataset-root", type=Path, required=True)

    check = subparsers.add_parser("preflight")
    check.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "build-pointing":
        report = build_pointing(args.dataset_root, verify_files=args.verify_files)
    elif args.command == "download-registration":
        report = download_registration_source(args.dataset_root)
    elif args.command == "download-odm-domains":
        report = download_odm_domains(args.dataset_root)
    elif args.command == "audit-registration-poses":
        report = audit_registration_poses(
            args.dataset_root,
            crop_source_pixels=args.crop_source_pixels,
        )
    elif args.command == "build-registration":
        report = build_registration(
            args.dataset_root,
            crop_source_pixels=args.crop_source_pixels,
            output_pixels=args.output_pixels,
            verify_source_images=not args.skip_source_image_sha256,
        )
    elif args.command == "build-critical-lots":
        report = build_critical_lots(args.dataset_root)
    elif args.command == "preflight":
        report = preflight(args.dataset_root)
    else:  # pragma: no cover - argparse rejects unknown commands
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
