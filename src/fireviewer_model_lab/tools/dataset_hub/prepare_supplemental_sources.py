from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

BUFFER_SIZE = 4 * 1024 * 1024
PACKAGE_FORMAT = "firewarning-train-bundle-zip-v1"
SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
FIRESPREAD_ARCHIVE_MD5 = "dc4de320f20b1f05cbd3bee4b688168b"
FIRESPREAD_SOURCE_URL = "https://zenodo.org/records/18200075"
FIRESPREAD_DOWNLOAD_URL = (
    "https://zenodo.org/api/records/18200075/files/FireSpread_MedEU.zip/content"
)
BOREAL_DATASET_ID = "1dce1023-493a-4d63-a906-f2a44f831898"
BOREAL_SOURCE_URL = f"https://etsin.fairdata.fi/dataset/{BOREAL_DATASET_ID}"
BOREAL_SITE_SPLITS = {
    "heinola": "train",
    "karkkila": "train",
    "ruokolahti": "validation",
    "evo": "test",
}
CRISISFACTS_COMMIT = "61813075b8e9f34c5c061a97871e3e17c8d86962"
CRISISFACTS_SOURCE_URL = "https://crisisfacts.github.io/"
CRISISFACTS_EXPECTED_SHA256 = {
    "CrisisFACTs-2022.facts.json": (
        "6b91b5b0d4982c89f97a2cfed0113cfe0d587c89e187e70d390dbf0a10947852"
    ),
    "CrisisFACTs-2022.topics.json": (
        "53eda10c6fb6fb9391fc2169c80338f950f3d47678f7a2904ec6ec6153c3ffef"
    ),
    "LICENSE": "d67e07654da8bada8892745a520269e92e0908a8ef8e8ed391ad2650efbd19e1",
    "README.md": "a7e462fd203fc561965c606c2e36983e124f9918f13e4d2bca37eb7fff1b2e47",
}
CRISISFACTS_WILDFIRE_SPLITS = {
    "CrisisFACTS-001": "train",
    "CrisisFACTS-002": "train",
    "CrisisFACTS-003": "validation",
    "CrisisFACTS-006": "test",
}
IMSR_SOURCE_URL = (
    "https://figshare.com/articles/dataset/"
    "Dataset_of_United_States_Incident_Management_Situation_Reports_from_2007_to_2021/24243184"
)
IMSR_DOI = "10.6084/m9.figshare.24243184.v3"
IMSR_EXPECTED_MD5 = {
    "national_activity.csv": "a2732f92e7b534065f9bc50fd47fb84a",
    "gacc_activity.csv": "a0238661115f602c21a1ed6d65b01cec",
    "wildfire_activity.csv": "5988a850ea7637753428be5ee98bafc3",
    "resource_summary.csv": "6d88f8ca0057d1fa64d443e7f65e31a2",
}
TARTANAIR_REVISION = "0d2d145e973832742a2aaa04b7d2ebffc8d82817"
TARTANAIR_LANDING_PAGE = "https://tartanair.org/"
TARTANAIR_MIRROR_PAGE = "https://huggingface.co/datasets/theairlabcmu/tartanair2"
TARTANAIR_ENVIRONMENT_SPLITS = {
    "DesertGasStation": "train",
    "TerrainBlending": "train",
    "WaterMillDay": "train",
    "SeasideTown": "validation",
    "SeasonalForestAutumn": "test",
}
TARTANAIR_ENVIRONMENT_TYPES = {
    "DesertGasStation": "rural",
    "TerrainBlending": "nature",
    "WaterMillDay": "rural",
    "SeasideTown": "rural",
    "SeasonalForestAutumn": "nature",
}
DIODE_LANDING_PAGE = "https://diode-dataset.org/"
DIODE_DATA_LIST_SHA256 = "d95e6480691c151c002a32ebc3fed9831c387001ecb29a49b7b4fecadb46062f"
DIODE_ARCHIVE_MD5 = {
    "train.tar.gz": "3a94632398fe1d002d89f11743f748b1",
    "val.tar.gz": "5c895d09201b88973c8fe4552a67dd85",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def safe_relative_path(raw: str) -> PurePosixPath:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {raw}")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"Invalid relative path: {raw}")
    return path


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC validation failed: {archive_path}")
        seen: set[str] = set()
        for info in archive.infolist():
            relative = safe_relative_path(info.filename)
            key = relative.as_posix()
            if key in seen:
                raise ValueError(f"Duplicate ZIP entry: {key}")
            seen.add(key)
            if info.is_dir():
                continue
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, BUFFER_SIZE)


def deterministic_group_splits(group_ids: Iterable[str]) -> dict[str, str]:
    ranked = sorted(
        set(group_ids),
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(ranked) < 3:
        raise ValueError("At least three independent groups are required for train/validation/test")
    train_end = max(1, math.floor(len(ranked) * 0.80))
    validation_end = max(train_end + 1, math.floor(len(ranked) * 0.90))
    validation_end = min(validation_end, len(ranked) - 1)
    assignments: dict[str, str] = {}
    for index, group_id in enumerate(ranked):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        assignments[group_id] = split
    return assignments


def _transform_coordinates(value: Any, transformer: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Invalid geometry coordinate structure")
    if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
        longitude, latitude = transformer.transform(float(value[0]), float(value[1]))
        if not (math.isfinite(longitude) and math.isfinite(latitude)):
            raise ValueError("Non-finite transformed coordinate")
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError("Transformed coordinate outside WGS84 bounds")
        return [longitude, latitude, *list(value[2:])]
    return [_transform_coordinates(item, transformer) for item in value]


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload))


def _firespread_properties(record: dict[str, Any]) -> dict[str, Any]:
    prop_step = record.get("Prop_step")
    burned_area = record.get("BA (ha)")
    acquisition_date = str(record.get("Acqu_date") or "").strip() or None
    acquisition_time = str(record.get("Acqu_time") or "").strip() or None
    acquisition_local = acquisition_date
    if acquisition_date and acquisition_time:
        acquisition_local = f"{acquisition_date}T{acquisition_time}"
    return {
        "acquisition_local": acquisition_local,
        "acquisition_timezone": None,
        "burned_area_ha": float(burned_area) if burned_area is not None else None,
        "cloud_score": float(record["Clouds"]) if record.get("Clouds") is not None else None,
        "info": str(record.get("Info") or "").strip() or None,
        "land_cover_percent": {
            "bare": record.get("Bare"),
            "crop": record.get("Crop"),
            "crop_vegetation": record.get("Crop/Veg"),
            "grassland": record.get("Grassland"),
            "mixed_trees": record.get("Tree-Mixed"),
            "mixed_vegetation": record.get("Veg-Mixed"),
            "needleleaf_trees": record.get("Tree-Needl"),
            "broadleaf_trees": record.get("Tree-Broad"),
            "shrub": record.get("Shrub"),
            "sparse_vegetation": record.get("Sparse Veg"),
            "urban": record.get("Urban"),
        },
        "propagation_step": int(prop_step) if prop_step is not None else None,
        "quality": str(record.get("Quality") or "").strip() or None,
        "smoke_score": float(record["Smoke"]) if record.get("Smoke") is not None else None,
        "usable_as_cumulative_burn_target": bool(
            prop_step is not None and float(prop_step) > 0 and burned_area is not None
        ),
    }


def prepare_firespread(source_zip: Path, output_root: Path, *, force: bool) -> dict[str, Any]:
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    observed_md5 = md5_file(source_zip)
    if observed_md5 != FIRESPREAD_ARCHIVE_MD5:
        raise ValueError(
            f"FireSpread archive MD5 mismatch: {observed_md5} != {FIRESPREAD_ARCHIVE_MD5}"
        )
    if output_root.exists():
        if not force:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    try:
        import shapefile  # type: ignore[import-not-found]
        from pyproj import Transformer
    except ImportError as error:
        raise RuntimeError("FireSpread conversion requires pyshp and pyproj") from error

    with tempfile.TemporaryDirectory(prefix="firespread-source-") as temp:
        extracted = Path(temp)
        safe_extract_zip(source_zip, extracted)
        required = {
            "FireSpread_MedEU.cpg",
            "FireSpread_MedEU.dbf",
            "FireSpread_MedEU.prj",
            "FireSpread_MedEU.shp",
            "FireSpread_MedEU.shx",
            "feature_description.pdf",
        }
        observed = {path.name for path in extracted.iterdir() if path.is_file()}
        if observed != required:
            raise ValueError(
                f"Unexpected FireSpread source inventory: missing={sorted(required - observed)} "
                f"extra={sorted(observed - required)}"
            )
        source_dir = output_root / "source"
        shutil.copytree(extracted, source_dir)

        reader = shapefile.Reader(str(extracted / "FireSpread_MedEU.shp"), encoding="utf-8")
        try:
            if reader.shapeTypeName != "POLYGON":
                raise ValueError(f"Unexpected FireSpread shape type: {reader.shapeTypeName}")
            transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
            events: dict[str, list[dict[str, Any]]] = defaultdict(list)
            source_records = 0
            for source_index, shape_record in enumerate(reader.iterShapeRecords()):
                record = dict(shape_record.record.as_dict())
                event_id = str(int(record["EFFIS_id"]))
                if shape_record.shape.shapeType == shapefile.NULL:
                    geometry = None
                else:
                    geometry = dict(shape_record.shape.__geo_interface__)
                    geometry["coordinates"] = _transform_coordinates(
                        geometry["coordinates"], transformer
                    )
                properties = _firespread_properties(record)
                if geometry is None:
                    properties["usable_as_cumulative_burn_target"] = False
                    properties["exclusion_reason"] = "source_geometry_missing"
                properties.update(
                    {
                        "event_id": event_id,
                        "source_record_index": source_index,
                    }
                )
                events[event_id].append(
                    {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": properties,
                    }
                )
                source_records += 1
        finally:
            reader.close()

    assignments = deterministic_group_splits(f"firespread:{event_id}" for event_id in events)
    manifest_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    split_observations: Counter[str] = Counter()
    usable_observations = 0
    quarantined_target_observations = 0
    duplicate_event_payloads: dict[str, list[str]] = defaultdict(list)
    geometry_owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sequence_anomalies: list[dict[str, Any]] = []
    event_dir = output_root / "events"
    event_dir.mkdir()

    for event_id in sorted(events, key=int):
        features = sorted(
            events[event_id],
            key=lambda feature: (
                feature["properties"]["acquisition_local"] or "9999",
                feature["properties"]["source_record_index"],
            ),
        )
        previous_step: int | None = None
        previous_area: float | None = None
        event_anomalies: list[dict[str, Any]] = []
        event_usable = 0
        for feature in features:
            properties = feature["properties"]
            geometry = feature["geometry"]
            if geometry is not None:
                geometry_digest = hashlib.sha256(_canonical_json_bytes(geometry)).hexdigest()
                geometry_owners[geometry_digest].append(
                    {
                        "event_id": event_id,
                        "source_record_index": properties["source_record_index"],
                    }
                )
            if not properties["usable_as_cumulative_burn_target"]:
                continue
            step = int(properties["propagation_step"])
            area = float(properties["burned_area_ha"])
            observation_anomalies: list[dict[str, Any]] = []
            if previous_step is not None and step <= previous_step:
                observation_anomalies.append(
                    {"kind": "non_increasing_step", "previous": previous_step, "current": step}
                )
            if previous_area is not None and area < previous_area:
                observation_anomalies.append(
                    {
                        "kind": "decreasing_cumulative_area",
                        "previous": previous_area,
                        "current": area,
                    }
                )
            if observation_anomalies:
                properties["usable_as_cumulative_burn_target"] = False
                properties["exclusion_reason"] = "sequence_inconsistency"
                quarantined_target_observations += 1
                event_anomalies.extend(
                    {
                        **item,
                        "source_record_index": properties["source_record_index"],
                    }
                    for item in observation_anomalies
                )
                continue
            event_usable += 1
            previous_step = step
            previous_area = area
        if event_anomalies:
            sequence_anomalies.append({"event_id": event_id, "items": event_anomalies})

        payload = {
            "type": "FeatureCollection",
            "fireviewer_contract": {
                "schema_version": SCHEMA_VERSION,
                "task": "fire_progression_and_front_inference",
                "source_id": "firespread-medeu-v1",
                "event_id": event_id,
                "source_crs": "EPSG:3035",
                "output_crs": "EPSG:4326",
                "target_semantics": "cumulative_burned_area_polygon_not_active_fire_front",
            },
            "features": features,
        }
        relative = Path("events") / f"effis-{event_id}.geojson"
        artifact_path = output_root / relative
        _write_json(artifact_path, payload)
        artifact_sha256 = sha256_file(artifact_path)
        duplicate_event_payloads[artifact_sha256].append(event_id)
        group_id = f"firespread:{event_id}"
        split = assignments[group_id]
        split_counts[split] += 1
        split_observations[split] += len(features)
        usable_observations += event_usable
        manifest_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"firespread-medeu:effis-{event_id}",
                "source_id": "firespread-medeu-v1",
                "source_record_id": event_id,
                "task": "fire_progression_and_front_inference",
                "split": split,
                "split_group": group_id,
                "license": "CC-BY-4.0",
                "provenance": {
                    "landing_page": FIRESPREAD_SOURCE_URL,
                    "download_url": FIRESPREAD_DOWNLOAD_URL,
                    "archive_md5": observed_md5,
                },
                "artifact": {
                    "path": relative.as_posix(),
                    "sha256": artifact_sha256,
                    "media_type": "application/geo+json",
                },
                "event_observations": len(features),
                "usable_target_observations": event_usable,
                "target_semantics": "cumulative_burned_area_polygon_not_active_fire_front",
            }
        )

    duplicates = {
        digest: event_ids
        for digest, event_ids in duplicate_event_payloads.items()
        if len(event_ids) > 1
    }
    if duplicates:
        raise ValueError(f"Duplicate FireSpread event artifacts: {duplicates}")
    duplicate_geometries = {
        digest: owners for digest, owners in geometry_owners.items() if len(owners) > 1
    }
    geometry_split_leakage = []
    for digest, owners in duplicate_geometries.items():
        splits = {assignments[f"firespread:{owner['event_id']}"] for owner in owners}
        if len(splits) > 1:
            geometry_split_leakage.append({"sha256": digest, "owners": owners})
    if geometry_split_leakage:
        raise ValueError(
            f"FireSpread exact geometry leaks across splits: {len(geometry_split_leakage)}"
        )
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in manifest_rows
        ),
        encoding="utf-8",
        newline="\n",
    )

    source_inventory = []
    for path in iter_files(output_root / "source"):
        source_inventory.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_id": "firespread-medeu-v1",
        "title": "FireSpread_MedEU",
        "landing_page": FIRESPREAD_SOURCE_URL,
        "download_url": FIRESPREAD_DOWNLOAD_URL,
        "license": "CC-BY-4.0",
        "archive": {
            "filename": source_zip.name,
            "size_bytes": source_zip.stat().st_size,
            "md5": observed_md5,
            "sha256": sha256_file(source_zip),
        },
        "files": source_inventory,
    }
    _write_json(output_root / "SOURCE_MANIFEST.json", source_manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "firespread-medeu-v1",
        "files_verified": True,
        "source_records": source_records,
        "event_groups": len(events),
        "manifest_rows": len(manifest_rows),
        "usable_target_observations": usable_observations,
        "quarantined_target_observations": quarantined_target_observations,
        "split_event_counts": dict(sorted(split_counts.items())),
        "split_observation_counts": dict(sorted(split_observations.items())),
        "split_group_leakage": 0,
        "exact_duplicate_event_artifacts": 0,
        "exact_duplicate_geometries": len(duplicate_geometries),
        "exact_geometry_split_leakage": 0,
        "sequence_anomalies": sequence_anomalies,
        "manifest_sha256": sha256_file(manifest_path),
        "target_semantics": "cumulative_burned_area_polygon_not_active_fire_front",
    }
    _write_json(output_root / "VALIDATION_REPORT.json", report)
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _boreal_site(filename: str) -> str:
    normalized = filename.lower()
    for site in BOREAL_SITE_SPLITS:
        if normalized.startswith(site):
            return site
    raise ValueError(f"Cannot determine Boreal collection site from {filename}")


def _materialize_payload(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _validate_yolo_label(path: Path, *, allow_empty: bool) -> int:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines and not allow_empty:
        raise ValueError(f"Unexpected empty Boreal label: {path}")
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
        class_id = float(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id != 0 or any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(f"Out-of-range YOLO label at {path}:{line_number}")
    return len(lines)


def _boreal_raw_path(raw_root: Path, pathname: str) -> Path:
    relative = safe_relative_path(pathname.lstrip("/"))
    return raw_root.joinpath(*relative.parts)


def _boreal_inventory(raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    metadata = raw_root / "_fireviewer_metadata"
    report_path = metadata / "DOWNLOAD_REPORT.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Boreal download is incomplete: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("profile") != "images" or report.get("complete") is not True:
        raise ValueError("Boreal images profile is not fully downloaded and verified")
    inventory_path = metadata / "OFFICIAL_IMAGES_INVENTORY.jsonl"
    rows = _read_jsonl(inventory_path)
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        pathname = str(row["pathname"])
        source = _boreal_raw_path(raw_root, pathname)
        if not source.is_file() or source.stat().st_size != int(row["size"]):
            raise ValueError(f"Missing or truncated Boreal source file: {pathname}")
        if sha256_file(source) != str(row["sha256"]):
            raise ValueError(f"Boreal source SHA-256 mismatch: {pathname}")
        by_path[pathname] = row
    return rows, by_path


def _boreal_payload_ref(
    *,
    raw_root: Path,
    output_root: Path,
    row: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    source = _boreal_raw_path(raw_root, str(row["pathname"]))
    suffix = source.suffix.lower() or ".bin"
    digest = str(row["sha256"])
    relative = Path("payload") / digest[:2] / f"{digest}{suffix}"
    _materialize_payload(source, output_root / relative)
    return {
        "path": relative.as_posix(),
        "sha256": digest,
        "size_bytes": int(row["size"]),
        "role": role,
        "official_pathname": str(row["pathname"]),
    }


def _boreal_detection_pairs(
    rows: list[dict[str, Any]], by_path: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for image in rows:
        pathname = str(image["pathname"])
        if not pathname.lower().endswith(".jpg"):
            continue
        name = PurePosixPath(pathname).name
        if "/Boreal-Forest-Fire-Subset-A/" in pathname:
            label_path = pathname.replace("-Images/", "-Labels/").rsplit(".", 1)[0] + ".txt"
            subset = "A"
            upstream_split = None
            is_negative = "/Empty-Images/" in pathname
        elif "/Boreal-Forest-Fire-Subset-C/images/" in pathname:
            label_path = pathname.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            subset = "C"
            upstream_split = PurePosixPath(pathname).parent.name
            is_negative = False
        else:
            continue
        label = by_path.get(label_path)
        if label is None:
            excluded.append(
                {
                    "image": pathname,
                    "reason": "official_detection_label_missing",
                }
            )
            continue
        # The official paper documents exactly 256 Subset-A negatives and the
        # release places those images under Empty-Images with zero-byte YOLO
        # files.  A few other Subset-A files and some Subset-C detection labels
        # are also empty, but they are outside that documented negative set.
        # Do not silently turn those inconsistencies into negative supervision.
        if int(label["size"]) == 0 and not is_negative:
            excluded.append(
                {
                    "image": pathname,
                    "label": label_path,
                    "reason": "official_empty_detection_label_outside_documented_negative_set",
                    "upstream_subset": subset,
                }
            )
            continue
        pairs.append(
            {
                "image": image,
                "label": label,
                "site": _boreal_site(name),
                "subset": subset,
                "upstream_split": upstream_split,
                "negative": is_negative,
            }
        )
    return pairs, excluded


def _boreal_segmentation_pairs(
    rows: list[dict[str, Any]], by_path: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for image in rows:
        pathname = str(image["pathname"])
        if "/Boreal-Forest-Fire-Subset-C/images/" not in pathname or not pathname.endswith(".jpg"):
            continue
        name = PurePosixPath(pathname).name
        sam_path = pathname.replace("/images/", "/sam_masks/").rsplit(".", 1)[0] + ".png"
        manual_path = pathname.replace("/images/", "/manual_masks/").rsplit(".", 1)[0] + ".png"
        manual = by_path.get(manual_path)
        sam = by_path.get(sam_path)
        mask = manual or sam
        if mask is None:
            excluded.append(
                {
                    "image": pathname,
                    "reason": "official_segmentation_mask_missing",
                }
            )
            continue
        pairs.append(
            {
                "image": image,
                "mask": mask,
                "site": _boreal_site(name),
                "upstream_split": PurePosixPath(pathname).parent.name,
                "annotation_provenance": (
                    "human_pixel_mask" if manual is not None else "sam_generated_from_manual_box"
                ),
                "annotation_strength": "strong" if manual is not None else "weak",
            }
        )
    return pairs, excluded


def prepare_boreal_images(
    raw_root: Path,
    output_root: Path,
    *,
    task: str,
    force: bool,
) -> dict[str, Any]:
    if task not in {"detection", "segmentation"}:
        raise ValueError(f"Unsupported Boreal image task: {task}")
    rows, by_path = _boreal_inventory(raw_root)
    pairs, excluded = (
        _boreal_detection_pairs(rows, by_path)
        if task == "detection"
        else _boreal_segmentation_pairs(rows, by_path)
    )
    if output_root.exists():
        if not force:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    source_id = f"boreal-forest-fire-{task}-v1"
    task_id = "wildfire_smoke_detection" if task == "detection" else "wildfire_smoke_segmentation"
    sample_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    site_counts: Counter[str] = Counter()
    strong_masks: Counter[str] = Counter()
    weak_masks: Counter[str] = Counter()
    media_owners: dict[str, list[dict[str, str]]] = defaultdict(list)
    used_official_paths: set[str] = set()
    payload_hashes: set[str] = set()
    total_boxes = 0

    for pair in pairs:
        image = pair["image"]
        site = str(pair["site"])
        split = BOREAL_SITE_SPLITS[site]
        image_ref = _boreal_payload_ref(
            raw_root=raw_root,
            output_root=output_root,
            row=image,
            role="media",
        )
        annotation_key = "label" if task == "detection" else "mask"
        annotation = pair[annotation_key]
        annotation_ref = _boreal_payload_ref(
            raw_root=raw_root,
            output_root=output_root,
            row=annotation,
            role="annotation",
        )
        used_official_paths.update({str(image["pathname"]), str(annotation["pathname"])})
        payload_hashes.update({str(image["sha256"]), str(annotation["sha256"])})
        media_owners[str(image["sha256"])].append({"site": site, "split": split})
        stem = PurePosixPath(str(image["pathname"])).stem
        subset = str(pair.get("subset") or "C")
        upstream_split = pair.get("upstream_split") or "all"
        sample_id = f"boreal:{task}:{subset.lower()}:{upstream_split}:{stem}"
        details: dict[str, Any]
        if task == "detection":
            label_path = _boreal_raw_path(raw_root, str(annotation["pathname"]))
            box_count = _validate_yolo_label(label_path, allow_empty=bool(pair["negative"]))
            total_boxes += box_count
            details = {
                "annotation_format": "yolo_normalized_xywh",
                "class_map": {"0": "smoke"},
                "box_count": box_count,
                "negative": bool(pair["negative"]),
                "annotation_provenance": "human_bounding_box",
            }
        else:
            provenance = str(pair["annotation_provenance"])
            strength = str(pair["annotation_strength"])
            if strength == "strong":
                strong_masks[split] += 1
            else:
                weak_masks[split] += 1
            details = {
                "annotation_format": "binary_png_mask",
                "class_map": {"1": "smoke"},
                "annotation_provenance": provenance,
                "annotation_strength": strength,
            }
        artifact_payload = {
            "schema_version": SCHEMA_VERSION,
            "source_id": source_id,
            "sample_id": sample_id,
            "task": task_id,
            "collection_site": site,
            "split": split,
            "upstream_subset": subset,
            "upstream_split": upstream_split,
            "image": image_ref,
            "annotation": annotation_ref,
            **details,
        }
        artifact_relative = (
            Path("samples") / site / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"
        )
        artifact_path = output_root / artifact_relative
        _write_json(artifact_path, artifact_payload)
        sample_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "source_id": source_id,
                "source_record_id": str(image["pathname"]),
                "task": task_id,
                "split": split,
                "split_group": f"boreal-site:{site}",
                "license": "CC-BY-4.0",
                "provenance": {
                    "landing_page": BOREAL_SOURCE_URL,
                    "dataset_id": BOREAL_DATASET_ID,
                    "collection_site": site,
                    "upstream_split": upstream_split,
                },
                "artifact": {
                    "path": artifact_relative.as_posix(),
                    "sha256": sha256_file(artifact_path),
                    "media_type": "application/json",
                },
                "referenced_payloads": [image_ref, annotation_ref],
            }
        )
        split_counts[split] += 1
        site_counts[site] += 1

    cross_site_media = {
        digest: owners
        for digest, owners in media_owners.items()
        if len({owner["site"] for owner in owners}) > 1
    }
    if cross_site_media:
        raise ValueError(f"Boreal media duplicates across sites: {len(cross_site_media)}")
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in sample_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    source_dir = output_root / "source"
    source_dir.mkdir()
    official_inventory_path = source_dir / "OFFICIAL_SELECTED_INVENTORY.jsonl"
    official_inventory_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
            if str(row["pathname"]) in used_official_paths
        ),
        encoding="utf-8",
        newline="\n",
    )
    _write_json(source_dir / "EXCLUDED_RECORDS.json", excluded)
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "title": (
            "Boreal Forest Fire: UAV-collected Wildfire Detection and Smoke Segmentation Dataset"
        ),
        "landing_page": BOREAL_SOURCE_URL,
        "license": "CC-BY-4.0",
        "task": task_id,
        "official_inventory": {
            "selected_source_files": len(used_official_paths),
            "sha256": sha256_file(official_inventory_path),
        },
        "split_policy": {
            "unit": "collection_site",
            "assignments": BOREAL_SITE_SPLITS,
            "upstream_split_ignored_for_fireviewer_split": True,
        },
        "annotation_policy": {
            "sam_masks_are_weak_labels": True,
            "manual_masks_are_strong_labels": True,
        },
        "normalized_payload": {
            "unique_sha256": len(payload_hashes),
            "exact_media_duplicates_across_sites": 0,
        },
    }
    _write_json(output_root / "SOURCE_MANIFEST.json", source_manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": source_id,
        "task": task_id,
        "files_verified": True,
        "samples": len(sample_rows),
        "excluded_samples": len(excluded),
        "split_counts": dict(sorted(split_counts.items())),
        "site_counts": dict(sorted(site_counts.items())),
        "split_group_leakage": 0,
        "exact_media_duplicates_across_sites": 0,
        "unique_payload_sha256": len(payload_hashes),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if task == "detection":
        report["total_smoke_boxes"] = total_boxes
    else:
        report["strong_human_masks_by_split"] = dict(sorted(strong_masks.items()))
        report["weak_sam_masks_by_split"] = dict(sorted(weak_masks.items()))
    _write_json(output_root / "VALIDATION_REPORT.json", report)
    return report


def _replace_output_root(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def _verify_named_files(
    raw_root: Path, expected: dict[str, str], *, algorithm: str
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name, expected_digest in expected.items():
        path = raw_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path) if algorithm == "sha256" else md5_file(path)
        if digest != expected_digest:
            raise ValueError(
                f"Source {algorithm.upper()} mismatch for {name}: {digest} != {expected_digest}"
            )
        observed[name] = {
            "size_bytes": path.stat().st_size,
            algorithm: digest,
        }
    return observed


def prepare_crisisfacts(raw_root: Path, output_root: Path, *, force: bool) -> dict[str, Any]:
    verified = _verify_named_files(raw_root, CRISISFACTS_EXPECTED_SHA256, algorithm="sha256")
    facts = json.loads((raw_root / "CrisisFACTs-2022.facts.json").read_text(encoding="utf-8"))
    topics = json.loads((raw_root / "CrisisFACTs-2022.topics.json").read_text(encoding="utf-8"))
    if not isinstance(facts, list) or not isinstance(topics, list):
        raise ValueError("Unexpected CrisisFACTS JSON structure")
    topic_by_event = {str(topic["eventID"]): topic for topic in topics}
    selected_events = {
        event_id
        for event_id, topic in topic_by_event.items()
        if str(topic.get("type")) == "Wildfire"
    }
    if selected_events != set(CRISISFACTS_WILDFIRE_SPLITS):
        raise ValueError(f"Unexpected CrisisFACTS wildfire inventory: {selected_events}")
    _replace_output_root(output_root, force=force)
    source_id = "crisisfacts-wildfire-2022-v1"
    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    fact_counts: Counter[str] = Counter()
    source_identifiers: set[str] = set()
    duplicate_facts_removed = 0

    for event in facts:
        event_id = str(event.get("eventID"))
        if event_id not in selected_events:
            continue
        topic = topic_by_event[event_id]
        split = CRISISFACTS_WILDFIRE_SPLITS[event_id]
        facts_by_request = event.get("factsByRequest")
        if not isinstance(facts_by_request, dict):
            raise ValueError(f"Missing factsByRequest for {event_id}")
        requests = event.get("summaryRequests")
        if not isinstance(requests, list):
            raise ValueError(f"Missing summaryRequests for {event_id}")
        for request in requests:
            request_id = str(request["requestID"])
            date = str(request["dateString"])
            daily_facts = facts_by_request.get(request_id, [])
            if not isinstance(daily_facts, list):
                raise ValueError(f"Invalid fact list for {request_id}")
            canonical: dict[str, dict[str, Any]] = {}
            for fact in daily_facts:
                text = " ".join(str(fact.get("fact") or "").split())
                if not text:
                    continue
                source_identifier = str(fact.get("source") or "").strip()
                if source_identifier:
                    source_identifiers.add(source_identifier)
                key = text.casefold()
                if key in canonical:
                    duplicate_facts_removed += 1
                    identifiers = canonical[key]["source_identifiers"]
                    if source_identifier and source_identifier not in identifiers:
                        identifiers.append(source_identifier)
                else:
                    canonical[key] = {
                        "text": text,
                        "source_identifiers": [source_identifier] if source_identifier else [],
                    }
            if not canonical:
                continue
            sample_id = f"crisisfacts:{event_id}:{date}"
            artifact_payload = {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "sample_id": sample_id,
                "task": "daily_wildfire_situational_fact_synthesis",
                "split": split,
                "event": {
                    "event_id": event_id,
                    "title": str(topic["title"]),
                    "date": date,
                    "description": str(topic.get("description") or ""),
                    "reference_url": str(topic.get("url") or ""),
                },
                "input_contract": {
                    "available": "event metadata and source identifiers only",
                    "excluded": "raw social-media posts, news articles and Facebook content",
                },
                "target_facts": list(canonical.values()),
            }
            relative = (
                Path("samples")
                / event_id
                / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"
            )
            artifact = output_root / relative
            _write_json(artifact, artifact_payload)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "source_record_id": request_id,
                    "task": "daily_wildfire_situational_fact_synthesis",
                    "split": split,
                    "split_group": f"crisisfacts-event:{event_id}",
                    "license": "MIT",
                    "provenance": {
                        "landing_page": CRISISFACTS_SOURCE_URL,
                        "repository_commit": CRISISFACTS_COMMIT,
                        "event_id": event_id,
                        "date": date,
                        "raw_source_content_redistributed": False,
                    },
                    "artifact": {
                        "path": relative.as_posix(),
                        "sha256": sha256_file(artifact),
                        "media_type": "application/json",
                    },
                }
            )
            split_counts[split] += 1
            fact_counts[split] += len(canonical)

    if not rows:
        raise ValueError("No CrisisFACTS wildfire samples were produced")
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    source_dir = output_root / "source"
    source_dir.mkdir()
    for name in CRISISFACTS_EXPECTED_SHA256:
        shutil.copy2(raw_root / name, source_dir / name)
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "title": "CrisisFACTS 2022 wildfire daily fact annotations",
        "landing_page": CRISISFACTS_SOURCE_URL,
        "license": "MIT",
        "license_scope": (
            "Pinned utilities repository and its derived annotations. Raw platform "
            "streams are excluded and are not redistributed."
        ),
        "repository_commit": CRISISFACTS_COMMIT,
        "task": "daily_wildfire_situational_fact_synthesis",
        "source_files": verified,
        "selected_events": sorted(selected_events),
        "split_policy": {
            "unit": "event",
            "assignments": CRISISFACTS_WILDFIRE_SPLITS,
        },
        "redistribution_policy": {
            "raw_social_media": False,
            "raw_news_articles": False,
            "facebook_content": False,
            "source_identifiers_only": True,
        },
    }
    _write_json(output_root / "SOURCE_MANIFEST.json", source_manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": source_id,
        "files_verified": True,
        "events": len(selected_events),
        "samples": len(rows),
        "facts": sum(fact_counts.values()),
        "source_identifiers": len(source_identifiers),
        "duplicate_facts_removed": duplicate_facts_removed,
        "split_counts": dict(sorted(split_counts.items())),
        "fact_counts_by_split": dict(sorted(fact_counts.items())),
        "split_group_leakage": 0,
        "raw_source_content_redistributed": False,
        "manifest_sha256": sha256_file(manifest),
    }
    _write_json(output_root / "VALIDATION_REPORT.json", report)
    return report


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    # Figshare v1.06 contains Windows-1252 bytes (for example 0xBD in
    # "2 1/2 MILE") despite the paper's general Unicode-normalization claim.
    # Decode strictly as CP1252: never replace or discard undecodable bytes.
    with path.open(encoding="cp1252", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _imsr_event_proxy(row: dict[str, str]) -> str:
    date = str(row.get("imsr_date") or "")
    year = date[:4]
    unit = " ".join(str(row.get("unit") or "UNKNOWN").upper().split())
    name = " ".join(str(row.get("fire_name") or "UNKNOWN").upper().split())
    return f"{year}|{unit}|{name}"


def prepare_imsr(raw_root: Path, output_root: Path, *, force: bool) -> dict[str, Any]:
    verified = _verify_named_files(raw_root, IMSR_EXPECTED_MD5, algorithm="md5")
    national_rows = _read_csv_rows(raw_root / "national_activity.csv")
    gacc_rows = _read_csv_rows(raw_root / "gacc_activity.csv")
    wildfire_rows = _read_csv_rows(raw_root / "wildfire_activity.csv")
    resource_rows = _read_csv_rows(raw_root / "resource_summary.csv")
    national_by_date = {row["imsr_date"]: row for row in national_rows}
    gacc_by_key = {(row["imsr_date"], row["gacc"]): row for row in gacc_rows}
    resource_by_key = {(row["imsr_date"], row["gacc"]): row for row in resource_rows}
    wildfire_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for wildfire in wildfire_rows:
        wildfire_by_event[_imsr_event_proxy(wildfire)].append(wildfire)
    event_proxies = set(wildfire_by_event)
    assignments = deterministic_group_splits(
        f"imsr-incident:{event_proxy}" for event_proxy in event_proxies
    )
    _replace_output_root(output_root, force=force)
    source_id = "imsr-2007-2021-structured-v1"
    rows: list[dict[str, Any]] = []
    shard_counts: Counter[str] = Counter()
    incident_counts: Counter[str] = Counter()
    daily_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    missing_context: Counter[str] = Counter()
    conflicting_event_days = 0
    sequences_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event_proxy, event_rows in sorted(wildfire_by_event.items()):
        split = assignments[f"imsr-incident:{event_proxy}"]
        days: list[dict[str, Any]] = []
        rows_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
        for wildfire in event_rows:
            rows_by_date[str(wildfire["imsr_date"])].append(wildfire)
        for date, candidates in sorted(rows_by_date.items()):
            gacc_values = {str(candidate["gacc"]) for candidate in candidates}
            gacc = next(iter(gacc_values)) if len(gacc_values) == 1 else ""
            national = national_by_date.get(date)
            gacc_context = gacc_by_key.get((date, gacc)) if gacc else None
            resources = resource_by_key.get((date, gacc)) if gacc else None
            if national is None:
                missing_context["national"] += 1
            if gacc_context is None:
                missing_context["gacc"] += 1
            if resources is None:
                missing_context["resource"] += 1
            if len(candidates) > 1:
                conflicting_event_days += 1
            days.append(
                {
                    "date": date,
                    "wildfire_activity_candidates": candidates,
                    "source_conflict": len(candidates) > 1,
                    "gacc_activity": gacc_context,
                    "resource_summary": resources,
                    "national_activity": national,
                }
            )
        sequences_by_split[split].append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "task": "structured_daily_wildfire_situation_and_resources",
                "split": split,
                "event_proxy": {
                    "value": event_proxy,
                    "limitation": (
                        "IMSR has no stable incident identifier; proxy uses "
                        "report year, unit and fire name"
                    ),
                },
                "days": days,
            }
        )
        incident_counts[split] += 1
        daily_counts[split] += len(days)

    shard_size = 256
    for split in SPLITS:
        sequences = sequences_by_split[split]
        for offset in range(0, len(sequences), shard_size):
            shard_index = offset // shard_size
            shard = sequences[offset : offset + shard_size]
            sample_id = f"imsr:{split}:shard-{shard_index:04d}"
            relative = Path("samples") / split / f"shard-{shard_index:04d}.jsonl"
            artifact = output_root / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
                    for item in shard
                ),
                encoding="utf-8",
                newline="\n",
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "source_record_id": f"{split}:{shard_index}",
                    "task": "structured_daily_wildfire_situation_and_resources",
                    "split": split,
                    "split_group": f"imsr-shard:{split}:{shard_index}",
                    "license": "CC-BY-4.0",
                    "provenance": {
                        "landing_page": IMSR_SOURCE_URL,
                        "doi": IMSR_DOI,
                    },
                    "artifact": {
                        "path": relative.as_posix(),
                        "sha256": sha256_file(artifact),
                        "media_type": "application/x-ndjson",
                    },
                    "sequence_count": len(shard),
                    "day_count": sum(len(item["days"]) for item in shard),
                }
            )
            shard_counts[split] += 1

    event_counts.update(incident_counts)
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    source_dir = output_root / "source"
    source_dir.mkdir()
    for name in IMSR_EXPECTED_MD5:
        shutil.copy2(raw_root / name, source_dir / name)
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "title": "Dataset of United States Incident Management Situation Reports, 2007-2021",
        "landing_page": IMSR_SOURCE_URL,
        "doi": IMSR_DOI,
        "license": "CC-BY-4.0",
        "task": "structured_daily_wildfire_situation_and_resources",
        "source_files": verified,
        "source_csv_encoding": "windows-1252-strict",
        "excluded_source_file": "2007-2021-IMSR-PDFs.zip",
        "excluded_source_reason": (
            "The four official structured CSV files are sufficient; "
            "the 1.1 GB PDF archive is not duplicated"
        ),
        "split_policy": {
            "unit": "incident_proxy",
            "incident_proxy": "report year + unit + normalized fire name",
            "random_rows_or_days": False,
        },
        "storage": {
            "format": "jsonl shards of incident sequences",
            "incident_sequences_per_shard": shard_size,
        },
        "known_limitations": [
            "United States operational vocabulary and structures differ from France",
            "The source does not expose a stable incident identifier",
            "Airtanker details are not present in IMSR and require SIT-209 cross-reference",
        ],
    }
    _write_json(output_root / "SOURCE_MANIFEST.json", source_manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": source_id,
        "files_verified": True,
        "shards": len(rows),
        "incident_sequences": len(event_proxies),
        "daily_records": len(wildfire_rows),
        "incident_days": sum(daily_counts.values()),
        "conflicting_event_days_preserved": conflicting_event_days,
        "incident_proxy_groups": len(event_proxies),
        "split_counts": dict(sorted(shard_counts.items())),
        "incident_counts_by_split": dict(sorted(incident_counts.items())),
        "daily_counts_by_split": dict(sorted(daily_counts.items())),
        "event_counts_by_split": dict(sorted(event_counts.items())),
        "missing_context_rows": dict(sorted(missing_context.items())),
        "split_group_leakage": 0,
        "manifest_sha256": sha256_file(manifest),
    }
    _write_json(output_root / "VALIDATION_REPORT.json", report)
    return report


def _materialize_stream_payload(
    *,
    source: Any,
    output_root: Path,
    original_name: str,
    role: str,
    media_type: str,
) -> dict[str, Any]:
    staging = output_root / "payload" / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{hashlib.sha256(original_name.encode()).hexdigest()}.partial"
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as output:
        while True:
            chunk = source.read(BUFFER_SIZE)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    sha256 = digest.hexdigest()
    suffix = PurePosixPath(original_name).suffix.lower()
    destination_relative = Path("payload") / role / sha256[:2] / f"{sha256}{suffix}"
    destination = output_root / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != size or sha256_file(destination) != sha256:
            raise ValueError(f"Conflicting normalized payload: {destination_relative}")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    return {
        "path": destination_relative.as_posix(),
        "sha256": sha256,
        "size_bytes": size,
        "role": role,
        "media_type": media_type,
        "source_name": original_name,
    }


def _read_pose_lines(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> list[list[float]]:
    with archive.open(info, "r") as stream:
        text = stream.read().decode("utf-8")
    poses: list[list[float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid TartanAir pose at {info.filename}:{line_number}")
        poses.append(values)
    if not poses:
        raise ValueError(f"Empty TartanAir pose file: {info.filename}")
    return poses


def _tartanair_archive_paths(raw_root: Path) -> dict[tuple[str, str], Path]:
    from download_supplemental_archives import TARTANAIR_ARCHIVES

    paths: dict[tuple[str, str], Path] = {}
    for spec in TARTANAIR_ARCHIVES:
        relative = PurePosixPath(spec.relative_path)
        environment = relative.parts[-3]
        modality = relative.name.split("_", maxsplit=1)[0]
        path = raw_root.joinpath(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != spec.size:
            raise ValueError(f"TartanAir archive size mismatch: {path}")
        if sha256_file(path) != spec.checksum:
            raise ValueError(f"TartanAir archive SHA-256 mismatch: {path}")
        paths[(environment, modality)] = path
    expected = {
        (environment, modality)
        for environment in TARTANAIR_ENVIRONMENT_SPLITS
        for modality in ("image", "depth")
    }
    if set(paths) != expected:
        raise ValueError("TartanAir pinned archive inventory is incomplete")
    return paths


def _tartanair_members(
    archive: zipfile.ZipFile, environment: str, modality: str
) -> tuple[dict[tuple[str, int], zipfile.ZipInfo], dict[str, zipfile.ZipInfo]]:
    frames: dict[tuple[str, int], zipfile.ZipInfo] = {}
    poses: dict[str, zipfile.ZipInfo] = {}
    frame_pattern = re.compile(r"^(\d+)_lcam_front(?:_depth)?\.png$")
    for info in archive.infolist():
        if info.is_dir():
            continue
        relative = safe_relative_path(info.filename)
        parts = relative.parts
        if len(parts) < 4 or parts[0] != environment or parts[1] != "Data_easy":
            raise ValueError(f"Unexpected TartanAir ZIP path: {info.filename}")
        trajectory = parts[2]
        if relative.name == "pose_lcam_front.txt":
            if trajectory in poses:
                raise ValueError(f"Duplicate TartanAir pose: {environment}/{trajectory}")
            poses[trajectory] = info
            continue
        expected_directory = f"{modality}_lcam_front"
        if parts[-2] != expected_directory:
            raise ValueError(f"Unexpected TartanAir modality path: {info.filename}")
        match = frame_pattern.fullmatch(relative.name)
        if match is None:
            raise ValueError(f"Unexpected TartanAir frame name: {info.filename}")
        key = (trajectory, int(match.group(1)))
        if key in frames:
            raise ValueError(f"Duplicate TartanAir frame: {key}")
        frames[key] = info
    if not frames or not poses:
        raise ValueError(f"TartanAir {environment}/{modality} archive is incomplete")
    return frames, poses


def prepare_tartanair_rural_nature(
    raw_root: Path, output_root: Path, *, force: bool
) -> dict[str, Any]:
    archive_paths = _tartanair_archive_paths(raw_root)
    _replace_output_root(output_root, force=force)
    source_id = "tartanair-v2-rural-nature-depth-pose-v1"
    manifest_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    environment_counts: Counter[str] = Counter()
    media_owners: dict[str, set[str]] = defaultdict(set)
    archive_records: list[dict[str, Any]] = []

    for environment, split in TARTANAIR_ENVIRONMENT_SPLITS.items():
        image_path = archive_paths[(environment, "image")]
        depth_path = archive_paths[(environment, "depth")]
        archive_records.extend(
            {
                "environment": environment,
                "modality": modality,
                "path": path.relative_to(raw_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for modality, path in (("image", image_path), ("depth", depth_path))
        )
        with (
            zipfile.ZipFile(image_path, "r", allowZip64=True) as image_zip,
            zipfile.ZipFile(depth_path, "r", allowZip64=True) as depth_zip,
        ):
            image_frames, image_poses = _tartanair_members(image_zip, environment, "image")
            depth_frames, depth_poses = _tartanair_members(depth_zip, environment, "depth")
            if set(image_frames) != set(depth_frames):
                raise ValueError(f"TartanAir RGB/depth frame mismatch: {environment}")
            if set(image_poses) != set(depth_poses):
                raise ValueError(f"TartanAir pose inventory mismatch: {environment}")
            poses_by_trajectory = {
                trajectory: _read_pose_lines(image_zip, info)
                for trajectory, info in image_poses.items()
            }
            for trajectory, frame_index in sorted(image_frames):
                poses = poses_by_trajectory.get(trajectory)
                if poses is None or frame_index >= len(poses):
                    raise ValueError(
                        f"TartanAir pose missing for {environment}/{trajectory}/{frame_index}"
                    )
                with image_zip.open(image_frames[(trajectory, frame_index)], "r") as stream:
                    image_ref = _materialize_stream_payload(
                        source=stream,
                        output_root=output_root,
                        original_name=image_frames[(trajectory, frame_index)].filename,
                        role="media",
                        media_type="image/png",
                    )
                with depth_zip.open(depth_frames[(trajectory, frame_index)], "r") as stream:
                    depth_ref = _materialize_stream_payload(
                        source=stream,
                        output_root=output_root,
                        original_name=depth_frames[(trajectory, frame_index)].filename,
                        role="metric_depth",
                        media_type="image/png",
                    )
                media_owners[str(image_ref["sha256"])].add(environment)
                sample_id = f"tartanair:{environment}:{trajectory}:{frame_index:06d}"
                artifact_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "source_id": source_id,
                    "sample_id": sample_id,
                    "task": "synthetic_outdoor_metric_depth_and_camera_pose",
                    "split": split,
                    "environment": environment,
                    "environment_type": TARTANAIR_ENVIRONMENT_TYPES[environment],
                    "difficulty": "easy",
                    "trajectory": trajectory,
                    "frame_index": frame_index,
                    "camera": {
                        "name": "lcam_front",
                        "model": "pinhole",
                        "width_px": 640,
                        "height_px": 640,
                        "fx_px": 320.0,
                        "fy_px": 320.0,
                        "cx_px": 320.0,
                        "cy_px": 320.0,
                        "distortion": [0.0, 0.0, 0.0, 0.0],
                        "pose_ned_xyz_xyzw": poses[frame_index],
                    },
                    "rgb": image_ref,
                    "metric_depth": {
                        **depth_ref,
                        "encoding": "little_endian_float32_view_of_rgba_png",
                        "unit": "meter",
                    },
                }
                artifact_relative = (
                    Path("samples") / environment / trajectory / f"{frame_index:06d}.json"
                )
                artifact_path = output_root / artifact_relative
                _write_json(artifact_path, artifact_payload)
                manifest_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "source_record_id": sample_id,
                        "task": "synthetic_outdoor_metric_depth_and_camera_pose",
                        "split": split,
                        "split_group": f"tartanair-environment:{environment}",
                        "license": "CC-BY-4.0",
                        "provenance": {
                            "landing_page": TARTANAIR_LANDING_PAGE,
                            "mirror_page": TARTANAIR_MIRROR_PAGE,
                            "mirror_revision": TARTANAIR_REVISION,
                            "environment": environment,
                            "trajectory": trajectory,
                            "synthetic": True,
                        },
                        "artifact": {
                            "path": artifact_relative.as_posix(),
                            "sha256": sha256_file(artifact_path),
                            "media_type": "application/json",
                        },
                        "referenced_payloads": [image_ref, depth_ref],
                    }
                )
                split_counts[split] += 1
                environment_counts[environment] += 1
        print(
            f"TartanAir normalized {environment}: {environment_counts[environment]} frames",
            flush=True,
        )
    cross_environment_duplicates = {
        digest: owners for digest, owners in media_owners.items() if len(owners) > 1
    }
    if cross_environment_duplicates:
        raise ValueError(
            f"TartanAir RGB duplicates across environments: {len(cross_environment_duplicates)}"
        )
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in manifest_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    source = output_root / "source"
    source.mkdir()
    _write_json(source / "PINNED_ARCHIVES.json", archive_records)
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "title": "TartanAir V2 rural and nature outdoor RGB-depth-pose subset",
        "landing_page": TARTANAIR_LANDING_PAGE,
        "mirror_page": TARTANAIR_MIRROR_PAGE,
        "mirror_revision": TARTANAIR_REVISION,
        "license": "CC-BY-4.0",
        "license_audit_note": (
            "Official TartanAir V2 documentation states CC-BY-4.0 while the pinned "
            "Hugging Face mirror card states BSD-3-Clause; retain both records."
        ),
        "task": "synthetic_outdoor_metric_depth_and_camera_pose",
        "split_policy": {
            "unit": "environment",
            "assignments": TARTANAIR_ENVIRONMENT_SPLITS,
        },
        "scope": {
            "difficulty": ["easy"],
            "camera": ["lcam_front"],
            "modalities": ["rgb", "metric_depth", "camera_pose"],
            "environment_types": ["rural", "nature"],
            "outdoor_only": True,
        },
        "limitations": [
            "Synthetic domain only",
            "Camera poses are local NED coordinates, not geographic coordinates",
        ],
    }
    _write_json(output_root / "SOURCE_MANIFEST.json", source_manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": source_id,
        "files_verified": True,
        "samples": len(manifest_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "environment_counts": dict(sorted(environment_counts.items())),
        "split_group_leakage": 0,
        "exact_media_duplicates_across_environments": 0,
        "manifest_sha256": sha256_file(manifest_path),
    }
    _write_json(output_root / "VALIDATION_REPORT.json", report)
    shutil.rmtree(output_root / "payload" / ".staging", ignore_errors=True)
    return report


def _diode_csv_rows(data_list_zip: Path) -> list[dict[str, str]]:
    if sha256_file(data_list_zip) != DIODE_DATA_LIST_SHA256:
        raise ValueError("DIODE data_list.zip SHA-256 mismatch")
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(data_list_zip, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("DIODE data list CRC validation failed")
        for partition in ("train", "val"):
            name = f"data_list/{partition}_outdoor.csv"
            text = archive.read(name).decode("utf-8")
            for values in csv.reader(text.splitlines()):
                if len(values) != 4:
                    raise ValueError(f"Invalid DIODE CSV row in {name}")
                image, depth, mask, _normal = values
                normalized = [value.removeprefix("./") for value in (image, depth, mask)]
                for value in normalized:
                    safe_relative_path(value)
                    if f"/{partition}/outdoor/" not in f"/{value}":
                        raise ValueError(f"Unexpected DIODE outdoor path: {value}")
                parts = PurePosixPath(normalized[0]).parts
                rows.append(
                    {
                        "partition": partition,
                        "image": normalized[0],
                        "depth": normalized[1],
                        "mask": normalized[2],
                        "scene": parts[2],
                        "scan": parts[3],
                    }
                )
    if not rows:
        raise ValueError("DIODE outdoor index is empty")
    return rows


def _normalize_tar_member_name(name: str) -> str:
    normalized = name.replace("\\", "/").removeprefix("./")
    return safe_relative_path(normalized).as_posix()


def prepare_diode_outdoor(raw_root: Path, output_root: Path, *, force: bool) -> dict[str, Any]:
    data_list_zip = raw_root / "data_list.zip"
    rows = _diode_csv_rows(data_list_zip)
    archives = {
        partition: raw_root / "archives" / f"{partition}.tar.gz" for partition in ("train", "val")
    }
    for partition, path in archives.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = DIODE_ARCHIVE_MD5[f"{partition}.tar.gz"]
        if md5_file(path) != expected:
            raise ValueError(f"DIODE {partition} archive MD5 mismatch")
    groups = sorted({f"{row['partition']}:{row['scene']}" for row in rows})
    assignments = deterministic_group_splits(groups)
    _replace_output_root(output_root, force=force)
    expected_paths = {row[key] for row in rows for key in ("image", "depth", "mask")}
    path_roles = {row["image"]: ("media", "image/png") for row in rows}
    path_roles.update({row["depth"]: ("metric_depth", "application/x-npy") for row in rows})
    path_roles.update({row["mask"]: ("validity_mask", "application/x-npy") for row in rows})
    payload_refs: dict[str, dict[str, Any]] = {}
    for _partition, archive_path in archives.items():
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = _normalize_tar_member_name(member.name)
                if name not in expected_paths:
                    continue
                if name in payload_refs:
                    raise ValueError(f"Duplicate DIODE archive member: {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable DIODE archive member: {name}")
                role, media_type = path_roles[name]
                with stream:
                    payload_refs[name] = _materialize_stream_payload(
                        source=stream,
                        output_root=output_root,
                        original_name=name,
                        role=role,
                        media_type=media_type,
                    )
                if len(payload_refs) % 2000 == 0:
                    print(
                        f"DIODE outdoor: {len(payload_refs)}/{len(expected_paths)} payloads",
                        flush=True,
                    )
    missing = expected_paths - set(payload_refs)
    if missing:
        raise ValueError(f"DIODE outdoor payloads missing: {len(missing)}")
    source_id = "diode-outdoor-rgb-depth-v1"
    manifest_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    for row in rows:
        group = f"{row['partition']}:{row['scene']}"
        split = assignments[group]
        stem = PurePosixPath(row["image"]).stem
        sample_id = f"diode:{row['partition']}:{row['scene']}:{row['scan']}:{stem}"
        refs = [payload_refs[row[key]] for key in ("image", "depth", "mask")]
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "source_id": source_id,
            "sample_id": sample_id,
            "task": "outdoor_metric_depth",
            "split": split,
            "site": group,
            "scan": row["scan"],
            "rgb": refs[0],
            "metric_depth": {**refs[1], "unit": "meter", "dtype": "float32"},
            "depth_validity_mask": {**refs[2], "dtype": "bool"},
        }
        artifact_relative = Path("samples") / row["scene"] / row["scan"] / f"{stem}.json"
        artifact_path = output_root / artifact_relative
        _write_json(artifact_path, artifact)
        manifest_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "source_id": source_id,
                "source_record_id": row["image"],
                "task": "outdoor_metric_depth",
                "split": split,
                "split_group": f"diode-site:{group}",
                "license": "MIT",
                "provenance": {
                    "landing_page": DIODE_LANDING_PAGE,
                    "upstream_partition": row["partition"],
                    "scene": row["scene"],
                    "scan": row["scan"],
                },
                "artifact": {
                    "path": artifact_relative.as_posix(),
                    "sha256": sha256_file(artifact_path),
                    "media_type": "application/json",
                },
                "referenced_payloads": refs,
            }
        )
        split_counts[split] += 1
        scene_counts[group] += 1
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in manifest_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    source = output_root / "source"
    source.mkdir()
    shutil.copy2(data_list_zip, source / "data_list.zip")
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "title": "DIODE outdoor RGB, metric depth and depth validity masks",
        "landing_page": DIODE_LANDING_PAGE,
        "license": "MIT",
        "task": "outdoor_metric_depth",
        "split_policy": {
            "unit": "upstream_partition_and_scene",
            "assignments": assignments,
            "upstream_train_validation_ignored_for_fireviewer_split": True,
        },
        "scope": {
            "outdoor_only": True,
            "surface_normals_excluded": True,
            "official_downloaded_test_partition_unavailable": True,
        },
        "source_archives": {
            name: {"md5": digest, "verified": True} for name, digest in DIODE_ARCHIVE_MD5.items()
        },
    }
    _write_json(output_root / "SOURCE_MANIFEST.json", source_manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": source_id,
        "files_verified": True,
        "samples": len(manifest_rows),
        "payloads": len(payload_refs),
        "sites": len(groups),
        "split_counts": dict(sorted(split_counts.items())),
        "site_counts": dict(sorted(scene_counts.items())),
        "split_group_leakage": 0,
        "manifest_sha256": sha256_file(manifest_path),
    }
    _write_json(output_root / "VALIDATION_REPORT.json", report)
    shutil.rmtree(output_root / "payload" / ".staging", ignore_errors=True)
    return report


def validate_normalized_source(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "manifest.jsonl"
    source_manifest_path = source_root / "SOURCE_MANIFEST.json"
    validation_report_path = source_root / "VALIDATION_REPORT.json"
    for required in (manifest_path, source_manifest_path, validation_report_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    sample_ids: set[str] = set()
    artifact_digests: set[str] = set()
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    media_payload_splits: dict[str, set[str]] = defaultdict(set)
    verified_payload_paths: dict[str, str] = {}
    rows = 0
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required_fields = {
                "sample_id",
                "source_id",
                "task",
                "split",
                "split_group",
                "license",
                "provenance",
                "artifact",
            }
            missing = required_fields - set(row)
            if missing:
                raise ValueError(f"Missing manifest fields at line {line_number}: {missing}")
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id at line {line_number}: {sample_id}")
            sample_ids.add(sample_id)
            split = str(row["split"])
            if split not in SPLITS:
                raise ValueError(f"Invalid split at line {line_number}: {split}")
            split_group = str(row["split_group"])
            split_groups[split_group].add(split)
            split_counts[split] += 1
            artifact = row["artifact"]
            relative = safe_relative_path(str(artifact["path"]))
            artifact_path = source_root.joinpath(*relative.parts)
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            digest = sha256_file(artifact_path)
            if digest != str(artifact["sha256"]):
                raise ValueError(f"Artifact SHA-256 mismatch: {relative.as_posix()}")
            if digest in artifact_digests:
                raise ValueError(f"Duplicate artifact content: {relative.as_posix()}")
            artifact_digests.add(digest)
            for payload in row.get("referenced_payloads", []):
                payload_relative = safe_relative_path(str(payload["path"]))
                payload_path = source_root.joinpath(*payload_relative.parts)
                if not payload_path.is_file():
                    raise FileNotFoundError(payload_path)
                expected_digest = str(payload["sha256"])
                previous_digest = verified_payload_paths.get(payload_relative.as_posix())
                if previous_digest is None:
                    observed_digest = sha256_file(payload_path)
                    if observed_digest != expected_digest:
                        raise ValueError(
                            f"Referenced payload SHA-256 mismatch: {payload_relative.as_posix()}"
                        )
                    if payload_path.stat().st_size != int(payload["size_bytes"]):
                        raise ValueError(
                            f"Referenced payload size mismatch: {payload_relative.as_posix()}"
                        )
                    verified_payload_paths[payload_relative.as_posix()] = observed_digest
                elif previous_digest != expected_digest:
                    raise ValueError(f"Conflicting payload digest: {payload_relative.as_posix()}")
                if payload.get("role") == "media":
                    media_payload_splits[expected_digest].add(split)
            rows += 1
    leakage = [group for group, splits in split_groups.items() if len(splits) > 1]
    if leakage:
        raise ValueError(f"Split-group leakage detected: {len(leakage)} groups")
    media_leakage = [
        digest for digest, payload_splits in media_payload_splits.items() if len(payload_splits) > 1
    ]
    if media_leakage:
        raise ValueError(f"Exact media leakage across splits: {len(media_leakage)} payloads")
    if rows == 0 or any(split_counts[split] == 0 for split in SPLITS):
        raise ValueError("A normalized source must contain non-empty train/validation/test splits")
    return {
        "rows": rows,
        "unique_sample_ids": len(sample_ids),
        "unique_artifact_sha256": len(artifact_digests),
        "split_counts": dict(sorted(split_counts.items())),
        "split_groups": len(split_groups),
        "split_group_leakage": 0,
        "referenced_payloads_verified": len(verified_payload_paths),
        "exact_media_split_leakage": 0,
        "files_verified": True,
    }


def _zip_info(name: str, *, compressed: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _copy_stream(source: Any, target: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = source.read(BUFFER_SIZE)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        target.write(chunk)


def _load_current_validation(source_root: Path) -> dict[str, Any]:
    """Reuse a complete validation only while its manifest is unchanged."""
    report_path = source_root / "VALIDATION_REPORT.json"
    manifest_path = source_root / "manifest.jsonl"
    if report_path.is_file() and manifest_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("files_verified") is True and report.get("manifest_sha256") == sha256_file(
            manifest_path
        ):
            return report
    return validate_normalized_source(source_root)


def build_train_bundle(
    *,
    train_id: str,
    source_roots: list[Path],
    output_dir: Path,
    entrypoints: list[dict[str, Any]],
    training_ready: bool,
    blocking_reasons: list[str],
    force: bool,
) -> dict[str, Any]:
    if not train_id or PurePosixPath(train_id).name != train_id:
        raise ValueError(f"Invalid train_id: {train_id}")
    if not source_roots:
        raise ValueError("At least one source is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{train_id}.zip"
    if output_path.exists() and not force:
        raise FileExistsError(output_path)
    source_reports = []
    source_ids: set[str] = set()
    cross_source_artifacts: dict[str, list[str]] = defaultdict(list)
    manifest_file_digests: dict[str, dict[str, str]] = {}
    for source_root in source_roots:
        print(f"bundle {train_id}: checking source {source_root.name}", flush=True)
        report = _load_current_validation(source_root)
        source_manifest = json.loads(
            (source_root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
        )
        source_id = str(source_manifest["source_id"])
        if source_id in source_ids:
            raise ValueError(f"Duplicate source_id: {source_id}")
        source_ids.add(source_id)
        expected_files: dict[str, str] = {}
        with (source_root / "manifest.jsonl").open(encoding="utf-8") as manifest:
            for line in manifest:
                if not line.strip():
                    continue
                row = json.loads(line)
                digest = str(row["artifact"]["sha256"])
                cross_source_artifacts[digest].append(f"{source_id}:{row['sample_id']}")
                declared_files = [row["artifact"], *row.get("referenced_payloads", [])]
                for declared in declared_files:
                    relative = str(declared["path"])
                    declared_digest = str(declared["sha256"])
                    previous = expected_files.setdefault(relative, declared_digest)
                    if previous != declared_digest:
                        raise ValueError(f"Conflicting manifest digest for {source_id}:{relative}")
        manifest_file_digests[source_id] = expected_files
        source_reports.append(
            {
                "source_id": source_id,
                "license": source_manifest["license"],
                "landing_page": source_manifest["landing_page"],
                "validation": report,
            }
        )
    duplicates = {
        digest: owners for digest, owners in cross_source_artifacts.items() if len(owners) > 1
    }
    if duplicates:
        first_digest, owners = next(iter(duplicates.items()))
        raise ValueError(f"Cross-source duplicate artifact {first_digest}: {owners}")

    train_bundle = {
        "schema_version": SCHEMA_VERSION,
        "package_format": PACKAGE_FORMAT,
        "train_id": train_id,
        "training_ready": training_ready,
        "blocking_reasons": blocking_reasons,
        "entrypoints": entrypoints,
        "sources": source_reports,
        "integrity": {
            "cross_source_exact_duplicates": 0,
            "cross_source_split_leakage": 0,
        },
    }
    generated: dict[str, bytes] = {"TRAIN_BUNDLE.json": _canonical_json_bytes(train_bundle)}
    expected: dict[str, str] = {}
    checksum_lines: list[str] = []

    temporary = output_path.with_suffix(".zip.partial")
    temporary.unlink(missing_ok=True)
    print(f"bundle {train_id}: writing ZIP", flush=True)
    with zipfile.ZipFile(temporary, mode="w", allowZip64=True) as archive:
        for source_root in source_roots:
            source_manifest = json.loads(
                (source_root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
            )
            source_id = str(source_manifest["source_id"])
            seen_declared_files: set[str] = set()
            for path in iter_files(source_root):
                relative = path.relative_to(source_root).as_posix()
                entry = f"{train_id}/sources/{source_id}/{relative}"
                compressed = path.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".mp4",
                    ".npy",
                    ".npz",
                    ".png",
                    ".tif",
                    ".tiff",
                    ".zip",
                }
                with (
                    path.open("rb") as source,
                    archive.open(
                        _zip_info(entry, compressed=compressed), mode="w", force_zip64=True
                    ) as target,
                ):
                    digest = _copy_stream(source, target)
                declared_digest = manifest_file_digests[source_id].get(relative)
                if declared_digest is not None:
                    if digest != declared_digest:
                        raise ValueError(f"Source changed after validation: {source_id}:{relative}")
                    seen_declared_files.add(relative)
                expected[entry] = digest
                checksum_lines.append(f"{digest}  sources/{source_id}/{relative}\n")
            missing = set(manifest_file_digests[source_id]) - seen_declared_files
            if missing:
                raise ValueError(
                    f"Manifest payload missing from source {source_id}: {min(missing)}"
                )
        generated["PAYLOAD_CHECKSUMS.sha256"] = "".join(sorted(checksum_lines)).encode("ascii")
        for relative, content in sorted(generated.items()):
            entry = f"{train_id}/{relative}"
            archive.writestr(_zip_info(entry, compressed=True), content)
            expected[entry] = hashlib.sha256(content).hexdigest()
    os.replace(temporary, output_path)

    seen: set[str] = set()
    print(f"bundle {train_id}: verifying ZIP once", flush=True)
    with zipfile.ZipFile(output_path, mode="r", allowZip64=True) as archive:
        for info in archive.infolist():
            safe_relative_path(info.filename)
            if info.filename in seen:
                raise ValueError(f"Duplicate ZIP entry: {info.filename}")
            seen.add(info.filename)
            digest = hashlib.sha256()
            with archive.open(info, "r") as stream:
                for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected.get(info.filename):
                raise ValueError(f"ZIP entry SHA-256 mismatch: {info.filename}")
    if seen != set(expected):
        raise ValueError("ZIP inventory mismatch")
    report = {
        "schema_version": SCHEMA_VERSION,
        "package_format": PACKAGE_FORMAT,
        "train_id": train_id,
        "source_validation": source_reports,
        "integrity": train_bundle["integrity"],
        "zip_validation": {
            "zip_sha256": sha256_file(output_path),
            "zip_size_bytes": output_path.stat().st_size,
            "entry_count": len(seen),
            "crc_verified": True,
            "entry_sha256_verified": True,
            "single_train_root": train_id,
        },
    }
    _write_json(output_dir / f"{train_id}.validation.json", report)
    (output_dir / f"{train_id}.zip.sha256").write_text(
        f"{report['zip_validation']['zip_sha256']}  {train_id}.zip\n", encoding="ascii"
    )
    print(f"bundle {train_id}: complete", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare additional public datasets into FireViewer train bundles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    firespread = subparsers.add_parser("firespread")
    firespread.add_argument("--source-zip", type=Path, required=True)
    firespread.add_argument("--output-root", type=Path, required=True)
    firespread.add_argument("--force", action="store_true")

    boreal = subparsers.add_parser("boreal-images")
    boreal.add_argument("--raw-root", type=Path, required=True)
    boreal.add_argument("--output-root", type=Path, required=True)
    boreal.add_argument("--task", choices=("detection", "segmentation"), required=True)
    boreal.add_argument("--force", action="store_true")

    crisisfacts = subparsers.add_parser("crisisfacts")
    crisisfacts.add_argument("--raw-root", type=Path, required=True)
    crisisfacts.add_argument("--output-root", type=Path, required=True)
    crisisfacts.add_argument("--force", action="store_true")

    imsr = subparsers.add_parser("imsr")
    imsr.add_argument("--raw-root", type=Path, required=True)
    imsr.add_argument("--output-root", type=Path, required=True)
    imsr.add_argument("--force", action="store_true")

    tartanair = subparsers.add_parser("tartanair-rural-nature")
    tartanair.add_argument("--raw-root", type=Path, required=True)
    tartanair.add_argument("--output-root", type=Path, required=True)
    tartanair.add_argument("--force", action="store_true")

    diode = subparsers.add_parser("diode-outdoor")
    diode.add_argument("--raw-root", type=Path, required=True)
    diode.add_argument("--output-root", type=Path, required=True)
    diode.add_argument("--force", action="store_true")

    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--train-id", required=True)
    bundle.add_argument("--source-root", action="append", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    bundle.add_argument("--training-ready", action="store_true")
    bundle.add_argument("--blocking-reason", action="append", default=[])
    bundle.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "firespread":
        report = prepare_firespread(
            args.source_zip.resolve(), args.output_root.resolve(), force=args.force
        )
    elif args.command == "boreal-images":
        report = prepare_boreal_images(
            args.raw_root.resolve(),
            args.output_root.resolve(),
            task=args.task,
            force=args.force,
        )
    elif args.command == "crisisfacts":
        report = prepare_crisisfacts(
            args.raw_root.resolve(), args.output_root.resolve(), force=args.force
        )
    elif args.command == "imsr":
        report = prepare_imsr(args.raw_root.resolve(), args.output_root.resolve(), force=args.force)
    elif args.command == "tartanair-rural-nature":
        report = prepare_tartanair_rural_nature(
            args.raw_root.resolve(), args.output_root.resolve(), force=args.force
        )
    elif args.command == "diode-outdoor":
        report = prepare_diode_outdoor(
            args.raw_root.resolve(), args.output_root.resolve(), force=args.force
        )
    elif args.command == "bundle":
        report = build_train_bundle(
            train_id=args.train_id,
            source_roots=[path.resolve() for path in args.source_root],
            output_dir=args.output_dir.resolve(),
            entrypoints=[],
            training_ready=args.training_ready,
            blocking_reasons=list(args.blocking_reason),
            force=args.force,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
