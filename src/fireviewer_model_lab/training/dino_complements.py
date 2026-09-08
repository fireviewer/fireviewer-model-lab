"""Plan and acquire bounded DINOv3 dataset complements.

The module deliberately separates source acquisition from admission to a training
manifest. Downloaded frames are not training labels: every source keeps explicit
annotation, geometry, grouping, and redistribution gates.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from fireviewer_model_lab.training.remote_zip import require_http_url

REGISTRY_PATH = Path(__file__).parent / "registries" / "dino-complements-v1.json"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported DINO complement registry schema")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("DINO complement registry has no sources")
    source_ids: set[str] = set()
    for source in sources:
        source_id = str(source.get("source_id", ""))
        if not source_id:
            raise ValueError("DINO complement source_id is required")
        if source_id in source_ids:
            raise ValueError(f"duplicate DINO complement source_id: {source_id}")
        source_ids.add(source_id)
        if "split_policy" not in source:
            raise ValueError(f"split_policy is required for {source_id}")
    return registry


def find_source(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    for source in registry["sources"]:
        if source["source_id"] == source_id:
            return source
    raise KeyError(f"unknown DINO complement source: {source_id}")


def build_plan(registry: dict[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for source in registry["sources"]:
        assets = source.get("assets", [])
        sources.append(
            {
                "source_id": source["source_id"],
                "acquisition_status": source["acquisition"]["status"],
                "multitask_status": source["multitask_role"]["status"],
                "cross_view_status": source["cross_view_role"]["status"],
                "asset_count": len(assets),
                "archive_bytes": sum(int(asset["expected_bytes"]) for asset in assets),
                "split_policy": source["split_policy"],
                "blockers": list(source.get("blockers", [])),
            }
        )
    ready_downloads = [
        item["source_id"]
        for item in sources
        if item["acquisition_status"] == "ready" and item["asset_count"] > 0
    ]
    return {
        "schema_version": 1,
        "campaign_id": registry["campaign_id"],
        "sources": sources,
        "ready_downloads": ready_downloads,
        "training_admission_rule": (
            "A source is admitted only after derived labels, group-disjoint splits, "
            "and source-specific quality gates are materialized."
        ),
    }


def probe_assets(source: dict[str, Any], *, timeout_seconds: float = 30.0) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for asset in source.get("assets", []):
        url = require_http_url(str(asset["url"]))
        request = urllib.request.Request(url, method="HEAD")  # noqa: S310
        with urllib.request.urlopen(  # noqa: S310 - URL validated above
            request, timeout=timeout_seconds
        ) as response:
            observed = int(response.headers.get("Content-Length", "0"))
            expected = int(asset["expected_bytes"])
            results.append(
                {
                    "filename": asset["filename"],
                    "status": int(response.status),
                    "expected_bytes": expected,
                    "observed_bytes": observed,
                    "size_matches": observed == expected,
                    "accept_ranges": response.headers.get("Accept-Ranges"),
                }
            )
    return results


def _download_asset(asset: dict[str, Any], destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / str(asset["filename"])
    partial_path = final_path.with_suffix(final_path.suffix + ".part")
    expected = int(asset["expected_bytes"])
    if final_path.is_file():
        observed = final_path.stat().st_size
        if observed != expected:
            raise ValueError(
                f"existing archive has unexpected size: {final_path} ({observed} != {expected})"
            )
        return {"path": str(final_path), "bytes": observed, "status": "already_present"}

    offset = partial_path.stat().st_size if partial_path.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    url = require_http_url(str(asset["url"]))
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(  # noqa: S310 - URL validated above
        request, timeout=60.0
    ) as response:
        append = offset > 0 and int(response.status) == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial_path.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    observed = partial_path.stat().st_size
    if observed != expected:
        raise OSError(f"incomplete archive: {partial_path} ({observed} != {expected})")
    os.replace(partial_path, final_path)
    return {"path": str(final_path), "bytes": observed, "status": "downloaded"}


def download_assets(
    source: dict[str, Any], destination: Path, *, maximum_assets: int | None = None
) -> list[dict[str, Any]]:
    assets = list(source.get("assets", []))
    if not assets:
        raise ValueError(f"source has no direct HTTP assets: {source['source_id']}")
    if source["acquisition"]["strategy"] != "resumable_direct_http":
        raise ValueError(f"source is not a direct HTTP acquisition: {source['source_id']}")
    if maximum_assets is not None:
        if maximum_assets <= 0:
            raise ValueError("maximum_assets must be positive")
        assets = assets[:maximum_assets]
    return [_download_asset(asset, destination) for asset in assets]


def _safe_zip_member(destination: Path, member: zipfile.ZipInfo) -> Path:
    target = (destination / member.filename).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe ZIP member: {member.filename}") from exc
    unix_mode = member.external_attr >> 16
    if stat.S_ISLNK(unix_mode):
        raise ValueError(f"ZIP symlink is forbidden: {member.filename}")
    return target


def extract_archives(
    source: dict[str, Any],
    archive_root: Path,
    output_root: Path,
    *,
    delete_archives: bool = False,
    maximum_assets: int | None = None,
) -> list[dict[str, Any]]:
    assets = list(source.get("assets", []))
    if maximum_assets is not None:
        if maximum_assets <= 0:
            raise ValueError("maximum_assets must be positive")
        assets = assets[:maximum_assets]
    reports: list[dict[str, Any]] = []
    for asset in assets:
        archive = archive_root / str(asset["filename"])
        if not archive.is_file():
            raise FileNotFoundError(archive)
        expected = int(asset["expected_bytes"])
        if archive.stat().st_size != expected:
            raise ValueError(f"archive size mismatch: {archive}")
        sequence_root = output_root / str(asset["sequence_group"])
        extracted_files = 0
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = _safe_zip_member(sequence_root, member)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source_stream, target.open("wb") as output_stream:
                    shutil.copyfileobj(source_stream, output_stream, length=8 * 1024 * 1024)
                extracted_files += 1
        if extracted_files == 0:
            raise ValueError(f"archive extracted no files: {archive}")
        if delete_archives:
            archive.unlink()
        reports.append(
            {
                "archive": str(archive),
                "sequence_group": asset["sequence_group"],
                "extracted_files": extracted_files,
                "archive_deleted": delete_archives,
            }
        )
    return reports
