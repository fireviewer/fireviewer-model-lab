from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import requests

DATASET_ID = "1dce1023-493a-4d63-a906-f2a44f831898"
METAX_FILES_URL = f"https://metax.fairdata.fi/v3/datasets/{DATASET_ID}/files?pagination=false"
AUTHORIZE_URL = "https://etsin.fairdata.fi/api/v3/download/authorize"
LANDING_PAGE = f"https://etsin.fairdata.fi/dataset/{DATASET_ID}"
BUFFER_SIZE = 4 * 1024 * 1024
PROFILES = ("images", "videos", "all")
MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class OfficialFile:
    pathname: str
    size: int
    sha256: str
    storage_identifier: str

    @property
    def relative_path(self) -> PurePosixPath:
        value = PurePosixPath(self.pathname.lstrip("/"))
        if value.is_absolute() or not value.parts or ".." in value.parts:
            raise ValueError(f"Unsafe official pathname: {self.pathname}")
        return value


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_official_files(payload: Any) -> list[OfficialFile]:
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    files: list[OfficialFile] = []
    paths: set[str] = set()
    for row in rows:
        pathname = str(row["pathname"])
        if pathname in paths:
            raise ValueError(f"Duplicate official pathname: {pathname}")
        paths.add(pathname)
        checksum = str(row["checksum"])
        if not checksum.startswith("sha256:") or len(checksum) != 71:
            raise ValueError(f"Unsupported checksum for {pathname}: {checksum}")
        official = OfficialFile(
            pathname=pathname,
            size=int(row["size"]),
            sha256=checksum.removeprefix("sha256:"),
            storage_identifier=str(row.get("storage_identifier") or ""),
        )
        files.append(official)
    if not files:
        raise ValueError("The official Boreal inventory is empty")
    return sorted(files, key=lambda item: item.pathname)


def select_profile(files: Iterable[OfficialFile], profile: str) -> list[OfficialFile]:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    selected = []
    for item in files:
        is_video_subset = "/Boreal-Forest-Fire-Subset-B/" in item.pathname
        if profile == "images" and is_video_subset:
            continue
        if profile == "videos" and not is_video_subset:
            continue
        selected.append(item)
    if not selected:
        raise ValueError(f"Profile {profile} selected no files")
    return selected


def fetch_inventory(session: requests.Session) -> list[OfficialFile]:
    response = session.get(METAX_FILES_URL, timeout=180)
    response.raise_for_status()
    return parse_official_files(response.json())


def _authorize(session: requests.Session, pathname: str) -> str:
    response = session.post(
        AUTHORIZE_URL,
        json={"cr_id": DATASET_ID, "file": pathname},
        timeout=60,
    )
    response.raise_for_status()
    url = str(response.json().get("url") or "")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "download.fairdata.fi":
        raise ValueError(f"Unexpected Fairdata download URL for {pathname}")
    return url


def _download_one(
    item: OfficialFile,
    destination_root: Path,
    *,
    session_factory: Any = requests.Session,
) -> dict[str, Any]:
    destination = destination_root.joinpath(*item.relative_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == item.size:
        observed = _sha256(destination)
        if observed == item.sha256:
            return {"pathname": item.pathname, "status": "cache_hit", "size": item.size}
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        digest = hashlib.sha256()
        downloaded = 0
        try:
            with session_factory() as session:
                signed_url = _authorize(session, item.pathname)
                with session.get(signed_url, stream=True, timeout=(60, 300)) as response:
                    response.raise_for_status()
                    with partial.open("wb") as output:
                        for chunk in response.iter_content(BUFFER_SIZE):
                            if not chunk:
                                continue
                            output.write(chunk)
                            digest.update(chunk)
                            downloaded += len(chunk)
            if downloaded != item.size:
                raise ValueError(f"Size mismatch for {item.pathname}: {downloaded} != {item.size}")
            if digest.hexdigest() != item.sha256:
                raise ValueError(f"SHA-256 mismatch for {item.pathname}")
            os.replace(partial, destination)
            return {"pathname": item.pathname, "status": "downloaded", "size": item.size}
        except Exception as error:  # retries are bounded and reported
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 15))
    raise RuntimeError(
        f"Download failed after {MAX_ATTEMPTS} attempts: {item.pathname}"
    ) from last_error


def _write_inventory(path: Path, files: Iterable[OfficialFile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for item in files:
            output.write(
                _canonical_json_bytes(
                    {
                        "pathname": item.pathname,
                        "size": item.size,
                        "sha256": item.sha256,
                        "storage_identifier": item.storage_identifier,
                    }
                )
            )


def download_profile(
    *,
    output_root: Path,
    profile: str,
    max_workers: int,
) -> dict[str, Any]:
    if max_workers < 1 or max_workers > 16:
        raise ValueError("max_workers must be between 1 and 16")
    with requests.Session() as session:
        official_files = fetch_inventory(session)
    selected = select_profile(official_files, profile)
    metadata_root = output_root / "_fireviewer_metadata"
    _write_inventory(metadata_root / "OFFICIAL_FULL_INVENTORY.jsonl", official_files)
    _write_inventory(metadata_root / f"OFFICIAL_{profile.upper()}_INVENTORY.jsonl", selected)
    summary = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "landing_page": LANDING_PAGE,
        "license": "CC-BY-4.0",
        "profile": profile,
        "selected_files": len(selected),
        "selected_bytes": sum(item.size for item in selected),
        "official_files": len(official_files),
        "official_bytes": sum(item.size for item in official_files),
    }
    (metadata_root / "DOWNLOAD_PLAN.json").write_bytes(_canonical_json_bytes(summary))

    lock = threading.Lock()
    completed = 0
    downloaded_files = 0
    cache_hits = 0
    completed_bytes = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, item, output_root): item for item in selected}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                completed += 1
                completed_bytes += int(result["size"])
                downloaded_files += result["status"] == "downloaded"
                cache_hits += result["status"] == "cache_hit"
                if completed == 1 or completed % 100 == 0 or completed == len(selected):
                    elapsed = max(time.monotonic() - started, 0.001)
                    gib = completed_bytes / (1024**3)
                    rate = completed_bytes / elapsed / (1024**2)
                    print(
                        f"boreal download: {completed}/{len(selected)} files "
                        f"{gib:.2f} GiB verified {rate:.1f} MiB/s",
                        flush=True,
                    )
    report = {
        **summary,
        "files_verified": completed,
        "bytes_verified": completed_bytes,
        "downloaded_files": downloaded_files,
        "cache_hits": cache_hits,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "complete": completed == len(selected) and completed_bytes == summary["selected_bytes"],
    }
    (metadata_root / "DOWNLOAD_REPORT.json").write_bytes(_canonical_json_bytes(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume and verify the official Boreal Forest Fire download."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="images")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = download_profile(
        output_root=args.output_root.resolve(),
        profile=args.profile,
        max_workers=args.max_workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
