from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import requests

BUFFER_SIZE = 8 * 1024 * 1024
MAX_ATTEMPTS = 6
TARTANAIR_REVISION = "0d2d145e973832742a2aaa04b7d2ebffc8d82817"
TARTANAIR_REPO = "theairlabcmu/tartanair2"
TARTANAIR_LANDING_PAGE = "https://tartanair.org/"
TARTANAIR_MIRROR_PAGE = f"https://huggingface.co/datasets/{TARTANAIR_REPO}"
DIODE_LANDING_PAGE = "https://diode-dataset.org/"


@dataclass(frozen=True)
class ArchiveSpec:
    relative_path: str
    url: str
    size: int
    checksum: str
    checksum_algorithm: str
    source_id: str

    @property
    def safe_path(self) -> PurePosixPath:
        value = PurePosixPath(self.relative_path)
        if value.is_absolute() or not value.parts or ".." in value.parts:
            raise ValueError(f"Unsafe archive path: {self.relative_path}")
        return value


def _tartanair_spec(environment: str, modality: str, size: int, sha256: str) -> ArchiveSpec:
    source_path = f"{environment}/Data_easy/{modality}_lcam_front.zip"
    return ArchiveSpec(
        relative_path=f"archives/{source_path}",
        url=(
            f"https://huggingface.co/datasets/{TARTANAIR_REPO}/resolve/"
            f"{TARTANAIR_REVISION}/{source_path}"
        ),
        size=size,
        checksum=sha256,
        checksum_algorithm="sha256",
        source_id="tartanair-v2-rural-nature",
    )


TARTANAIR_ARCHIVES = (
    _tartanair_spec(
        "DesertGasStation",
        "image",
        1_308_096_024,
        "8fdb4996e40b02886444d15b396e0c28d588e43c07dd7a8989f7a5a53952b51f",
    ),
    _tartanair_spec(
        "DesertGasStation",
        "depth",
        730_304_733,
        "06645df4c66cd87e8e9c77268914f1d7f7da215929a86fa2bd22e8006a83b141",
    ),
    _tartanair_spec(
        "TerrainBlending",
        "image",
        3_360_547_539,
        "d21e3c12f7511175ad508cf1c2ba563e1fb55a47669cea43e09f61df64b50017",
    ),
    _tartanair_spec(
        "TerrainBlending",
        "depth",
        1_700_825_486,
        "e6882d47cebf9f0863aac4de521a7bae8778440b1145131d2c435b9fbd7fcbbd",
    ),
    _tartanair_spec(
        "WaterMillDay",
        "image",
        6_090_341_242,
        "04b386cc1fe39be251fab464033515cd81e3c3c576d9ea7c355aa5871c51908b",
    ),
    _tartanair_spec(
        "WaterMillDay",
        "depth",
        2_523_661_567,
        "7f885576bd8b3e7feecdf8df6d883ba07fdbaf84e425677327c0da22fd66fe59",
    ),
    _tartanair_spec(
        "SeasideTown",
        "image",
        3_929_710_508,
        "710dd11fa0ca5f1350adf88564f0b698f8f0122e48827f62ed92d2f23427254a",
    ),
    _tartanair_spec(
        "SeasideTown",
        "depth",
        1_388_959_160,
        "68c1b319b75b671d0c0f4ea1ebf0b6061603fd2a8aa1a01abf225dec692801d9",
    ),
    _tartanair_spec(
        "SeasonalForestAutumn",
        "image",
        8_895_725_830,
        "ae278f2a576cc7d7dd210425054fe712a3c7cbdd50ee8d8ffe98d558da8d3c99",
    ),
    _tartanair_spec(
        "SeasonalForestAutumn",
        "depth",
        4_605_883_026,
        "c96e14890173d8d233e3d6e7d40f17ae9df6b197ddcd049be424471e79beb501",
    ),
)


DIODE_ARCHIVES = (
    ArchiveSpec(
        relative_path="archives/train.tar.gz",
        url="https://diode-dataset.s3.amazonaws.com/train.tar.gz",
        size=86_747_209_190,
        checksum="3a94632398fe1d002d89f11743f748b1",
        checksum_algorithm="md5",
        source_id="diode-outdoor",
    ),
    ArchiveSpec(
        relative_path="archives/val.tar.gz",
        url="https://diode-dataset.s3.amazonaws.com/val.tar.gz",
        size=2_774_625_282,
        checksum="5c895d09201b88973c8fe4552a67dd85",
        checksum_algorithm="md5",
        source_id="diode-outdoor",
    ),
)


PROFILES = {
    "tartanair-rural-nature": TARTANAIR_ARCHIVES,
    "diode-outdoor": DIODE_ARCHIVES,
}


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _checksum(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
    elif algorithm == "md5":
        digest = hashlib.md5(usedforsecurity=False)
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_download_response(response: requests.Response, offset: int) -> tuple[str, int]:
    if offset and response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {offset}-"):
            raise ValueError(f"Invalid resumed Content-Range: {content_range}")
        return "ab", offset
    if response.status_code == 200:
        return "wb", 0
    response.raise_for_status()
    raise ValueError(f"Unexpected download status: {response.status_code}")


def download_archive(spec: ArchiveSpec, output_root: Path) -> dict[str, Any]:
    destination = output_root.joinpath(*spec.safe_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == spec.size
        and _checksum(destination, spec.checksum_algorithm) == spec.checksum
    ):
        return {"path": spec.relative_path, "status": "cache_hit", "size": spec.size}
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and partial.stat().st_size > spec.size:
        partial.unlink()
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            print(
                f"supplemental download: connecting {spec.relative_path} "
                f"attempt={attempt} offset={offset}",
                flush=True,
            )
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(
                spec.url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(60, 300),
            ) as response:
                mode, accepted_offset = _validate_download_response(response, offset)
                written = accepted_offset
                next_report = written + 512 * 1024 * 1024
                with partial.open(mode) as output:
                    for chunk in response.iter_content(BUFFER_SIZE):
                        if chunk:
                            output.write(chunk)
                            written += len(chunk)
                            if written >= next_report:
                                percent = written * 100 / spec.size
                                print(
                                    f"supplemental download: {spec.relative_path} "
                                    f"{written}/{spec.size} bytes ({percent:.1f}%)",
                                    flush=True,
                                )
                                next_report = written + 512 * 1024 * 1024
            observed_size = partial.stat().st_size
            if observed_size != spec.size:
                raise ValueError(
                    f"Incomplete archive {spec.relative_path}: {observed_size} != {spec.size}"
                )
            observed_checksum = _checksum(partial, spec.checksum_algorithm)
            if observed_checksum != spec.checksum:
                partial.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum mismatch for {spec.relative_path}: "
                    f"{observed_checksum} != {spec.checksum}"
                )
            os.replace(partial, destination)
            return {
                "path": spec.relative_path,
                "status": "downloaded" if accepted_offset == 0 else "resumed",
                "size": spec.size,
            }
        except Exception as error:
            last_error = error
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Download failed: {spec.relative_path}") from last_error


def download_profile(profile: str, output_root: Path, max_workers: int) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    if max_workers < 1 or max_workers > 4:
        raise ValueError("max_workers must be between 1 and 4")
    specs = PROFILES[profile]
    metadata_root = output_root / "_fireviewer_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    inventory = [
        {
            "relative_path": spec.relative_path,
            "url": spec.url,
            "size": spec.size,
            "checksum": spec.checksum,
            "checksum_algorithm": spec.checksum_algorithm,
            "source_id": spec.source_id,
        }
        for spec in specs
    ]
    with (metadata_root / "ARCHIVE_INVENTORY.jsonl").open("wb") as output:
        for row in inventory:
            output.write(_canonical_json_bytes(row))
    provenance = {
        "schema_version": 1,
        "profile": profile,
        "selected_archives": len(specs),
        "selected_bytes": sum(spec.size for spec in specs),
        "landing_page": (
            TARTANAIR_LANDING_PAGE if profile.startswith("tartanair") else DIODE_LANDING_PAGE
        ),
        "mirror_page": TARTANAIR_MIRROR_PAGE if profile.startswith("tartanair") else None,
        "mirror_revision": TARTANAIR_REVISION if profile.startswith("tartanair") else None,
        "license": "CC-BY-4.0" if profile.startswith("tartanair") else "MIT",
        "license_note": (
            "Official TartanAir V2 documentation states CC-BY-4.0; the pinned Hugging Face "
            "mirror card currently states BSD-3-Clause. Both records must remain "
            "in the final audit."
            if profile.startswith("tartanair")
            else "Official DIODE dataset and code license."
        ),
    }
    (metadata_root / "DOWNLOAD_PLAN.json").write_bytes(_canonical_json_bytes(provenance))
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_archive, spec, output_root): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed = sum(item["size"] for item in results)
            print(
                f"{profile}: {len(results)}/{len(specs)} archives "
                f"{completed / (1024**3):.2f} GiB verified",
                flush=True,
            )
    report = {
        **provenance,
        "archives_verified": len(results),
        "bytes_verified": sum(item["size"] for item in results),
        "complete": len(results) == len(specs),
        "results": sorted(results, key=lambda item: item["path"]),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (metadata_root / "DOWNLOAD_REPORT.json").write_bytes(_canonical_json_bytes(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume and verify large supplemental FireViewer source archives."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = download_profile(args.profile, args.output_root.resolve(), args.max_workers)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
