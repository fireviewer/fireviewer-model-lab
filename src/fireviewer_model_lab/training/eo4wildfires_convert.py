from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import tarfile
import urllib.request
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

UPSTREAM_REVISION = "d194cca045aa6b0dc0e53b0ee45573324efa5c4f"
UPSTREAM_ROOT = (
    f"https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires/resolve/{UPSTREAM_REVISION}"
)
DEFAULT_ARCHIVE = f"{UPSTREAM_ROOT}/eo4wildfires.tar.gz?download=true"
DEFAULT_SPLITS = {
    "train": f"{UPSTREAM_ROOT}/files_train.csv.gz?download=true",
    "validation": f"{UPSTREAM_ROOT}/files_val.csv.gz?download=true",
    "test": f"{UPSTREAM_ROOT}/files_test.csv.gz?download=true",
}
EXPECTED_SPLIT_COUNTS = {"train": 20_307, "validation": 5_077, "test": 6_346}
NASA_VARIABLES = {
    "RH2M",
    "T2M",
    "PRECTOTCORR",
    "WS2M",
    "FRSNO",
    "GWETROOT",
    "SNODP",
    "PRECSNOLAND",
    "GWETTOP",
}
REQUIRED_VARIABLES = {
    "S1_GRD_A",
    "S1_GRD_D",
    "S2A",
    "BURNED_AREA",
    "burned_mask",
    "x",
    "y",
} | NASA_VARIABLES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _drop_page_cache(path: Path) -> None:
    """Release Linux page-cache pages without deleting or truncating the file."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _open_binary(location: str) -> BinaryIO:
    if location.startswith(("https://", "http://")):
        request = urllib.request.Request(  # noqa: S310
            location, headers={"User-Agent": "firewarning-eo4/1"}
        )
        return urllib.request.urlopen(request, timeout=120)  # noqa: S310
    return Path(location).open("rb")


def load_official_splits(
    locations: dict[str, str], *, enforce_declared_counts: bool = True
) -> dict[str, tuple[str, int]]:
    assignments: dict[str, tuple[str, int]] = {}
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        location = locations[split]
        with (
            closing(_open_binary(location)) as raw,
            gzip.GzipFile(fileobj=raw) as compressed,
        ):
            reader = csv.reader(io.TextIOWrapper(compressed, encoding="utf-8-sig"))
            names = [row[0].strip() for row in reader if row and row[0].strip()]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate scene inside official {split} split")
        counts[split] = len(names)
        for ordinal, name in enumerate(names):
            if PurePosixPath(name).name != name or not name.endswith(".nc"):
                raise ValueError(f"Invalid official scene name: {name}")
            previous = assignments.setdefault(name, (split, ordinal))
            if previous != (split, ordinal):
                raise ValueError(f"Cross-split leakage for {name}: {previous[0]} and {split}")
    if enforce_declared_counts and counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Official split-count drift: {counts} != {EXPECTED_SPLIT_COUNTS}")
    return assignments


def inspect_netcdf(payload: bytes) -> dict[str, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - pod dependency gate
        raise RuntimeError("h5py and numpy are required for EO4Wildfires conversion") from exc

    try:
        dataset = h5py.File(io.BytesIO(payload), "r")
    except OSError as exc:
        raise ValueError("Scene is not a readable NetCDF4/HDF5 payload") from exc
    with dataset:
        names = set(dataset.keys())
        missing = sorted(REQUIRED_VARIABLES - names)
        if missing:
            raise ValueError(f"NetCDF scene misses required variables: {', '.join(missing)}")
        variables = {
            name: {"shape": list(dataset[name].shape), "dtype": str(dataset[name].dtype)}
            for name in sorted(REQUIRED_VARIABLES)
        }
        mask = np.asarray(dataset["burned_mask"][...])
        finite = np.isfinite(mask)
        positive = finite & (mask > 0)
        coordinates: dict[str, dict[str, float | int]] = {}
        for name in ("x", "y"):
            values = np.asarray(dataset[name][...])
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                raise ValueError(f"Coordinate {name} has no finite value")
            coordinates[name] = {
                "count": int(finite_values.size),
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
            }
        crs_evidence: dict[str, str] = {}
        owners = [("root", dataset), *[(name, dataset[name]) for name in names]]
        for owner_name, owner in owners:
            for attribute in ("crs", "crs_wkt", "spatial_ref", "grid_mapping"):
                if attribute in owner.attrs:
                    value = owner.attrs[attribute]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    crs_evidence[f"{owner_name}.{attribute}"] = str(value)
        return {
            "variables": variables,
            "coordinates": coordinates,
            "burned_mask": {
                "pixels": int(mask.size),
                "finite_pixels": int(finite.sum()),
                "positive_pixels": int(positive.sum()),
                "positive_fraction": float(positive.sum() / mask.size) if mask.size else 0.0,
            },
            "crs_evidence": crs_evidence,
        }


class HubUploader:
    def __init__(self, repo_id: str, prefix: str, *, delete_uploaded: bool) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - pod dependency gate
            raise RuntimeError("huggingface-hub is required for remote materialization") from exc
        self.api = HfApi(token=os.environ.get("HF_TOKEN"))
        self.repo_id = repo_id
        self.prefix = prefix.strip("/")
        self.delete_uploaded = delete_uploaded

    def upload(self, path: Path, relative_path: str) -> dict[str, Any]:
        remote_path = f"{self.prefix}/{relative_path}" if self.prefix else relative_path
        self.api.upload_file(
            path_or_fileobj=path,
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Materialize EO4Wildfires {relative_path}",
        )
        info = self.api.get_paths_info(
            self.repo_id, paths=[remote_path], repo_type="dataset", expand=True
        )
        if len(info) != 1 or int(info[0].size) != path.stat().st_size:
            raise RuntimeError(f"Remote size verification failed for {remote_path}")
        receipt = {
            "remote_path": remote_path,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        _drop_page_cache(path)
        if self.delete_uploaded:
            path.unlink()
        return receipt


@dataclass
class SplitShardWriter:
    split: str
    output_root: Path
    samples_per_shard: int
    uploader: HubUploader | None = None
    shard_index: int = 0
    shard_sample_count: int = 0
    total_samples: int = 0
    receipts: list[dict[str, Any]] = field(default_factory=list)
    _tar: tarfile.TarFile | None = None
    _gzip: gzip.GzipFile | None = None
    _raw: BinaryIO | None = None
    _tar_path: Path | None = None
    _records: list[dict[str, Any]] = field(default_factory=list)

    def _open(self) -> None:
        directory = self.output_root / self.split
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"eo4wildfires-{self.split}-{self.shard_index:05d}"
        self._tar_path = directory / f"{stem}.tar.gz"
        self._raw = self._tar_path.open("wb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._tar = tarfile.open(fileobj=self._gzip, mode="w|")  # noqa: SIM115

    def add(self, name: str, payload: bytes, record: dict[str, Any]) -> None:
        if self._tar is None:
            self._open()
        if self.shard_sample_count >= self.samples_per_shard:
            self.close_shard()
            self.shard_index += 1
            self._open()
        member = tarfile.TarInfo(f"eo4wildfires/{name}")
        member.size = len(payload)
        member.mode = 0o644
        member.mtime = 0
        assert self._tar is not None
        self._tar.addfile(member, io.BytesIO(payload))
        self._records.append(record)
        self.shard_sample_count += 1
        self.total_samples += 1

    def close_shard(self) -> None:
        if self._tar is None or self._tar_path is None:
            return
        self._tar.close()
        assert self._gzip is not None and self._raw is not None
        self._gzip.close()
        self._raw.close()
        manifest_path = self._tar_path.with_suffix("").with_suffix(".manifest.jsonl")
        manifest_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in self._records),
            encoding="utf-8",
        )
        receipt = {
            "split": self.split,
            "shard": self.shard_index,
            "samples": self.shard_sample_count,
            "archive": {
                "path": self._tar_path.name,
                "size_bytes": self._tar_path.stat().st_size,
                "sha256": _sha256(self._tar_path),
            },
            "manifest": {
                "path": manifest_path.name,
                "size_bytes": manifest_path.stat().st_size,
                "sha256": _sha256(manifest_path),
            },
        }
        receipt_path = self._tar_path.with_suffix("").with_suffix(".receipt.json")
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if self.uploader:
            remote_directory = f"shards/{self.split}"
            receipt["uploads"] = [
                self.uploader.upload(self._tar_path, f"{remote_directory}/{self._tar_path.name}"),
                self.uploader.upload(manifest_path, f"{remote_directory}/{manifest_path.name}"),
                self.uploader.upload(receipt_path, f"{remote_directory}/{receipt_path.name}"),
            ]
        self.receipts.append(receipt)
        self._tar = None
        self._gzip = None
        self._raw = None
        self._tar_path = None
        self._records = []
        self.shard_sample_count = 0

    def close(self) -> None:
        self.close_shard()


def convert(
    *,
    archive: str,
    split_locations: dict[str, str],
    output_root: Path,
    samples_per_shard: int,
    uploader: HubUploader | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    assignments = load_official_splits(split_locations, enforce_declared_counts=max_samples is None)
    writers = {
        split: SplitShardWriter(split, output_root, samples_per_shard, uploader)
        for split in EXPECTED_SPLIT_COUNTS
    }
    seen: set[str] = set()
    schema_counts: Counter[str] = Counter()
    processed = 0
    started_at = datetime.now(UTC)
    try:
        with (
            closing(_open_binary(archive)) as raw,
            tarfile.open(fileobj=raw, mode="r|gz") as source,
        ):
            for member in source:
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"Unsupported archive member: {member.name}")
                normalized = PurePosixPath(member.name.replace("\\", "/"))
                if normalized.is_absolute() or ".." in normalized.parts:
                    raise ValueError(f"Unsafe archive member: {member.name}")
                name = normalized.name
                if not name.endswith(".nc"):
                    continue
                if name in seen:
                    raise ValueError(f"Duplicate archive scene: {name}")
                assignment = assignments.get(name)
                if assignment is None:
                    raise ValueError(f"Archive scene absent from official splits: {name}")
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Unreadable archive scene: {name}")
                with extracted:
                    payload = extracted.read()
                split, ordinal = assignment
                metadata = inspect_netcdf(payload)
                schema_key = json.dumps(metadata["variables"], sort_keys=True)
                schema_counts[schema_key] += 1
                record = {
                    "id": Path(name).stem,
                    "filename": name,
                    "split": split,
                    "official_ordinal": ordinal,
                    "source_revision": UPSTREAM_REVISION,
                    "source_member": normalized.as_posix(),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload, usedforsecurity=False).hexdigest(),
                    **metadata,
                }
                writers[split].add(name, payload, record)
                seen.add(name)
                processed += 1
                if processed % 250 == 0:
                    print(
                        f"eo4wildfires conversion: {processed}/{len(assignments)} scenes",
                        flush=True,
                    )
                if max_samples is not None and processed >= max_samples:
                    break
    finally:
        for writer in writers.values():
            writer.close()
    if max_samples is None:
        missing = sorted(set(assignments) - seen)
        if missing:
            raise ValueError(f"Archive misses {len(missing)} official scenes; first={missing[0]}")
    report = {
        "schema_version": 1,
        "source": {
            "repository": "AUA-Informatics-Lab/eo4wildfires",
            "revision": UPSTREAM_REVISION,
            "archive": archive,
        },
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "processed_scenes": processed,
        "official_split_counts": EXPECTED_SPLIT_COUNTS,
        "materialized_split_counts": {
            split: writer.total_samples for split, writer in writers.items()
        },
        "distinct_variable_schemas": len(schema_counts),
        "shards": [receipt for writer in writers.values() for receipt in writer.receipts],
        "complete": max_samples is None and processed == len(assignments),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "conversion-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if uploader:
        report["report_upload"] = uploader.upload(report_path, "conversion-report.json")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize official EO4Wildfires event splits")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--train-split", default=DEFAULT_SPLITS["train"])
    parser.add_argument("--validation-split", default=DEFAULT_SPLITS["validation"])
    parser.add_argument("--test-split", default=DEFAULT_SPLITS["test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--hf-repo")
    parser.add_argument(
        "--hf-prefix",
        default="additional/v1/satellite_burnscar_multisensor_v1/eo4wildfires/materialized",
    )
    parser.add_argument("--delete-uploaded", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if args.delete_uploaded and not args.hf_repo:
        raise ValueError("--delete-uploaded requires --hf-repo")
    uploader = (
        HubUploader(args.hf_repo, args.hf_prefix, delete_uploaded=args.delete_uploaded)
        if args.hf_repo
        else None
    )
    report = convert(
        archive=args.archive,
        split_locations={
            "train": args.train_split,
            "validation": args.validation_split,
            "test": args.test_split,
        },
        output_root=args.output,
        samples_per_shard=args.samples_per_shard,
        uploader=uploader,
        max_samples=args.max_samples,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
