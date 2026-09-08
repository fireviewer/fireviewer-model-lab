from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_INCLUDED_ROOTS = ("corpus", "sources", "training", "incoming", "evaluation")
DEFAULT_EXCLUDED_ROOTS = ("models", "_staging")
ROOT_METADATA_FILES = (
    "README.md",
    "dataset-index.json",
    "critical-lots-report.json",
    "spatial-training-preflight.json",
)


@dataclass(frozen=True)
class ShardReceipt:
    path: str
    sha256: str
    size_bytes: int
    file_count: int
    source_bytes: int


@dataclass(frozen=True)
class DatasetReceipt:
    dataset_id: str
    source_path: str
    file_count: int
    source_bytes: int
    shards: tuple[ShardReceipt, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive FireWarning datasets into resumable Hugging Face upload shards."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Add a dataset outside source-root, for example prepared/incidents=D:/path.",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=4 * 1024**3,
        help="Approximate maximum payload bytes per uncompressed tar shard.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current_root, dir_names, file_names in os.walk(root):
        dir_names.sort()
        file_names.sort()
        current = Path(current_root)
        for name in file_names:
            path = current / name
            if path.is_symlink():
                raise RuntimeError(f"Symbolic links are not accepted: {path}")
            if not path.is_file():
                raise RuntimeError(f"Unsupported dataset entry: {path}")
            files.append(path)
    return files


def normalize_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def shard_groups(files: Iterable[Path], max_shard_bytes: int) -> list[list[Path]]:
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in files:
        size = path.stat().st_size
        estimate = size + 4096
        if current and current_bytes + estimate > max_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += estimate
    if current:
        groups.append(current)
    return groups


def receipt_is_valid(receipt_path: Path, staging_root: Path) -> bool:
    if not receipt_path.is_file():
        return False
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        shards = payload["shards"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    for shard in shards:
        local_path = staging_root / shard["path"]
        if not local_path.is_file() or local_path.stat().st_size != shard["size_bytes"]:
            return False
        if sha256_file(local_path) != shard["sha256"]:
            return False
    return True


def archive_dataset(
    *,
    dataset_id: str,
    source_path: Path,
    staging_root: Path,
    max_shard_bytes: int,
    force: bool,
) -> DatasetReceipt:
    destination = staging_root / "datasets" / Path(dataset_id)
    receipt_path = destination / "manifest.json"
    if not force and receipt_is_valid(receipt_path, staging_root):
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        return DatasetReceipt(
            dataset_id=payload["dataset_id"],
            source_path=payload["source_path"],
            file_count=payload["file_count"],
            source_bytes=payload["source_bytes"],
            shards=tuple(ShardReceipt(**item) for item in payload["shards"]),
        )

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    files = iter_regular_files(source_path)
    groups = shard_groups(files, max_shard_bytes)
    receipts: list[ShardReceipt] = []
    for shard_index, group in enumerate(groups):
        shard_name = f"shard-{shard_index:05d}.tar"
        final_path = destination / shard_name
        with tempfile.NamedTemporaryFile(
            prefix=f".{shard_name}.", suffix=".partial", dir=destination, delete=False
        ) as temp_stream:
            temp_path = Path(temp_stream.name)
        try:
            with tarfile.open(temp_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in group:
                    relative = path.relative_to(source_path).as_posix()
                    archive.add(
                        path,
                        arcname=f"payload/{relative}",
                        recursive=False,
                        filter=normalize_tar_info,
                    )
            os.replace(temp_path, final_path)
        finally:
            temp_path.unlink(missing_ok=True)

        receipt = ShardReceipt(
            path=final_path.relative_to(staging_root).as_posix(),
            sha256=sha256_file(final_path),
            size_bytes=final_path.stat().st_size,
            file_count=len(group),
            source_bytes=sum(path.stat().st_size for path in group),
        )
        receipts.append(receipt)
        print(
            "archived",
            dataset_id,
            shard_name,
            f"files={receipt.file_count}",
            f"bytes={receipt.size_bytes}",
            f"sha256={receipt.sha256}",
            flush=True,
        )

    dataset_receipt = DatasetReceipt(
        dataset_id=dataset_id,
        source_path=str(source_path),
        file_count=len(files),
        source_bytes=sum(path.stat().st_size for path in files),
        shards=tuple(receipts),
    )
    receipt_path.write_text(
        json.dumps(asdict(dataset_receipt), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_receipt


def parse_extra_datasets(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --extra-dataset value: {value}")
        dataset_id, raw_path = value.split("=", 1)
        parsed.append((dataset_id.strip("/"), Path(raw_path)))
    return parsed


def discover_datasets(source_root: Path, extras: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    datasets: list[tuple[str, Path]] = []
    for root_name in DEFAULT_INCLUDED_ROOTS:
        root = source_root / root_name
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if child.is_dir():
                datasets.append((f"{root_name}/{child.name}", child))
    datasets.extend(extras)
    return datasets


def write_repo_readme(staging_root: Path) -> None:
    content = """---
license: other
pretty_name: FireWarning private training corpus
---

# FireWarning private training corpus

Private archival copy of the FireWarning training and evaluation datasets.

- Dataset payloads are stored as deterministic, uncompressed tar shards.
- Every logical dataset has a `manifest.json` with byte counts and SHA-256 checksums.
- `repository-manifest.json` records the complete upload inventory.
- Licenses, provenance, consent and publication rights remain dataset-specific.
- Private storage does not grant permission to redistribute source media.

Extract a shard into an empty dataset directory with `tar -xf shard-00000.tar`.
"""
    (staging_root / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    staging_root = args.staging_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if source_root == staging_root or source_root in staging_root.parents:
        raise ValueError("The staging directory must not be inside the source dataset directory.")

    staging_root.mkdir(parents=True, exist_ok=True)
    write_repo_readme(staging_root)
    metadata_root = staging_root / "metadata" / "datasetfire"
    metadata_root.mkdir(parents=True, exist_ok=True)
    for name in ROOT_METADATA_FILES:
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, metadata_root / name)
    for root_name in DEFAULT_INCLUDED_ROOTS:
        source = source_root / root_name
        if not source.is_dir():
            continue
        destination = metadata_root / "root-files" / root_name
        for child in source.iterdir():
            if child.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, destination / child.name)

    extras = parse_extra_datasets(args.extra_dataset)
    dataset_entries = discover_datasets(source_root, extras)
    receipts: list[DatasetReceipt] = []
    for dataset_id, path in dataset_entries:
        if path.is_file():
            raise ValueError(f"Extra dataset must be a directory: {path}")
        print(f"packing dataset={dataset_id} source={path}", flush=True)
        receipts.append(
            archive_dataset(
                dataset_id=dataset_id,
                source_path=path.resolve(),
                staging_root=staging_root,
                max_shard_bytes=args.max_shard_bytes,
                force=args.force,
            )
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(source_root),
        "included_roots": list(DEFAULT_INCLUDED_ROOTS),
        "excluded_roots": list(DEFAULT_EXCLUDED_ROOTS),
        "dataset_count": len(receipts),
        "file_count": sum(item.file_count for item in receipts),
        "source_bytes": sum(item.source_bytes for item in receipts),
        "datasets": [asdict(item) for item in receipts],
    }
    (staging_root / "repository-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({key: manifest[key] for key in ("dataset_count", "file_count", "source_bytes")})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
