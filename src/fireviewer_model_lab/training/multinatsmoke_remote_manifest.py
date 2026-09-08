"""Build a payload-free manifest for selected MultiNatSmoke ZIP members.

The command reads only the remote ZIP directory through HTTP range requests. It
does not download or extract the 43.7 GB archive and it never creates labels.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
KINDS = {"images", "masks"}
SPLITS = {"Train", "Test"}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _open_source(stack: contextlib.ExitStack, source: str | Path) -> BinaryIO:
    text = str(source)
    if text.startswith(("https://", "http://")):
        import fsspec

        return stack.enter_context(
            fsspec.open(text, "rb", block_size=8 * 1024 * 1024, cache_type="blockcache")
        )
    return stack.enter_context(Path(source).open("rb"))


def build_remote_manifest(
    *,
    source: str | Path,
    output_dir: Path,
    repository: str,
    revision: str,
    archive_lfs_sha256: str,
    expected_archive_bytes: int,
    allowed_sources: set[str],
) -> dict[str, Any]:
    if not HF_REVISION_RE.fullmatch(revision):
        raise ValueError("repository revision must be a full 40-character hexadecimal SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", archive_lfs_sha256):
        raise ValueError("archive LFS SHA-256 must be lowercase hexadecimal")
    if not allowed_sources:
        raise ValueError("at least one source family must be selected")

    members: dict[tuple[str, str, str], dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    source_counts: Counter[tuple[str, str, str]] = Counter()
    selected_compressed_bytes = 0
    selected_uncompressed_bytes = 0
    archive_members = 0

    with contextlib.ExitStack() as stack:
        handle = _open_source(stack, source)
        with zipfile.ZipFile(handle) as archive:
            archive_members = len(archive.infolist())
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parts = [part for part in info.filename.replace("\\", "/").split("/") if part]
                if (
                    len(parts) != 5
                    or parts[0] != "MultiNatSmokeDataset"
                    or parts[1] not in SPLITS
                    or parts[2] not in allowed_sources
                    or parts[3] not in KINDS
                ):
                    continue
                split, source_name, kind = parts[1], parts[2], parts[3]
                sample_key = PurePosixPath(parts[4]).stem
                key = (split, source_name, sample_key)
                if kind in members[key]:
                    raise ValueError(f"duplicate {kind} member for {key}")
                members[key][kind] = info
                source_counts[(split, source_name, kind)] += 1
                selected_compressed_bytes += info.compress_size
                selected_uncompressed_bytes += info.file_size

    rows: list[dict[str, Any]] = []
    missing_pairs: list[str] = []
    for (split, source_name, sample_key), pair in sorted(members.items()):
        if set(pair) != KINDS:
            missing_pairs.append(f"{split}/{source_name}/{sample_key}:{sorted(pair)}")
            continue
        image = pair["images"]
        mask = pair["masks"]
        rows.append(
            {
                "sample_id": hashlib.sha256(
                    f"{revision}:{split}:{source_name}:{sample_key}".encode("utf-8")
                ).hexdigest(),
                "source_family": source_name,
                "upstream_split": split.casefold(),
                "source_record_id": sample_key,
                "image_member": image.filename,
                "mask_member": mask.filename,
                "image_crc32": f"{image.CRC:08x}",
                "mask_crc32": f"{mask.CRC:08x}",
                "image_bytes": image.file_size,
                "mask_bytes": mask.file_size,
                "image_compressed_bytes": image.compress_size,
                "mask_compressed_bytes": mask.compress_size,
                "annotation_provenance": "published_multinatsmoke_mask",
                "training_eligible": False,
                "admission_status": "pending_payload_hash_dedup_semantic_and_rights_gates",
            }
        )

    errors: list[str] = []
    if missing_pairs:
        errors.append("missing_image_mask_pairs")
    observed_archive_bytes: int | None = None
    if isinstance(source, Path) or not str(source).startswith(("https://", "http://")):
        observed_archive_bytes = Path(source).stat().st_size
        if observed_archive_bytes != expected_archive_bytes:
            errors.append("archive_size_mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "multinatsmoke_selected_members.jsonl"
    _write_jsonl(manifest_path, rows)
    summary = {
        "schema_version": 1,
        "kind": "multinatsmoke_remote_zip_index",
        "repository": repository,
        "revision": revision,
        "archive_lfs_sha256_declared": archive_lfs_sha256,
        "expected_archive_bytes": expected_archive_bytes,
        "observed_archive_bytes": observed_archive_bytes,
        "archive_members": archive_members,
        "allowed_sources": sorted(allowed_sources),
        "source_member_counts": {
            f"{split}/{source_name}/{kind}": count
            for (split, source_name, kind), count in sorted(source_counts.items())
        },
        "selected_pairs": len(rows),
        "selected_compressed_bytes": selected_compressed_bytes,
        "selected_uncompressed_bytes": selected_uncompressed_bytes,
        "missing_pair_count": len(missing_pairs),
        "missing_pair_examples": missing_pairs[:20],
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "range_index_only": str(source).startswith(("https://", "http://")),
        "payload_downloaded": False,
        "labels_generated": False,
        "training_eligible": False,
        "errors": errors,
        "next_action": "selective_payload_extraction_then_exact_hash_phash_semantic_and_rights_gates",
    }
    _write_json(output_dir / "multinatsmoke_remote_index_summary.json", summary)
    if errors:
        raise ValueError(f"remote archive index failed: {errors}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--archive-lfs-sha256", required=True)
    parser.add_argument("--expected-archive-bytes", required=True, type=int)
    parser.add_argument("--allowed-source", action="append", required=True)
    args = parser.parse_args()
    report = build_remote_manifest(
        source=args.source,
        output_dir=args.output_dir,
        repository=args.repository,
        revision=args.revision.casefold(),
        archive_lfs_sha256=args.archive_lfs_sha256.casefold(),
        expected_archive_bytes=args.expected_archive_bytes,
        allowed_sources=set(args.allowed_source),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
