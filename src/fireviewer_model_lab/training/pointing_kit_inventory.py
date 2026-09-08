"""Inventory the KIT Industrial Burner Flames archive in a cloud-only SageMaker job."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

SOURCE_ID = "kit-industrial-burner-flames"
SOURCE_RECORD_URL = "https://publikationen.bibliothek.kit.edu/1000159497"
SOURCE_ARCHIVE_URL = (
    "https://www.radar-service.eu/radar/en/dataset/UoXePcvHXuiShQHq/"
    "datasetBinaryTreeStream"
)
SOURCE_LICENSE = "CC-BY-4.0"
EXPECTED_ARCHIVE_BYTES = 383_521_792
CATALOG_BAGIT_ARCHIVE_BYTES = 383_631_872
EXPECTED_HUMAN_MASKS = 200
ARCHIVE_FILENAME = "kit-industrial-burner-flames.tar"

IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff"})
ANNOTATION_SUFFIXES = frozenset({".json", ".mat", ".npy", ".npz", ".xml"})
TEXT_SUFFIXES = frozenset(
    {"", ".bib", ".cff", ".csv", ".json", ".md", ".rst", ".txt", ".xml", ".yaml", ".yml"}
)
MASK_TOKENS = frozenset(
    {
        "annotation",
        "annotations",
        "gt",
        "label",
        "labels",
        "mask",
        "masks",
        "segmentation",
        "segmentations",
    }
)
RELABELLED_TOKENS = frozenset(
    {
        "corrected",
        "correction",
        "human",
        "manual",
        "relabel",
        "relabeled",
        "relabeling",
        "relabelled",
        "relabelling",
    }
)
README_TOKENS = frozenset({"readme", "read_me"})
LICENSE_TOKENS = frozenset({"copying", "licence", "license"})
METADATA_TOKENS = frozenset(
    {
        "authors",
        "citation",
        "codemeta",
        "description",
        "manifest",
        "metadata",
    }
)

MAX_MEMBER_COUNT = 500_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 32 * 1024**3
MAX_SINGLE_MEMBER_BYTES = 16 * 1024**3
MAX_METADATA_FILE_BYTES = 2 * 1024**2
MAX_METADATA_TOTAL_BYTES = 16 * 1024**2
MAX_METADATA_FILES = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _url_without_query(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def download_archive(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
) -> dict[str, Any]:
    """Download an immutable-size archive while computing its SHA-256 receipt."""
    if expected_bytes <= 0:
        raise ValueError("expected archive size must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    downloaded = 0
    request = urllib.request.Request(  # noqa: S310 - fixed HTTPS dataset endpoint
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "FireViewer-Pointing-Corpus/2.0",
        },
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=180) as response,  # noqa: S310
            partial.open("wb") as output,
        ):
            resolved_url = response.geturl()
            if urlsplit(resolved_url).scheme.casefold() != "https":
                raise ValueError("archive download redirected away from HTTPS")
            header_value = response.headers.get("Content-Length")
            response_bytes = int(header_value) if header_value and header_value.isdigit() else None
            if response_bytes is not None and response_bytes != expected_bytes:
                raise ValueError(
                    f"archive Content-Length drift: {response_bytes} != {expected_bytes}"
                )
            while chunk := response.read(4 * 1024 * 1024):
                downloaded += len(chunk)
                if downloaded > expected_bytes:
                    raise ValueError(
                        f"archive byte-count exceeds expectation: {downloaded} > {expected_bytes}"
                    )
                digest.update(chunk)
                output.write(chunk)
            response_metadata = {
                "content_disposition": response.headers.get("Content-Disposition"),
                "content_length": response_bytes,
                "content_type": response.headers.get("Content-Type"),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "resolved_url_without_query": _url_without_query(resolved_url),
            }
        if downloaded != expected_bytes:
            raise ValueError(f"archive byte-count drift: {downloaded} != {expected_bytes}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "bytes": downloaded,
        "expected_bytes": expected_bytes,
        "requested_url": url,
        "sha256": digest.hexdigest(),
        **response_metadata,
    }


def _normalized_member_name(name: str) -> str:
    if "\x00" in name:
        raise ValueError("unsafe tar member contains a NUL byte")
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"unsafe tar member path: {name}")
    if not normalized.parts or normalized.as_posix() in {"", "."}:
        raise ValueError(f"unsafe empty tar member path: {name}")
    if ":" in normalized.parts[0]:
        raise ValueError(f"unsafe drive-qualified tar member path: {name}")
    if len(normalized.as_posix()) > 1024:
        raise ValueError(f"tar member path is too long: {name[:80]}")
    return normalized.as_posix()


def _safe_target(destination: Path, member_name: str) -> Path:
    normalized = PurePosixPath(_normalized_member_name(member_name))
    root = destination.resolve()
    target = (root / Path(*normalized.parts)).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"tar member escapes extraction root: {member_name}")
    return target


def validate_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Fail closed before any member is extracted or inspected."""
    validated: list[tarfile.TarInfo] = []
    normalized_names: set[str] = set()
    total_bytes = 0
    for member in archive:
        if len(validated) >= MAX_MEMBER_COUNT:
            raise ValueError(f"tar member count exceeds limit: {MAX_MEMBER_COUNT}")
        normalized = _normalized_member_name(member.name)
        normalized_key = normalized.casefold()
        if normalized_key in normalized_names:
            raise ValueError(f"duplicate tar member path: {member.name}")
        normalized_names.add(normalized_key)
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise ValueError(f"unsupported tar member type: {member.name}")
        if member.size < 0 or member.size > MAX_SINGLE_MEMBER_BYTES:
            raise ValueError(f"unsafe tar member size: {member.name}:{member.size}")
        if member.isfile():
            total_bytes += member.size
            if total_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "tar uncompressed byte count exceeds limit: "
                    f"{total_bytes} > {MAX_TOTAL_UNCOMPRESSED_BYTES}"
                )
        validated.append(member)
    return validated


def _tokens(path: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", path.casefold()))


def classify_member(member: tarfile.TarInfo) -> dict[str, Any]:
    normalized = _normalized_member_name(member.name)
    pure_path = PurePosixPath(normalized)
    suffix = pure_path.suffix.casefold()
    name_tokens = _tokens(pure_path.name)
    path_tokens = _tokens(normalized)
    lowered = normalized.casefold().replace("-", "_").replace(" ", "_")
    is_file = member.isfile()
    is_image = is_file and suffix in IMAGE_SUFFIXES
    is_mask = is_file and (
        bool(path_tokens & MASK_TOKENS)
        or "ground_truth" in lowered
        or "groundtruth" in lowered
    ) and suffix in IMAGE_SUFFIXES | ANNOTATION_SUFFIXES
    is_relabelled = is_file and bool(path_tokens & RELABELLED_TOKENS)
    is_readme = is_file and bool(name_tokens & README_TOKENS)
    is_license = is_file and bool(name_tokens & LICENSE_TOKENS)
    is_metadata = is_file and (
        is_readme
        or is_license
        or bool(name_tokens & METADATA_TOKENS)
        or suffix in {".bib", ".cff", ".csv", ".json", ".xml", ".yaml", ".yml"}
    )
    return {
        "path": normalized,
        "type": "file" if is_file else "directory",
        "size_bytes": member.size,
        "suffix": suffix or "<none>",
        "depth": len(pure_path.parts),
        "image_candidate": is_image,
        "source_image_candidate": is_image and not is_mask,
        "mask_candidate": is_mask,
        "relabelled_candidate": is_relabelled,
        "relabelled_mask_candidate": is_relabelled and is_mask,
        "readme_candidate": is_readme,
        "license_candidate": is_license,
        "metadata_candidate": is_metadata,
    }


def _metadata_priority(record: dict[str, Any]) -> tuple[int, str]:
    if record["license_candidate"]:
        priority = 0
    elif record["readme_candidate"]:
        priority = 1
    else:
        priority = 2
    return priority, str(record["path"])


def _extract_metadata(
    archive_path: Path,
    records: list[dict[str, Any]],
    destination: Path,
) -> list[dict[str, Any]]:
    candidates = sorted(
        (record for record in records if record["metadata_candidate"]),
        key=_metadata_priority,
    )
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    for record in candidates:
        size = int(record["size_bytes"])
        if size > MAX_METADATA_FILE_BYTES:
            continue
        if len(selected) >= MAX_METADATA_FILES or selected_bytes + size > MAX_METADATA_TOTAL_BYTES:
            break
        selected.append(record)
        selected_bytes += size

    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = {member.name: member for member in archive}
        for record in selected:
            path = str(record["path"])
            source_member = members.get(path)
            if source_member is None:
                source_member = next(
                    (
                        member
                        for member in members.values()
                        if _normalized_member_name(member.name) == path
                    ),
                    None,
                )
            if source_member is None or not source_member.isfile():
                raise ValueError(f"validated metadata member disappeared: {path}")
            target = _safe_target(destination, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(source_member)
            if source is None:
                raise ValueError(f"unreadable metadata member: {path}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            raw = target.read_bytes()
            extracted.append(
                {
                    "path": path,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "text_preview": raw.decode("utf-8", errors="replace")[:4096]
                    if PurePosixPath(path).suffix.casefold() in TEXT_SUFFIXES
                    else None,
                }
            )
    return extracted


def _relabelled_evidence(
    records: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    mask_records = [record for record in records if record["mask_candidate"]]
    relabelled_records = [record for record in mask_records if record["relabelled_candidate"]]
    parent_counts = Counter(
        str(PurePosixPath(str(record["path"])).parent) for record in mask_records
    )
    relabelled_parent_counts = Counter(
        str(PurePosixPath(str(record["path"])).parent) for record in relabelled_records
    )
    exact_200_groups = sorted(
        {parent for parent, count in parent_counts.items() if count == EXPECTED_HUMAN_MASKS}
    )
    relabelled_exact_200_groups = sorted(
        {
            parent
            for parent, count in relabelled_parent_counts.items()
            if count == EXPECTED_HUMAN_MASKS
        }
    )
    metadata_text = "\n".join(
        str(item.get("text_preview") or "") for item in metadata
    ).casefold()
    metadata_mentions_relabelling = bool(
        re.search(r"\b200\b", metadata_text)
        and re.search(r"re[ -]?labell?ed|manual|human|corrected", metadata_text)
    )

    identified = False
    strategy = "not_identified_from_archive_structure"
    selected_count = 0
    if len(relabelled_records) == EXPECTED_HUMAN_MASKS:
        identified = True
        strategy = "exact_path_marked_relabelled_mask_count"
        selected_count = len(relabelled_records)
    elif len(relabelled_exact_200_groups) == 1:
        identified = True
        strategy = "single_path_marked_parent_group_with_exact_count"
        selected_count = EXPECTED_HUMAN_MASKS
    elif len(mask_records) == EXPECTED_HUMAN_MASKS and metadata_mentions_relabelling:
        identified = True
        strategy = "exact_total_mask_count_plus_metadata_statement"
        selected_count = len(mask_records)

    return {
        "expected_human_masks": EXPECTED_HUMAN_MASKS,
        "identified": identified,
        "identification_strategy": strategy,
        "selected_candidate_count": selected_count,
        "path_marked_relabelled_mask_candidates": len(relabelled_records),
        "metadata_mentions_relabelled_200": metadata_mentions_relabelling,
        "exact_200_mask_parent_groups": exact_200_groups,
        "relabelled_exact_200_parent_groups": relabelled_exact_200_groups,
        "candidate_examples": [str(record["path"]) for record in relabelled_records[:20]],
    }


def inventory_tar(
    archive_path: Path,
    *,
    inventory_path: Path,
    metadata_output_dir: Path,
) -> dict[str, Any]:
    """Validate the tar, write a full JSONL inventory, and extract only bounded metadata."""
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = validate_tar_members(archive)
    records = [classify_member(member) for member in members]
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    metadata = _extract_metadata(archive_path, records, metadata_output_dir)
    suffix_counts = Counter(
        str(record["suffix"]) for record in records if record["type"] == "file"
    )
    classification_counts = {
        key: sum(bool(record[key]) for record in records)
        for key in (
            "image_candidate",
            "source_image_candidate",
            "mask_candidate",
            "relabelled_candidate",
            "relabelled_mask_candidate",
            "readme_candidate",
            "license_candidate",
            "metadata_candidate",
        )
    }
    metadata_text = "\n".join(
        str(item.get("text_preview") or "") for item in metadata
    ).casefold()
    return {
        "archive_members": len(records),
        "archive_files": sum(record["type"] == "file" for record in records),
        "archive_directories": sum(record["type"] == "directory" for record in records),
        "uncompressed_file_bytes": sum(
            int(record["size_bytes"]) for record in records if record["type"] == "file"
        ),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "classification_counts": classification_counts,
        "metadata_extracted": metadata,
        "embedded_cc_by_4_signal": bool(
            re.search(r"cc[ -]?by[ -]?4(?:\.0)?|creative commons attribution 4", metadata_text)
        ),
        "relabelled_subset": _relabelled_evidence(records, metadata),
        "inventory_file": inventory_path.name,
        "inventory_sha256": _sha256(inventory_path),
    }


def run_inventory(
    *,
    output_dir: Path,
    work_dir: Path,
    expected_bytes: int = EXPECTED_ARCHIVE_BYTES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    staged_archive = work_dir / ARCHIVE_FILENAME
    archive_receipt = download_archive(
        SOURCE_ARCHIVE_URL,
        staged_archive,
        expected_bytes=expected_bytes,
    )
    archive_output = output_dir / "source" / ARCHIVE_FILENAME
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(staged_archive, archive_output)
    inventory_path = output_dir / "kit_inventory.jsonl"
    tar_summary = inventory_tar(
        archive_output,
        inventory_path=inventory_path,
        metadata_output_dir=output_dir / "source_metadata",
    )
    summary = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_record_url": SOURCE_RECORD_URL,
        "source_archive_url": SOURCE_ARCHIVE_URL,
        "source_archive_variant": "public_raw_dataset_binary_tree",
        "catalog_bagit_archive_bytes": CATALOG_BAGIT_ARCHIVE_BYTES,
        "declared_license": SOURCE_LICENSE,
        "archive_output": f"source/{ARCHIVE_FILENAME}",
        "archive_receipt": archive_receipt,
        "tar_inventory": tar_summary,
        "cloud_only_processing": True,
        "reviews_admitted": False,
        "strict_rows_admitted": 0,
        "training_eligible_rows": 0,
        "publication_allowed": False,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "audit_status": "inventory_only_pending_pixel_and_semantic_validation",
        "next_action": "audit_the_identified_human_mask_subset_without_admitting_rows",
    }
    summary_path = output_dir / "kit_inventory_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, default=EXPECTED_ARCHIVE_BYTES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_inventory(
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        expected_bytes=args.expected_bytes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
