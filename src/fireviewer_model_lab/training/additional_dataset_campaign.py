from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REGISTRY_PATH = Path(__file__).parent / "registries" / "additional-sources-v1.json"
MIRROR_POLICIES = {"mirror_normalized"}
REFERENCE_POLICIES = {
    "reference_only",
    "blocked_no_redistribution",
    "direct_benchmark_only",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported additional dataset registry schema")
    batches = registry.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("The registry must declare at least one batch")
    batch_ids: set[str] = set()
    source_ids: set[str] = set()
    for batch in batches:
        batch_id = str(batch["batch_id"])
        if batch_id in batch_ids:
            raise ValueError(f"Duplicate batch_id: {batch_id}")
        batch_ids.add(batch_id)
        if not str(batch["target_prefix"]).startswith("additional/v1/"):
            raise ValueError(f"Invalid target prefix for {batch_id}")
        for source in batch["sources"]:
            source_id = str(source["source_id"])
            if source_id in source_ids:
                raise ValueError(f"Duplicate source_id: {source_id}")
            source_ids.add(source_id)
            policy = str(source["payload_policy"])
            if policy not in MIRROR_POLICIES | REFERENCE_POLICIES:
                raise ValueError(f"Unsupported payload policy for {source_id}: {policy}")
            if policy in MIRROR_POLICIES and not source.get("revision"):
                raise ValueError(f"Mirrored source must be revision-pinned: {source_id}")
    return registry


def find_batch(registry: dict[str, Any], batch_id: str) -> dict[str, Any]:
    for batch in registry["batches"]:
        if batch["batch_id"] == batch_id:
            return batch
    raise KeyError(f"Unknown batch: {batch_id}")


def remote_only_workspace(registry: dict[str, Any], workspace: Path) -> Path:
    resolved = workspace.resolve()
    if not registry["payload_execution_policy"]["remote_only"]:
        return resolved
    allowed = [
        Path(path).resolve()
        for path in registry["payload_execution_policy"]["allowed_workspace_roots"]
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise ValueError(
            "Payload preparation is remote-only; workspace must be below one of: "
            + ", ".join(str(path) for path in allowed)
        )
    return resolved


def _repo_file_metadata(info: Any) -> tuple[int, int]:
    siblings = list(info.siblings or [])
    total_bytes = 0
    for sibling in siblings:
        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            size = getattr(lfs, "size", None) if lfs is not None else None
        if size is None:
            raise ValueError(f"Missing remote size for {sibling.rfilename}")
        total_bytes += int(size)
    return len(siblings), total_bytes


def audit_sources(registry: dict[str, Any], *, batch_id: str | None = None) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("huggingface-hub is required for the online audit") from exc

    api = HfApi()
    batches = [find_batch(registry, batch_id)] if batch_id else registry["batches"]
    results: list[dict[str, Any]] = []
    for batch in batches:
        for source in batch["sources"]:
            revision = source.get("revision")
            if not revision:
                results.append(
                    {
                        "source_id": source["source_id"],
                        "status": "external_unpinned_reference",
                        "payload_policy": source["payload_policy"],
                    }
                )
                continue
            info = api.dataset_info(source["repository"], revision=revision, files_metadata=True)
            if info.sha != revision:
                raise ValueError(
                    f"Revision mismatch for {source['source_id']}: {info.sha} != {revision}"
                )
            file_count, total_bytes = _repo_file_metadata(info)
            expected_files = source.get("expected_repo_files")
            expected_bytes = source.get("expected_repo_bytes")
            if expected_files is not None and file_count != int(expected_files):
                raise ValueError(
                    f"File-count drift for {source['source_id']}: {file_count} != {expected_files}"
                )
            if expected_bytes is not None and total_bytes != int(expected_bytes):
                raise ValueError(
                    f"Byte-count drift for {source['source_id']}: {total_bytes} != {expected_bytes}"
                )
            results.append(
                {
                    "source_id": source["source_id"],
                    "repository": source["repository"],
                    "revision": info.sha,
                    "repo_files": file_count,
                    "repo_bytes": total_bytes,
                    "payload_policy": source["payload_policy"],
                    "status": "pinned_remote_verified",
                }
            )
    return {
        "campaign_id": registry["campaign_id"],
        "checked_at": datetime.now(UTC).isoformat(),
        "sources": results,
    }


def _safe_member_path(destination: Path, member_name: str) -> Path:
    normalized = PurePosixPath(member_name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Archive member escapes destination: {member_name}")
    target = (destination / Path(*normalized.parts)).resolve()
    if destination.resolve() != target and destination.resolve() not in target.parents:
        raise ValueError(f"Archive member escapes destination: {member_name}")
    return target


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = _safe_member_path(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported tar member: {member.name}")
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable tar member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)


def verify_expected_assets(source: dict[str, Any], source_dir: Path) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for relative, expected in source.get("expected_assets", {}).items():
        path = source_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(expected["size_bytes"]):
            raise ValueError(f"Size mismatch for {relative}: {size}")
        if digest != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative}: {digest}")
        verified.append({"path": relative, "size_bytes": size, "sha256": digest})
    return verified


def inspect_tree(root: Path) -> dict[str, Any]:
    suffix_counts: Counter[str] = Counter()
    file_count = 0
    total_bytes = 0
    image_count = 0
    text_samples: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        suffix = path.suffix.casefold() or "<none>"
        suffix_counts[suffix] += 1
        if suffix in IMAGE_SUFFIXES:
            image_count += 1
        elif suffix in {".csv", ".json", ".txt", ".yaml", ".yml"} and len(text_samples) < 20:
            text_samples.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sample": path.read_text(encoding="utf-8", errors="replace")[:2048],
                }
            )
    return {
        "root": root.name,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "image_count": image_count,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "text_samples": text_samples,
    }


def _alarmod_annotations(label_path: Path) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {label_path}:{line_number}")
        class_id = int(fields[0])
        center_x, center_y, width, height = (float(value) for value in fields[1:])
        if class_id != 0:
            raise ValueError(f"Unexpected class {class_id} at {label_path}:{line_number}")
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, width, height)):
            raise ValueError(f"Out-of-range YOLO value at {label_path}:{line_number}")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Empty YOLO box at {label_path}:{line_number}")
        annotations.append(
            {
                "class_id": 0,
                "class_name": "fire",
                "bbox_xywh_normalized": [center_x, center_y, width, height],
                "point_xy_normalized": [center_x, center_y],
            }
        )
    return annotations


def _alarmod_frame_group(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) != 4 or parts[0] != "image":
        raise ValueError(f"Unexpected Alarmod image stem: {stem}")
    int(parts[1])
    int(parts[2])
    int(parts[3])
    return parts[1]


def build_alarmod_manifest(prepared_dir: Path, source: dict[str, Any]) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("Pillow is required to validate Alarmod images") from exc

    records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    frame_groups: dict[str, set[str]] = {}
    content_hash_splits: dict[str, str] = {}
    for split in ("train", "validation"):
        split_dir = prepared_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images = sorted(images_dir.glob("*.jpg"))
        labels = sorted(labels_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ValueError(
                f"Alarmod image/label mismatch in {split}: "
                f"missing_labels={len(image_stems - label_stems)} "
                f"missing_images={len(label_stems - image_stems)}"
            )
        frame_groups[split] = set()
        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            annotations = _alarmod_annotations(label_path)
            frame_group = _alarmod_frame_group(image_path.stem)
            frame_groups[split].add(frame_group)
            with Image.open(image_path) as image:
                image.verify()
                size = image.size
            if size != (1280, 720):
                raise ValueError(f"Unexpected Alarmod image size for {image_path}: {size}")
            digest = sha256_file(image_path)
            previous_split = content_hash_splits.setdefault(digest, split)
            if previous_split != split:
                raise ValueError(f"Exact image duplicate crosses Alarmod splits: {image_path.name}")
            image_relative = image_path.relative_to(prepared_dir).as_posix()
            label_relative = label_path.relative_to(prepared_dir).as_posix()
            records.append(
                {
                    "sample_id": f"alarmod_forest_fire:{split}:{image_path.stem}",
                    "source_id": "alarmod_forest_fire",
                    "source_record_id": image_path.stem,
                    "split": split,
                    "split_group": f"alarmod_frame:{frame_group}",
                    "image_relpath": image_relative,
                    "label_relpath": label_relative,
                    "sha256": digest,
                    "width": 1280,
                    "height": 720,
                    "annotations": annotations,
                    "negative": not annotations,
                    "source_asset": {
                        "dataset": source["repository"],
                        "revision": source["revision"],
                        "license": source["declared_license"],
                        "derived_from": "FLAME file 9 images and file 10 masks",
                    },
                }
            )
            split_counts[split] += 1
            if annotations:
                positive_counts[split] += 1
            box_counts[split] += len(annotations)

    overlap = frame_groups["train"] & frame_groups["validation"]
    if overlap:
        raise ValueError(f"Alarmod source-frame groups cross splits: {sorted(overlap)[:10]}")
    expected_rows = {key: int(value) for key, value in source["declared_rows"].items()}
    if dict(split_counts) != expected_rows:
        raise ValueError(f"Unexpected Alarmod split counts: {dict(split_counts)}")
    manifest_path = prepared_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "rows": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "positive_counts": dict(sorted(positive_counts.items())),
        "negative_counts": {
            split: split_counts[split] - positive_counts[split] for split in split_counts
        },
        "box_counts": dict(sorted(box_counts.items())),
        "frame_group_counts": {
            split: len(groups) for split, groups in sorted(frame_groups.items())
        },
        "cross_split_frame_groups": 0,
        "cross_split_exact_duplicates": 0,
        "manifest_sha256": sha256_file(manifest_path),
    }


def build_tar_inventory(archive_path: Path, output_path: Path) -> dict[str, Any]:
    suffix_counts: Counter[str] = Counter()
    top_level_counts: Counter[str] = Counter()
    member_count = 0
    payload_bytes = 0
    with (
        output_path.open("w", encoding="utf-8", newline="\n") as output,
        tarfile.open(archive_path, mode="r:gz") as archive,
    ):
        for member in archive:
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported tar member: {member.name}")
            _safe_member_path(output_path.parent / "virtual-extraction-root", member.name)
            if member.isdir():
                continue
            normalized = PurePosixPath(member.name.replace("\\", "/"))
            suffix = normalized.suffix.casefold() or "<none>"
            suffix_counts[suffix] += 1
            top_level_counts[normalized.parts[0]] += 1
            member_count += 1
            payload_bytes += int(member.size)
            output.write(
                json.dumps(
                    {
                        "path": normalized.as_posix(),
                        "size_bytes": int(member.size),
                        "suffix": suffix,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "member_files": member_count,
        "uncompressed_payload_bytes": payload_bytes,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "top_level_counts": dict(sorted(top_level_counts.items())),
        "inventory_sha256": sha256_file(output_path),
    }


def acquire_source(source: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    if source["payload_policy"] not in MIRROR_POLICIES:
        raise ValueError(
            f"Payload acquisition forbidden by policy for {source['source_id']}: "
            f"{source['payload_policy']}"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("huggingface-hub is required for acquisition") from exc

    source_dir.mkdir(parents=True, exist_ok=False)
    snapshot_download(
        repo_id=source["repository"],
        repo_type="dataset",
        revision=source["revision"],
        allow_patterns=source.get("allow_patterns"),
        local_dir=source_dir,
    )
    verified = verify_expected_assets(source, source_dir)
    return {
        "source_id": source["source_id"],
        "repository": source["repository"],
        "revision": source["revision"],
        "verified_assets": verified,
    }


def prepare_source(source: dict[str, Any], source_dir: Path, prepared_dir: Path) -> dict[str, Any]:
    prepared_dir.mkdir(parents=True, exist_ok=False)
    source_id = source["source_id"]
    validation: dict[str, Any] | None = None
    if source_id == "alarmod_forest_fire":
        safe_extract_zip(source_dir / "train.zip", prepared_dir)
        safe_extract_zip(source_dir / "val.zip", prepared_dir)
        (prepared_dir / "val").rename(prepared_dir / "validation")
        validation = build_alarmod_manifest(prepared_dir, source)
        readme = source_dir / "README.md"
        if readme.is_file():
            shutil.copy2(readme, prepared_dir / "UPSTREAM_README.md")
    elif source_id == "eo4wildfires":
        upstream_dir = prepared_dir / "upstream"
        upstream_dir.mkdir()
        source_archive = source_dir / "eo4wildfires.tar.gz"
        target_archive = upstream_dir / source_archive.name
        try:
            os.link(source_archive, target_archive)
        except OSError:
            shutil.copy2(source_archive, target_archive)
        validation = build_tar_inventory(target_archive, prepared_dir / "tar-inventory.jsonl")
        if validation["member_files"] != int(source["declared_events"]):
            raise ValueError(
                f"EO4Wildfires event-count mismatch: {validation['member_files']} "
                f"!= {source['declared_events']}"
            )
        if validation["suffix_counts"] != {".nc": int(source["declared_events"])}:
            raise ValueError(
                f"EO4Wildfires archive contains unexpected payloads: {validation['suffix_counts']}"
            )
        for path in source_dir.glob("*.csv"):
            shutil.copy2(path, prepared_dir / path.name)
        readme = source_dir / "README.md"
        if readme.is_file():
            shutil.copy2(readme, prepared_dir / "UPSTREAM_README.md")
    else:  # pragma: no cover - registry prevents this branch in v1
        raise ValueError(f"No normalizer implemented for {source_id}")

    inspection = inspect_tree(prepared_dir)
    report = {
        "schema_version": 1,
        "source_id": source_id,
        "repository": source["repository"],
        "revision": source["revision"],
        "declared_license": source["declared_license"],
        "payload_policy": source["payload_policy"],
        "split_policy": source["split_policy"],
        "training_gate": source["training_gate"],
        "inspection": inspection,
        "validation": validation,
        "promotion_status": (
            "validated_training_ready"
            if source_id == "alarmod_forest_fire" and validation is not None
            else "archived_validated_requires_schema_conversion"
            if source_id == "eo4wildfires" and validation is not None
            else "inspected_not_training_ready"
        ),
    }
    (prepared_dir / "preparation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def write_reference_batch(
    registry: dict[str, Any], batch: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "campaign_id": registry["campaign_id"],
        "batch_id": batch["batch_id"],
        "payload_included": False,
        "sources": batch["sources"],
    }
    (output_dir / "reference-registry.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def upload_repo_root(
    registry: dict[str, Any], batch: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("huggingface-hub is required for upload") from exc

    api = HfApi()
    repo_id = registry["target_repository"]
    target_root = repo_root / Path(*PurePosixPath(batch["target_prefix"]).parts)
    local_files = {
        path.relative_to(repo_root).as_posix(): path
        for path in target_root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if not local_files:
        raise ValueError(f"No local files found below {target_root}")
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=repo_root,
        print_report=False,
    )
    info = api.dataset_info(repo_id, files_metadata=True)
    if not info.private:
        raise RuntimeError(f"Target dataset unexpectedly became public: {repo_id}")
    expected_prefix = str(batch["target_prefix"]).rstrip("/") + "/"
    uploaded = {
        item.rfilename: item
        for item in info.siblings or []
        if item.rfilename.startswith(expected_prefix)
    }
    if set(uploaded) != set(local_files):
        raise RuntimeError(
            f"Remote inventory mismatch below {expected_prefix}: "
            f"local={len(local_files)} remote={len(uploaded)}"
        )
    size_mismatches = [
        name
        for name, path in local_files.items()
        if int(uploaded[name].size or -1) != path.stat().st_size
    ]
    if size_mismatches:
        raise RuntimeError(f"Remote size mismatch: {size_mismatches[:10]}")
    return {
        "repository": repo_id,
        "commit": info.sha,
        "target_prefix": batch["target_prefix"],
        "remote_files": len(uploaded),
        "remote_bytes": sum(int(item.size or 0) for item in uploaded.values()),
        "inventory_verified": True,
    }


def plan(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_id": registry["campaign_id"],
        "target_repository": registry["target_repository"],
        "batches": [
            {
                "batch_id": batch["batch_id"],
                "target_prefix": batch["target_prefix"],
                "sources": [
                    {
                        "source_id": source["source_id"],
                        "payload_policy": source["payload_policy"],
                        "revision": source.get("revision"),
                    }
                    for source in batch["sources"]
                ],
            }
            for batch in registry["batches"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare additional FireWarning datasets")
    parser.add_argument(
        "command",
        choices=("plan", "audit", "prepare-downloaded", "prepare-reference", "upload"),
    )
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--batch")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    if args.command == "plan":
        result = plan(registry)
    elif args.command == "audit":
        result = audit_sources(registry, batch_id=args.batch)
    elif args.command == "prepare-downloaded":
        if not args.batch or args.source_dir is None or args.output is None:
            parser.error("prepare-downloaded requires --batch, --source-dir and --output")
        batch = find_batch(registry, args.batch)
        mirrors = [
            source for source in batch["sources"] if source["payload_policy"] in MIRROR_POLICIES
        ]
        if len(mirrors) != 1:
            raise ValueError("prepare-downloaded requires exactly one mirrored source")
        verified = verify_expected_assets(mirrors[0], args.source_dir)
        result = prepare_source(mirrors[0], args.source_dir, args.output)
        result["verified_source_assets"] = verified
    elif args.command == "prepare-reference":
        if not args.batch or args.output is None:
            parser.error("prepare-reference requires --batch and --output")
        batch = find_batch(registry, args.batch)
        if any(source["payload_policy"] in MIRROR_POLICIES for source in batch["sources"]):
            raise ValueError("prepare-reference accepts metadata-only batches")
        write_reference_batch(registry, batch, args.output)
        result = {"batch_id": args.batch, "output": str(args.output)}
    else:
        if not args.batch or args.repo_root is None:
            parser.error("upload requires --batch and --repo-root")
        batch = find_batch(registry, args.batch)
        result = upload_repo_root(registry, batch, args.repo_root)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")


if __name__ == "__main__":
    main()
