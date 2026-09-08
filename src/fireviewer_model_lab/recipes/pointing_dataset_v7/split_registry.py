"""Persistent image/group split locks. Never infer independence from a filename alone."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "fireviewer.pointing-persistent-split-registry.v1"
SPLITS = ("train", "validation", "test")


def canonical_source_group(value: str) -> str:
    lowered = value.casefold()
    if "hpwren" in lowered or "figlib" in lowered:
        return "hpwren:" + lowered.replace("/", ":").split(":")[-1]
    return lowered


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def put_lock(mapping: dict, key: str, split: str) -> None:
    if not key or split not in SPLITS:
        raise ValueError(f"Missing identity or invalid split: {key!r}/{split!r}")
    if key in mapping and mapping[key] != split:
        raise ValueError(f"Conflicting frozen split for {key}: {mapping[key]} versus {split}")
    mapping[key] = split


def make_registry(rows: list[dict], parent: dict | None = None) -> dict:
    registry = {"schema": SCHEMA, "sha256": dict((parent or {}).get("sha256", {})),
                "groups": dict((parent or {}).get("groups", {})),
                "image_groups": dict((parent or {}).get("image_groups", {})),
                "source_groups": {key: list(value) for key, value in (parent or {}).get("source_groups", {}).items()}}
    for row in rows:
        sha = str(row["sha256"])
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise ValueError(f"Invalid SHA-256 identity: {sha}")
        put_lock(registry["sha256"], sha, row["split"])
        put_lock(registry["groups"], row["split_group_id"], row["split"])
        registry["image_groups"][sha] = row["split_group_id"]
        if row.get("source_group_id"):
            source_group = canonical_source_group(row["source_group_id"])
            registry["source_groups"][source_group] = sorted(set(registry["source_groups"].get(source_group, [])) | {row["split"]})
    registry["counts"] = dict(Counter(registry["sha256"].values()))
    registry["policy"] = "Never move a known image or group between splits; retain retired training exposures."
    registry["limitation"] = "Exact bytes and declared groups only; physical event identity requires separate review."
    return registry


def load_registry(path: Path) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != SCHEMA or not registry.get("sha256") or not registry.get("groups"):
        raise ValueError("Invalid or empty persistent split registry")
    if any(split not in SPLITS for mapping in (registry["sha256"], registry["groups"]) for split in mapping.values()):
        raise ValueError("Invalid frozen split")
    return registry


def assign_locked(rows: list[dict], registry: dict | None = None, *, new_split: str | None = None) -> dict[str, str]:
    """Connected groups (including renamed copies) share one immutable assignment."""
    if new_split is not None and new_split not in SPLITS:
        raise ValueError(new_split)
    registry = registry or {"sha256": {}, "groups": {}}
    source_locks: dict[str, set[str]] = defaultdict(set)
    for group, splits in registry.get("source_groups", {}).items():
        source_locks[canonical_source_group(group)].update(splits)
    parents: dict[str, str] = {}
    by_sha: dict[str, str] = {}
    by_source_group: dict[str, str] = {}

    def find(group: str) -> str:
        parents.setdefault(group, group)
        if parents[group] != group:
            parents[group] = find(parents[group])
        return parents[group]

    for row in rows:
        group = row["split_group_id"]
        if not group:
            raise ValueError("Missing split group")
        root = find(group)
        old = by_sha.setdefault(row["sha256"], group)
        other = find(old)
        if root != other:
            parents[max(root, other)] = min(root, other)
        if row.get("source_group_id"):
            old_group = by_source_group.setdefault(canonical_source_group(row["source_group_id"]), group)
            root, other = find(group), find(old_group)
            if root != other:
                parents[max(root, other)] = min(root, other)
    members: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        members[find(row["split_group_id"])].append(row)
    assignments = {}
    for root, component in members.items():
        locked = {split for row in component for split in (
            registry["sha256"].get(row["sha256"]), registry["groups"].get(row["split_group_id"])
        ) if split is not None}
        for row in component:
            if row.get("source_group_id"):
                locked.update(source_locks.get(canonical_source_group(row["source_group_id"]), []))
        if len(locked) > 1:
            raise ValueError(f"Conflicting frozen splits in connected group {root}: {sorted(locked)}")
        # Independent of corpus cardinality, source aliases, row ordering and label corrections.
        bucket = int(hashlib.sha256(f"fireviewer-split-v1:{root}".encode()).hexdigest(), 16) % 10000
        split = next(iter(locked)) if locked else new_split or (
            "test" if bucket < 1000 else "validation" if bucket < 2000 else "train")
        for row in component:
            assignments[row["split_group_id"]] = split
    return assignments


def assert_frozen_holdouts_retained(rows: list[dict], registry: dict) -> None:
    present = {row["sha256"] for row in rows}
    missing = {sha for sha, split in registry["sha256"].items() if split != "train"} - present
    if missing:
        raise ValueError(f"Refusing to trim frozen evaluation images: {len(missing)} missing")


def verify_coco(coco_root: Path, registry: dict, *, require_exact: bool = True) -> dict:
    """Check actual image bytes, annotations, counts and declared groups before GPU use."""
    receipt_path = coco_root / "coco_view_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "ready":
        raise ValueError("COCO view is not ready")
    dataset_root = Path(receipt["dataset_root"])
    for filename, field in (("selection_manifest.jsonl", "selection_manifest_sha256"), ("report.json", "dataset_report_sha256")):
        if digest(dataset_root / filename) != receipt[field]:
            raise ValueError(f"Dataset identity mismatch: {filename}")
    observed, groups, annotation_hashes = {}, {}, {}
    source_groups: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        folder = "valid" if split == "validation" else split
        path = coco_root / folder / "_annotations.coco.json"
        annotation_hashes[split] = digest(path)
        expected = receipt["splits"][split]
        if annotation_hashes[split] != expected["annotation_sha256"]:
            raise ValueError(f"Annotation hash mismatch: {split}")
        coco = json.loads(path.read_text(encoding="utf-8"))
        if {row["id"]: row["name"] for row in coco["categories"]} != {0: "fire", 1: "smoke"}:
            raise ValueError("Unexpected COCO categories")
        if len(coco["images"]) != expected["images"] or len(coco["annotations"]) != expected["annotations"]:
            raise ValueError(f"COCO cardinality mismatch: {split}")
        for row in coco["images"]:
            path = (coco_root / folder / row["file_name"]).resolve()
            if not path.is_relative_to((coco_root / folder).resolve()):
                raise ValueError("Image path escapes split")
            sha = digest(path)
            if sha != row["fireviewer_sha256"]:
                raise ValueError(f"Image identity mismatch: {path}")
            if sha in observed:
                raise ValueError(f"Duplicate image across COCO: {sha}")
            observed[sha] = split
            group = registry["image_groups"].get(sha)
            put_lock(groups, group, split)
            if not row.get("fireviewer_source_group_id"):
                raise ValueError("Missing source group in COCO")
            source_groups[canonical_source_group(row["fireviewer_source_group_id"])].add(split)
            if registry["sha256"].get(sha) != split or registry["groups"].get(group) != split:
                raise ValueError(f"COCO violates frozen split: {sha}/{group}/{split}")
    if require_exact and observed != registry["sha256"]:
        raise ValueError("COCO image membership differs from frozen registry")
    source_conflicts = {key: sorted(value) for key, value in source_groups.items() if len(value) > 1}
    return {"status": "passed" if not source_conflicts else "identity_verified_source_group_conflicts",
            "image_bytes_verified": len(observed), "groups": len(groups),
            "split_counts": dict(Counter(observed.values())), "annotation_sha256": annotation_hashes,
            "exact_or_split_group_overlap": 0, "source_group_conflicts": source_conflicts,
            "coco_receipt_sha256": digest(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    registry = make_registry(read_rows(args.manifest))
    registry["source_manifest"] = str(args.manifest.resolve())
    registry["source_manifest_sha256"] = digest(args.manifest)
    registry["verification"] = verify_coco(args.coco_root.resolve(), registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(registry["verification"], indent=2))


if __name__ == "__main__":
    main()
