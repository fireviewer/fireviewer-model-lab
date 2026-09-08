"""Acquire, merge, and validate the DINOv3 multi-task FireViewer v4 dataset."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).parent / "registries" / "dinov3-multitask-v4.json"
QUALITY_RANK = {
    "human_strong": 6,
    "sensor_derived": 5,
    "strong": 5,
    "weak": 3,
    "temporal_negative": 3,
    "negative": 2,
    "abstention": 1,
}


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported DINOv3 v4 campaign registry")
    if value.get("status") != "active":
        raise ValueError(
            "DINOv3 v4 weak-label campaign is superseded; use the strict composition registry"
        )
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise ValueError("DINOv3 v4 campaign must declare exactly four accessible sources")
    source_ids = [str(source.get("source_id", "")) for source in sources]
    if len(set(source_ids)) != len(source_ids) or not all(source_ids):
        raise ValueError("DINOv3 v4 campaign source IDs are invalid")
    return value


def find_source(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    for source in registry["sources"]:
        if source["source_id"] == source_id:
            return source
    raise KeyError(source_id)


def acquire_hf_sources(
    *,
    registry: dict[str, Any],
    campaign_root: Path,
    token: str,
    source_ids: tuple[str, ...] = ("fireviewer-v3-baseline", "pyronear-pyro-sdis"),
    workers: int = 1,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    results: list[dict[str, Any]] = []
    allowed = {"fireviewer-v3-baseline", "pyronear-pyro-sdis"}
    if not source_ids or not set(source_ids).issubset(allowed):
        raise ValueError(f"invalid Hugging Face source selection: {source_ids}")
    if workers <= 0:
        raise ValueError("Hugging Face acquisition workers must be positive")
    for source_id in source_ids:
        source = find_source(registry, source_id)
        destination = campaign_root / "sources" / source_id
        destination.mkdir(parents=True, exist_ok=True)
        resolved = snapshot_download(
            repo_id=str(source["repository"]),
            repo_type="dataset",
            revision=str(source["revision"]),
            token=token,
            local_dir=destination,
            max_workers=workers,
        )
        files = [
            path for path in destination.rglob("*") if path.is_file() and ".cache" not in path.parts
        ]
        results.append(
            {
                "source_id": source_id,
                "repository": source["repository"],
                "revision": source["revision"],
                "destination": resolved,
                "files": len(files),
                "bytes": sum(path.stat().st_size for path in files),
            }
        )
    return {"schema_version": 1, "sources": results}


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty campaign manifest: {path}")
    return rows


def adapt_baseline_manifest(
    *, campaign_root: Path, baseline_root: Path, output: Path
) -> dict[str, Any]:
    rows = _read_manifest(baseline_root / "manifest.jsonl")
    for row in rows:
        for key, value in list(row.items()):
            if key.endswith("_relpath") and value:
                relative = (
                    Path("sources") / "fireviewer-v3-baseline" / Path(str(value))
                ).as_posix()
                if not (campaign_root / relative).is_file():
                    raise FileNotFoundError(campaign_root / relative)
                row[key] = relative
        row["origin_dataset"] = "fireviewer/dinov3-multitask-fireviewer-v3-dataset"
        row["sample_weight"] = 4.0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return {"rows": len(rows), "manifest": str(output)}


def _quality(row: dict[str, Any]) -> tuple[int, float]:
    strength = str(row.get("annotation_strength", "weak"))
    quality = str(row.get("mask_quality", ""))
    if quality == "human_strong":
        strength = "human_strong"
    return QUALITY_RANK.get(strength, 0), float(row.get("sample_weight", 1.0))


def _asset_paths(row: dict[str, Any]) -> Iterable[str]:
    for key, value in row.items():
        if key.endswith("_relpath") and value:
            yield str(value)


def merge_manifests(
    *, campaign_root: Path, manifests: list[Path], output_root: Path
) -> dict[str, Any]:
    by_sample: dict[str, dict[str, Any]] = {}
    by_image_digest: dict[str, dict[str, Any]] = {}
    duplicate_images = 0
    for manifest in manifests:
        for row in _read_manifest(manifest):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"missing sample_id in {manifest}")
            if sample_id in by_sample:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            for relative in _asset_paths(row):
                target = (campaign_root / relative).resolve()
                if not target.is_relative_to(campaign_root.resolve()) or not target.is_file():
                    raise FileNotFoundError(f"missing or unsafe campaign asset: {relative}")
            digest = str(row.get("image_sha256", ""))
            if digest and digest in by_image_digest:
                duplicate_images += 1
                existing = by_image_digest[digest]
                if _quality(row) > _quality(existing):
                    del by_sample[str(existing["sample_id"])]
                    by_sample[sample_id] = row
                    by_image_digest[digest] = row
                continue
            by_sample[sample_id] = row
            if digest:
                by_image_digest[digest] = row

    rows = sorted(by_sample.values(), key=lambda row: str(row["sample_id"]))
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split for {row['sample_id']}: {split}")
        group = str(row.get("split_group", ""))
        if not group:
            raise ValueError(f"missing split_group for {row['sample_id']}")
        group_splits.setdefault(group, set()).add(split)
        if "visual_abstention_reason" not in row:
            raise ValueError(f"missing abstention field for {row['sample_id']}")
    leaks = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaks:
        raise ValueError(f"split-group leakage: {leaks[:20]}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "dataset_id": "fireviewer/dinov3-multitask-fireviewer-v4-dataset",
        "rows": len(rows),
        "input_manifests": [str(path) for path in manifests],
        "source_counts": dict(sorted(Counter(str(row["source_id"]) for row in rows).items())),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "strength_counts": dict(
            sorted(Counter(str(row.get("annotation_strength", "unknown")) for row in rows).items())
        ),
        "abstention_rows": sum(row.get("visual_abstention_reason") is not None for row in rows),
        "duplicate_images_removed": duplicate_images,
        "split_group_leakage": leaks,
        "manifest": str(manifest),
        "training_ready": not leaks
        and set(row["split"] for row in rows) == {"train", "validation", "test"},
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
