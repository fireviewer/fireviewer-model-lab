from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

PUBLIC_REPOSITORY = "Charlbi/firewarning-train-bundles-v1"
PRIVATE_REPOSITORY = "Charlbi/firewarning-training-corpus"

PUBLIC_ARTIFACTS = {
    "media-filter-fire-smoke-v1.zip": (
        32_772_718_037,
        "ed6ee4cf71a2538cfce7b95e6e548528d94b4cea095c8d03ccfc4c0e5ba4c8ae",
    ),
    "burned-area-segmentation-v1.zip": (
        30_950_528_241,
        "3d22beb4d21aa051e003f76b859ec46750b4fd6fba9f6c446a99f16fbad7ce33",
    ),
    "cross-view-localization-v1.zip": (
        15_895_751_194,
        "5f5e14083da209bd978117e2c7470f8f63decdb4ba3223914da7e7164aa64f5f",
    ),
    "fire-pointing-lora-v1.zip": (
        32_589_764_131,
        "a43cda497078b89960c300fce26280171441cf6111c208f96adfcfdecdc9762b",
    ),
    "cross-view-registration-v1.zip": (
        2_299_599_606,
        "a7e7d9205c34f96cf1843a7b7a9455e81a704c2926b9ba648bb00ecbddbc2f1e",
    ),
    "media-triage-fire-smoke-v1.zip": (
        31_688_094_244,
        "0ce5a30979007468efa8dd2f2ec5b20547943e080a6ed36318297023456ece34",
    ),
    "orchestrator-gates-sft-v1.zip": (
        17_444,
        "3ca4813cae6f04b806fce16be7a08a6854409dcd96df166d62a346e6d9a809ff",
    ),
}

FINALIZED_PRIVATE_PREFIXES = (
    "datasets/corpus/fasdd/",
    "datasets/corpus/pyro-sdis-v0.1.0/",
    "additional/v1/detection_uav_pointing_v1/alarmod_forest_fire/",
    "datasets/corpus/hls-burn-scars-v1/",
    "additional/v1/satellite_burnscar_multisensor_v1/eo4wildfires/",
    "datasets/sources/justzoomin-selective/",
    "datasets/corpus/cross-view-coarse-localizer-v0.1.0/",
    "datasets/corpus/streetview-global-context-v1/",
    "datasets/sources/aerialextrematch-localization/",
    "datasets/sources/odm-cross-view/",
    "datasets/corpus/fire-pointing-v0.1.0/",
    "datasets/corpus/wikimedia-candidates-v0.1.0/",
    "datasets/corpus/cross-view-registration-v0.1.0/",
)

DISCARDABLE_PRIVATE_PREFIXES = (
    "datasets/training/qwen3-vl-4b-spatial/",
    "datasets/training/cross-view-coarse-localizer-dinov2-v0.2.0/",
)

PRUNABLE_PRIVATE_PREFIXES = (
    *FINALIZED_PRIVATE_PREFIXES,
    *DISCARDABLE_PRIVATE_PREFIXES,
)

PRUNE_REASONS = {
    **{
        prefix: "source_replaced_by_verified_public_training_bundle"
        for prefix in FINALIZED_PRIVATE_PREFIXES
    },
    "datasets/training/qwen3-vl-4b-spatial/": (
        "reproducible_nonportable_derived_annotations_replaced_by_"
        "public_fire_pointing_bundle_entrypoint"
    ),
    "datasets/training/cross-view-coarse-localizer-dinov2-v0.2.0/": (
        "orphaned_partial_feature_cache_without_checkpoint_or_resume_metadata"
    ),
}

MANAGED_DOCUMENTS = {
    "README.md",
    "repository-manifest.json",
    "metadata/datasetfire/dataset-index.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune private source shards replaced by verified public train bundles."
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def verify_public_repository(api: HfApi, cache_dir: Path) -> str:
    info = api.dataset_info(PUBLIC_REPOSITORY, files_metadata=True)
    if info.private:
        raise RuntimeError(f"Public repository is private: {PUBLIC_REPOSITORY}")
    files = {item.rfilename: item for item in info.siblings}
    for path, (expected_size, expected_sha256) in PUBLIC_ARTIFACTS.items():
        item = files.get(path)
        if item is None or item.lfs is None:
            raise RuntimeError(f"Missing public LFS artifact: {path}")
        if item.size != expected_size or item.lfs.sha256 != expected_sha256:
            raise RuntimeError(f"Public artifact mismatch: {path}")

    index_path = hf_hub_download(
        PUBLIC_REPOSITORY,
        "train-bundles-v1.json",
        repo_type="dataset",
        revision=info.sha,
        cache_dir=cache_dir,
    )
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    indexed = {
        item["artifact"]["path"]: (
            item["artifact"]["size_bytes"],
            item["artifact"]["sha256"],
        )
        for item in index["bundles"]
    }
    if indexed != PUBLIC_ARTIFACTS:
        raise RuntimeError("The public bundle index does not match the public artifacts.")
    return info.sha


def load_private_manifest(cache_dir: Path, revision: str) -> dict:
    path = hf_hub_download(
        PRIVATE_REPOSITORY,
        "repository-manifest.json",
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def logical_manifest_path(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/manifest.json"


def render_readme(public_revision: str) -> bytes:
    content = f"""---
license: other
pretty_name: FireWarning private evaluation and unfinished corpus
---

# FireWarning private evaluation and unfinished corpus

This private repository now contains only evaluation, critical-validation,
operational-reference, unfinished-training and retained auxiliary artifacts.

The seven finalized training bundles were migrated to the public dataset
[`{PUBLIC_REPOSITORY}`](https://huggingface.co/datasets/{PUBLIC_REPOSITORY})
and verified at revision `{public_revision}` before their former private source
shards were removed. Reproducible derived annotations with non-portable local
paths and an orphaned partial feature cache were also removed after an explicit
content audit.

- `repository-manifest.json` is the current retained inventory.
- `metadata/datasetfire/dataset-index.json` lists retained logical datasets.
- Critical and operational evaluation sets remain private and are not training members.
- Licenses, provenance, consent and publication rights remain source-specific.
- Private storage does not grant permission to redistribute source media.
"""
    return content.encode("utf-8")


def build_documents(
    *,
    private_revision: str,
    public_revision: str,
    retained_files: list,
    deleted_files: list,
    source_manifest: dict,
) -> tuple[bytes, bytes]:
    retained_paths = {item.rfilename for item in retained_files}
    logical_datasets = [
        item
        for item in source_manifest.get("datasets", [])
        if logical_manifest_path(item["dataset_id"]) in retained_paths
    ]
    logical_datasets.sort(key=lambda item: item["dataset_id"])

    index = {
        "schema_version": 2,
        "repository": PRIVATE_REPOSITORY,
        "role": "private_evaluation_reference_and_unfinished_data",
        "finalized_training_repository": PUBLIC_REPOSITORY,
        "finalized_training_revision": public_revision,
        "dataset_count": len(logical_datasets),
        "datasets": [
            {
                "dataset_id": item["dataset_id"],
                "file_count": item["file_count"],
                "source_bytes": item["source_bytes"],
                "manifest_path": logical_manifest_path(item["dataset_id"]),
            }
            for item in logical_datasets
        ],
    }

    remote_inventory = []
    for item in sorted(retained_files, key=lambda value: value.rfilename):
        if item.rfilename in MANAGED_DOCUMENTS:
            continue
        entry = {"path": item.rfilename, "size_bytes": int(item.size or 0)}
        if item.lfs is not None:
            entry["sha256"] = item.lfs.sha256
        remote_inventory.append(entry)

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": PRIVATE_REPOSITORY,
        "parent_revision": private_revision,
        "migration": {
            "public_repository": PUBLIC_REPOSITORY,
            "public_revision": public_revision,
            "verified_artifacts": {
                path: {"size_bytes": size, "sha256": sha256}
                for path, (size, sha256) in PUBLIC_ARTIFACTS.items()
            },
            "deleted_prefixes": list(PRUNABLE_PRIVATE_PREFIXES),
            "deletion_reasons": PRUNE_REASONS,
            "deleted_file_count": len(deleted_files),
            "deleted_size_bytes": sum(int(item.size or 0) for item in deleted_files),
        },
        "retained_dataset_count": len(logical_datasets),
        "retained_file_count_excluding_managed_documents": len(remote_inventory),
        "retained_size_bytes_excluding_managed_documents": sum(
            item["size_bytes"] for item in remote_inventory
        ),
        "datasets": logical_datasets,
        "remote_inventory": remote_inventory,
    }
    return (
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def main() -> int:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    public_revision = verify_public_repository(api, args.cache_dir)
    private_info = api.dataset_info(PRIVATE_REPOSITORY, files_metadata=True)
    if not private_info.private:
        raise RuntimeError(f"Private repository is public: {PRIVATE_REPOSITORY}")

    deleted_files = [
        item
        for item in private_info.siblings
        if item.rfilename.startswith(PRUNABLE_PRIVATE_PREFIXES)
    ]
    retained_files = [
        item
        for item in private_info.siblings
        if not item.rfilename.startswith(PRUNABLE_PRIVATE_PREFIXES)
    ]
    source_manifest = load_private_manifest(args.cache_dir, private_info.sha)
    manifest_bytes, index_bytes = build_documents(
        private_revision=private_info.sha,
        public_revision=public_revision,
        retained_files=retained_files,
        deleted_files=deleted_files,
        source_manifest=source_manifest,
    )
    expected_verified_artifacts = {
        path: {"size_bytes": size, "sha256": digest}
        for path, (size, digest) in PUBLIC_ARTIFACTS.items()
    }
    migration = source_manifest.get("migration", {})
    managed_documents_need_update = (
        migration.get("public_revision") != public_revision
        or migration.get("verified_artifacts") != expected_verified_artifacts
    )

    summary = {
        "private_revision": private_info.sha,
        "public_revision": public_revision,
        "delete_file_count": len(deleted_files),
        "delete_size_bytes": sum(int(item.size or 0) for item in deleted_files),
        "expected_retained_file_count": len(retained_files),
        "managed_documents_need_update": managed_documents_need_update,
        "apply": args.apply,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not deleted_files and not managed_documents_need_update:
        print("No finalized or discardable private artifact remains to delete.")
        return 0
    if not args.apply:
        return 0

    operations = [
        CommitOperationDelete(path_in_repo=item.rfilename, is_folder=False)
        for item in deleted_files
    ]
    operations.extend(
        (
            CommitOperationAdd("README.md", render_readme(public_revision)),
            CommitOperationAdd("repository-manifest.json", manifest_bytes),
            CommitOperationAdd("metadata/datasetfire/dataset-index.json", index_bytes),
        )
    )
    commit_message = (
        "Remove finalized and obsolete private artifacts"
        if deleted_files
        else "Refresh private corpus public bundle index"
    )
    commit = api.create_commit(
        repo_id=PRIVATE_REPOSITORY,
        repo_type="dataset",
        operations=operations,
        parent_commit=private_info.sha,
        commit_message=commit_message,
        commit_description=(
            f"Verified public revision {public_revision}; removed "
            f"{len(deleted_files)} replaced or obsolete private files."
        ),
    )
    result = api.dataset_info(
        PRIVATE_REPOSITORY,
        revision=commit.oid,
        files_metadata=True,
    )
    leftovers = [
        item.rfilename
        for item in result.siblings
        if item.rfilename.startswith(PRUNABLE_PRIVATE_PREFIXES)
    ]
    if leftovers:
        raise RuntimeError(f"Finalized private files remain: {leftovers[:5]}")
    if len(result.siblings) != len(retained_files):
        raise RuntimeError(
            f"Unexpected retained file count: {len(result.siblings)} != {len(retained_files)}"
        )
    print(
        json.dumps(
            {
                "commit": commit.oid,
                "retained_file_count": len(result.siblings),
                "leftover_finalized_files": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
