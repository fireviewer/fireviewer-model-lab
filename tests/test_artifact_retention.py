from __future__ import annotations

from pathlib import Path

from fireviewer_model_lab.artifact_retention import (
    ArtifactClass,
    audit_artifacts,
    audit_manifest,
    scratch_budget_bytes,
)


def _write(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_audit_preserves_runtime_models_and_classifies_unused_artifacts(tmp_path: Path) -> None:
    active = tmp_path / "data" / "models" / "active-model.pt"
    legacy = tmp_path / "data" / "models" / "old-model.ckpt"
    unused_dataset = tmp_path / "data" / "datasets" / "unused.parquet"
    cached = tmp_path / "data" / ".cache" / "download.bin"
    _write(active)
    _write(legacy)
    _write(unused_dataset)
    _write(cached)
    source = tmp_path / "src" / "package" / "provider.py"
    source.parent.mkdir(parents=True)
    source.write_text('MODEL = "data/models/active-model.pt"\n', encoding="utf-8")
    records = audit_artifacts(tmp_path, roots=[tmp_path / "data"], minimum_bytes=1)
    by_path = {item.path: item for item in records}
    assert by_path["data/models/active-model.pt"].classification is ArtifactClass.ACTIVE_LOCAL
    assert by_path["data/models/old-model.ckpt"].classification is ArtifactClass.LEGACY_MODEL
    assert by_path["data/datasets/unused.parquet"].classification is ArtifactClass.UNUSED_DATASET
    assert by_path["data/.cache/download.bin"].classification is ArtifactClass.REBUILDABLE_CACHE


def test_remote_catalog_prevents_rearchiving_and_manifest_has_bounded_scratch(
    tmp_path: Path,
) -> None:
    remote_model = tmp_path / "data" / "models" / "remote.pth"
    _write(remote_model)
    catalog = {
        "data/models/remote.pth": {
            "repo_id": "fireviewer/fireviewer-legacy-models",
            "revision": "1" * 40,
            "path": "models/remote/v1/artifacts/remote.pth",
            "byte_count": remote_model.stat().st_size,
        }
    }
    records = audit_artifacts(
        tmp_path,
        roots=[tmp_path / "data"],
        minimum_bytes=1,
        remote_catalog=catalog,
    )
    assert records[0].classification is ArtifactClass.REMOTE_AVAILABLE
    manifest = audit_manifest(tmp_path, records)
    assert manifest["scratch_budget_bytes"] == scratch_budget_bytes(tmp_path)
    assert manifest["scratch_budget_bytes"] <= 20 * 1024**3
