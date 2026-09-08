"""Conservative inventory and Hugging Face archival policy for heavy artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

MODEL_SUFFIXES = frozenset({".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"})
DATASET_SUFFIXES = frozenset({".arrow", ".csv", ".h5", ".hdf5", ".jsonl", ".parquet"})
TEXT_SUFFIXES = frozenset(
    {".cfg", ".json", ".md", ".ps1", ".py", ".sh", ".toml", ".yaml", ".yml"}
)
MAX_REFERENCE_FILE_BYTES = 2 * 1024**2
MAX_SCRATCH_BYTES = 20 * 1024**3


class ArtifactClass(StrEnum):
    ACTIVE_LOCAL = "ACTIVE_LOCAL"
    REMOTE_AVAILABLE = "REMOTE_AVAILABLE"
    LEGACY_MODEL = "LEGACY_MODEL"
    REBUILDABLE_CACHE = "REBUILDABLE_CACHE"
    UNUSED_DATASET = "UNUSED_DATASET"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    byte_count: int
    classification: ArtifactClass
    consumers: tuple[str, ...]
    remote: dict[str, Any] | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "byte_count": self.byte_count,
            "classification": self.classification.value,
            "consumers": list(self.consumers),
            "remote": self.remote,
            "reason": self.reason,
        }


def scratch_budget_bytes(path: Path) -> int:
    return max(1, min(MAX_SCRATCH_BYTES, shutil.disk_usage(path).free // 10))


def _candidate_reference_files(repo_root: Path) -> tuple[Path, ...]:
    roots = ("src", "training", "benchmark", "scripts")
    paths: list[Path] = [repo_root / "pyproject.toml"]
    paths.extend(repo_root.glob("Dockerfile*"))
    for name in roots:
        root = repo_root / name
        if not root.is_dir():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in TEXT_SUFFIXES
            and path.stat().st_size <= MAX_REFERENCE_FILE_BYTES
        )
    return tuple(sorted(set(paths)))


def _reference_index(repo_root: Path) -> dict[Path, str]:
    result: dict[Path, str] = {}
    for path in _candidate_reference_files(repo_root):
        try:
            result[path] = path.read_text(encoding="utf-8", errors="ignore").casefold()
        except OSError:
            continue
    return result


def _consumers(
    artifact: Path,
    *,
    repo_root: Path,
    reference_index: dict[Path, str],
) -> tuple[str, ...]:
    relative = artifact.relative_to(repo_root).as_posix().casefold()
    needles = {relative}
    generic_names = {
        "checkpoint.ckpt",
        "config.json",
        "model.bin",
        "model.onnx",
        "model.pt",
        "model.pth",
        "model.safetensors",
    }
    if artifact.name.casefold() not in generic_names:
        needles.add(artifact.name.casefold())
    parents = artifact.relative_to(repo_root).parts
    if len(parents) >= 2:
        needles.add("/".join(parents[-2:]).casefold())
    matches = [
        path.relative_to(repo_root).as_posix()
        for path, content in reference_index.items()
        if path != artifact and any(needle in content for needle in needles if len(needle) >= 6)
    ]
    return tuple(sorted(set(matches)))


def _is_runtime_consumer(value: str) -> bool:
    normalized = value.casefold()
    return normalized.startswith("src/") or normalized.startswith("dockerfile")


def _classify(
    artifact: Path,
    *,
    consumers: tuple[str, ...],
    remote: dict[str, Any] | None,
) -> tuple[ArtifactClass, str]:
    parts = {item.casefold() for item in artifact.parts}
    if parts.intersection({".cache", ".hf", ".pytest_cache", "__pycache__", "cache", "caches"}):
        return ArtifactClass.REBUILDABLE_CACHE, "artifact is stored in a rebuildable cache"
    runtime_consumers = tuple(item for item in consumers if _is_runtime_consumer(item))
    if runtime_consumers:
        return ArtifactClass.ACTIVE_LOCAL, "referenced by current runtime or container source"
    suffix = artifact.suffix.casefold()
    if remote is not None:
        return ArtifactClass.REMOTE_AVAILABLE, "an immutable remote artifact is recorded"
    if suffix in MODEL_SUFFIXES:
        return ArtifactClass.LEGACY_MODEL, "model artifact has no current runtime consumer"
    if suffix in DATASET_SUFFIXES:
        return ArtifactClass.UNUSED_DATASET, "dataset shard has no current runtime consumer"
    return ArtifactClass.UNKNOWN, "artifact requires manual identification before removal"


def load_remote_catalog(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("artifacts", []) if isinstance(payload, dict) else []
    return {
        str(item["local_path"]).replace("\\", "/").casefold(): dict(item)
        for item in records
        if isinstance(item, dict) and isinstance(item.get("local_path"), str)
    }


def audit_artifacts(
    repo_root: Path,
    *,
    roots: Sequence[Path],
    minimum_bytes: int = 50 * 1024**2,
    remote_catalog: dict[str, dict[str, Any]] | None = None,
) -> tuple[ArtifactRecord, ...]:
    repo_root = repo_root.resolve()
    references = _reference_index(repo_root)
    remote_catalog = remote_catalog or {}
    records: list[ArtifactRecord] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if not resolved.is_relative_to(repo_root) or not resolved.exists():
            continue
        candidates: Iterable[Path] = (resolved,) if resolved.is_file() else resolved.rglob("*")
        for artifact in candidates:
            if not artifact.is_file() or artifact in seen:
                continue
            seen.add(artifact)
            try:
                size = artifact.stat().st_size
            except OSError:
                continue
            if size < minimum_bytes:
                continue
            relative = artifact.relative_to(repo_root).as_posix()
            remote = remote_catalog.get(relative.casefold())
            consumers = _consumers(artifact, repo_root=repo_root, reference_index=references)
            classification, reason = _classify(
                artifact,
                consumers=consumers,
                remote=remote,
            )
            records.append(
                ArtifactRecord(
                    path=relative,
                    byte_count=size,
                    classification=classification,
                    consumers=consumers,
                    remote=remote,
                    reason=reason,
                )
            )
    return tuple(sorted(records, key=lambda item: (-item.byte_count, item.path)))


def audit_manifest(
    repo_root: Path,
    records: Sequence[ArtifactRecord],
) -> dict[str, Any]:
    totals: dict[str, int] = {value.value: 0 for value in ArtifactClass}
    for record in records:
        totals[record.classification.value] += record.byte_count
    return {
        "schema": "fireviewer.local-artifact-audit.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root.resolve()),
        "scratch_budget_bytes": scratch_budget_bytes(repo_root),
        "totals_by_class_bytes": totals,
        "artifacts": [item.to_dict() for item in records],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, action="append")
    parser.add_argument("--remote-catalog", type=Path)
    parser.add_argument("--minimum-mib", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve()
    roots = args.root or [repo_root / "data"]
    records = audit_artifacts(
        repo_root,
        roots=[path if path.is_absolute() else repo_root / path for path in roots],
        minimum_bytes=args.minimum_mib * 1024**2,
        remote_catalog=load_remote_catalog(args.remote_catalog),
    )
    payload = audit_manifest(repo_root, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact_count": len(records), "output": str(args.output)}))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "ArtifactClass",
    "ArtifactRecord",
    "audit_artifacts",
    "audit_manifest",
    "load_remote_catalog",
    "scratch_budget_bytes",
]
