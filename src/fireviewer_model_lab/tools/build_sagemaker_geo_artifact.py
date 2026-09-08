from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tarfile
from hashlib import sha256
from pathlib import Path

MEGALOC_REVISION = "37bb43d65dd6388d1578052de5eb0bcdceb497e7"
PRITHVI_REVISION = "a3f2c410e45b8ac7417976614528a872f024d831"
PRITHVI_REPOSITORY = "models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M-BurnScars"


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_deterministic_tar_gz(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw_stream,
        gzip.GzipFile(fileobj=raw_stream, mode="wb", compresslevel=1, mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)


def build(*, artifact_root: Path, output_root: Path) -> dict[str, object]:
    megaloc_source = artifact_root / "models" / "megaloc" / MEGALOC_REVISION
    prithvi_source = (
        artifact_root
        / "models"
        / "prithvi-cache"
        / PRITHVI_REPOSITORY
        / "snapshots"
        / PRITHVI_REVISION
    )
    sources = {
        "megaloc/config.json": megaloc_source / "config.json",
        "megaloc/megaloc_model.py": megaloc_source / "megaloc_model.py",
        "megaloc/model.safetensors": megaloc_source / "model.safetensors",
        "megaloc/bundle-manifest.json": megaloc_source / "bundle-manifest.json",
        (
            f"prithvi-cache/{PRITHVI_REPOSITORY}/snapshots/{PRITHVI_REVISION}/"
            "Prithvi_EO_V2_300M_BurnScars.pt"
        ): prithvi_source / "Prithvi_EO_V2_300M_BurnScars.pt",
        (
            f"prithvi-cache/{PRITHVI_REPOSITORY}/snapshots/{PRITHVI_REVISION}/"
            "burn_scars_config.yaml"
        ): prithvi_source / "burn_scars_config.yaml",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model artifact inputs: {missing}")

    staging = output_root / "model-root.staging"
    model_root = output_root / "model-root"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    entries: list[dict[str, object]] = []
    for relative, source in sources.items():
        target = staging / relative
        _link_or_copy(source, target)
        entries.append(
            {
                "path": relative,
                "byte_size": target.stat().st_size,
                "sha256": _digest(target),
            }
        )
    manifest = {
        "schema": "fireviewer.sagemaker-geo-model-artifact.v1",
        "megaloc_revision": MEGALOC_REVISION,
        "prithvi_revision": PRITHVI_REVISION,
        "files": entries,
    }
    manifest_path = staging / "fireviewer-model-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if model_root.exists():
        shutil.rmtree(model_root)
    staging.replace(model_root)

    archive_path = output_root / "fireviewer-geo-model.tar.gz"
    partial_archive = output_root / "fireviewer-geo-model.tar.gz.part"
    partial_archive.unlink(missing_ok=True)
    _write_deterministic_tar_gz(model_root, partial_archive)
    partial_archive.replace(archive_path)
    receipt = {
        "schema": "fireviewer.sagemaker-geo-artifact-receipt.v1",
        "archive": str(archive_path),
        "archive_byte_size": archive_path.stat().st_size,
        "archive_sha256": _digest(archive_path),
        "manifest_sha256": _digest(model_root / manifest_path.name),
        "megaloc_revision": MEGALOC_REVISION,
        "prithvi_revision": PRITHVI_REVISION,
    }
    receipt_path = output_root / "fireviewer-geo-model.receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                artifact_root=args.artifact_root.resolve(strict=True),
                output_root=args.output_root.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
