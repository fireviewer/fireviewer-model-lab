#!/usr/bin/env python3
"""Stage and publish a FireViewer training dataset on Hugging Face Hub.

The staging tree uses hard links, so preparing an upload does not duplicate the
dataset payload on disk. The tool intentionally does not recompute hashes and
does not download the remote dataset after publication.
"""

from __future__ import annotations

# Generated dataset cards intentionally contain human-readable Markdown lines.
# ruff: noqa: E501
import argparse
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def _iter_relpaths(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.endswith("_relpath") and nested:
                if not isinstance(nested, str):
                    raise ValueError(f"Expected a string for {key}")
                yield nested
            else:
                yield from _iter_relpaths(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_relpaths(nested)


def collect_asset_relpaths(rows: Iterable[dict[str, Any]]) -> list[str]:
    paths: set[str] = set()
    for row in rows:
        paths.update(_iter_relpaths(row))
    return sorted(paths)


def _safe_source(data_root: Path, relpath: str) -> tuple[Path, Path]:
    posix = PurePosixPath(relpath.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe manifest path: {relpath}")
    relative = Path(*posix.parts)
    source = (data_root / relative).resolve(strict=True)
    root = data_root.resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError(f"Manifest path escapes data root: {relpath}")
    if not source.is_file():
        raise ValueError(f"Manifest asset is not a file: {relpath}")
    return source, relative


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        source = row.get("source_id") or row.get("source") or "unknown"
        counts[str(source)] += 1
    return dict(sorted(counts.items()))


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("split", "unspecified")) for row in rows).items()))


def _dataset_card(
    *,
    title: str,
    dataset_id: str,
    private: bool,
    rows: list[dict[str, Any]],
    notice: str,
) -> str:
    access = (
        "This repository is private because at least one upstream source does not provide "
        "a sufficiently explicit redistribution grant. Access does not grant permission to "
        "redistribute upstream assets."
        if private
        else "This repository is public. Users must retain the attribution and usage conditions recorded per sample."
    )
    split_lines = "\n".join(f"- `{name}`: {count}" for name, count in _split_counts(rows).items())
    source_lines = "\n".join(f"- `{name}`: {count}" for name, count in _source_counts(rows).items())
    return f"""---
pretty_name: {title}
task_categories:
- image-segmentation
- image-to-image
tags:
- fireviewer
- wildfire
- computer-vision
---

# {title}

Training-specific FireViewer dataset published for `{dataset_id}`.

## Access and rights

{access}

{notice.strip()}

Licensing, consent, citation, and redistribution fields in `manifest.jsonl` remain authoritative for each sample. No additional rights are granted by this dataset card.

## Contents

- Rows: {len(rows)}
- Manifest: `manifest.jsonl`
- Metadata: `dataset-info.json`

### Splits

{split_lines}

### Sources

{source_lines}

All paths stored in the manifest are relative to the repository root. This publication was uploaded from a validated local manifest without downloading a second verification copy from the Hub.
"""


def stage_dataset(
    *,
    data_root: Path,
    manifest: Path,
    staging: Path,
    dataset_id: str,
    title: str,
    private: bool,
    notice: str,
) -> dict[str, Any]:
    if staging.exists() and any(staging.iterdir()):
        raise FileExistsError(f"Staging directory is not empty: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest(manifest)
    relpaths = collect_asset_relpaths(rows)
    total_bytes = 0
    linked = 0
    for relpath in relpaths:
        source, relative = _safe_source(data_root, relpath)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        total_bytes += source.stat().st_size
        linked += 1

    shutil.copyfile(manifest, staging / "manifest.jsonl")
    info = {
        "dataset_id": dataset_id,
        "private": private,
        "rows": len(rows),
        "asset_files": linked,
        "asset_bytes": total_bytes,
        "split_counts": _split_counts(rows),
        "source_counts": _source_counts(rows),
        "manifest_paths_relative_to_repo_root": True,
        "staging_uses_hardlinks": True,
        "remote_redownload_verification": False,
    }
    (staging / "dataset-info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        _dataset_card(
            title=title,
            dataset_id=dataset_id,
            private=private,
            rows=rows,
            notice=notice,
        ),
        encoding="utf-8",
    )
    return info


def _read_token(path: Path) -> str:
    token = path.read_text(encoding="utf-8-sig").strip()
    if not token:
        raise ValueError(f"Token file is empty: {path}")
    return token


def publish_dataset(
    *, repo_id: str, staging: Path, token_file: Path, private: bool, workers: int
) -> dict[str, Any]:
    token = _read_token(token_file)
    api = HfApi(token=token)
    url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staging,
        private=private,
        num_workers=workers,
        print_report=True,
        print_report_every=30,
    )
    result = confirm_dataset(repo_id=repo_id, token=token)
    result["url"] = str(url)
    return result


def confirm_dataset(*, repo_id: str, token: str) -> dict[str, Any]:
    info = HfApi(token=token).dataset_info(repo_id, files_metadata=True)
    siblings = info.siblings or []
    files = [item.rfilename for item in siblings]
    required = {"README.md", "dataset-info.json", "manifest.jsonl"}
    missing = sorted(required.difference(files))
    if missing:
        raise RuntimeError(f"Remote dataset is missing required files: {missing}")
    return {
        "repo_id": info.id,
        "private": info.private,
        "sha": info.sha,
        "files": len(files),
        "bytes": sum((item.size or 0) for item in siblings),
        "required_files_present": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--data-root", type=Path, required=True)
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--staging", type=Path, required=True)
    stage.add_argument("--dataset-id", required=True)
    stage.add_argument("--title", required=True)
    stage.add_argument("--private", action="store_true")
    stage.add_argument("--notice", default="")

    publish = subparsers.add_parser("publish")
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--staging", type=Path, required=True)
    publish.add_argument("--token-file", type=Path, required=True)
    publish.add_argument("--private", action="store_true")
    publish.add_argument("--workers", type=int, default=8)

    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--repo-id", required=True)
    confirm.add_argument("--token-file", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "stage":
        result = stage_dataset(
            data_root=args.data_root,
            manifest=args.manifest,
            staging=args.staging,
            dataset_id=args.dataset_id,
            title=args.title,
            private=args.private,
            notice=args.notice,
        )
    elif args.command == "publish":
        result = publish_dataset(
            repo_id=args.repo_id,
            staging=args.staging,
            token_file=args.token_file,
            private=args.private,
            workers=args.workers,
        )
    else:
        result = confirm_dataset(
            repo_id=args.repo_id,
            token=_read_token(args.token_file),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
