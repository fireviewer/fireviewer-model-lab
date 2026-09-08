from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ALLOWED_MEDIA_HOST = "upload.wikimedia.org"
MAX_MEDIA_BYTES = 16 * 1024 * 1024
MAX_RATE_LIMIT_RETRIES = 3


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize an allowlisted Wikimedia Commons selection with digests."
    )
    parser.add_argument("selection", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("receipt", type=Path)
    return parser.parse_args()


def _safe_target(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or posix.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("media target must be a safe relative JPEG path")
    resolved_root = root.resolve()
    target = (resolved_root / Path(*posix.parts)).resolve()
    if target == resolved_root or resolved_root not in target.parents:
        raise ValueError("media target leaves the configured output root")
    return target


def _download(url: str, target: Path) -> tuple[int, str, str]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_MEDIA_HOST
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("media URL is outside the allowed Wikimedia upload host")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.part")
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    try:
        request = Request(  # noqa: S310 - scheme and host validated above
            url,
            headers={
                "Accept": "image/jpeg",
                "User-Agent": "Mozilla/5.0 FireViewerBenchmark/0.1",
                "Referer": "https://commons.wikimedia.org/",
            },
        )
        with urlopen(request, timeout=90) as response:  # noqa: S310 - HTTPS host allowlisted
            final = urlsplit(response.url)
            if final.scheme != "https" or final.hostname != ALLOWED_MEDIA_HOST:
                raise ValueError("media download redirected outside the allowed host")
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_MEDIA_BYTES:
                        raise ValueError("media payload exceeds the 16 MiB safety cap")
                    sha1.update(chunk)
                    sha256.update(chunk)
                    output.write(chunk)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return size, sha1.hexdigest(), sha256.hexdigest()


def _hash_existing(target: Path) -> tuple[int, str, str]:
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with target.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_MEDIA_BYTES:
                raise ValueError("existing media exceeds the 16 MiB safety cap")
            sha1.update(chunk)
            sha256.update(chunk)
    return size, sha1.hexdigest(), sha256.hexdigest()


def _materialize(url: str, target: Path) -> tuple[int, str, str]:
    if target.is_file():
        return _hash_existing(target)
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return _download(url, target)
        except HTTPError as exc:
            if exc.code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            retry_after = exc.headers.get("Retry-After", "5")
            delay_seconds = min(int(retry_after) if retry_after.isdigit() else 5, 20)
            time.sleep(delay_seconds)
    raise RuntimeError("unreachable Commons materialization retry state")


def main() -> int:
    args = _arguments()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema") != "fireviewer.commons-media-selection.v1":
        raise ValueError("unexpected Commons media selection schema")
    items = selection.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Commons media selection is empty")

    materialized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Commons media selection contains a non-object item")
        target = _safe_target(args.output_root, str(item["target"]))
        size, sha1, sha256 = _materialize(str(item["direct_url"]), target)
        if size != item["expected_size_bytes"] or sha1 != item["expected_sha1"]:
            target.unlink(missing_ok=True)
            raise ValueError(f"Commons identity mismatch for {item['media_id']}")
        materialized.append(
            {
                "case_id": item["case_id"],
                "media_id": item["media_id"],
                "target": item["target"],
                "size_bytes": size,
                "commons_sha1": sha1,
                "sha256": sha256,
                "license": item["license"],
                "license_url": item["license_url"],
                "author": item["author"],
                "source_page_url": item["source_page_url"],
                "direct_url": item["direct_url"],
            }
        )

    receipt = {
        "schema": "fireviewer.commons-media-materialization.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "materialized_count": len(materialized),
        "output_root": args.output_root.as_posix(),
        "items": materialized,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
