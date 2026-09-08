from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import imagehash
from PIL import Image


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    return config


def data_root(config: dict[str, Any]) -> Path:
    return Path(config["data_root"]).resolve()


def ensure_layout(root: Path) -> None:
    for relative in (
        "audit",
        "candidates",
        "manifests",
        "raw/photos",
        "raw/posters",
        "raw/videos",
        "receipts/flickr",
        "review/contact_sheets",
        "review/decisions",
        "sequences",
        "release/annotations",
        "release/images",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash_file(path: Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("RGB"), hash_size=8))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows

