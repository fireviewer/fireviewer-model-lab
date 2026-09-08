from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import data_root, ensure_layout, load_config, write_json


SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PHASH_RE = re.compile(r"^[0-9a-fA-F]{16}$")


def walk_values(value: Any, key: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from walk_values(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child, key)
    else:
        yield key, value


def parse_file(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item


def candidate_files(config: dict[str, Any]) -> list[Path]:
    settings = config["independence"]
    active_root = Path(settings["active_manifest_root"])
    historical_root = Path(settings["historical_manifest_root"])
    name_re = re.compile(settings["historical_manifest_name_regex"])
    files: set[Path] = set()
    if active_root.exists():
        files.update(path for path in active_root.rglob("*") if path.suffix.lower() in {".json", ".jsonl"})
    if historical_root.exists():
        files.update(
            path
            for path in historical_root.rglob("*")
            if path.is_file() and name_re.search(path.name)
        )
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = data_root(config)
    ensure_layout(root)

    sha_values: set[str] = set()
    phash_values: set[str] = set()
    source_values: Counter[str] = Counter()
    files = candidate_files(config)
    parsed_records = 0
    for index, path in enumerate(files, 1):
        for record in parse_file(path):
            parsed_records += 1
            for key, value in walk_values(record):
                if not isinstance(value, str):
                    continue
                normalized = value.strip()
                lower_key = key.lower()
                if "phash" in lower_key and PHASH_RE.fullmatch(normalized):
                    phash_values.add(normalized.lower())
                elif "sha256" in lower_key and SHA_RE.fullmatch(normalized):
                    sha_values.add(normalized.lower())
                if lower_key in {"source", "source_id", "source_dataset", "source_family", "family"}:
                    source_values[normalized] += 1
        if index % 20 == 0 or index == len(files):
            print(f"indexed {index}/{len(files)} manifest files", flush=True)

    audit = root / "audit"
    (audit / "used_sha256.txt").write_text("\n".join(sorted(sha_values)) + "\n", encoding="ascii")
    (audit / "used_phash64.txt").write_text("\n".join(sorted(phash_values)) + "\n", encoding="ascii")
    write_json(
        audit / "used_sources.json",
        {
            "blocked_source_families": config["independence"]["blocked_source_families"],
            "observed_source_values": dict(source_values.most_common()),
        },
    )
    summary = {
        "manifest_files": len(files),
        "parsed_records": parsed_records,
        "unique_sha256_values": len(sha_values),
        "unique_phash64_values": len(phash_values),
        "unique_source_values": len(source_values),
    }
    write_json(audit / "exclusion_index_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
