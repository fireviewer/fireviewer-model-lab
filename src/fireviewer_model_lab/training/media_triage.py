"""Fail-closed preflight for the FireWarning fire/smoke/normal triage corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MANIFESTS = (
    Path("corpus/fasdd/triage-manifest.jsonl"),
    Path("corpus/pyro-sdis-v0.1.0/triage-manifest.jsonl"),
)
EXPECTED_ROWS = 155_044
EXPECTED_SPLITS = {"test": 23_906, "train": 107_449, "validation": 23_689}
EXPECTED_CLASSES = {
    "fire": 12_755,
    "fire_and_smoke": 27_955,
    "normal": 57_294,
    "smoke": 57_040,
}
EXPECTED_LABELS = {
    "fire": ["fire"],
    "fire_and_smoke": ["fire", "smoke"],
    "normal": ["normal"],
    "smoke": ["smoke"],
}
DENIED_TOKENS = (
    "operational-reference-a",
    "operational-reference-b",
    "operational_incident",
    "critical_lot",
)


class MediaTriageError(RuntimeError):
    """Raised when a triage corpus invariant is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MediaTriageError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise MediaTriageError(f"non-object row at {path}:{line_number}")
            yield row


def preflight(dataset_root: Path, *, verify_hashes: bool = False) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    rows = 0
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = defaultdict(set)
    seen_samples: set[tuple[str, str]] = set()
    verified_media: dict[Path, str] = {}

    for relative_manifest in MANIFESTS:
        manifest = dataset_root / relative_manifest
        if not manifest.is_file():
            raise MediaTriageError(f"missing triage manifest: {manifest}")
        for line_number, row in enumerate(_iter_jsonl(manifest), start=1):
            serialized = json.dumps(row, ensure_ascii=False).lower()
            if any(token in serialized for token in DENIED_TOKENS):
                raise MediaTriageError(f"operational or critical row at {manifest}:{line_number}")
            source_id = str(row.get("source_id", ""))
            sample_id = str(row.get("sample_id", ""))
            key = (source_id, sample_id)
            if not source_id or not sample_id or key in seen_samples:
                raise MediaTriageError(f"missing or duplicate sample at {manifest}:{line_number}")
            seen_samples.add(key)
            primary_class = str(row.get("primary_class", ""))
            labels = row.get("labels")
            if labels != EXPECTED_LABELS.get(primary_class):
                raise MediaTriageError(f"invalid triage label at {manifest}:{line_number}")
            split = str(row.get("split", ""))
            split_group = str(row.get("split_group", ""))
            if split not in EXPECTED_SPLITS or not split_group:
                raise MediaTriageError(f"invalid split contract at {manifest}:{line_number}")
            image = (manifest.parent / str(row.get("image_relpath", ""))).resolve()
            if manifest.parent.resolve() not in image.parents or not image.is_file():
                raise MediaTriageError(f"missing or escaping media at {manifest}:{line_number}")
            expected_sha256 = str(row.get("sha256", ""))
            if len(expected_sha256) != 64:
                raise MediaTriageError(f"invalid media hash at {manifest}:{line_number}")
            if verify_hashes:
                observed = verified_media.get(image)
                if observed is None:
                    observed = _sha256_file(image)
                    verified_media[image] = observed
                if observed != expected_sha256:
                    raise MediaTriageError(f"media hash mismatch at {manifest}:{line_number}")
            split_counts[split] += 1
            class_counts[primary_class] += 1
            split_groups[f"{source_id}:{split_group}"].add(split)
            rows += 1

    leaking_groups = [group for group, splits in split_groups.items() if len(splits) > 1]
    if leaking_groups:
        raise MediaTriageError(f"triage split-group leakage: {len(leaking_groups)} groups")
    if rows != EXPECTED_ROWS:
        raise MediaTriageError(f"triage row mismatch: {rows} != {EXPECTED_ROWS}")
    if dict(sorted(split_counts.items())) != EXPECTED_SPLITS:
        raise MediaTriageError("triage split counts differ from the pinned contract")
    if dict(sorted(class_counts.items())) != EXPECTED_CLASSES:
        raise MediaTriageError("triage class counts differ from the pinned contract")
    return {
        "dataset_ready": True,
        "training_ready": False,
        "blocking_reasons": [
            "triage_classifier_trainer_not_implemented",
            "independent_double_validated_triage_test_missing",
        ],
        "rows": rows,
        "split_counts": dict(sorted(split_counts.items())),
        "primary_class_counts": dict(sorted(class_counts.items())),
        "split_groups": len(split_groups),
        "split_group_leakage": 0,
        "verified_media": len(verified_media),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("preflight")
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = preflight(args.dataset_root, verify_hashes=args.verify_hashes)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
