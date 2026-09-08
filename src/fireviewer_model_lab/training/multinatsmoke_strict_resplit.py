"""Filter and deterministically resplit an audited MultiNatSmoke source family."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _components(rows: list[dict[str, Any]], maximum_distance: int) -> list[list[int]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        lroot, rroot = find(left), find(right)
        if lroot != rroot:
            parent[max(lroot, rroot)] = min(lroot, rroot)

    exact: dict[str, int] = {}
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        digest = str(row["image_sha256"])
        if digest in exact:
            union(index, exact[digest])
        else:
            exact[digest] = index
        signature = str(row["image_phash64"])
        # Four high bits keep the candidate set bounded; neighboring buckets are
        # included because a <=6-bit match can cross one high-nibble boundary.
        prefix = int(signature[0], 16)
        candidates: set[int] = set()
        for value in range(16):
            if (prefix ^ value).bit_count() <= maximum_distance:
                candidates.update(buckets[f"{value:x}"])
        for previous in candidates:
            if _hamming(signature, str(rows[previous]["image_phash64"])) <= maximum_distance:
                union(index, previous)
        buckets[signature[0]].append(index)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)
    return [grouped[key] for key in sorted(grouped)]


def _assign_split(cluster_id: str) -> str:
    value = int(hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.65:
        return "train"
    if value < 0.80:
        return "validation"
    return "test"


def strict_resplit(
    *, audit_path: Path, rights_receipt: Path, output_dir: Path, maximum_phash_distance: int = 6
) -> dict[str, Any]:
    rights = json.loads(rights_receipt.read_text(encoding="utf-8"))
    if rights.get("source_family") != "D-Fire" or rights.get("commercial_use_allowed") is not True:
        raise ValueError("primary D-Fire rights receipt does not authorize commercial use")
    if not str(rights.get("immutable_source_revision") or ""):
        raise ValueError("rights receipt lacks an immutable source revision")
    audited = _read_jsonl(audit_path)
    candidates = [
        row for row in audited
        if not row.get("errors") and row.get("base_candidate", {}).get("valid") is True
    ]
    clusters = _components(candidates, maximum_phash_distance)
    output: list[dict[str, Any]] = []
    for members in clusters:
        cluster_rows = [candidates[index] for index in members]
        representative = min(cluster_rows, key=lambda row: str(row["image_sha256"]))
        cluster_id = hashlib.sha256(
            "\n".join(sorted(str(row["image_sha256"]) for row in cluster_rows)).encode("utf-8")
        ).hexdigest()
        output.append(
            {
                "sample_id": representative["sample_id"],
                "source_family": "D-Fire",
                "source_record_id": representative["source_record_id"],
                "image_sha256": representative["image_sha256"],
                "mask_sha256": representative["mask_sha256"],
                "phash64": representative["image_phash64"],
                "split_group": f"perceptual:{cluster_id}",
                "split": _assign_split(cluster_id),
                "point_type": "smoke_column_base",
                "point_x": representative["base_candidate"]["point_x"],
                "point_y": representative["base_candidate"]["point_y"],
                "cluster_members": len(cluster_rows),
                "rights_receipt": rights,
                "training_eligible": False,
                "admission_status": "pending_cross_corpus_and_semantic_stability_gates",
            }
        )
    output.sort(key=lambda row: str(row["sample_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "dfire_strict_resplit.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
        newline="\n",
    )
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in output:
        split_groups[str(row["split_group"])].add(str(row["split"]))
    report = {
        "schema_version": 1,
        "audited_rows": len(audited),
        "payload_valid_rows": len(candidates),
        "strict_representative_rows": len(output),
        "excluded_payload_rows": len(audited) - len(candidates),
        "deduplicated_rows": len(candidates) - len(output),
        "split_counts": dict(sorted(Counter(row["split"] for row in output).items())),
        "cross_split_group_leaks": sum(len(splits) > 1 for splits in split_groups.values()),
        "maximum_phash_distance": maximum_phash_distance,
        "training_eligible_rows": 0,
        "next_gate": "cross_corpus_benchmark_exclusion_and_mask_base_semantic_stability",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (output_dir / "dfire_strict_resplit_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rights-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-phash-distance", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(strict_resplit(audit_path=args.audit, rights_receipt=args.rights_receipt, output_dir=args.output_dir, maximum_phash_distance=args.maximum_phash_distance), indent=2))


if __name__ == "__main__":
    main()
