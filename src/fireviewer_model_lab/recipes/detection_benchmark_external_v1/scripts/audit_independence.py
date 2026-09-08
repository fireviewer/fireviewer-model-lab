from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import data_root, ensure_layout, load_config, read_jsonl, write_json, write_jsonl


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def load_set(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing exclusion index: {path}")
    return {line.strip().lower() for line in path.read_text(encoding="ascii").splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = data_root(config)
    ensure_layout(root)
    candidates_path = root / "candidates" / "still_candidates.jsonl"
    candidates = read_jsonl(candidates_path)
    used_sha = load_set(root / "audit" / "used_sha256.txt")
    used_phash = load_set(root / "audit" / "used_phash64.txt")
    threshold = int(config["independence"]["phash_hamming_threshold"])

    buckets: dict[int, list[str]] = defaultdict(list)
    prefix_bits = 12
    suffix_bits = 64 - prefix_bits
    for value in used_phash:
        buckets[int(value, 16) >> suffix_bits].append(value)

    audited: list[dict[str, Any]] = []
    exact_collisions = 0
    perceptual_collisions = 0
    internal_exact: dict[str, list[str]] = defaultdict(list)
    internal_phash: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        sha = row["sha256"].lower()
        phash = row["phash"].lower()
        internal_exact[sha].append(row["sample_id"])
        internal_phash[phash].append(row["sample_id"])
        exact = sha in used_sha
        nearest: dict[str, Any] | None = None
        candidate_prefix = int(phash, 16) >> suffix_bits
        # Hamming distance <= 6 cannot guarantee an identical 12-bit prefix.
        # Search every indexed hash; benchmark v1 is small and this remains
        # deterministic. The buckets are retained for future larger indexes.
        for existing in used_phash:
            distance = hamming(phash, existing)
            if nearest is None or distance < nearest["distance"]:
                nearest = {"phash": existing, "distance": distance}
                if distance == 0:
                    break
        perceptual = nearest is not None and nearest["distance"] <= threshold
        exact_collisions += int(exact)
        perceptual_collisions += int(perceptual)
        audited.append(
            {
                **row,
                "independence_audit": {
                    "exact_sha256_collision": exact,
                    "nearest_used_phash": nearest,
                    "phash_hamming_threshold": threshold,
                    "perceptual_collision": perceptual,
                    "passed": not exact and not perceptual,
                },
            }
        )

    duplicate_sha = {key: value for key, value in internal_exact.items() if len(value) > 1}
    duplicate_phash = {key: value for key, value in internal_phash.items() if len(value) > 1}
    write_jsonl(root / "audit" / "still_candidates.audited.jsonl", audited)
    summary = {
        "candidate_count": len(audited),
        "exact_training_collisions": exact_collisions,
        "perceptual_training_collisions": perceptual_collisions,
        "internal_exact_duplicate_groups": duplicate_sha,
        "internal_identical_phash_groups": duplicate_phash,
        "passed_candidates": sum(1 for row in audited if row["independence_audit"]["passed"]),
        "status": "pass" if exact_collisions == 0 and perceptual_collisions == 0 else "fail_closed",
    }
    write_json(root / "audit" / "independence_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
