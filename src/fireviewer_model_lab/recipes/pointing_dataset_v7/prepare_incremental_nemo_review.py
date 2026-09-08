#!/usr/bin/env python3
"""Carry prior NEMO decisions by SHA and isolate only newly sampled views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    args = parser.parse_args()
    prior_rows = read(args.prior / "candidate_manifest.jsonl")
    prior_decisions = {row["candidate_id"]: row for row in read(args.prior / "review_decisions.jsonl")}
    by_sha = {row["sha256"]: prior_decisions[row["candidate_id"]] for row in prior_rows}
    expanded = read(args.expanded / "candidate_manifest.jsonl")
    carried, new = [], []
    for row in expanded:
        decision = by_sha.get(row["sha256"])
        if decision:
            carried.append({**decision, "candidate_id": row["candidate_id"], "carried_from_sha256": row["sha256"]})
        else:
            new.append(row)
    write(args.expanded / "carried_review_decisions.jsonl", carried)
    write(args.expanded / "new_candidate_manifest.jsonl", new)
    # A dedicated review artifact lets the existing renderer operate unchanged.
    review = args.expanded / "incremental_review"
    review.mkdir(exist_ok=True)
    write(review / "candidate_manifest.jsonl", new)
    print(json.dumps({"expanded": len(expanded), "carried": len(carried), "new_visual_review_required": len(new)}, indent=2))


if __name__ == "__main__":
    main()
