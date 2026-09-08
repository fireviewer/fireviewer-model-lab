#!/usr/bin/env python3
"""Admit only the visually confirmed, ground-view subset of local DFire gaps."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ACCEPT = {
    101, 107, 108, 113, 114, 115, 118, 119, 120, 122, 124, 125, 128, 129,
    136, 143, 145, 146, 148, 151, 154, 155, 156, 157, 158, 159, 163, 164,
    165, 166, 170, 171, 173, 174, 176, 177, 178, 179, 180, 181, 184, 186,
    187, 188, 189, 193, 194, 196, 197, 200,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.artifact / "candidate_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {int(row["candidate_id"].rsplit("-", 1)[-1]) for row in rows}
    if not ACCEPT.issubset(known):
        raise RuntimeError("DFire pool changed; fail closed")
    decisions = []
    for row in rows:
        number = int(row["candidate_id"].rsplit("-", 1)[-1])
        accepted = number in ACCEPT
        decisions.append({
            "candidate_id": row["candidate_id"],
            "decision": "accept" if accepted else "reject",
            "reason": (
                "manual_full_frame_review_confirmed_ground_view_no_foreground_person_and_usable_boxes"
                if accepted else
                ("reviewed_but_aerial_person_risk_news_ui_or_annotation_not_exploitable" if number <= 200 else "not_reviewed_not_admitted")
            ),
            "training_admitted": accepted,
        })
    output = args.artifact / "review_decisions.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
    counts = Counter(row["decision"] for row in decisions)
    report = {
        "schema": "fireviewer.pointing-v7-dfire-ground-subset-review.v1",
        "candidate_count": len(rows),
        "accepted": counts["accept"],
        "rejected": counts["reject"],
        "reviewed_range": [1, 200],
        "unreviewed_are_rejected": True,
        "status": "bounded_ground_subset_review_complete",
    }
    (args.artifact / "review_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
