#!/usr/bin/env python3
"""Freeze the bounded review of local hard-negative candidates 1..300."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Confirmed after review of sheets 0001..0012.  They contain no visible fire
# or smoke and remain useful in a pointing detector: roads, vegetation,
# buildings, adverse skies, sunsets and other realistic false-positive cues.
# Aerial views, tight product shots, selfies, risky scenes and contaminated
# examples are absent from this allowlist.
ACCEPT = {
    4, 5, 8, 9, 18, 20, 27, 29, 31, 33, 39, 49, 59, 63, 75, 76,
    79, 81, 84, 92,
    101, 106, 108, 115, 123, 134, 144, 148, 151, 158, 160, 165,
    168, 182, 184, 187, 190, 194, 196, 204,
    207, 222, 227, 228, 232, 235, 241, 243, 249, 253, 257, 259,
    261, 266, 269, 277,
    *range(285, 301),
}
REVIEWED = set(range(1, 301))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.artifact / "candidate_manifest.jsonl")
    numeric = {
        int(row["candidate_id"].rsplit("-", 1)[-1]): row for row in rows
    }
    if len(rows) != 822 or set(numeric) != set(range(1, 823)):
        raise RuntimeError("local candidate pool changed; review is invalid")
    if not ACCEPT.issubset(REVIEWED):
        raise RuntimeError("allowlist contains an unreviewed candidate")
    if any(numeric[number]["gap_bucket"] != "hard_negative" for number in ACCEPT):
        raise RuntimeError("allowlist contains a positive candidate")
    if any(numeric[number].get("annotations") for number in ACCEPT):
        raise RuntimeError("allowlisted negative unexpectedly has annotations")

    decisions = []
    for number, row in sorted(numeric.items()):
        if number in ACCEPT:
            decision = "accept"
            reason = "manual_review_confirmed_contextual_no_fire_no_smoke"
            scope = "review_sheets_0001_0012"
        elif number in REVIEWED:
            decision = "reject"
            reason = "out_of_domain_tight_aerial_person_risk_or_not_useful"
            scope = "review_sheets_0001_0012"
        else:
            decision = "not_reviewed"
            reason = "fail_closed_outside_bounded_review_scope"
            scope = "not_reviewed_not_training_admissible"
        decisions.append({
            "candidate_id": row["candidate_id"],
            "decision": decision,
            "reason": reason,
            "review_scope": scope,
            "training_admitted": decision == "accept",
        })

    (args.artifact / "review_decisions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    counts = Counter(row["decision"] for row in decisions)
    report = {
        "schema": "fireviewer.pointing-v7-local-hard-negative-review.v1",
        "candidate_count": len(rows),
        "reviewed_count": len(REVIEWED),
        "accepted": counts["accept"],
        "rejected": counts["reject"],
        "not_reviewed": counts["not_reviewed"],
        "automatic_admission": False,
        "status": "bounded_visual_review_complete_fail_closed",
    }
    (args.artifact / "review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
