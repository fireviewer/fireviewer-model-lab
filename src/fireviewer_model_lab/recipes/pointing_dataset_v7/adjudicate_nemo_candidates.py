#!/usr/bin/env python3
"""Freeze the exhaustive visual adjudication of the bounded NEMO pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Rejected after the 40 context sheets plus dedicated target zooms for every
# distant/tiny candidate.  Ambiguous pixels are rejected even when the source
# video annotation says smoke: a training image must be visually defensible on
# its own.
REJECT = {
    21, 54, 55,
    346, 347, 348, 349, 350, 358, 361, 370, 373, 375, 380, 384, 386, 387,
    396, 399, 402, 404, 410, 419, 420, 421, 422, 424, 425, 426, 429, 436,
    438, 440, 441, 449, 450, 454, 455, 456, 459, 460, 461, 465, 467, 468,
    469, 470, 473, 474, 478, 480, 483, 485, 486, 487, 489, 493, 495, 496,
    499, 500, 501, 506, 546,
    721, 723, 724, 726, 728, 733, 734, 737, 738, 742, 743, 744, 747, 748,
    752, 753, 754, 756, 757, 758, 762, 763, 765, 766, 775, 776, 777, 778,
    780, 784, 785, 789, 790, 792, 798,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.artifact / "candidate_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions = []
    for row in rows:
        number = int(row["candidate_id"].rsplit("-", 1)[-1])
        accepted = number not in REJECT
        decisions.append({
            "candidate_id": row["candidate_id"],
            "decision": "accept" if accepted else "reject",
            "reason": (
                "manual_context_sheet_and_target_zoom_confirmed_real_smoke_and_usable_box"
                if accepted
                else "target_ambiguous_non_smoke_obstruction_ui_or_not_visually_confirmable"
            ),
            "review_scope": "all_context_sheets_and_dedicated_distant_tiny_target_zooms",
            "training_admitted": accepted,
        })
    if len(rows) != 799 or not REJECT.issubset({int(row["candidate_id"].rsplit("-", 1)[-1]) for row in rows}):
        raise RuntimeError("NEMO candidate pool changed; adjudication must fail closed")
    output = args.artifact / "review_decisions.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
    counts = Counter(row["decision"] for row in decisions)
    report = {
        "schema": "fireviewer.pointing-v7-nemo-review.v1",
        "candidate_count": len(rows),
        "accepted": counts["accept"],
        "rejected": counts["reject"],
        "status": "exhaustive_visual_review_complete",
        "automatic_admission": False,
    }
    (args.artifact / "review_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
