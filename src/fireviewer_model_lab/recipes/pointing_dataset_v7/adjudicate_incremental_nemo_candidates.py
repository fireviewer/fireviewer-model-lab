#!/usr/bin/env python3
"""Freeze visual decisions for the third-view NEMO expansion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Conservative rejection after review of every context/target pair.  These are
# UI captures, glare/obstruction, or targets that cannot be distinguished from
# terrain, cloud, road, water, or structures at source resolution.
REJECT_NEW = {
    80, 119, 123, 125, 134, 141, 150, 155, 159, 162, 170, 174, 218,
    295, 304, 310, 314, 320, 334, 348, 351, 352, 354, 358, 360, 363,
    365, 367, 371, 373, 394, 396, 399, 403, 408, 413, 419, 424, 450,
    461, 476, 479, 483, 487, 493,
    509, 510, 536, 539, 548, 550, 559, 564, 568, 585, 591, 595, 598,
    602, 606, 609, 613, 636, 640, 656, 663, 676, 693, 696,
    714, 718, 724, 728, 738, 743, 744, 747, 748, 750, 758, 763, 769,
    775, 784, 790, 793, 799, 807, 819, 822, 825, 832, 834, 837, 843,
    849, 857, 861, 864, 867, 871, 872, 875, 876, 882, 885, 890, 892,
    900, 901, 905, 906, 908, 910, 911, 912, 916, 917, 920, 923, 926,
    936, 942, 943, 946, 947, 949, 950, 952, 953, 954, 955, 962, 967,
    968, 972, 976, 979, 980, 984, 986, 992, 995,
    1019, 1023, 1034, 1050, 1060, 1079,
}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    expanded = read(args.artifact / "candidate_manifest.jsonl")
    carried = {row["candidate_id"]: row for row in read(args.artifact / "carried_review_decisions.jsonl")}
    new_ids = {row["candidate_id"] for row in read(args.artifact / "new_candidate_manifest.jsonl")}
    numeric_new = {int(value.rsplit("-", 1)[-1]) for value in new_ids}
    if not REJECT_NEW.issubset(numeric_new):
        raise RuntimeError(f"decision IDs absent from incremental pool: {sorted(REJECT_NEW - numeric_new)}")
    decisions = []
    for row in expanded:
        candidate_id = row["candidate_id"]
        if candidate_id in carried:
            decision = dict(carried[candidate_id])
            decision["review_scope"] = "carried_by_exact_sha_from_exhaustive_r2_review"
        elif candidate_id in new_ids:
            number = int(candidate_id.rsplit("-", 1)[-1])
            accepted = number not in REJECT_NEW
            decision = {
                "candidate_id": candidate_id,
                "decision": "accept" if accepted else "reject",
                "reason": (
                    "manual_context_and_target_zoom_confirmed_real_smoke_and_usable_box"
                    if accepted else
                    "target_ambiguous_non_smoke_obstruction_ui_or_not_visually_confirmable"
                ),
                "review_scope": "incremental_third_view_context_and_target_zoom",
                "training_admitted": accepted,
            }
        else:
            raise RuntimeError(f"candidate has no decision authority: {candidate_id}")
        decisions.append(decision)
    counts = Counter(row["decision"] for row in decisions)
    output = args.artifact / "review_decisions.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
    report = {
        "schema": "fireviewer.pointing-v7-nemo-expanded-review.v1",
        "candidate_count": len(expanded),
        "carried_exact_sha_decisions": len(carried),
        "new_visual_decisions": len(new_ids),
        "accepted": counts["accept"],
        "rejected": counts["reject"],
        "status": "exhaustive_visual_review_complete",
    }
    (args.artifact / "review_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
