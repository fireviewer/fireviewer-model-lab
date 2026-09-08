#!/usr/bin/env python3
"""Freeze exhaustive visual decisions for the final NEMO expansion pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Every candidate was reviewed in a context/target pair.  The rejection ledger
# is deliberately conservative: cloud, fog, rain shafts, glare, UI captures,
# structures, heavy foreground obstruction and pixels that are not visibly
# defensible as smoke are excluded even when the source event says smoke.
REJECT = {
    13, 14, 15, 35, 36, 37, 47, 49, 50, 51, 52, 54, 55, 58, 59,
    63, 64, 65, 76, 84, 87, 98, 117, 118, 119,
    136, 139, 142, 143, 168,
    181, 182, 183, 184, 185, 186, 192, 196, 197, 205, 206, 207,
    208, 209, 220, 221, 222, 223, 225, 226, 227, 238, 239, 240,
    247, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
    261, 262, 263, 264, 265, 266, 267, 269, 270, 271, 278, 279,
    280, 281, 282, 283, 284, 285, 292, 294, 295, 296, 297, 298,
    299, 300,
    301, 302, 303, 304, 305, 306, 307, 308, 311, 313, 314, 315,
    316, 318, 319, 322, 323, 324, 326, 331, 332, 333, 334, 335,
    336, 337, 338, 339, 341, 342, 343, 344, 345, 350, 351, 353,
    354, 355, 356, 357, 358, 359, 360,
    361, 362, 363, 364, 365, 367, 368, 375, 376, 380, 381, 382,
    383, 384, 385, 390, 391, 392, 393, 394, 395, 396, 399, 400,
    401, 402, 404, 405, 406, 410, 411, 412, 413, 416, 417, 420,
    424, 428, 430, 431, 432, 434, 436, 437, 438, 439, 440, 441,
    444, 445, 447, 448, 450, 451, 456, 457, 459, 460, 461, 462,
    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474,
    476, 477, 478, 480,
    486, 487, 488, 489, 490, 491, 492, 493, 494, 500, 501,
    505, 506, 507, 508, 509, 510, 511, 512, 513, 515, 516, 517,
    521, 525, 528, 529, 539, 544, 545, 546, 547, 548, 552,
    559, 560, 562, 564, 565,
    567, 568, 569, 570, 571, 572, 573, 574, 575, 576,
    *range(577, 613), 614,
}


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
    numeric_ids = {
        int(row["candidate_id"].rsplit("-", 1)[-1]) for row in rows
    }
    if len(rows) != 614 or numeric_ids != set(range(1, 615)):
        raise RuntimeError("NEMO final candidate pool changed; review is invalid")
    if not REJECT.issubset(numeric_ids):
        raise RuntimeError("rejection ledger contains an absent candidate")

    decisions = []
    accepted_by_bucket: Counter[str] = Counter()
    rejected_by_bucket: Counter[str] = Counter()
    for row in rows:
        number = int(row["candidate_id"].rsplit("-", 1)[-1])
        accepted = number not in REJECT
        bucket = str(row["gap_bucket"])
        (accepted_by_bucket if accepted else rejected_by_bucket)[bucket] += 1
        decisions.append({
            "candidate_id": row["candidate_id"],
            "decision": "accept" if accepted else "reject",
            "reason": (
                "manual_context_and_target_zoom_confirmed_real_smoke_and_usable_box"
                if accepted
                else "ambiguous_non_smoke_cloud_fog_rain_glare_ui_structure_or_obstruction"
            ),
            "review_scope": "all_52_context_and_target_zoom_sheets",
            "training_admitted": accepted,
        })

    output = args.artifact / "review_decisions.jsonl"
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    counts = Counter(row["decision"] for row in decisions)
    report = {
        "schema": "fireviewer.pointing-v7-nemo-final-review.v1",
        "candidate_count": len(rows),
        "accepted": counts["accept"],
        "rejected": counts["reject"],
        "accepted_by_bucket": dict(sorted(accepted_by_bucket.items())),
        "rejected_by_bucket": dict(sorted(rejected_by_bucket.items())),
        "automatic_admission": False,
        "status": "exhaustive_visual_review_complete",
    }
    (args.artifact / "review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
