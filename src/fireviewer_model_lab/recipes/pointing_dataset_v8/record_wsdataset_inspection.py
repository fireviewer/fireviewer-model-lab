"""Record only the 24 candidates actually inspected on 2026-08-27.

No source-wide clearance, ground-truth correction or training admission is inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.pointing_dataset_v7.split_registry import digest, read_rows

# Explicitly inspected full-context/all-box-detail packets, not a range presumed
# reviewed merely because it was generated.
REVIEWED_PAGES = {1, 2, 3, 4, 5, 6, 7, 8, 12, 18, 24, 25}
PLUME_REVIEW = set(range(57, 73)) | {79, 80, 91, 92, 103, 104, 105}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root
    rows = read_rows(root / "candidate_manifest.jsonl")
    by_id = {r["candidate_id"]: r for r in rows}
    packets = json.loads((root / "review/index.json").read_text())
    seen = {}
    for packet in packets:
        page = int(Path(packet["packet"]).stem.rsplit("-", 1)[1])
        if page not in REVIEWED_PAGES:
            continue
        if digest(root / "review" / packet["packet"]) != packet["sha256"]:
            raise ValueError("Reviewed packet changed")
        for identifier in packet["candidate_ids"]:
            seen[identifier] = packet
    decisions = []
    for identifier, packet in seen.items():
        row = by_id[identifier]
        n = int(identifier.rsplit("-", 1)[1])
        positive = n in PLUME_REVIEW
        reason = ("visible_smoke_extends_beyond_upstream_box_or_plume_extent_is_ambiguous_requires_annotation_adjudication"
                  if positive else "urban_elevated_view_negative_and_capture_suitability_not_fully_established")
        decisions.append({"candidate_id": identifier, "sha256": row["sha256"], "reviewer": "Codex_visual_inspection",
                          "review_date": "2026-08-27", "review_method": "scaled_full_context_and_all_bbox_detail_packet",
                          "decision": "needs_reannotation" if positive else "needs_review",
                          "reason": reason, "packet": packet["packet"], "packet_sha256": packet["sha256"],
                          "v8_training_admitted": False, "annotation_review_complete": False})
    if len(decisions) != 24 or sum(d["decision"] == "needs_reannotation" for d in decisions) != 23:
        raise ValueError("Inspection record no longer matches actually viewed packets")
    mapped = {d["candidate_id"]: d for d in decisions}
    curated = [r | ({"review_status": mapped[r["candidate_id"]]["decision"], "v8_inspection": mapped[r["candidate_id"]]}
                    if r["candidate_id"] in mapped else {}) for r in rows]
    for name, records in (("visual_inspection_decisions.jsonl", decisions), ("curation_manifest.jsonl", curated)):
        (root / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    report = {"schema": "fireviewer.wsdataset-v8-bounded-visual-inspection.v1", "images_inspected": 24,
              "annotation_adjudication_required": 23, "other_pending": 1, "not_visually_inspected": len(rows) - 24,
              "new_admitted_images": 0, "candidate_manifest_sha256": digest(root / "candidate_manifest.jsonl"),
              "observations": ["Repeated telecom-mast/tennis-court/hillside scene occurs across multiple official video IDs.",
                               "Another repeated scene shows a roadside compound and forest; video IDs must not be counted as independent incidents.",
                               "Some boxes cover only the smoke origin/dense core while a visible plume extends outside.",
                               "No statement of source-wide annotation defect or full-source visual review is made."]}
    (root / "visual_inspection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
