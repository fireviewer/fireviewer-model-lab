"""Apply explicit per-image visual decisions; no unlisted image is admitted."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8.audit_coverage import SEMANTIC_VALUES, geometries


def read_visual_batch(path, mapped):
    """Compact transcription of explicit inspected-image decisions, not a rule.

    A/N/P certify checked scene suitability and every target. No default A, no
    wildcard/range, and no review is inferred for absent indices.
    """
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("|")
        if len(cells) != 9:
            raise ValueError("Review batch needs exactly nine explicit fields")
        index, action, light, visibility, framing, obstruction, group, confuser, reason = cells
        row = mapped[int(index)]
        decision = {"review_index": int(index), "decision": {"A": "accept_source_boxes", "N": "accept_negative", "P": "accept_reviewed_proposal",
                    "R": "reject", "U": "needs_annotation"}[action], "reason": reason}
        if action in {"A", "N", "P"}:
            decision.update({"lighting_review": {"D": "daylight", "L": "low_light", "N": "night", "B": "backlit"}[light],
                "visibility_review": {"C": "clear", "L": "low_visibility", "N": "not_applicable"}[visibility],
                "framing_review": {"W": "well_framed", "P": "poor_but_usable"}[framing],
                "obstruction_review": {"C": "clear", "P": "partial_usable"}[obstruction],
                "scene_and_all_targets_checked": True})
            if group == "@camera":
                if not row.get("camera") or not row.get("partner"):
                    raise ValueError("No upstream fixed camera metadata for explicit camera grouping")
                decision["scene_group_id"] = "Pyro-SDIS:" + row["partner"] + ":" + row["camera"]
                decision["scene_group_evidence"] = "Visually inspected fixed-camera context; same camera/heading grouped across dates, not counted as new incidents."
            else:
                prefix = "WUI" if row.get("source_family") == "WUI-Fire-Detection" else row.get("source_family", "unknown")
                decision["scene_group_id"] = prefix + ":" + group
                decision["scene_group_evidence"] = "Visually inspected context grouped as: " + group + "; original incident identity is unverified."
            if action == "N":
                decision["negative_confuser"] = confuser
            if action == "P":
                decision["proposal_objects_sha256"] = row.get("proposal_objects_sha256")
                decision["annotation_proposal_model_sha256"] = row.get("annotation_proposal_model_sha256")
        result.append(decision)
    return result


def apply_decision(row, decision, packet):
    result = copy.deepcopy(row)
    status = decision["decision"]
    if status not in {"accept_source_boxes", "accept_negative", "accept_corrected", "accept_reviewed_proposal", "needs_annotation", "reject"}:
        raise ValueError("Unknown explicit review decision")
    if not decision.get("reason"):
        raise ValueError("A review observation is required")
    result["v8_corpus_admitted"] = False
    result["v8_training_admitted"] = False
    result["visual_review"] = {"reviewer": "Codex_visual_inspection", "review_date": decision.get("review_date", "2026-08-27"),
        "method": "explicit_packet_inspection; accepted_images_require_whole_scene_and_all_targets_checked; not_native_full_frame_for_large_images",
        "packet": packet["packet"], "packet_sha256": packet["sha256"], "image_sha256": row["sha256"],
        "decision": decision}
    if status in {"reject", "needs_annotation"}:
        result["review_status"] = status
        return result
    for name, allowed in SEMANTIC_VALUES.items():
        if decision.get(name) not in allowed:
            raise ValueError(f"Missing explicit semantic review: {name}")
        result[name] = decision[name]
    if not decision.get("scene_group_id") or not decision.get("scene_group_evidence"):
        raise ValueError("A reviewed scene grouping and observation are required")
    if decision.get("scene_and_all_targets_checked") is not True:
        raise ValueError("No explicit complete scene/annotation check")
    if len(row["objects"]["bbox"]) > 3 and not decision.get("all_targets_overlay_sha256"):
        raise ValueError("More than three targets require an additional inspected overlay covering every target")
    result["annotation_origin_state"] = row.get("annotation_state")
    if status == "accept_source_boxes":
        if row.get("annotation_state") in {"classification_only_not_detection_boxes", "model_proposals_NOT_ground_truth"} or not row["objects"]["bbox"]:
            raise ValueError("No source detection annotations to accept")
    elif status == "accept_reviewed_proposal":
        objects_sha = hashlib.sha256(json.dumps(row["objects"], sort_keys=True).encode()).hexdigest()
        if (row.get("annotation_state") != "model_proposals_NOT_ground_truth" or not row["objects"]["bbox"]
                or decision.get("proposal_objects_sha256") != objects_sha
                or not row.get("annotation_proposal_model_sha256")
                or decision.get("annotation_proposal_model_sha256") != row["annotation_proposal_model_sha256"]):
            raise ValueError("Reviewed model proposals must be explicitly bound to their objects and model")
        result["annotation_proposal_visually_reviewed"] = True
    elif status == "accept_negative":
        if row["objects"]["bbox"]:
            raise ValueError("Cannot silently discard positive annotations")
        if not decision.get("negative_confuser"):
            raise ValueError("Negative subtype review is required")
        result["negative_confuser"] = decision["negative_confuser"]
        result["negative_confuser_verified"] = True
    else:
        if decision.get("corrected_overlay_verified") is not True or not decision.get("corrected_overlay_sha256"):
            raise ValueError("Corrected annotations require an actually inspected bound overlay")
        result["upstream_objects"] = copy.deepcopy(row["objects"])
        result["objects"] = decision["corrected_objects"]
    if result["framing_review"] in {"too_tight", "unusable"} or result["obstruction_review"] == "blocked":
        raise ValueError("Unsuitable framing or obstruction cannot be admitted")
    if result["objects"]["bbox"] and result["visibility_review"] == "not_applicable":
        raise ValueError("Positive targets require an explicit visibility review")
    if not result["objects"]["bbox"] and status != "accept_negative":
        raise ValueError("Empty corrected annotations require a separate explicit negative decision")
    if row.get("synthetic") is True or row.get("augmentation_of"):
        raise ValueError("Synthetic/augmented examples are not real-case extension admissions")
    result["annotation_state"] = "reviewed_detection_boxes_or_verified_negative"
    geometries(result)
    result.update({"annotation_state": "reviewed_detection_boxes_or_verified_negative",
        "annotation_review_complete": True, "annotation_exploitable": True,
        "person_risk_reviewed_clear": True, "unsafe_capture_reviewed_clear": True, "aerial": False,
        "scene_group_id": decision["scene_group_id"], "scene_group_verified": True,
        "scene_group_evidence": decision["scene_group_evidence"], "physical_event_verified": False,
        "negative_verified": not result["objects"]["bbox"], "is_negative": not result["objects"]["bbox"],
        "review_status": "accepted_v8_review", "v8_corpus_admitted": True, "v8_training_admitted": True,
        "split": "train", "training_admitted": True})
    if not result.get("source_group_id"):
        result["source_group_id"] = decision["scene_group_id"]
    result["split_group_id"] = result["source_group_id"]
    result["image_id"] = int(result["sha256"][:13], 16)
    result["file_name"] = result["sha256"] + Path(result["source_image"]).suffix
    result["caption"] = "Reviewed visible targets: fire=%d, smoke=%d." % (
        result["objects"]["category"].count(0), result["objects"]["category"].count(1))
    result["annotation_sha256"] = hashlib.sha256(json.dumps(result["objects"], sort_keys=True).encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.review_root / "review_manifest.jsonl")
    mapped = {r["review_index"]: r for r in rows}
    packets = json.loads((args.review_root / "packets.json").read_text())
    packet_map = {i: packet for packet in packets for i in packet["review_indices"]}
    decisions = read_rows(args.decisions)
    for path in sorted((args.review_root / "batches").glob("*.tsv")):
        decisions.extend(read_visual_batch(path, mapped))
    if len(decisions) != len({d["review_index"] for d in decisions}):
        raise ValueError("Duplicate manual decision for same image")
    curated, checked_packets = [], set()
    for decision in decisions:
        row = mapped[decision["review_index"]]
        packet = packet_map[decision["review_index"]]
        if digest(Path(row["source_image"])) != row["sha256"]:
            raise ValueError("Reviewed source image changed")
        if packet["packet"] not in checked_packets:
            if digest(args.review_root / "pages" / packet["packet"]) != packet["sha256"]:
                raise ValueError("Reviewed packet changed")
            checked_packets.add(packet["packet"])
        if decision.get("corrected_overlay_sha256"):
            overlay = Path(decision["corrected_overlay"])
            if digest(overlay) != decision["corrected_overlay_sha256"]:
                raise ValueError("Corrected overlay changed")
        if decision.get("all_targets_overlay_sha256"):
            if digest(Path(decision["all_targets_overlay"])) != decision["all_targets_overlay_sha256"]:
                raise ValueError("Additional target overlay changed")
        curated.append(apply_decision(row, decision, packet))
    accepted = [r for r in curated if r["v8_corpus_admitted"]]
    for name, records in (("curation_manifest.jsonl", curated), ("admitted_manifest.jsonl", accepted)):
        (args.review_root / name).write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
    report = {"reviewed_images": len(curated), "admitted_images": len(accepted),
              "decisions": dict(Counter(r["visual_review"]["decision"]["decision"] for r in curated)),
              "unreviewed_queue_images": len(rows)-len(curated), "image_bytes_copied": 0,
              "annotation_instances": dict(Counter(c for r in accepted for c in r["objects"]["category"])),
              "reviewed_scene_groups": len({r["scene_group_id"] for r in accepted}),
              "decisions_sha256": digest(args.decisions),
              "batch_sha256": {p.name: digest(p) for p in sorted((args.review_root / "batches").glob("*.tsv"))}}
    (args.review_root / "curation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
