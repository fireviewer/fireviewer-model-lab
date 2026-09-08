"""Assemble explicitly reviewed V8 additions into one zero-copy, reloadable view.

This does not relax the coverage contract, manufacture reviews, or authorize a
training run. A technically reloadable COCO view can still be incomplete.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image, ImageOps
from pycocotools.coco import COCO

from training.pointing_dataset_v7.prepare_audited_v7 import write_json
from training.pointing_dataset_v7.split_registry import canonical_source_group, digest, make_registry, read_rows
from training.pointing_dataset_v8.audit_coverage import POLICY, SEMANTIC_VALUES, audit, family, geometries
from training.pointing_dataset_v8.prepare_extension_review import known_fingerprints
from training.pointing_dataset_v8.record_extension_review import apply_decision, read_visual_batch
from training.pointing_dataset_v8.source_identity import (bind_camera_reviews, camera_views,
    frozen_camera_views, load_camera_reviews, retire_camera_leaks, separate_validation_from_test)

SCHEMA = "fireviewer.pointing-v8-reviewed-extension.v1"
SPLITS = {"train": "train", "validation": "valid", "test": "test"}


def apply_condition_reviews(rows: list[dict], ledgers: list[Path]) -> tuple[list[dict], dict]:
    """Review legacy conditions without re-annotating or re-admitting images.

    Partial observations remain partial. Unsuitable observations reach the
    coverage gate unchanged; this function never removes images to hide them.
    """
    result = copy.deepcopy(rows)
    mapped = {row["sha256"]: row for row in result}
    seen, checked_packets = set(), set()
    inputs, resolved, unknown = {}, Counter(), Counter()
    for ledger in ledgers:
        packets_path = ledger.parent / "packets.json"
        packets = json.loads(packets_path.read_text(encoding="utf-8"))
        packet_map = {str(Path(p["packet"]).resolve()): p for p in packets}
        if len(packet_map) != len(packets):
            raise ValueError("Duplicate condition-review packet")
        ledger_sha = digest(ledger)
        inputs[str(ledger.resolve())] = {"sha256": ledger_sha, "packets_sha256": digest(packets_path)}
        for decision in read_rows(ledger):
            sha = decision["sha256"]
            if sha in seen:
                raise ValueError("Duplicate condition review for one image")
            seen.add(sha)
            row = mapped.get(sha)
            if row is None or row.get("split") != "train" or row.get("review_status") != "accepted_existing_review":
                raise ValueError("Condition reviews are restricted to retained legacy train images")
            annotation_sha = hashlib.sha256(json.dumps(row["objects"], sort_keys=True).encode()).hexdigest()
            if decision.get("annotation_sha256") != annotation_sha or digest(Path(row["source_image"])) != sha:
                raise ValueError("Condition review image or annotation binding changed")
            packet_path = str(Path(decision["packet"]).resolve())
            packet = packet_map.get(packet_path)
            if (packet is None or sha not in packet.get("image_sha256s", [])
                    or decision.get("packet_sha256") != packet["sha256"]):
                raise ValueError("Condition review image is not bound to its inspected packet")
            if packet_path not in checked_packets:
                if digest(Path(packet_path)) != packet["sha256"]:
                    raise ValueError("Condition-review packet changed")
                checked_packets.add(packet_path)
            if not decision.get("reason") or not decision.get("review_date"):
                raise ValueError("Explicit condition observation and review date required")
            for field, allowed in SEMANTIC_VALUES.items():
                if field not in decision:
                    raise ValueError("Every condition must be explicitly resolved or unknown")
                value = decision[field]
                if value is None or value == "unknown":
                    unknown[field] += 1
                    continue
                if value not in allowed:
                    raise ValueError("Invalid explicit condition value")
                if field == "visibility_review" and value == "not_applicable" and row["objects"]["bbox"]:
                    raise ValueError("Positive targets cannot have not-applicable visibility")
                if row.get(field) in allowed and row[field] != value:
                    raise ValueError("Conflicting existing condition review")
                row[field] = value
                resolved[field] += 1
            if decision.get("negative_confuser"):
                if row["objects"]["bbox"] or row.get("negative_verified") is not True:
                    raise ValueError("Negative subtype review requires an existing verified negative")
                row["negative_confuser"] = decision["negative_confuser"]
                row["negative_confuser_verified"] = True
            row["legacy_conditions_review"] = {
                "ledger": str(ledger.resolve()), "ledger_sha256": ledger_sha,
                "annotation_sha256": annotation_sha, "decision": copy.deepcopy(decision),
                "scope": "conditions_only_existing_images_annotations_and_admission_unchanged",
            }
        if digest(ledger) != ledger_sha or digest(packets_path) != inputs[str(ledger.resolve())]["packets_sha256"]:
            raise ValueError("Condition-review inputs changed during verification; retry a stable snapshot")
    return result, {"reviewed_images": len(seen), "resolved_fields": dict(resolved),
                    "unknown_fields": dict(unknown), "packets_verified": len(checked_packets), "inputs": inputs}


def retire_unsuitable_condition_reviews(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Exclude only explicit unsuitable legacy observations; retain their proof."""
    kept, excluded = [], []
    for row in rows:
        evidence = row.get("legacy_conditions_review")
        decision = (evidence or {}).get("decision", {})
        pending_person_review = decision.get("requires_person_review") is True
        unsuitable = (row.get("framing_review") in {"too_tight", "unusable"}
                      or row.get("obstruction_review") == "blocked"
                      or bool(decision.get("exclude_from_real_corpus_reason"))
                      or bool(decision.get("exclude_from_training_reason")))
        if row.get("split") == "train" and evidence and (unsuitable or pending_person_review):
            excluded.append({"sha256": row["sha256"], "source_image": row["source_image"],
                             "reason": "legacy_person_presence_unresolved" if pending_person_review and not unsuitable
                                       else "explicit_legacy_conditions_unsuitable",
                             "framing_review": row.get("framing_review"),
                             "obstruction_review": row.get("obstruction_review"),
                             "legacy_conditions_review": evidence, "source_image_deleted": False})
        else:
            kept.append(row)
    return kept, excluded


def verified_reviews(root: Path) -> list[dict]:
    """Recreate admissions from explicit decisions, not editable admitted flags."""
    mapped = {r["review_index"]: r for r in read_rows(root / "review_manifest.jsonl")}
    report = json.loads((root / "curation_report.json").read_text())
    decisions_path = root / "manual_decisions.jsonl"
    if digest(decisions_path) != report["decisions_sha256"]:
        raise ValueError("Manual review ledger differs from curation report")
    decisions = read_rows(decisions_path)
    batches = sorted((root / "batches").glob("*.tsv"))
    if {p.name: digest(p) for p in batches} != report["batch_sha256"]:
        raise ValueError("Visual review batches changed; rerun the recorder")
    for path in batches:
        decisions.extend(read_visual_batch(path, mapped))
    if len(decisions) != len({d["review_index"] for d in decisions}):
        raise ValueError("Duplicate explicit decision in one review root")
    packets = json.loads((root / "packets.json").read_text())
    packet_map = {index: p for p in packets for index in p["review_indices"]}
    checked_packets, result = set(), []
    for decision in decisions:
        row = mapped[decision["review_index"]]
        packet = packet_map[decision["review_index"]]
        if packet["packet"] not in checked_packets:
            if digest(root / "pages" / packet["packet"]) != packet["sha256"]:
                raise ValueError("The inspected packet changed")
            checked_packets.add(packet["packet"])
        if row["candidate_id"] not in packet["candidate_ids"]:
            raise ValueError("The image does not belong to its reviewed packet")
        for prefix in ("corrected_overlay", "all_targets_overlay"):
            if decision.get(prefix + "_sha256") and digest(Path(decision[prefix])) != decision[prefix + "_sha256"]:
                raise ValueError("An inspected annotation overlay changed")
        accepted = apply_decision(row, decision, packet)
        accepted["review_root"] = str(root.resolve())
        if accepted["v8_corpus_admitted"]:
            source = Path(row["source_image"])
            if digest(source) != row["sha256"]:
                raise ValueError("Reviewed image bytes changed")
            with Image.open(source) as im:
                im.load()
                if im.size != (row["width"], row["height"]):
                    raise ValueError("Reviewed image dimensions changed")
                actual = str(imagehash.phash(im)), str(imagehash.phash(ImageOps.mirror(im)))
            if actual != (row["phash"], row["phash_flipped"]):
                raise ValueError("Reviewed image fingerprints changed")
        result.append(accepted)
    return result


def priority(row):
    """Prefer gap coverage within a scene, never use a detector score."""
    small = sum(b[2] * b[3] / (row["width"] * row["height"]) <= .005 for b in row["objects"]["bbox"])
    score = (4 * (row.get("visibility_review") == "low_visibility")
             + 3 * (row.get("framing_review") == "poor_but_usable")
             + 2 * (row.get("lighting_review") in {"low_light", "night"}) + min(2, small))
    return -score, row["sha256"]


def select_additions(base, reviews, baseline, history, fingerprints, policy):
    """Global exact/mirrored-near screening and conservative scene/group caps."""
    baseline_ids = {r["sha256"] for r in baseline}
    if not baseline_ids <= {r["sha256"] for r in fingerprints}:
        raise ValueError("Fingerprint registry omits historical V7 exposures")
    rejected = {r["sha256"] for r in reviews if r["review_status"] == "reject"}
    candidates = sorted((r for r in reviews if r["v8_corpus_admitted"]), key=priority)
    hashes = {r["sha256"] for r in fingerprints} | {r["sha256"] for r in base}
    fingerprint_rows = {r["sha256"]: r for r in fingerprints}
    for row in base:
        if row["sha256"] in fingerprint_rows:
            continue
        if row.get("phash") and row.get("phash_flipped"):
            fingerprint_rows[row["sha256"]] = row
        else:
            source = Path(row["source_image"])
            if digest(source) != row["sha256"]:
                raise ValueError("Base image differs from its recorded identity")
            with Image.open(source) as image:
                image.load()
                fingerprint_rows[row["sha256"]] = row | {
                    "phash": str(imagehash.phash(image)),
                    "phash_flipped": str(imagehash.phash(ImageOps.mirror(image)))}
    index = [(int(r["phash"], 16), int(r["phash_flipped"], 16), r["sha256"])
             for r in fingerprint_rows.values()]
    scenes, groups, events = Counter(), Counter(), Counter()
    # A completion pass continues the same extension relative to V7. Previously
    # admitted V8 images consume the same scene/group budget as this new lot.
    for row in base:
        if row["sha256"] in baseline_ids or row.get("synthetic") is not False or row.get("augmentation_of"):
            continue
        scene = canonical_source_group(row.get("scene_group_id", ""))
        group = canonical_source_group(row.get("source_group_id", ""))
        event = row.get("physical_event_id") if row.get("physical_event_verified") is True else None
        if scene and row.get("scene_group_verified") is True:
            scenes[scene] += 1
        if group:
            groups[group] += 1
        if event:
            events[event] += 1
    selected, excluded = [], []
    heldout_cameras = frozen_camera_views(baseline + base + reviews, history)
    heldout_groups = {canonical_source_group(r.get(k, "")) for r in base if r["split"] != "train"
                      for k in ("source_group_id", "split_group_id", "scene_group_id") if r.get(k)}

    def reject(row, reason, match=None):
        excluded.append({"sha256": row["sha256"], "candidate_id": row.get("candidate_id"),
                         "reason": reason, "matching_sha256": match, "review_root": row.get("review_root")})

    for original in candidates:
        row = copy.deepcopy(original)
        sha = row["sha256"]
        row["source_family"] = family(row)
        source_group = canonical_source_group(row.get("source_group_id", ""))
        scene = canonical_source_group(row.get("scene_group_id", ""))
        event = row.get("physical_event_id") if row.get("physical_event_verified") is True else None
        reason = None
        if sha in rejected:
            reason = "explicit_rejection_in_another_review"
        elif sha in baseline_ids:
            reason = "not_new_historical_v7_image"
        elif camera_views(row) & heldout_cameras:
            reason = "historical_holdout_camera_view"
        elif row.get("source_dataset") == "WUI-Fire-Detection" and row["source_family"] == "DFire" and not row.get("original_source_group_bound"):
            reason = "dfire_mirror_without_original_group_binding"
        elif row.get("synthetic") is not False or row.get("augmentation_of"):
            reason = "real_case_provenance_not_established"
        elif not row.get("license") or not row.get("license_evidence") or not row.get("source_revision"):
            reason = "missing_source_revision_or_license_evidence"
        elif row.get("split") != "train" or row.get("scene_group_verified") is not True or not scene or not source_group:
            reason = "unverified_group_or_invalid_split"
        elif (history.get("sha256", {}).get(sha) in {"test", "validation"}
              or history.get("groups", {}).get(row.get("split_group_id", "")) in {"test", "validation"}
              or any(s != "train" for s in history.get("source_groups", {}).get(source_group, []))
              or any(canonical_source_group(row.get(k, "")) in heldout_groups
                     for k in ("source_group_id", "split_group_id", "scene_group_id") if row.get(k))):
            reason = "historical_holdout_group"
        if reason:
            reject(row, reason)
            continue
        match = sha if sha in hashes else None
        a, af = int(row["phash"], 16), int(row["phash_flipped"], 16)
        if match is None:
            match = next((s for b, bf, s in index if min((a ^ b).bit_count(), (af ^ b).bit_count(),
                                                        (a ^ bf).bit_count(), (af ^ bf).bit_count()) <= 4), None)
        if match:
            reject(row, "global_exact_overlap" if sha == match else "global_mirrored_phash_le4", match)
            continue
        if scenes[scene] >= policy["new_images_per_declared_group_max"]:
            reject(row, "reviewed_scene_cap")
            continue
        if groups[source_group] >= policy["new_images_per_declared_group_max"]:
            reject(row, "declared_source_group_cap")
            continue
        if event and events[event] >= policy["new_images_per_physical_event_max"]:
            reject(row, "verified_event_cap")
            continue
        geometries(row)
        # Dataset-level MIT is recorded, but not misrepresented as per-photo clearance.
        if row.get("source_dataset") == "WUI-Fire-Detection":
            row["redistribution_review"] = "dataset_MIT_declared_individual_third_party_photo_rights_not_verified"
        selected.append(row)
        hashes.add(sha)
        index.append((a, af, sha))
        scenes[scene] += 1
        groups[source_group] += 1
        if event:
            events[event] += 1
    return selected, excluded


def retire_unlisted_export_links(image_root: Path, expected: set[str], known_sources: list[dict], retired_root: Path) -> list[dict]:
    """Move only verified obsolete generated hardlinks, never source media.

    Retention is literal: no unlink/delete and no image copy. Unknown files,
    redirects or destination collisions fail closed rather than being cleaned.
    """
    scope = image_root.parent.parent.parent.resolve()
    retired = []
    for path in sorted(image_root.iterdir()):
        if path.name in expected:
            continue
        destination = retired_root / path.name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(scope)
                or not destination.resolve().is_relative_to(scope) or destination.exists()):
            raise ValueError("Refusing to move an unsafe or conflicting unlisted export path")
        originals = [Path(r["source_image"]).resolve() for r in known_sources if r["sha256"] == path.stem]
        source = next((p for p in originals if p != path.resolve() and p.is_file() and os.path.samefile(p, path)), None)
        if source is None or digest(source) != path.stem:
            raise ValueError("Unlisted export file is not a verified hardlink to a retained source")
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        if not os.path.samefile(source, destination):
            raise ValueError("Retired hardlink identity mismatch")
        retired.append({"file_name": path.name, "retained_link": str(destination.resolve()),
                        "source_image": str(source), "sha256": path.stem, "source_deleted": False})
    return retired


def export_and_reload(rows, root: Path, *, known_sources=None):
    """Real COCO/PIL reload and byte/annotation checks; hardlinks, no copies."""
    ids = [r["image_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("COCO image-id collision")
    receipts = {}
    retired = []
    for split, folder in SPLITS.items():
        selected = [r for r in rows if r["split"] == split]
        images, annotations = [], []
        for row in selected:
            src = Path(row["source_image"]).resolve()
            filename = row["sha256"] + src.suffix.lower()
            dst = root / folder / "images" / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if not os.path.samefile(src, dst):
                    raise ValueError("Refusing to replace an unrelated image")
            else:
                os.link(src, dst)  # cross-volume failure is intentional; never silently copy
            if not os.path.samefile(src, dst):
                raise ValueError("Hardlink identity mismatch")
            images.append({"id": row["image_id"], "file_name": "images/" + filename,
                           "width": row["width"], "height": row["height"], "fireviewer_sha256": row["sha256"],
                           "fireviewer_source_group_id": row["source_group_id"]})
            for box, label in geometries(row):
                annotations.append({"id": len(annotations) + 1, "image_id": row["image_id"], "category_id": label,
                                    "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
        path = root / folder / "_annotations.coco.json"
        write_json(path, {"info": {"status": "review_only_not_training_ready", "description": SCHEMA},
                          "images": images, "annotations": annotations,
                          "categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}]})
        loaded = COCO(str(path))
        if len(loaded.imgs) != len(selected) or len(loaded.anns) != len(annotations):
            raise ValueError("Reloaded COCO cardinality mismatch")
        for row in selected:
            im = loaded.imgs[row["image_id"]]
            image_path = root / folder / im["file_name"]
            if digest(image_path) != row["sha256"]:
                raise ValueError("Reloaded image identity mismatch")
            with Image.open(image_path) as image:
                image.load()
                if image.size != (im["width"], im["height"]):
                    raise ValueError("Reloaded image dimensions mismatch")
            anns = sorted(loaded.imgToAnns[row["image_id"]], key=lambda a: a["id"])
            if [(a["bbox"], a["category_id"]) for a in anns] != geometries(row):
                raise ValueError("Reloaded annotation differs from reviewed boxes")
        receipts[split] = {"images": len(images), "annotations": len(annotations), "hardlinks_verified": len(images),
                           "negative_images": sum(not r["objects"]["bbox"] for r in selected),
                           "annotation_sha256": digest(path), "all_images_decoded": True}
        image_root = root / folder / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        expected = {Path(image["file_name"]).name for image in images}
        retired += retire_unlisted_export_links(image_root, expected, known_sources or rows,
                                               root.parent / "retired_export_links" / folder)
        if {p.name for p in image_root.iterdir()} != expected:
            raise ValueError("Image directory membership differs from COCO")
    return {"status": "passed", "image_count": len(rows), "splits": receipts,
            "validation": "actual_pycocotools_COCO_and_PIL_all_images_annotations_and_bytes",
            "directories_match_coco_exactly": True, "retired_export_links": retired,
            "additional_image_payload_bytes": 0, "training_authorized": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("artifacts/local/fireviewer-pointing-v8-draft-20260827"))
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/local/fireviewer-pointing-v7-ready-local-5000-20260825-r7/selection_manifest.jsonl"))
    parser.add_argument("--history", type=Path, default=Path("artifacts/local/fireviewer-pointing-v7-audited-20260827/historical_split_registry.json"))
    parser.add_argument("--fingerprints", type=Path, default=Path("artifacts/local/pointing-v7-split-audit-groupwise-20260827/fingerprints.jsonl"))
    parser.add_argument("--external", type=Path, default=Path("fireviewer_bench/data/homefire-pointing-independent-v1-r3/samples.jsonl"))
    parser.add_argument("--review-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local/fireviewer-pointing-v8-extended-20260827"))
    parser.add_argument("--refresh-generated-view", action="store_true")
    parser.add_argument("--real-only-base", action="store_true",
                        help="Keep synthetic supplements separate from real-case completion accounting")
    parser.add_argument("--identity-reviews", type=Path,
                        help="Explicit image-bound camera grouping reviews, without annotation admission")
    parser.add_argument("--conditions-review", type=Path, action="append", default=[],
                        help="Explicit image/annotation/packet-bound legacy train condition decisions")
    parser.add_argument("--separate-validation-from-test", action="store_true",
                        help="Reserve validation images sharing known test cameras; preserve a full frozen evaluation reference")
    args = parser.parse_args()
    marker = args.output / "assembly_owner.json"
    base_path = args.base / "selection_manifest.jsonl"
    owner = {"schema": SCHEMA, "base_manifest_sha256": digest(base_path), "base_path": str(base_path.resolve())}
    if args.real_only_base:
        owner["real_only_base"] = True
    if args.output.exists():
        if not args.refresh_generated_view or not marker.is_file() or json.loads(marker.read_text()) != owner:
            raise ValueError("Refusing to overwrite an unrelated or unbound corpus")
        if (args.output / "validation_reserve_manifest.jsonl").exists() and not args.separate_validation_from_test:
            raise ValueError("The disjoint evaluation protocol must be retained explicitly on replay")
        prior_report = args.output / "coverage_report.json"
        if prior_report.exists():
            prior_conditions = json.loads(prior_report.read_text()).get("legacy_conditions_review", {}).get("inputs", {})
            if not set(prior_conditions) <= {str(p.resolve()) for p in args.conditions_review}:
                raise ValueError("Previously applied condition-review ledgers must be retained on replay")
    base, baseline = read_rows(base_path), read_rows(args.baseline)
    source_base_rows = base
    history = json.loads(args.history.read_text())
    source_base_count = len(base)
    separated_synthetic = [r for r in base if r.get("synthetic") is True or r.get("augmentation_of")] if args.real_only_base else []
    if separated_synthetic:
        separated_ids = {r["sha256"] for r in separated_synthetic}
        base = [r for r in base if r["sha256"] not in separated_ids]
    policy = json.loads(POLICY.read_text())
    reviews = [row for root in args.review_root for row in verified_reviews(root)]
    if args.identity_reviews:
        bindings = load_camera_reviews(args.identity_reviews)
        base = bind_camera_reviews(base, bindings)
        baseline = bind_camera_reviews(baseline, bindings)
        reviews = bind_camera_reviews(reviews, bindings)
    # Verify immutable legacy reviews before a newly identified holdout camera
    # retires their train rows. Retirement must not invalidate prior evidence.
    base, conditions_report = apply_condition_reviews(base, args.conditions_review)
    base, base_exclusions = retire_camera_leaks(base, baseline + reviews, history)
    camera_exclusion_count = len(base_exclusions)
    frozen_holdouts = [r for r in base if r["split"] != "train"]
    validation_reserve = []
    if args.separate_validation_from_test:
        base, validation_reserve = separate_validation_from_test(base)
    base, conditions_exclusions = retire_unsuitable_condition_reviews(base)
    base_exclusions += conditions_exclusions
    known = known_fingerprints(args.fingerprints, args.external)
    additions, excluded = select_additions(base, reviews, baseline, history, known, policy)
    selected = base + sorted(additions, key=lambda r: r["sha256"])
    if [r for r in selected if r["split"] != "train"] != [r for r in base if r["split"] != "train"]:
        raise ValueError("The frozen evaluation selection was changed")
    report = audit(selected, baseline, policy, history)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(marker, owner)
    if args.separate_validation_from_test:
        # Keep the original calibration/test protocol reloadable for paired
        # historical comparisons, without duplicating the image payload.
        frozen_reload = export_and_reload(frozen_holdouts, args.output / "frozen_evaluation_reference" / "coco")
        write_json(args.output / "frozen_evaluation_reference" / "reload_validation.json", frozen_reload)
        (args.output / "validation_reserve_manifest.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in validation_reserve), encoding="utf-8")
    for name, records in (("selection_manifest.jsonl", selected), ("new_admitted_manifest.jsonl", additions),
                          ("excluded_additions.jsonl", excluded), ("base_exclusions.jsonl", base_exclusions)):
        (args.output / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    report.update({"schema": SCHEMA, "draft_status": "incomplete_do_not_train" if not report["ready"] else "coverage_passed_pending_release_review",
                   "legacy_conditions_review": conditions_report,
                   "new_admitted_images": len(additions), "base_images": len(base), "selected_images": len(selected),
                   "source_base_images": source_base_count, "base_train_references_retired": len(base_exclusions),
                   "base_camera_train_references_retired": camera_exclusion_count,
                   "base_unsuitable_condition_references_retired": len(conditions_exclusions),
                   "synthetic_base_references_kept_separate": len(separated_synthetic),
                   "reviewed_candidates": len(reviews), "preassembly_admitted_images": sum(r["v8_corpus_admitted"] for r in reviews),
                   "reviewed_unique_images": len({r["sha256"] for r in reviews}),
                   "assembly_exclusions": dict(Counter(r["reason"] for r in excluded)),
                   "inputs": {"base_manifest_sha256": digest(base_path), "policy_sha256": digest(POLICY),
                              "identity_reviews_sha256": digest(args.identity_reviews) if args.identity_reviews else None,
                              "historical_registry_sha256": digest(args.history), "known_fingerprints_sha256": digest(args.fingerprints),
                              "external_manifest_sha256": digest(args.external),
                              "review_reports": {str(p.resolve()): digest(p / "curation_report.json") for p in args.review_root}},
                   "storage": {"additional_image_payload_bytes": 0, "automatic_cleanup": False, "mode": "NTFS_hardlinks_to_retained_sources"},
                   "split_policy": "New images train-only; historical held-out camera views excluded across dates and explicit visual aliases, including legacy train references. Frozen holdout membership and annotations are unchanged; historical exposure evidence is preserved.",
                   "initialization_caveat": "Removing V8 train references cannot undo earlier V7 weight exposure. Unseen-camera claims require generic pretrained initialization or a genuinely new independent evaluation panel.",
                   "training_started": False, "public_redistribution_cleared": False,
                   "public_redistribution_limit": "WUI declares MIT at dataset level; underlying third-party photo rights remain unverified."})
    if args.separate_validation_from_test:
        report.update({
            "evaluation_protocol": "v8_known_camera_disjoint_validation_test_v1",
            "validation_images_reserved_for_test_camera_overlap": len(validation_reserve),
            "reserved_validation_sha256": [r["sha256"] for r in validation_reserve],
            "frozen_evaluation_reference_preserved": True,
            "active_test_membership_and_annotations_unchanged": True,
            "split_policy": "New images train-only; historical holdout locks retained. Validation images sharing known test cameras are reserved, never trained. All historical validation/test images, IDs and annotations remain reloadable under frozen_evaluation_reference/coco. Active test is unchanged. Unknown camera independence remains unproven.",
        })
    # An incomplete report is written before materialization, so interruption is
    # never mistaken for successful export or training readiness.
    write_json(args.output / "coverage_report.json", report)
    write_json(args.output / "report.json", {"schema": SCHEMA, "status": "incomplete_do_not_train", "selected_count": len(selected)})
    reload = export_and_reload(selected, args.output / "coco", known_sources=source_base_rows + reviews)
    write_json(args.output / "reload_validation.json", reload)
    registry = make_registry(selected, history)
    registry["camera_views"] = {key: list(value) for key, value in history.get("camera_views", {}).items()}
    for row in baseline + selected:
        for view in camera_views(row):
            registry["camera_views"][view] = sorted(set(registry["camera_views"].get(view, [])) | {row["split"]})
    write_json(args.output / "historical_split_registry.json", registry)
    write_json(args.output / "coco" / "coco_view_receipt.json", {
        "schema": SCHEMA, "status": "review_only_not_training_ready", "dataset_root": str(args.output.resolve()),
        "selection_manifest_sha256": digest(args.output / "selection_manifest.jsonl"), "splits": reload["splits"],
        "storage_policy": report["storage"]})
    artifact_names = ["selection_manifest.jsonl", "new_admitted_manifest.jsonl", "base_exclusions.jsonl",
                      "excluded_additions.jsonl", "coverage_report.json", "reload_validation.json", "historical_split_registry.json"]
    if args.separate_validation_from_test:
        artifact_names += ["validation_reserve_manifest.jsonl", "frozen_evaluation_reference/reload_validation.json"]
    write_json(args.output / "assembly_receipt.json", {"schema": SCHEMA, "reload_status": reload["status"],
        "artifact_hashes": {p: digest(args.output / p) for p in artifact_names},
        "validation_code_sha256": {p.name: digest(p) for p in (
            Path(__file__), Path(__file__).with_name("source_identity.py"), Path(__file__).with_name("audit_coverage.py"),
            Path(__file__).with_name("record_extension_review.py"))},
        "training_authorized": False, "automatic_cleanup": False})
    print(json.dumps({"selected_images": len(selected), "new_images": len(additions), "reload": reload["status"],
                      "exclusions": report["assembly_exclusions"], "ready": report["ready"], "blocked_gates": report["blocked_gates"]}), flush=True)


if __name__ == "__main__":
    main()
