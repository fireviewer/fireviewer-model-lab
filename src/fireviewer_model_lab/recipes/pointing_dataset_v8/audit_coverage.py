"""Measure coverage without turning unknown reviews or candidate pools into admissions."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from training.pointing_dataset_v7.split_registry import canonical_source_group, digest, read_rows
from training.pointing_dataset_v8.reconcile_review_evidence import source_family
from training.pointing_dataset_v8.source_identity import camera_views, frozen_camera_views

POLICY = Path(__file__).with_name("coverage_policy.json")
ACCEPTED = {"accepted_existing_review", "accepted_v8_review"}
SEMANTIC_VALUES = {
    "framing_review": {"well_framed", "poor_but_usable", "too_tight", "unusable"},
    "obstruction_review": {"clear", "partial_usable", "blocked"},
    "visibility_review": {"clear", "low_visibility", "not_applicable"},
    "lighting_review": {"daylight", "low_light", "night", "backlit"},
}


def family(row: dict) -> str:
    return source_family(row.get("source_family", row.get("source_dataset", "unknown")), row.get("source_record_id", ""))


def admitted(row: dict) -> bool:
    # The V8 decision always overrides a legacy V7 training_admitted=true.
    decision = row.get("v8_corpus_admitted", row.get("v8_training_admitted"))
    return decision is True and row.get("review_status") in ACCEPTED


def geometries(row: dict) -> list[tuple[list[float], int]]:
    if row.get("annotation_state") in {"classification_only_not_detection_boxes", "unannotated"}:
        raise ValueError("Classification-only images are not annotated detection negatives")
    objects = row.get("objects", {})
    boxes, labels = objects.get("bbox", []), objects.get("category", [])
    if len(boxes) != len(labels):
        raise ValueError("bbox/category cardinality mismatch")
    width, height = row.get("width", 0), row.get("height", 0)
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)) or not math.isfinite(width * height) or min(width, height) <= 0:
        raise ValueError("invalid image dimensions")
    result = []
    for bbox, label in zip(boxes, labels):
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1) or len(bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
            raise ValueError("invalid category or bbox")
        x, y, w, h = bbox
        if x < -0.01 or y < -0.01 or min(w, h) <= 0 or x + w > width + .01 or y + h > height + .01:
            raise ValueError("bbox outside image or empty")
        result.append((bbox, label))
    return result


def coverage(rows: list[dict], *, small_ratio: float = .005) -> dict:
    stats: dict = {"images": len(rows), "classes": {}, "scenes": {}, "sources": {}, "geometry_errors": []}
    classes, scenes, sources, small = Counter(), Counter(), Counter(), Counter()
    declared_groups, physical_events, reviews, semantic_unknown = Counter(), Counter(), Counter(), Counter()
    reviewed_scenes = Counter()
    framing, obstruction, conditions, confusers = Counter(), Counter(), Counter(), Counter()
    scene_conditions = defaultdict(lambda: {"lighting_review": Counter(), "visibility_review": Counter()})
    off_center_proxy = 0
    for row in rows:
        sources[family(row)] += 1
        reviews[row.get("review_status", "unknown")] += 1
        group = row.get("source_group_id")
        if group:
            declared_groups[canonical_source_group(group)] += 1
        if row.get("scene_group_verified") is True and row.get("scene_group_id") and row.get("scene_group_evidence"):
            reviewed_scenes[row["scene_group_id"]] += 1
        else:
            semantic_unknown["scene_group"] += 1
        if row.get("physical_event_verified") is True and row.get("physical_event_id"):
            physical_events[row["physical_event_id"]] += 1
        else:
            semantic_unknown["physical_event"] += 1
        for field, target in (("framing_review", framing), ("obstruction_review", obstruction),
                              ("visibility_review", conditions), ("lighting_review", conditions)):
            value = row.get(field)
            if value not in SEMANTIC_VALUES[field]:
                semantic_unknown[field] += 1
            else:
                target[value] += 1
        try:
            objects = geometries(row)
        except (ValueError, TypeError) as exc:
            stats["geometry_errors"].append({"sha256": row.get("sha256"), "error": str(exc)})
            continue
        labels = {label for _, label in objects}
        scene = "negative" if not labels else "fire_and_smoke" if len(labels) == 2 else "fire_only" if 0 in labels else "smoke_only"
        scenes[scene] += 1
        for field in ("lighting_review", "visibility_review"):
            value = row.get(field)
            scene_conditions[scene][field][value if value in SEMANTIC_VALUES[field] else "unknown"] += 1
        if not labels:
            if row.get("negative_verified") is not True:
                semantic_unknown["negative_verification"] += 1
            if row.get("negative_confuser_verified") is True and row.get("negative_confuser"):
                name = row["negative_confuser"]
                confusers[{"sunset_reflections": "sunset_reflection"}.get(name, name)] += 1
            else:
                semantic_unknown["negative_confuser"] += 1
        small_labels = set()
        off_center = False
        for (x, y, w, h), label in objects:
            name = "fire" if label == 0 else "smoke"
            classes[name] += 1
            if w * h / (row["width"] * row["height"]) <= small_ratio:
                small_labels.add(name)
            cx, cy = (x + w / 2) / row["width"], (y + h / 2) / row["height"]
            off_center |= not (.25 <= cx <= .75 and .25 <= cy <= .75)
        small.update(small_labels)
        off_center_proxy += off_center
    total = max(len(rows), 1)
    return stats | {"classes": dict(classes), "scenes": dict(scenes), "sources": dict(sources),
                    "source_share_max": max(sources.values(), default=0) / total,
                    "top_three_source_share": sum(n for _, n in sources.most_common(3)) / total,
                    "negative_fraction": scenes["negative"] / total,
                    "small_box_images_by_class": dict(small), "small_relative_box_area_max": small_ratio,
                    "off_center_geometry_proxy_images": off_center_proxy,
                    "declared_groups": len(declared_groups), "declared_group_max_images": max(declared_groups.values(), default=0),
                    "verified_physical_events": len(physical_events), "verified_physical_event_max_images": max(physical_events.values(), default=0),
                    "reviewed_scene_groups": len(reviewed_scenes), "reviewed_scene_group_max_images": max(reviewed_scenes.values(), default=0),
                    "review_status": dict(reviews), "semantic_unknown": dict(semantic_unknown),
                    "framing_review": dict(framing), "obstruction_review": dict(obstruction),
                    "conditions_review": dict(conditions), "negative_confusers_review": dict(confusers),
                    "scene_condition_slices": {scene: {field: dict(counts) for field, counts in fields.items()}
                                               for scene, fields in scene_conditions.items()}}


def audit(rows: list[dict], baseline: list[dict], policy: dict, history: dict | None = None,
          evaluation: list[dict] | None = None) -> dict:
    baseline_ids = {r["sha256"] for r in baseline}
    baseline_families = {family(r) for r in baseline}
    accepted = [r for r in rows if admitted(r)]
    new = [r for r in accepted if r["sha256"] not in baseline_ids and r.get("synthetic") is False and not r.get("augmentation_of")]
    train = [r for r in accepted if r.get("split") == "train"]
    new_train = [r for r in new if r.get("split") == "train"]
    all_stats = coverage(rows, small_ratio=policy["small_relative_box_area_max"])
    train_stats = coverage(train, small_ratio=policy["small_relative_box_area_max"])
    new_stats = coverage(new, small_ratio=policy["small_relative_box_area_max"])
    new_train_stats = coverage(new_train, small_ratio=policy["small_relative_box_area_max"])
    new_sources = {name: count for name, count in new_stats["sources"].items() if name not in baseline_families}
    checks = []

    def minimum(name, actual, target):
        checks.append({"gate": name, "actual": actual, "min": target, "missing": max(0, target - actual), "passed": actual >= target})

    def maximum(name, actual, target):
        checks.append({"gate": name, "actual": actual, "max": target, "passed": actual <= target})

    minimum("admitted_images", len(accepted), policy["total_admitted_min"])
    maximum("admitted_images_storage_budget", len(accepted), policy["total_admitted_target_max"])
    minimum("genuinely_new_admitted_images", len(new), policy["new_admitted_min"])
    minimum("new_source_families_with_minimum_support", sum(n >= policy["new_images_per_new_source_min"] for n in new_sources.values()), policy["new_source_families_min"])
    maximum("train_largest_source_share", train_stats["source_share_max"], policy["train_source_share_max"])
    maximum("train_top_three_source_share", train_stats["top_three_source_share"], policy["train_top_three_source_share_max"])
    instance_count = sum(train_stats["classes"].values())
    for name in ("fire", "smoke"):
        minimum(f"train_{name}_instance_fraction", train_stats["classes"].get(name, 0) / max(instance_count, 1), policy["train_class_instance_fraction_min"])
    minimum("train_negative_fraction", train_stats["negative_fraction"], policy["train_negative_fraction_min"])
    maximum("train_negative_fraction_max", train_stats["negative_fraction"], policy["train_negative_fraction_max"])
    for name in ("fire", "smoke"):
        minimum(f"new_train_small_{name}_images", new_train_stats["small_box_images_by_class"].get(name, 0), policy[f"new_train_small_{name}_images_min"])
    minimum("new_train_negative_images", new_train_stats["scenes"].get("negative", 0), policy["new_train_negative_images_min"])
    minimum("new_reviewed_scene_groups", new_stats["reviewed_scene_groups"], policy["new_reviewed_scene_groups_min"])
    maximum("new_images_per_verified_physical_event", new_stats["verified_physical_event_max_images"], policy["new_images_per_physical_event_max"])
    maximum("new_images_per_declared_group", new_stats["declared_group_max_images"], policy["new_images_per_declared_group_max"])
    maximum("new_images_per_reviewed_scene", new_stats["reviewed_scene_group_max_images"], policy["new_images_per_declared_group_max"])
    maximum("new_unverified_scene_groups", new_stats["semantic_unknown"].get("scene_group", 0), 0)
    for field, value, key in (("conditions_review", "low_visibility", "new_low_visibility_images_min"),
                              ("conditions_review", "low_light", "new_low_light_images_min"),
                              ("framing_review", "poor_but_usable", "new_poor_framing_images_min")):
        count = new_stats[field].get(value, 0)
        if value == "low_light":
            count += new_stats[field].get("night", 0)
        minimum(key.removesuffix("_min"), count, policy[key])
    # The requested night-detection coverage needs real annotated positives.
    # Night backgrounds remain useful confusers, but cannot fulfil this target.
    low_light_positives = sum(
        counts.get("lighting_review", {}).get(light, 0)
        for scene, counts in new_train_stats["scene_condition_slices"].items()
        if scene != "negative" for light in ("low_light", "night"))
    minimum("new_low_light_positive_images", low_light_positives, policy["new_low_light_positive_images_min"])
    minimum("negative_confuser_types_with_support", sum(n >= policy["negative_images_per_confuser_min"] for n in train_stats["negative_confusers_review"].values()), policy["negative_confuser_types_min"])
    # Unknown physical-incident identity is not invented and does not invalidate
    # a background negative. Reviewed scene groups have their own evidence gate;
    # their count is never presented as a count of independent physical fires.
    mandatory_semantics = set(SEMANTIC_VALUES) | {"negative_verification"}
    maximum("admitted_unknown_required_semantics", sum(train_stats["semantic_unknown"].get(k, 0) for k in mandatory_semantics), 0)
    maximum("new_unknown_required_semantics", sum(new_stats["semantic_unknown"].get(k, 0) for k in mandatory_semantics), 0)
    train_count = max(len(train), 1)
    poor_fraction = train_stats["framing_review"].get("poor_but_usable", 0) / train_count
    minimum("train_poor_framing_fraction", poor_fraction, policy["poor_framing_fraction_min"])
    maximum("train_poor_framing_fraction_max", poor_fraction, policy["poor_framing_fraction_max"])
    maximum("train_obstructed_fraction", train_stats["obstruction_review"].get("partial_usable", 0) / train_count, policy["obstructed_fraction_max"])
    maximum("unusable_framing_or_obstruction", sum(train_stats["framing_review"].get(v, 0) for v in ("too_tight", "unusable")) + train_stats["obstruction_review"].get("blocked", 0), 0)
    maximum("new_unreviewed_safety_or_annotations", sum(
        r.get("person_risk_reviewed_clear") is not True or r.get("unsafe_capture_reviewed_clear") is not True
        or r.get("aerial") is not False or r.get("annotation_review_complete") is not True
        or r.get("annotation_exploitable") is not True for r in new), 0)
    maximum("invalid_annotation_images", len(all_stats["geometry_errors"]), 0)
    maximum("duplicate_image_identities", len(rows) - len({r["sha256"] for r in rows}), 0)
    collisions = []
    for row in train:
        if history and (history["sha256"].get(row["sha256"]) in {"test", "validation"} or
                        history.get("groups", {}).get(row.get("split_group_id", "")) in {"test", "validation"} or
                        any(s != "train" for s in history.get("source_groups", {}).get(canonical_source_group(row.get("source_group_id", "")), []))):
            collisions.append(row["sha256"])
    maximum("historical_holdouts_entering_train", len(collisions), 0)
    heldout_cameras = frozen_camera_views(baseline + rows, history)
    camera_train_collisions = [r["sha256"] for r in train if camera_views(r) & heldout_cameras]
    maximum("historical_camera_views_entering_train", len(camera_train_collisions), 0)
    camera_splits = defaultdict(set)
    for row in accepted:
        for camera in camera_views(row):
            camera_splits[camera].add(row.get("split"))
    camera_split_collisions = {key: sorted(value) for key, value in camera_splits.items() if len(value) > 1}
    maximum("cross_split_camera_views", len(camera_split_collisions), 0)
    groups = defaultdict(set)
    for row in accepted:
        for field in ("split_group_id", "source_group_id", "scene_group_id"):
            group = row.get(field)
            if group:
                groups[field + ":" + canonical_source_group(group)].add(row.get("split"))
    split_collisions = {key: sorted(value, key=str) for key, value in groups.items() if len(value) > 1}
    maximum("cross_split_source_groups", len(split_collisions), 0)
    maximum("invalid_admitted_split", sum(r.get("split") not in {"train", "validation", "test"} for r in accepted), 0)
    maximum("holdouts_marked_training_admitted", sum(r.get("split") != "train" and r.get("v8_training_admitted") is True for r in accepted), 0)
    eval_stats = coverage(evaluation or [], small_ratio=policy["small_relative_box_area_max"])
    ep = policy["independent_extra_evaluation"]
    eval_checks = []
    for name, actual, threshold in (("images", len(evaluation or []), ep["images_min"]),
                                     ("negative_images", eval_stats["scenes"].get("negative", 0), ep["negative_images_min"]),
                                     ("reviewed_scene_groups", eval_stats["reviewed_scene_groups"], ep["reviewed_scene_groups_min"]),
                                     ("fire_instances", eval_stats["classes"].get("fire", 0), ep["target_instances_per_class_min"]),
                                     ("smoke_instances", eval_stats["classes"].get("smoke", 0), ep["target_instances_per_class_min"])):
        eval_checks.append({"gate": name, "actual": actual, "min": threshold, "passed": actual >= threshold})
    eval_unverified = [r.get("sha256") for r in evaluation or [] if r.get("evaluation_admitted") is not True
                       or r.get("selection_blind_to_model_predictions") is not True
                       or (not r.get("objects", {}).get("bbox") and r.get("negative_verified") is not True)]
    train_hashes = {r["sha256"] for r in train}
    train_groups = {canonical_source_group(r.get("source_group_id", "")) for r in train}
    eval_leaks = [r.get("sha256") for r in evaluation or [] if r.get("sha256") in train_hashes
                  or canonical_source_group(r.get("source_group_id", "")) in train_groups]
    eval_checks.extend([{"gate": "unverified_or_prediction_selected_images", "actual": len(eval_unverified), "passed": not eval_unverified},
                        {"gate": "invalid_geometry", "actual": len(eval_stats["geometry_errors"]), "passed": not eval_stats["geometry_errors"]},
                        {"gate": "train_overlap", "actual": len(eval_leaks), "passed": not eval_leaks},
                        {"gate": "duplicate_evaluation_identities", "actual": len(evaluation or []) - len({r.get('sha256') for r in evaluation or []}),
                         "passed": len(evaluation or []) == len({r.get('sha256') for r in evaluation or []})}])
    # Separate splits must still have a full identity/near-duplicate audit before release.
    return {"schema": "fireviewer.pointing-v8-coverage-audit.v1", "ready": all(c["passed"] for c in checks),
            "scope": "coverage_gate_only_not_full_release_authority", "proposed": all_stats,
            "proposed_train": coverage([r for r in rows if r.get("split") == "train"]),
            "admitted_train": train_stats, "new_admitted": new_stats, "new_source_families": new_sources,
            "checks": checks, "blocked_gates": [c["gate"] for c in checks if not c["passed"]],
            "historical_holdout_collisions": collisions, "cross_split_source_groups": split_collisions,
            "historical_camera_train_collisions": camera_train_collisions, "cross_split_camera_views": camera_split_collisions,
            "extra_evaluation": {"ready": all(c["passed"] for c in eval_checks), "coverage": eval_stats, "checks": eval_checks},
            "notes": ["Small bounding-box area is not proof of physical distance or faint visibility.",
                      "Declared groups and reviewed scene groups are not verified independent physical events.",
                      "Unknown semantic labels are never counted as absent or passed.",
                      "The existing per-image V8 decision overrides legacy training flags.",
                      "Coverage targets are engineering criteria, not a guarantee of V8 detection performance."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--independent-evaluation", type=Path)
    args = parser.parse_args()
    policy = json.loads(POLICY.read_text())
    report = audit(read_rows(args.manifest), read_rows(args.baseline), policy, json.loads(args.history.read_text()),
                   read_rows(args.independent_evaluation) if args.independent_evaluation else None)
    report["inputs"] = {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in
                       (("manifest", args.manifest), ("baseline", args.baseline), ("history", args.history), ("policy", POLICY))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ready": report["ready"], "proposed": report["proposed"], "blocked_gates": report["blocked_gates"]}, indent=2))
    if args.require_ready and not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
