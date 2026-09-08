"""Export explicitly reviewed native Image Gen candidates into a TRAIN-only COCO.

This does not generate/edit pixels, fabricate labels, change the real V8 corpus,
or launch a training run. Synthetic examples never satisfy real coverage gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import imagehash
from PIL import Image, ImageOps
from pycocotools.coco import COCO

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8.audit_coverage import geometries


def validate_record(row, real_by_sha):
    if row.get("synthetic") is not True or row.get("split") != "train":
        raise ValueError("Synthetic supplement must be explicitly train-only")
    if row.get("origin") != "codex-image-gen" or not row.get("generation_context_id") or not row.get("prompt"):
        raise ValueError("Native generation provenance is required")
    if row.get("counts_as_new_real_case") is not False:
        raise ValueError("Synthetic must not count as new real data")
    for field in ("whole_scene_and_all_targets_reviewed", "annotations_exploitable", "safe_ground_view_reviewed",
                  "no_selfie_or_dominant_person_reviewed", "corrected_overlay_verified"):
        if row.get(field) is not True:
            raise ValueError("Missing explicit visual review: " + field)
    if not row.get("visual_review_note") or not row.get("corrected_overlay_sha256"):
        raise ValueError("Actual review observation and inspected overlay are required")
    if not row.get("synthetic_scene_group_id"):
        raise ValueError("Generated variants need a parent scene group")
    for field, allowed in {
        "lighting_review": {"daylight", "low_light", "night", "backlit"},
        "visibility_review": {"clear", "low_visibility"},
        "framing_review": {"well_framed", "poor_but_usable"},
        "obstruction_review": {"clear", "partial_usable"},
    }.items():
        if row.get(field) not in allowed:
            raise ValueError("Missing or invalid reviewed scene metadata: " + field)
    if not isinstance(row.get("reference_real_image_sha256"), list):
        raise ValueError("Explicit real-image reference list required, including [] for text-only generation")
    for sha in row["reference_real_image_sha256"]:
        if sha not in real_by_sha or real_by_sha[sha].get("split") != "train":
            raise ValueError("A synthetic reference is outside the admitted real training split")
    if row["sha256"] in real_by_sha:
        raise ValueError("A real source cannot be relabelled as generated")
    boxes = geometries(row)
    if not boxes:
        raise ValueError("This supplement requires reviewed positive targets")
    return boxes


def complete_real_fingerprints(real_rows, historical_rows):
    """Do not silently omit inherited real images lacking inline fingerprints."""
    historical = {}
    for row in historical_rows:
        value = (row.get("phash"), row.get("phash_flipped"))
        if row["sha256"] in historical and historical[row["sha256"]] != value:
            raise ValueError("Conflicting historical image fingerprints")
        historical[row["sha256"]] = value
    result = []
    for row in real_rows:
        inline = (row.get("phash"), row.get("phash_flipped"))
        inherited = historical.get(row["sha256"])
        if all(inline) and inherited and inline != inherited:
            raise ValueError("Inline and historical image fingerprints disagree")
        selected = inline if all(inline) else inherited
        if not selected or not all(isinstance(h, str) and len(h) == 16 for h in selected):
            raise ValueError("Missing complete fingerprint coverage for a real corpus image")
        result.append(tuple(int(h, 16) for h in selected))
    return result


def minimum_real_distance(phash, flipped, real_hashes):
    """Compare both orientations on both sides, without assuming hash symmetry."""
    return min((candidate ^ reference).bit_count()
               for candidate in (phash, flipped)
               for pair in real_hashes for reference in pair)


def validate_overlay_receipt(row, receipt):
    """The exported geometry must be the geometry of the inspected overlay."""
    if not receipt or receipt.get("image_sha256") != row["sha256"]:
        raise ValueError("Missing or mismatched inspected annotation receipt")
    if receipt.get("corrected_overlay_sha256") != row["corrected_overlay_sha256"]:
        raise ValueError("Inspected annotation receipt refers to a different overlay")
    if Path(receipt["corrected_overlay"]).resolve() != Path(row["corrected_overlay"]).resolve():
        raise ValueError("Inspected annotation overlay path differs from receipt")
    receipt_row = row | {"objects": receipt.get("corrected_objects")}
    if geometries(receipt_row) != geometries(row):
        raise ValueError("Exported annotations differ from the visually inspected geometry")


def check_synthetic_diversity(rows, maximum_distance=4):
    """New context ids do not make identical or mirrored pictures new scenes."""
    groups = Counter(r["synthetic_scene_group_id"] for r in rows)
    if max(groups.values(), default=0) > 8:
        raise ValueError("More than eight synthetic images in one parent scene")
    minimum = None
    pairs = 0
    for i, row in enumerate(rows):
        hashes = [int(row[k], 16) for k in ("phash", "phash_flipped")]
        for previous in rows[:i]:
            other = [int(previous[k], 16) for k in ("phash", "phash_flipped")]
            distance = min((a ^ b).bit_count() for a in hashes for b in other)
            pairs += 1
            minimum = distance if minimum is None else min(minimum, distance)
            if distance <= maximum_distance:
                raise ValueError("Synthetic near-duplicate or mirror requires exclusion/review")
    return {"pairs_checked": pairs, "minimum_phash_distance": minimum,
            "exclusion_distance_max": maximum_distance, "scene_groups": len(groups)}


def summarize_coverage(rows, small_area_max=.005):
    result = {"small_relative_box_area_max": small_area_max}
    for field in ("lighting_review", "visibility_review", "framing_review", "obstruction_review"):
        result[field] = dict(sorted(Counter(r[field] for r in rows).items()))
    for label, name in ((0, "fire"), (1, "smoke")):
        selected = [(r, [b for b, c in geometries(r) if c == label]) for r in rows]
        result[name + "_instances"] = sum(len(boxes) for _, boxes in selected)
        result[name + "_images"] = sum(bool(boxes) for _, boxes in selected)
        result["small_" + name + "_images"] = sum(
            any(b[2]*b[3]/(r["width"]*r["height"]) <= small_area_max for b in boxes)
            for r, boxes in selected)
        # Geometry projection only, not an image edit or a model measurement.
        result[name + "_images_with_target_min_side_below_4px_at_long_edge"] = {
            str(edge): sum(any(min(b[2:])*edge/max(r["width"], r["height"]) < 4 for b in boxes)
                           for r, boxes in selected) for edge in (640, 960)}
    return result


def prepare(manifest, real_root, output, fingerprints=None):
    rows = read_rows(manifest)
    if not rows or len(rows) != len({r["sha256"] for r in rows}):
        raise ValueError("Nonempty unique generated images required")
    real_selection = real_root / "selection_manifest.jsonl"
    real_rows = read_rows(real_selection)
    real_by_sha = {r["sha256"]: r for r in real_rows}
    frozen = {name: digest(real_root / "coco" / name / "_annotations.coco.json") for name in ("valid", "test")}
    images, annotations, checked = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    train = output / "coco" / "train"
    train.mkdir(parents=True, exist_ok=True)
    if (output / "coco" / "valid").exists() or (output / "coco" / "test").exists():
        raise ValueError("Synthetic validation/test directories are forbidden")
    fingerprint_sha = digest(fingerprints) if fingerprints else None
    coverage_path = real_root / "coverage_report.json"
    if fingerprints and coverage_path.is_file():
        expected_sha = json.loads(coverage_path.read_text(encoding="utf-8")).get("inputs", {}).get("known_fingerprints_sha256")
        if expected_sha and expected_sha != fingerprint_sha:
            raise ValueError("Historical fingerprints differ from the real corpus assembly input")
    real_hashes = complete_real_fingerprints(real_rows, read_rows(fingerprints) if fingerprints else [])
    overlay_receipts = {}
    for index, row in enumerate(rows, 1):
        boxes = validate_record(row, real_by_sha)
        receipt_path = Path(row["corrected_overlay"]).parent / "receipts.jsonl"
        if receipt_path not in overlay_receipts:
            receipt_rows = read_rows(receipt_path)
            if len(receipt_rows) != len({r["review_index"] for r in receipt_rows}):
                raise ValueError("Duplicate inspected annotation receipt index")
            overlay_receipts[receipt_path] = {r["review_index"]: r for r in receipt_rows}
        validate_overlay_receipt(row, overlay_receipts[receipt_path].get(row["review_index"]))
        source, native = Path(row["source_image"]), Path(row["native_tool_artifact"])
        if digest(source) != row["sha256"] or digest(native) != row["sha256"]:
            raise ValueError("Generated pixels differ from their native artifact")
        if digest(Path(row["corrected_overlay"])) != row["corrected_overlay_sha256"]:
            raise ValueError("Inspected synthetic annotation overlay changed")
        with Image.open(source) as image:
            image.load()
            if image.size != (row["width"], row["height"]):
                raise ValueError("Generated image dimensions changed")
            p = int(str(imagehash.phash(image)), 16)
            pf = int(str(imagehash.phash(ImageOps.mirror(image))), 16)
        minimum_distance = minimum_real_distance(p, pf, real_hashes)
        if minimum_distance <= 4:
            raise ValueError("Generated image is too close to a real corpus image")
        filename = row["sha256"] + source.suffix.lower()
        dest = train / filename
        if dest.exists() and digest(dest) != row["sha256"]:
            raise ValueError("Existing supplement export image changed")
        images.append({"id": index, "file_name": filename, "width": row["width"], "height": row["height"],
                       "synthetic": True, "source_scene": row["synthetic_scene_group_id"]})
        for box, label in boxes:
            annotations.append({"id": len(annotations)+1, "image_id": index, "category_id": label,
                                "bbox": box, "area": box[2]*box[3], "iscrowd": 0})
        checked.append(row | {"phash": f"{p:016x}", "phash_flipped": f"{pf:016x}",
                              "minimum_real_corpus_phash_distance": minimum_distance})
    diversity = check_synthetic_diversity(checked)
    expected = {i["file_name"] for i in images}
    actual = {p.name for p in train.iterdir() if p.is_file() and p.name != "_annotations.coco.json"}
    if not actual <= expected:
        raise ValueError("Stale or unexpected synthetic export files; nothing was deleted")
    # Validate the entire incoming batch before materializing any new export link.
    for row, item in zip(rows, images):
        dest = train / item["file_name"]
        if not dest.exists():
            os.link(Path(row["source_image"]).resolve(), dest)
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}],
            "info": {"description": "Explicitly synthetic train-only supplement; not an independent evaluation corpus."}}
    annotations_path = train / "_annotations.coco.json"
    annotations_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    reloaded = COCO(str(annotations_path))
    for item in reloaded.loadImgs(reloaded.getImgIds()):
        with Image.open(train / item["file_name"]) as image:
            image.load()
            if image.size != (item["width"], item["height"]):
                raise ValueError("Reloaded COCO dimensions differ")
    if frozen != {name: digest(real_root / "coco" / name / "_annotations.coco.json") for name in frozen}:
        raise ValueError("Real holdout annotation bytes changed during preparation")
    (output / "reviewed_synthetic_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in checked), encoding="utf-8")
    train_count = sum(r["split"] == "train" for r in real_rows)
    scenes = len({r["synthetic_scene_group_id"] for r in rows})
    # Cap both the synthetic share and reuse of each distinct generated scene.
    maximum_draws = min(scenes*8, int(train_count*.10/.90))
    report = {"status": "reviewed_synthetic_supplement_not_a_bulk_training_mix",
              "images": len(images), "annotations": len(annotations), "synthetic_scene_groups": scenes,
              "new_real_cases": 0, "validation_images": 0, "test_images": 0,
              "full_image_and_coco_reload_passed": True, "training_launched": False,
              "generator_model_revision": "not_exposed_by_builtin_tool",
              "real_selection_sha256": digest(real_selection), "real_holdout_coco_sha256": frozen,
              "real_images_checked_perceptually": len(real_hashes), "historical_fingerprints_sha256": fingerprint_sha,
              "minimum_real_corpus_phash_distances": [r["minimum_real_corpus_phash_distance"] for r in checked],
              "synthetic_diversity": diversity, "coverage": summarize_coverage(checked),
              "generator_source_families": 1, "physical_real_events": 0,
              "manual_manifest_sha256": digest(manifest), "coco_sha256": digest(annotations_path),
              "maximum_synthetic_draws_per_real_epoch_at_current_diversity": maximum_draws,
              "maximum_sample_fraction_at_current_diversity": maximum_draws/(train_count+maximum_draws),
              "mix_enabled": False,
              "required_next_gate": "User review and real fixed-validation ablation before enabling synthetic mixing; no claimed model gain."}
    (output / "supplement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, default=Path("artifacts/local/pointing-v7-split-audit-groupwise-20260827/fingerprints.jsonl"))
    args = parser.parse_args()
    prepare(args.manifest, args.real_root, args.output, args.fingerprints)


if __name__ == "__main__":
    main()
