"""Verify V8 and create a zero-copy, explicitly authorized experimental train view.

Coverage objectives remain unchanged. Only the documented coverage limitations
may be acknowledged; bad annotations, changed evidence and train/holdout leaks
cannot be waived. Original real/synthetic manifests and holdouts stay untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import imagehash
from PIL import Image, ImageOps
from pycocotools.coco import COCO

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8.audit_coverage import POLICY, audit, geometries
from training.pointing_dataset_v8.prepare_synthetic_supplement import (
    check_synthetic_diversity, complete_real_fingerprints, minimum_real_distance,
    validate_overlay_receipt, validate_record,
)


KNOWN_LIMITATIONS = {
    "admitted_images", "genuinely_new_admitted_images",
    "new_source_families_with_minimum_support", "train_top_three_source_share",
    "new_train_small_fire_images", "new_train_small_smoke_images",
    "new_low_visibility_images", "new_poor_framing_images",
    "negative_confuser_types_with_support", "admitted_unknown_required_semantics",
    "train_poor_framing_fraction", "cross_split_camera_views",
    "new_low_light_positive_images", "train_poor_framing_fraction_max",
    "train_obstructed_fraction",
}
FOLDERS = {"train": "train", "validation": "valid", "test": "test"}
CATEGORIES = [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}]


def check_limitations(report, *, acknowledge=False):
    unknown = set(report["blocked_gates"]) - KNOWN_LIMITATIONS
    if unknown:
        raise ValueError("Non-waivable V8 integrity gate: " + ", ".join(sorted(unknown)))
    if any("train" in splits for splits in report.get("cross_split_camera_views", {}).values()):
        raise ValueError("A training camera occurs in a holdout")
    if report["blocked_gates"] and not acknowledge:
        raise ValueError("Known incomplete coverage needs explicit run acknowledgement")


def merge_train_coco(real, synthetic):
    if real["categories"] != CATEGORIES or synthetic["categories"] != CATEGORIES:
        raise ValueError("Expected exactly category 0=fire and 1=smoke")
    images = [dict(i) for i in real["images"]]
    annotations = [dict(a) for a in real["annotations"]]
    names = {i["file_name"] for i in images}
    next_image = max((i["id"] for i in images), default=0) + 1
    next_annotation = max((a["id"] for a in annotations), default=0) + 1
    mapping = {}
    for source in synthetic["images"]:
        filename = "images/" + Path(source["file_name"]).name
        if filename in names or source["id"] in mapping:
            raise ValueError("Duplicate synthetic image in training view")
        names.add(filename)
        mapping[source["id"]] = next_image
        images.append(source | {"id": next_image, "file_name": filename, "synthetic": True})
        next_image += 1
    for source in synthetic["annotations"]:
        if source["image_id"] not in mapping:
            raise ValueError("Orphan synthetic annotation")
        annotations.append(source | {"id": next_annotation, "image_id": mapping[source["image_id"]]})
        next_annotation += 1
    if len({i["id"] for i in images}) != len(images) or len({a["id"] for a in annotations}) != len(annotations):
        raise ValueError("COCO identifier collision")
    return {"images": images, "annotations": annotations, "categories": CATEGORIES,
            "info": {"status": "verified_user_authorized_experimental_training",
                     "coverage_qualified": False, "synthetic_train_only": True}}, mapping


def verify_pixels(row):
    path = Path(row["source_image"])
    if digest(path) != row["sha256"]:
        raise ValueError("Image bytes changed: " + str(path))
    with Image.open(path) as image:
        image.load()
        if image.size != (row["width"], row["height"]):
            raise ValueError("Image dimensions changed: " + str(path))
    geometries(row)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_exclusions(rows, decisions):
    by_sha = {r["sha256"]: r for r in rows}
    excluded = set()
    for item in decisions:
        row = by_sha.get(item["sha256"])
        if not row or row["split"] != "train" or not item.get("reason"):
            raise ValueError("Exclusion must name a present training image with a reason")
        if row.get("source_group_id") != item.get("source_group_id"):
            raise ValueError("Exclusion source identity changed")
        if row["sha256"] in excluded:
            raise ValueError("Duplicate exclusion decision")
        excluded.add(row["sha256"])
    return [r for r in rows if r["sha256"] not in excluded]


def validate_authorization(record, manifest_sha256, limitations, execution_target):
    if (record.get("training_authorized") is not True or
            record.get("source_selection_manifest_sha256") != manifest_sha256 or
            record.get("execution_target") != execution_target or
            set(record.get("acknowledged_limitations", [])) != set(limitations) or
            not record.get("user_request") or not record.get("user_confirmation")):
        raise ValueError("Launch authorization does not bind this corpus, target and exact limitations")


def prepare(real_root, synthetic_root, baseline, fingerprints, output, authorization, acknowledge=False, exclusions=None,
            execution_target="local", authorization_record=None):
    roots = [p.resolve() for p in (real_root, synthetic_root) if p is not None]
    output = output.resolve()
    if not authorization.strip():
        raise ValueError("A user launch authorization is required")
    if execution_target not in {"local", "lightning_cpu_then_l4"}:
        raise ValueError("Unsupported training execution target")
    if execution_target != "local" and authorization_record is None:
        raise ValueError("Remote training needs a corpus-bound authorization record")
    if any(output == p or p in output.parents or output in p.parents for p in roots):
        raise ValueError("Training view must be isolated from immutable source corpora")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing a nonempty training-view output")
    seal = json.loads((real_root / "assembly_receipt.json").read_text(encoding="utf-8"))
    for name, expected in seal["artifact_hashes"].items():
        if digest(real_root / name) != expected:
            raise ValueError("Sealed real artifact changed: " + name)
    source_rows = read_rows(real_root / "selection_manifest.jsonl")
    exclusion_record = json.loads(exclusions.read_text(encoding="utf-8")) if exclusions else {"exclusions": []}
    if exclusions and exclusion_record["source_real_manifest_sha256"] != digest(real_root / "selection_manifest.jsonl"):
        raise ValueError("Prelaunch exclusions refer to another corpus")
    rows = apply_exclusions(source_rows, exclusion_record["exclusions"])
    real_by_sha = {r["sha256"]: r for r in source_rows}
    if len(real_by_sha) != len(source_rows):
        raise ValueError("Duplicate real image identity")
    history = json.loads((real_root / "historical_split_registry.json").read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    coverage = audit(rows, read_rows(baseline), policy, history)
    check_limitations(coverage, acknowledge=acknowledge)
    if authorization_record is not None:
        validate_authorization(json.loads(authorization_record.read_text(encoding="utf-8")),
                               digest(real_root / "selection_manifest.jsonl"), coverage["blocked_gates"], execution_target)
    print(json.dumps({"stage": "coverage_and_history_checked", "real_images": len(rows),
                      "acknowledged_limitations": coverage["blocked_gates"]}), flush=True)
    previous = json.loads((real_root / "coverage_report.json").read_text(encoding="utf-8"))
    if digest(fingerprints) != previous["inputs"]["known_fingerprints_sha256"]:
        raise ValueError("Historical fingerprint evidence changed")
    hashes = complete_real_fingerprints(source_rows, read_rows(fingerprints))
    retained_ids = {r["sha256"] for r in rows}
    train = [(r, h) for r, h in zip(source_rows, hashes) if r["split"] == "train" and r["sha256"] in retained_ids]
    heldout = [(r, h) for r, h in zip(source_rows, hashes) if r["split"] != "train"]
    for row, pair in train:
        for other, reference in heldout:
            if min((a ^ b).bit_count() for a in pair for b in reference) <= 4:
                raise ValueError("Real train/holdout perceptual collision: " + row["sha256"] + "/" + other["sha256"])
    print(json.dumps({"stage": "real_train_holdout_phash_passed", "pairs": len(train)*len(heldout)}), flush=True)
    real_coco = {}
    frozen = {}
    for split, folder in FOLDERS.items():
        source = real_root / "coco" / folder / "_annotations.coco.json"
        frozen[folder] = digest(source)
        coco = COCO(str(source))
        selected = [r for r in source_rows if r["split"] == split]
        if len(coco.imgs) != len(selected) or coco.dataset["categories"] != CATEGORIES:
            raise ValueError("Real COCO membership/categories differ")
        for row in selected:
            verify_pixels(row)
            item = coco.imgs[row["image_id"]]
            if (item["width"], item["height"]) != (row["width"], row["height"]):
                raise ValueError("Real COCO size differs")
            if not os.path.samefile(real_root / "coco" / folder / item["file_name"], row["source_image"]):
                raise ValueError("Real COCO source identity differs")
            anns = sorted(coco.imgToAnns[row["image_id"]], key=lambda a: a["id"])
            if [(a["bbox"], a["category_id"]) for a in anns] != geometries(row):
                raise ValueError("Real COCO boxes differ from reviewed annotations")
        real_coco[folder] = coco.dataset
        if split == "train":
            kept_image_ids = {r["image_id"] for r in rows if r["split"] == "train"}
            real_coco[folder] = coco.dataset | {
                "images": [i for i in coco.dataset["images"] if i["id"] in kept_image_ids],
                "annotations": [a for a in coco.dataset["annotations"] if a["image_id"] in kept_image_ids]}
        print(json.dumps({"stage": "full_real_split_reload_passed", "split": split, "images": len(selected)}), flush=True)
    synthetic, synth_by_sha, synth_report, mapping = [], {}, {}, {}
    synth_coco = None
    merged = real_coco["train"]
    if synthetic_root is not None:
        synthetic = read_rows(synthetic_root / "reviewed_synthetic_manifest.jsonl")
        synth_report = json.loads((synthetic_root / "supplement_report.json").read_text(encoding="utf-8"))
        synth_coco_path = synthetic_root / "coco" / "train" / "_annotations.coco.json"
        if (digest(synth_coco_path) != synth_report["coco_sha256"] or
                digest(synthetic_root / "manual_review.jsonl") != synth_report["manual_manifest_sha256"] or
                digest(real_root / "selection_manifest.jsonl") != synth_report["real_selection_sha256"]):
            raise ValueError("Synthetic report no longer binds its inputs")
        manual = read_rows(synthetic_root / "manual_review.jsonl")
        if {r["sha256"] for r in manual} != {r["sha256"] for r in synthetic} or len(synthetic) != len(manual):
            raise ValueError("Reviewed synthetic membership differs")
        receipts = {r["review_index"]: r for r in read_rows(synthetic_root / "corrected_overlays" / "receipts.jsonl")}
        synth_coco = COCO(str(synth_coco_path))
        if len(synth_coco.imgs) != len(synthetic):
            raise ValueError("Synthetic COCO image count differs")
        for row in synthetic:
            validate_record(row, real_by_sha)
            validate_overlay_receipt(row, receipts.get(row["review_index"]))
            verify_pixels(row)
            if digest(Path(row["native_tool_artifact"])) != row["sha256"] or digest(Path(row["corrected_overlay"])) != row["corrected_overlay_sha256"]:
                raise ValueError("Native synthetic pixels or inspected overlay changed")
            with Image.open(row["source_image"]) as im:
                pair = [int(str(imagehash.phash(im)), 16), int(str(imagehash.phash(ImageOps.mirror(im))), 16)]
            if pair != [int(row[k], 16) for k in ("phash", "phash_flipped")]:
                raise ValueError("Synthetic fingerprint differs")
            if minimum_real_distance(*pair, hashes) <= 4:
                raise ValueError("Synthetic image too close to real corpus")
        check_synthetic_diversity(synthetic)
        merged, mapping = merge_train_coco(real_coco["train"], synth_coco.dataset)
        synth_by_sha = {r["sha256"]: r for r in synthetic}
        for item in synth_coco.dataset["images"]:
            row = synth_by_sha[Path(item["file_name"]).stem]
            anns = sorted(synth_coco.imgToAnns[item["id"]], key=lambda a: a["id"])
            if [(a["bbox"], a["category_id"]) for a in anns] != geometries(row):
                raise ValueError("Synthetic COCO boxes differ from reviewed annotations")
    all_sources = real_by_sha | synth_by_sha
    output.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, folder in FOLDERS.items():
        target = output / folder
        (target / "images").mkdir(parents=True)
        coco = merged if split == "train" else real_coco[folder]
        for item in coco["images"]:
            row = all_sources[Path(item["file_name"]).stem]
            dest = target / item["file_name"]
            os.link(Path(row["source_image"]).resolve(), dest)
            if not os.path.samefile(row["source_image"], dest):
                raise ValueError("Training-view hardlink mismatch")
        if split == "train" and (synthetic_root is not None or exclusion_record["exclusions"]):
            write_json(target / "_annotations.coco.json", coco)
        else:
            shutil.copyfile(real_root / "coco" / folder / "_annotations.coco.json", target / "_annotations.coco.json")
            if digest(target / "_annotations.coco.json") != frozen[folder]:
                raise ValueError("Frozen holdout changed in training view")
        loaded = COCO(str(target / "_annotations.coco.json"))
        splits[split] = {"images": len(loaded.imgs), "annotations": len(loaded.anns),
                         "annotation_sha256": digest(target / "_annotations.coco.json"),
                         "hardlinks_verified": len(loaded.imgs)}
    combined_rows = list(rows)
    for item in synth_coco.dataset["images"] if synth_coco is not None else []:
        row = synth_by_sha[Path(item["file_name"]).stem]
        combined_rows.append(row | {"image_id": mapping[item["id"]], "mixing_authorized": True,
                                    "mixing_authorization": authorization, "training_admitted": True})
    (output / "selection_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in combined_rows), encoding="utf-8")
    write_json(output / "coverage_audit.json", coverage)
    report = {"schema": "fireviewer.pointing-v8-experimental-training-view.v1", "status": "verified_for_authorized_local_run" if execution_target == "local" else "verified_for_authorized_lightning_run",
              "local_training_authorization": authorization if execution_target == "local" else None,
              "training_authorization": authorization, "training_execution_target": execution_target,
              "authorization_sha256": hashlib.sha256(authorization.encode("utf-8")).hexdigest(),
              "authorization_record_sha256": digest(authorization_record) if authorization_record else None, "coverage_qualified": coverage["ready"],
              "acknowledged_limitations": coverage["blocked_gates"], "frozen_holdout_camera_overlap": coverage["cross_split_camera_views"],
              "source_real_images": len(source_rows), "real_images": len(rows), "synthetic_images": len(synthetic), "total_images": len(combined_rows),
              "prelaunch_excluded_real_images": len(source_rows)-len(rows),
              "prelaunch_exclusions": exclusion_record,
              "real_train_images": len(train), "synthetic_fraction_of_train": len(synthetic)/len(merged["images"]),
              "synthetic_sampling": "once_per_epoch_no_oversampling" if synthetic else "none_real_only", "initialization_required": "generic_pretrained_not_v7",
              "real_train_holdout_phash_pairs": len(train)*len(heldout), "full_image_reload_passed": True,
              "source_real_selection_sha256": digest(real_root / "selection_manifest.jsonl"),
              "source_synthetic_manual_sha256": synth_report.get("manual_manifest_sha256"),
              "source_synthetic_coco_sha256": synth_report.get("coco_sha256"), "real_coco_sha256": frozen,
              "splits": splits, "automatic_cleanup": False, "additional_image_payload_bytes": 0,
              "public_release_qualified": False, "independent_external_benchmark_qualified": False}
    write_json(output / "report.json", report)
    write_json(output / "coco_view_receipt.json", report | {
        "dataset_report_sha256": digest(output / "report.json"),
        "selection_manifest_sha256": digest(output / "selection_manifest.jsonl"),
        "dataset_root": str(real_root.resolve()), "output_root": str(output), "categories": CATEGORIES})
    if frozen != {folder: digest(real_root / "coco" / folder / "_annotations.coco.json") for folder in frozen}:
        raise ValueError("Original real COCO changed during preparation")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, help="Optional reviewed supplement; omit for real-only training")
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/local/fireviewer-pointing-v7-ready-local-5000-20260825-r7/selection_manifest.jsonl"))
    parser.add_argument("--fingerprints", type=Path, default=Path("artifacts/local/pointing-v7-split-audit-groupwise-20260827/fingerprints.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--execution-target", choices=("local", "lightning_cpu_then_l4"), default="local")
    parser.add_argument("--authorization-record", type=Path)
    parser.add_argument("--allow-known-coverage-limitations", action="store_true")
    args = parser.parse_args()
    prepare(args.real_root, args.synthetic_root, args.baseline, args.fingerprints,
            args.output, args.authorization, args.allow_known_coverage_limitations, args.exclusions,
            args.execution_target, args.authorization_record)


if __name__ == "__main__":
    main()
