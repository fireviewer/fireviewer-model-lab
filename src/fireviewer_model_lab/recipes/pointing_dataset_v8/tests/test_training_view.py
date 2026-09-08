import copy
import json
import os
import random
from pathlib import Path

import pytest
import yaml
import imagehash
from PIL import Image, ImageOps

from training.pointing_dataset_v8.prepare_training_view import (
    CATEGORIES, apply_exclusions, check_limitations, merge_train_coco,
    digest, prepare, validate_authorization, write_json,
)


def test_known_volume_limit_requires_acknowledgement():
    report = {"blocked_gates": ["admitted_images"]}
    with pytest.raises(ValueError, match="acknowledgement"):
        check_limitations(report)
    check_limitations(report, acknowledge=True)


@pytest.mark.parametrize("gate", ["new_low_light_positive_images", "train_poor_framing_fraction_max",
                                 "train_obstructed_fraction"])
def test_newly_acknowledged_coverage_limits_still_need_consent(gate):
    report = {"blocked_gates": [gate]}
    with pytest.raises(ValueError, match="acknowledgement"):
        check_limitations(report)
    check_limitations(report, acknowledge=True)


@pytest.mark.parametrize("change", [{"training_authorized": False},
    {"source_selection_manifest_sha256": "other"}, {"execution_target": "local"},
    {"acknowledged_limitations": []}, {"user_request": ""}, {"user_confirmation": ""}])
def test_authorization_binds_exact_corpus_target_and_limitations(change):
    record = {"training_authorized": True, "source_selection_manifest_sha256": "sealed",
              "execution_target": "lightning_cpu_then_l4", "acknowledged_limitations": ["admitted_images"],
              "user_request": "launch", "user_confirmation": "yes"}
    validate_authorization(record, "sealed", ["admitted_images"], "lightning_cpu_then_l4")
    with pytest.raises(ValueError, match="authorization"):
        validate_authorization(record | change, "sealed", ["admitted_images"], "lightning_cpu_then_l4")


def test_real_only_view_reloads_pixels_and_preserves_coco_bytes(tmp_path, monkeypatch):
    from training.pointing_dataset_v8 import prepare_training_view as module
    real, output = tmp_path / "real", tmp_path / "view"
    real.mkdir()
    rows, frozen = [], {}
    for index, (split, folder) in enumerate(module.FOLDERS.items(), 1):
        source = real / f"source{index}.png"
        Image.frombytes("RGB", (100, 100), random.Random(index).randbytes(30000)).save(source)
        sha = digest(source)
        with Image.open(source) as image:
            phash, flipped = str(imagehash.phash(image)), str(imagehash.phash(ImageOps.mirror(image)))
        target = real / "coco" / folder
        (target / "images").mkdir(parents=True)
        os.link(source, target / "images" / (sha + ".png"))
        rows.append({"sha256": sha, "image_id": index, "source_image": str(source),
                     "width": 100, "height": 100, "split": split, "objects": {"bbox": [], "category": []},
                     "phash": phash, "phash_flipped": flipped})
        write_json(target / "_annotations.coco.json", coco("images/" + sha + ".png", index, False))
        frozen[folder] = (target / "_annotations.coco.json").read_bytes()
    manifest = real / "selection_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    baseline, fingerprints = tmp_path / "baseline.jsonl", tmp_path / "fingerprints.jsonl"
    baseline.write_text("")
    fingerprints.write_text("")
    write_json(real / "historical_split_registry.json", {})
    write_json(real / "coverage_report.json", {"inputs": {"known_fingerprints_sha256": digest(fingerprints)}})
    write_json(real / "assembly_receipt.json", {"artifact_hashes": {"selection_manifest.jsonl": digest(manifest)}})
    coverage = {"ready": False, "blocked_gates": ["admitted_images"], "cross_split_camera_views": {}}
    monkeypatch.setattr(module, "audit", lambda *args: coverage)
    authorization = tmp_path / "authorization.json"
    write_json(authorization, {"training_authorized": True, "source_selection_manifest_sha256": digest(manifest),
        "execution_target": "lightning_cpu_then_l4", "acknowledged_limitations": coverage["blocked_gates"],
        "user_request": "launch", "user_confirmation": "yes"})
    report = prepare(real, None, baseline, fingerprints, output, "confirmed launch", acknowledge=True,
                     execution_target="lightning_cpu_then_l4", authorization_record=authorization)
    assert report["real_images"] == report["total_images"] == 3
    assert report["synthetic_images"] == 0
    assert report["coverage_qualified"] is False
    assert report["full_image_reload_passed"] is True
    assert report["authorization_record_sha256"] == digest(authorization)
    receipt = json.loads((output / "coco_view_receipt.json").read_text())
    assert receipt["dataset_report_sha256"] == digest(output / "report.json")
    for folder, original in frozen.items():
        assert (output / folder / "_annotations.coco.json").read_bytes() == original
        assert (real / "coco" / folder / "_annotations.coco.json").read_bytes() == original


@pytest.mark.parametrize("gate", ["invalid_annotation_images", "historical_holdouts_entering_train",
                                 "historical_camera_views_entering_train", "duplicate_image_identities"])
def test_integrity_failures_cannot_be_waived(gate):
    with pytest.raises(ValueError, match="Non-waivable"):
        check_limitations({"blocked_gates": [gate]}, acknowledge=True)


def test_train_camera_leak_cannot_hide_in_known_camera_warning():
    with pytest.raises(ValueError, match="training camera"):
        check_limitations({"blocked_gates": ["cross_split_camera_views"],
                           "cross_split_camera_views": {"camera": ["train", "test"]}}, acknowledge=True)
    check_limitations({"blocked_gates": ["cross_split_camera_views"],
                       "cross_split_camera_views": {"camera": ["validation", "test"]}}, acknowledge=True)


def coco(filename="real.jpg", image_id=100, annotations=True):
    return {"images": [{"id": image_id, "file_name": filename, "width": 100, "height": 100}],
            "annotations": [{"id": 12, "image_id": image_id, "category_id": 0,
                             "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}] if annotations else [],
            "categories": copy.deepcopy(CATEGORIES)}


def test_merge_reindexes_synthetic_preserves_negative_and_inputs():
    real, synth = coco("images/real.jpg", annotations=False), coco("synth.png", 1)
    frozen = copy.deepcopy((real, synth))
    merged, mapping = merge_train_coco(real, synth)
    assert (real, synth) == frozen
    assert len(merged["images"]) == 2 and len(merged["annotations"]) == 1
    assert mapping == {1: 101}
    assert merged["images"][1]["file_name"] == "images/synth.png"
    assert merged["images"][1]["synthetic"] is True
    assert merged["annotations"][0]["image_id"] == 101
    assert merged["annotations"][0]["bbox"] == [1, 2, 3, 4]


def test_merge_refuses_duplicate_pixels_by_filename():
    with pytest.raises(ValueError, match="Duplicate"):
        merge_train_coco(coco("images/same.png"), coco("same.png", 1))


def test_merge_refuses_wrong_categories():
    synth = coco("synth.png", 1)
    synth["categories"][0]["name"] = "person"
    with pytest.raises(ValueError, match="category"):
        merge_train_coco(coco(), synth)


@pytest.mark.parametrize("change", [{"sha256": "missing"}, {"reason": ""}, {"source_group_id": "wrong"}])
def test_exclusions_are_bound_to_present_reviewed_training_identity(change):
    rows = [{"sha256": "train", "split": "train", "source_group_id": "camera"}]
    decision = {"sha256": "train", "source_group_id": "camera", "reason": "heldout camera"} | change
    with pytest.raises(ValueError):
        apply_exclusions(rows, [decision])


def test_exclusions_never_remove_holdouts_or_mutate_originals():
    rows = [{"sha256": "train", "split": "train", "source_group_id": "camera"},
            {"sha256": "test", "split": "test", "source_group_id": "camera"}]
    decision = {"sha256": "train", "source_group_id": "camera", "reason": "heldout camera"}
    frozen = copy.deepcopy(rows)
    assert apply_exclusions(rows, [decision]) == [rows[1]]
    assert rows == frozen
    with pytest.raises(ValueError, match="training image"):
        apply_exclusions(rows, [decision | {"sha256": "test"}])


def test_v8_explicitly_disables_tiny_target_destroying_augmentations():
    path = Path(__file__).resolve().parents[2] / "vendor/DEIM/configs/fireviewer/deim_dfine_l_pointing_v8.yml"
    config = yaml.safe_load(path.read_text())
    loader = config["train_dataloader"]
    names = {op["type"] for op in loader["dataset"]["transforms"]["ops"]}
    assert not names & {"Mosaic", "RandomZoomOut", "RandomIoUCrop"}
    assert loader["collate_fn"]["base_size_repeat"] is None
    assert loader["collate_fn"]["mixup_prob"] == 0
    assert config["eval_spatial_size"] == [960, 960]
    assert loader["drop_last"] is False
    assert config["save_last_every_epoch"] is True
