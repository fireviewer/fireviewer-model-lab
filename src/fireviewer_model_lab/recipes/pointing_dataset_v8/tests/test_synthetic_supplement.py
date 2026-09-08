import copy

import pytest

from training.pointing_dataset_v8.prepare_synthetic_supplement import (
    check_synthetic_diversity, complete_real_fingerprints, minimum_real_distance,
    summarize_coverage, validate_overlay_receipt, validate_record,
)


def valid_row():
    return {"synthetic": True, "split": "train", "origin": "codex-image-gen", "generation_context_id": "context-1",
            "prompt": "Small visible flames at night", "counts_as_new_real_case": False,
            "whole_scene_and_all_targets_reviewed": True, "annotations_exploitable": True,
            "safe_ground_view_reviewed": True, "no_selfie_or_dominant_person_reviewed": True,
            "corrected_overlay_verified": True, "visual_review_note": "Flames and full smoke plume inspected",
            "corrected_overlay_sha256": "a"*64, "synthetic_scene_group_id": "synthetic:scene-1",
            "reference_real_image_sha256": [], "sha256": "b"*64, "width": 100, "height": 80,
            "lighting_review": "night", "visibility_review": "low_visibility",
            "framing_review": "well_framed", "obstruction_review": "clear",
            "objects": {"bbox": [[10, 20, 5, 10]], "category": [0], "area": [50]}}


def test_reviewed_text_only_generation_is_train_only():
    assert validate_record(valid_row(), {}) == [([10, 20, 5, 10], 0)]


@pytest.mark.parametrize("change", [{"split": "test"}, {"split": "validation"}, {"synthetic": False},
    {"counts_as_new_real_case": True}, {"origin": "unknown"}, {"whole_scene_and_all_targets_reviewed": False},
    {"reference_real_image_sha256": None}, {"corrected_overlay_verified": False}])
def test_missing_provenance_or_review_never_admitted(change):
    with pytest.raises(ValueError):
        validate_record(valid_row() | change, {})


def test_real_holdout_references_forbidden():
    row = valid_row() | {"reference_real_image_sha256": ["heldout"]}
    with pytest.raises(ValueError, match="outside"):
        validate_record(row, {"heldout": {"split": "test"}})
    with pytest.raises(ValueError, match="outside"):
        validate_record(row, {})
    validate_record(row, {"heldout": {"split": "train"}})


def test_no_real_relabel_and_geometry_checked():
    row = valid_row()
    with pytest.raises(ValueError, match="relabelled"):
        validate_record(row, {row["sha256"]: {"split": "train"}})
    row["objects"]["bbox"] = [[98, 20, 5, 10]]
    with pytest.raises(ValueError):
        validate_record(row, {})


def test_inherited_images_are_checked_using_historical_fingerprints():
    historical = {"sha256": "a"*64, "phash": "0"*16, "phash_flipped": "f"*16}
    inline = {"sha256": "b"*64, "phash": "1"*16, "phash_flipped": "e"*16}
    result = complete_real_fingerprints([{"sha256": historical["sha256"]}, inline], [historical])
    assert result == [(0, int("f"*16, 16)), (int("1"*16, 16), int("e"*16, 16))]


@pytest.mark.parametrize("historical", [[], [{"sha256": "a"*64, "phash": "0"*16}],
    [{"sha256": "a"*64, "phash": "0", "phash_flipped": "f"*16}]])
def test_missing_real_fingerprints_fail_closed(historical):
    with pytest.raises(ValueError, match="complete fingerprint coverage"):
        complete_real_fingerprints([{"sha256": "a"*64}], historical)


def test_conflicting_real_fingerprints_refused():
    row = {"sha256": "a"*64, "phash": "0"*16, "phash_flipped": "f"*16}
    changed = row | {"phash": "1"*16}
    with pytest.raises(ValueError, match="Conflicting"):
        complete_real_fingerprints([row], [row, changed])
    with pytest.raises(ValueError, match="disagree"):
        complete_real_fingerprints([row], [changed])


@pytest.mark.parametrize("field", ["lighting_review", "visibility_review", "framing_review", "obstruction_review"])
def test_scene_coverage_cannot_be_inferred_from_prompt(field):
    row = valid_row()
    row[field] = "unknown"
    with pytest.raises(ValueError, match="scene metadata"):
        validate_record(row, {})


def test_synthetic_mirror_cannot_be_a_new_context():
    a = {"synthetic_scene_group_id": "a", "phash": "0"*16, "phash_flipped": "f"*16}
    b = {"synthetic_scene_group_id": "b", "phash": "f"*16, "phash_flipped": "0"*16}
    with pytest.raises(ValueError, match="near-duplicate"):
        check_synthetic_diversity([a, b])


def test_distinct_images_and_parent_scene_cap():
    a = {"synthetic_scene_group_id": "a", "phash": "0"*16, "phash_flipped": "f"*16}
    b = {"synthetic_scene_group_id": "b", "phash": "a"*16, "phash_flipped": "5"*16}
    assert check_synthetic_diversity([a, b]) == {
        "pairs_checked": 1, "minimum_phash_distance": 32, "exclusion_distance_max": 4, "scene_groups": 2}
    with pytest.raises(ValueError, match="eight"):
        check_synthetic_diversity([a]*9)


def test_actual_geometry_not_prompt_drives_small_target_counts():
    a = valid_row() | {"width": 1600, "height": 900,
                       "objects": {"bbox": [[30, 40, 25, 8], [30, 20, 100, 200]], "category": [0, 1]}}
    b = valid_row() | {"lighting_review": "daylight", "width": 1600, "height": 900,
                       "objects": {"bbox": [[30, 40, 20, 12], [30, 20, 40, 60]], "category": [0, 1]}}
    report = summarize_coverage([a, b])
    assert report["small_fire_images"] == 2
    assert report["small_smoke_images"] == 1
    assert report["lighting_review"] == {"daylight": 1, "night": 1}
    assert report["fire_images_with_target_min_side_below_4px_at_long_edge"] == {"640": 1, "960": 0}


def test_both_mirrored_orientations_are_compared_against_real_images():
    assert minimum_real_distance(int("a"*16, 16), int("f"*16, 16),
                                 [(0, int("f"*16, 16))]) == 0


@pytest.mark.parametrize("change", [{}, {"image_sha256": "wrong"},
    {"corrected_overlay_sha256": "wrong"}, {"corrected_overlay": "another.png"},
    {"corrected_objects": {"bbox": [[11, 20, 5, 10]], "category": [0]}}])
def test_export_boxes_are_bound_to_the_inspected_overlay_receipt(tmp_path, change):
    row = valid_row() | {"corrected_overlay": str(tmp_path / "overlay.png")}
    receipt = {"image_sha256": row["sha256"], "corrected_overlay": row["corrected_overlay"],
               "corrected_overlay_sha256": row["corrected_overlay_sha256"],
               "corrected_objects": copy.deepcopy(row["objects"])}
    if change:
        with pytest.raises(ValueError):
            validate_overlay_receipt(row, receipt | change)
    else:
        validate_overlay_receipt(row, receipt)


def test_missing_annotation_receipt_is_not_accepted():
    with pytest.raises(ValueError, match="receipt"):
        validate_overlay_receipt(valid_row(), None)
