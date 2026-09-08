import json
from copy import deepcopy

import pytest

from training.pointing_dataset_v8.audit_coverage import POLICY, admitted, audit, coverage, geometries


def row(**kwargs):
    return {"sha256": "a" * 64, "source_dataset": "FireBench", "source_record_id": "HPWREN-FIgLib/e/image.jpg",
            "width": 1000, "height": 1000, "objects": {"bbox": [[10, 20, 20, 20]], "category": [1]},
            "split": "train", "source_group_id": "FireBench:HPWREN-FIgLib/e", **kwargs}


def test_legacy_flag_is_not_v8_admission():
    assert not admitted(row(training_admitted=True, v8_training_admitted=False, review_status="accepted_existing_review"))
    assert not admitted(row(v8_training_admitted=True, review_status="conflicting_reviews"))
    assert admitted(row(v8_training_admitted=True, review_status="accepted_v8_review"))


def test_source_alias_and_unknown_semantics_are_not_diversity_proof():
    result = coverage([row(poor_framing=True, aerial=False)])
    assert result["sources"] == {"HPWREN": 1}
    assert result["verified_physical_events"] == 0
    assert result["semantic_unknown"]["framing_review"] == 1
    assert result["small_box_images_by_class"] == {"smoke": 1}
    assert result["framing_review"] == {}


@pytest.mark.parametrize("objects", [
    {"bbox": [[0, 0, 1, 1]], "category": []},
    {"bbox": [[-1, 0, 1, 1]], "category": [0]},
    {"bbox": [[999, 0, 2, 1]], "category": [0]},
    {"bbox": [[0, 0, float("nan"), 1]], "category": [0]},
    {"bbox": [[0, 0, 1, 1]], "category": [2]},
    {"bbox": [[0, 0, 1, 1]], "category": [True]},
])
def test_invalid_boxes_fail(objects):
    with pytest.raises(ValueError):
        geometries(row(objects=objects))


def test_synthetic_and_duplicate_baseline_are_not_new_cases():
    item = row(v8_training_admitted=True, review_status="accepted_v8_review", synthetic=False)
    synthetic = row(sha256="b" * 64, v8_training_admitted=True, review_status="accepted_v8_review", synthetic=True)
    result = audit([item, synthetic], [deepcopy(item)], json.loads(POLICY.read_text()))
    assert result["new_admitted"]["images"] == 0
    assert not result["ready"]


def test_frozen_evaluation_source_never_enters_train_under_new_hash():
    item = row(v8_training_admitted=True, review_status="accepted_v8_review", synthetic=False)
    history = {"sha256": {}, "source_groups": {"hpwren:e": ["test"]}}
    result = audit([item], [], json.loads(POLICY.read_text()), history)
    assert result["historical_holdout_collisions"] == [item["sha256"]]


def test_camera_leak_is_detected_despite_new_date_and_legacy_alias():
    item = row(v8_training_admitted=True, review_status="accepted_v8_review", synthetic=False,
               source_dataset="Pyro-SDIS", source_record_id="sdis-07_brison-200_2024-08-01T11-00-00.jpg",
               source_group_id="Pyro-SDIS:sdis-07:brison-200:2024-08-01")
    history = {"sha256": {}, "source_groups": {"pyrosdis:sdis-07:brison-200:20240120T160320": ["test"]}}
    result = audit([item], [], json.loads(POLICY.read_text()), history)
    assert result["historical_camera_train_collisions"] == [item["sha256"]]
    assert not next(g for g in result["checks"] if g["gate"] == "historical_camera_views_entering_train")["passed"]


def test_night_negatives_are_not_reported_as_night_detection_positives():
    smoke = row(lighting_review="night", visibility_review="low_visibility")
    negative = row(sha256="b" * 64, lighting_review="night", visibility_review="not_applicable",
                   negative_verified=True, objects={"bbox": [], "category": []})
    result = coverage([smoke, negative])
    assert result["conditions_review"]["night"] == 2
    assert result["scene_condition_slices"]["smoke_only"]["lighting_review"] == {"night": 1}
    assert result["scene_condition_slices"]["negative"]["lighting_review"] == {"night": 1}


def test_night_backgrounds_synthetic_and_holdout_images_cannot_fill_positive_training_gate():
    flags = {"v8_corpus_admitted": True, "review_status": "accepted_v8_review", "synthetic": False,
             "lighting_review": "night"}
    negative = row(**flags, negative_verified=True, objects={"bbox": [], "category": []})
    positive = row(**(flags | {"sha256": "b" * 64, "lighting_review": "low_light"}))
    synthetic = row(**(flags | {"sha256": "c" * 64, "synthetic": True}))
    heldout = row(**(flags | {"sha256": "d" * 64, "split": "validation"}))
    historical = row(**(flags | {"sha256": "e" * 64}))
    policy = json.loads(POLICY.read_text())
    result = audit([negative, positive, synthetic, heldout, historical], [historical], policy)
    gate = next(g for g in result["checks"] if g["gate"] == "new_low_light_positive_images")
    assert gate["actual"] == 1 and gate["missing"] == policy["new_low_light_positive_images_min"] - 1
    without_positive = audit([negative, synthetic, heldout], [], policy)
    assert next(g for g in without_positive["checks"] if g["gate"] == "new_low_light_positive_images")["actual"] == 0


def test_spelling_alias_is_not_an_additional_confuser_family():
    a = row(negative_confuser_verified=True, negative_confuser="sunset_reflection",
            objects={"bbox": [], "category": []})
    b = row(negative_confuser_verified=True, negative_confuser="sunset_reflections",
            objects={"bbox": [], "category": []})
    assert coverage([a, b])["negative_confusers_review"] == {"sunset_reflection": 2}


def test_unknown_incident_is_not_fabricated_and_does_not_invalidate_negative():
    item = row(v8_training_admitted=True, review_status="accepted_v8_review", synthetic=False,
               framing_review="well_framed", obstruction_review="clear", visibility_review="not_applicable",
               lighting_review="daylight", negative_verified=True, objects={"bbox": [], "category": []})
    result = audit([item], [], json.loads(POLICY.read_text()))
    gates = {r["gate"]: r for r in result["checks"]}
    assert gates["admitted_unknown_required_semantics"]["passed"]
    assert not gates["new_reviewed_scene_groups"]["passed"]
    assert result["new_admitted"]["verified_physical_events"] == 0


def test_framing_obstruction_safety_and_source_split_are_enforced():
    item = row(v8_training_admitted=True, review_status="accepted_v8_review", synthetic=False,
               framing_review="poor_but_usable", obstruction_review="partial_usable")
    other = row(sha256="b" * 64, split="test", v8_training_admitted=True, review_status="accepted_v8_review")
    result = audit([item, other], [], json.loads(POLICY.read_text()))
    gates = {r["gate"]: r for r in result["checks"]}
    assert not gates["train_poor_framing_fraction_max"]["passed"]
    assert not gates["train_obstructed_fraction"]["passed"]
    assert not gates["new_unreviewed_safety_or_annotations"]["passed"]
    assert not gates["cross_split_source_groups"]["passed"]


def test_evaluation_is_separate_and_rejects_train_overlap_and_prediction_selection():
    item = row(v8_training_admitted=True, review_status="accepted_v8_review", synthetic=False)
    result = audit([item], [], json.loads(POLICY.read_text()), evaluation=[item])
    gates = {r["gate"]: r for r in result["extra_evaluation"]["checks"]}
    assert not gates["train_overlap"]["passed"]
    assert not gates["unverified_or_prediction_selected_images"]["passed"]
    assert not result["extra_evaluation"]["ready"]


def test_scene_review_is_not_a_claim_of_known_physical_incident():
    item = row(scene_group_verified=True, scene_group_id="camera-and-landmark-cluster-1",
               scene_group_evidence="explicit_visual_and_source_review_record")
    result = coverage([item])
    assert result["reviewed_scene_groups"] == 1
    assert result["verified_physical_events"] == 0
    no_proof = row(scene_group_verified=True, scene_group_id="invented")
    assert coverage([no_proof])["reviewed_scene_groups"] == 0


def test_renaming_split_group_does_not_hide_shared_source_group():
    a = row(v8_corpus_admitted=True, review_status="accepted_v8_review", split_group_id="one")
    b = row(sha256="b" * 64, v8_corpus_admitted=True, review_status="accepted_v8_review",
            split="test", split_group_id="two", v8_training_admitted=False)
    result = audit([a, b], [], json.loads(POLICY.read_text()))
    assert result["cross_split_source_groups"]


def test_unannotated_classification_image_is_not_counted_as_negative():
    item = row(annotation_state="classification_only_not_detection_boxes", objects={"bbox": [], "category": []})
    result = coverage([item])
    assert result["scenes"] == {}
    assert len(result["geometry_errors"]) == 1
