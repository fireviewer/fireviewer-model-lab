import hashlib
import json

import pytest

from training.pointing_dataset_v8.record_extension_review import apply_decision, read_visual_batch


def source_row(state="source_boxes"):
    return {"sha256": "a" * 64, "source_image": "test.jpg", "width": 100, "height": 100,
            "source_family": "WUI-Fire-Detection", "annotation_state": state,
            "objects": {"bbox": [[10, 10, 20, 20]], "category": [1], "area": [400]}}


def decision(action="accept_source_boxes"):
    return {"decision": action, "reason": "Explicit whole-scene and target inspection.",
            "lighting_review": "daylight", "visibility_review": "clear", "framing_review": "well_framed",
            "obstruction_review": "clear", "scene_group_id": "scene-a", "scene_group_evidence": "same visible hillside",
            "scene_and_all_targets_checked": True}


PACKET = {"packet": "page.png", "sha256": "b" * 64}


def test_explicit_review_date_keeps_new_and_historical_evidence_distinct():
    reviewed = decision()
    assert apply_decision(source_row(), reviewed, PACKET)["visual_review"]["review_date"] == "2026-08-27"
    reviewed["review_date"] = "2026-08-28"
    assert apply_decision(source_row(), reviewed, PACKET)["visual_review"]["review_date"] == "2026-08-28"


@pytest.mark.parametrize("state", ["classification_only_not_detection_boxes", "model_proposals_NOT_ground_truth"])
def test_source_acceptance_refuses_unreviewed_model_or_classification_labels(state):
    with pytest.raises(ValueError, match="No source detection"):
        apply_decision(source_row(state), decision(), PACKET)


def test_proposal_acceptance_binds_objects_and_model():
    row = source_row("model_proposals_NOT_ground_truth")
    row["annotation_proposal_model_sha256"] = "c" * 64
    row["proposal_objects_sha256"] = hashlib.sha256(json.dumps(row["objects"], sort_keys=True).encode()).hexdigest()
    reviewed = decision("accept_reviewed_proposal")
    with pytest.raises(ValueError, match="explicitly bound"):
        apply_decision(row, reviewed, PACKET)
    reviewed.update({key: row[key] for key in ("proposal_objects_sha256", "annotation_proposal_model_sha256")})
    accepted = apply_decision(row, reviewed, PACKET)
    assert accepted["v8_corpus_admitted"] is True
    assert accepted["annotation_proposal_visually_reviewed"] is True
    assert row["annotation_state"] == "model_proposals_NOT_ground_truth"
    reviewed["proposal_objects_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        apply_decision(row, reviewed, PACKET)


def test_negative_is_not_inferred_from_empty_proposals():
    row = source_row("model_proposals_NOT_ground_truth")
    row["objects"] = {"bbox": [], "category": [], "area": []}
    with pytest.raises(ValueError):
        apply_decision(row, decision("accept_reviewed_proposal"), PACKET)
    reviewed = decision("accept_negative")
    reviewed["negative_confuser"] = "cloud_fog"
    assert apply_decision(row, reviewed, PACKET)["negative_verified"] is True


@pytest.mark.parametrize("field", ["lighting_review", "scene_group_id", "scene_group_evidence", "scene_and_all_targets_checked"])
def test_review_evidence_must_be_explicit(field):
    reviewed = decision()
    del reviewed[field]
    with pytest.raises(ValueError):
        apply_decision(source_row(), reviewed, PACKET)


def test_corrected_requires_overlay():
    with pytest.raises(ValueError, match="actually inspected"):
        apply_decision(source_row(), decision("accept_corrected"), PACKET)


def test_cannot_silently_discard_positive_boxes():
    with pytest.raises(ValueError, match="discard positive"):
        apply_decision(source_row(), decision("accept_negative"), PACKET)


def test_renderer_three_zoom_limit_requires_extra_review():
    row = source_row()
    row["objects"] = {"bbox": [[10, 10, 2, 2]] * 4, "category": [1] * 4, "area": [4] * 4}
    with pytest.raises(ValueError, match="More than three"):
        apply_decision(row, decision(), PACKET)


def test_no_implicit_batch_reviews(tmp_path):
    path = tmp_path / "reviews.tsv"
    path.write_text("1|N|D|N|W|C|hill|cloud_fog|Clouds, no visible targets.\n", encoding="utf-8")
    mapped = {i: source_row() for i in (1, 2, 3)}
    result = read_visual_batch(path, mapped)
    assert [r["review_index"] for r in result] == [1]
    path.write_text("1-3|N|D|N|W|C|hill|cloud_fog|Not a real individual review.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_visual_batch(path, mapped)


@pytest.mark.parametrize("field,value", [("framing_review", "too_tight"), ("framing_review", "unusable"),
                                        ("obstruction_review", "blocked"), ("visibility_review", "not_applicable")])
def test_unsuitable_review_cannot_be_admitted(field, value):
    reviewed = decision()
    reviewed[field] = value
    with pytest.raises(ValueError):
        apply_decision(source_row(), reviewed, PACKET)


def test_synthetic_is_not_real_extension():
    row = source_row()
    row["synthetic"] = True
    with pytest.raises(ValueError, match="Synthetic"):
        apply_decision(row, decision(), PACKET)
