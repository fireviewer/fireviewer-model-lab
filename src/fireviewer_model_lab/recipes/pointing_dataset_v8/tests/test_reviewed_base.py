from copy import deepcopy

import pytest

from training.pointing_dataset_v8.audit_coverage import admitted
from training.pointing_dataset_v8.build_reviewed_base import select


def sample(**kwargs):
    return {"sha256": "a" * 64, "width": 100, "height": 100,
            "objects": {"bbox": [[2, 3, 4, 5]], "category": [1]},
            "split": "train", "source_group_id": "dfire:group1",
            "review_status": "accepted_existing_review", "annotation_exploitable": True,
            "person_risk_reviewed_clear": True, **kwargs}


def test_reuse_keeps_only_audited_review_consistent_and_cleared_images():
    good = sample()
    conflict = sample(sha256="b" * 64, review_status="conflicting_reviews")
    person = sample(sha256="c" * 64, person_risk_reviewed_clear=False)
    retired = sample(sha256="d" * 64)
    selected, excluded = select([good, conflict, person, retired], [good, conflict, person])
    assert selected == [good]
    assert len(excluded) == 3


def test_reuse_rejects_historically_mixed_group_even_when_image_was_train():
    item = sample()
    selected, excluded = select([item], [item], {"source_groups": {"dfire:group1": ["train", "test"]}})
    assert not selected
    assert excluded[0]["reason"] == "historically_mixed_split_group"


def test_reuse_cannot_change_annotated_boxes_or_splits():
    original = sample()
    changed = deepcopy(original)
    changed["objects"]["bbox"][0][0] = 6
    with pytest.raises(ValueError, match="silent annotation"):
        select([changed], [original])


def test_corpus_admission_does_not_turn_frozen_test_into_training():
    item = sample(split="test", v8_corpus_admitted=True, v8_training_admitted=False)
    assert admitted(item)
    assert item["v8_training_admitted"] is False
