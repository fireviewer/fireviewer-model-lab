from training.pointing_dataset_v8.reconcile_review_evidence import reconcile, source_family


def test_mirrors_and_hyphenated_dfire_are_not_new_sources():
    assert source_family("D-Fire") == "DFire"
    assert source_family("D_Fire") == "DFire"
    assert source_family("WUI-Fire-Detection", "/FINAL_DATASET/1/Day/Longe/Fire/WEB10991.jpg") == "DFire"
    assert source_family("WUI-Fire-Detection", "/FINAL_DATASET/0/Day/Longe/AoF06724.jpg") == "DFire"
    assert source_family("WUI-Fire-Detection", "/FINAL_DATASET/0/Day/Longe/forest.jpg") == "WUI-Fire-Detection"


def sample():
    return {"source_dataset": "fasdd_v9", "objects": {"bbox": [[0, 0, 10, 10]]},
            "negative_verified": True, "person_risk_reviewed_clear": False}


def test_positive_is_not_verified_negative_and_unknown_review_stays_unknown():
    result = reconcile(sample(), [])
    assert not result["is_negative"] and not result["negative_verified"]
    assert result["negative_review_status"] == "not_applicable"
    assert result["review_status"] == "needs_review"
    assert not result["person_risk_reviewed_clear"]
    assert not result["v8_training_admitted"]


def test_person_clearance_requires_explicit_positive_evidence_and_no_conflict():
    accepted = {"decision": "accepted", "zero_human_foreground_confirmed": True}
    assert reconcile(sample(), [accepted])["person_risk_reviewed_clear"]
    conflict = reconcile(sample(), [accepted, {"decision": "reject"}])
    assert conflict["review_status"] == "conflicting_reviews"
    assert not conflict["person_risk_reviewed_clear"]
    assert source_family("FASDD-v9") == source_family("fasdd") == "FASDD"


def test_duplicate_exclusion_does_not_invalidate_original_visual_review():
    result = reconcile(sample(), [{"decision": "accept"}, {"decision": "reject", "reason": "exact_duplicate_of_v5"}])
    assert result["review_status"] == "accepted_existing_review"
    assert source_family("FireBench", "figlib/HPWREN-FIgLib/event/image.jpg") == "HPWREN"
