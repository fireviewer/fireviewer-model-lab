import json

from training.pointing_dataset_v7.adjudicate_local_hard_negatives import ACCEPT, REVIEWED
from training.pointing_dataset_v7.adjudicate_nemo_final_expansion import REJECT
from training.pointing_dataset_v7.build_v7_expanded_ready import accepted_by_id, normalize_candidate


def test_review_ledgers_are_bounded_and_fail_closed() -> None:
    assert len(ACCEPT) == 72
    assert ACCEPT <= REVIEWED == set(range(1, 301))
    assert len(REJECT) == 297
    assert REJECT <= set(range(1, 615))


def test_only_explicit_accept_decisions_are_admitted(tmp_path) -> None:
    decisions = [
        {"candidate_id": "accepted", "decision": "accept"},
        {"candidate_id": "rejected", "decision": "reject"},
        {"candidate_id": "pending", "decision": "not_reviewed"},
    ]
    (tmp_path / "review_decisions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8"
    )
    assert accepted_by_id(tmp_path) == {"accepted"}


def test_reviewed_empty_annotation_sample_is_a_verified_negative() -> None:
    raw = {
        "candidate_id": "negative-1",
        "sha256": "0" * 64,
        "width": 640,
        "height": 480,
        "annotations": [],
        "source_record_id": "record-1",
        "split": "train",
        "split_group": "group-1",
        "license": "CC-BY-SA-4.0",
        "gap_bucket": "hard_negative",
        "source_image": "unused.jpg",
    }
    row = normalize_candidate(raw, "reviewed", "source", "revision", "manual_review")
    assert row["objects"] == {"bbox": [], "category": [], "area": []}
    assert row["negative_verified"] is True
    assert row["training_admitted"] is True
