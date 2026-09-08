import json

import pytest

from training.pointing_dataset_v8.prepare_extension_review import deduplicate, load_candidates


def test_existing_manifest_is_reused_without_admission_or_rewriting(tmp_path):
    candidate = {"candidate_id": "existing-1", "source_image": "original.jpg",
                 "objects": {"bbox": [], "category": []}, "negative_verified": False,
                 "v8_training_admitted": False, "review_status": "needs_visual_review"}
    path = tmp_path / "candidate_manifest.jsonl"
    payload = json.dumps(candidate) + "\n"
    path.write_text(payload, encoding="utf-8")
    assert load_candidates(tmp_path) == [candidate]
    assert path.read_text(encoding="utf-8") == payload
    assert not (tmp_path / "receipts").exists()


def test_receipts_remain_authoritative_when_present(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "one.json").write_text('{"candidate_id":"receipt"}')
    (tmp_path / "candidate_manifest.jsonl").write_text('{"candidate_id":"stale"}\n')
    assert load_candidates(tmp_path) == [{"candidate_id": "receipt"}]


def test_missing_acquisition_is_not_silently_an_empty_review(tmp_path):
    with pytest.raises(ValueError, match="No acquisition"):
        load_candidates(tmp_path)


def test_mirrored_to_mirrored_near_overlap_is_not_queued_again():
    candidate = {"candidate_id": "new", "sha256": "new-sha", "phash": "aaaaaaaa55555555",
                 "phash_flipped": "ffffffff00000000"}
    known = {"sha256": "old-sha", "phash": "cccc3333cccc3333", "phash_flipped": "ffffffff00000000"}
    added, excluded = deduplicate([candidate], [known], [])
    assert not added and excluded[0]["matching_sha256"] == "old-sha"
