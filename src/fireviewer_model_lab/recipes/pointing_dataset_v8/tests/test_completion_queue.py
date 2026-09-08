import json

from training.pointing_dataset_v8.prepare_completion_queue import collect_candidates


def test_prior_rejection_and_existing_admission_are_not_reintroduced(tmp_path):
    rows = [{"sha256": str(i), "candidate_id": f"c{i}", "review_index": i,
             "license": "CC-BY-4.0", "source_family": "WUI-Fire-Detection",
             "objects": {"bbox": [], "category": []}} for i in range(1, 5)]
    (tmp_path / "review_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    decisions = [{"sha256": "1", "visual_review": {"decision": {"decision": "reject"}}},
                 {"sha256": "2", "visual_review": {"decision": {"decision": "needs_annotation"}}},
                 {"sha256": "3", "visual_review": {"decision": {"decision": "accept_corrected"}}}]
    (tmp_path / "curation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decisions))
    kept, excluded = collect_candidates([tmp_path], [{"sha256": "4"}], {})
    assert [row["sha256"] for row in kept] == ["2"]
    assert not kept[0]["v8_corpus_admitted"]
    assert not kept[0]["training_admitted"]
    assert {row["reason"] for row in excluded} == {
        "prior_explicit_rejection_not_overridden", "already_reviewed_acceptance_or_global_exclusion", "already_in_active_corpus"}
