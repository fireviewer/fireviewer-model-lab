import json

from fireviewer_model_lab.training.multinatsmoke_strict_resplit import strict_resplit


def test_strict_resplit_deduplicates_and_never_promotes_early(tmp_path):
    rows = [
        {"sample_id": "a", "source_record_id": "a", "image_sha256": "1" * 64, "mask_sha256": "a" * 64, "image_phash64": "0000000000000000", "errors": [], "base_candidate": {"valid": True, "point_x": 0.4, "point_y": 0.8}},
        {"sample_id": "b", "source_record_id": "b", "image_sha256": "2" * 64, "mask_sha256": "b" * 64, "image_phash64": "0000000000000001", "errors": [], "base_candidate": {"valid": True, "point_x": 0.5, "point_y": 0.9}},
        {"sample_id": "bad", "source_record_id": "bad", "image_sha256": "3" * 64, "mask_sha256": "c" * 64, "image_phash64": "ffffffffffffffff", "errors": ["empty_or_full_mask"], "base_candidate": {"valid": False}},
    ]
    audit = tmp_path / "audit.jsonl"
    audit.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    rights = tmp_path / "rights.json"
    rights.write_text(json.dumps({"source_family": "D-Fire", "commercial_use_allowed": True, "immutable_source_revision": "license-sha"}), encoding="utf-8")

    report = strict_resplit(audit_path=audit, rights_receipt=rights, output_dir=tmp_path / "out")

    assert report["audited_rows"] == 3
    assert report["payload_valid_rows"] == 2
    assert report["strict_representative_rows"] == 1
    assert report["cross_split_group_leaks"] == 0
    output = json.loads((tmp_path / "out/dfire_strict_resplit.jsonl").read_text())
    assert output["training_eligible"] is False
