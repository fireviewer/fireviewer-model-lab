import json
import pytest
from training.pointing_dataset_v7.split_registry import assign_locked, assert_frozen_holdouts_retained, digest, make_registry, verify_coco


def row(n, group, split="train"):
    return {"sha256": f"{n:064x}", "split_group_id": group, "split": split}


def test_frozen_partitions_survive_extension_metadata_and_row_order():
    base = [row(1, "train"), row(2, "valid", "validation"), row(3, "test", "test")]
    registry = make_registry(base)
    extension = [row(i, f"new-{i}") for i in range(4, 100)]
    result = assign_locked(list(reversed(base + extension)), registry)
    assert all(result[r["split_group_id"]] == r["split"] for r in base)
    assert assign_locked(extension) == {k: v for k, v in result.items() if k.startswith("new-")}


def test_same_bytes_in_renamed_group_cannot_enter_training():
    registry = make_registry([row(1, "original", "test")])
    assert assign_locked([row(1, "renamed"), row(2, "renamed")], registry, new_split="train") == {"renamed": "test"}


def test_conflicting_group_and_image_locks_fail_closed():
    registry = make_registry([row(1, "a", "train"), row(2, "b", "test")])
    with pytest.raises(ValueError, match="Conflicting"):
        assign_locked([row(1, "b")], registry)
    with pytest.raises(ValueError, match="Conflicting"):
        make_registry([row(1, "a", "test")], registry)


def test_removed_training_exposure_remains_locked():
    registry = make_registry([row(1, "old"), row(2, "test", "test")])
    updated = make_registry([row(2, "test", "test")], registry)
    assert assign_locked([row(1, "recovered")], updated) == {"recovered": "train"}
    with pytest.raises(ValueError, match="trim frozen"):
        assert_frozen_holdouts_retained([row(1, "old")], registry)


def test_hash_linked_new_groups_share_assignment():
    rows = [row(1, "a"), row(1, "b"), row(2, "b"), row(2, "c")]
    result = assign_locked(rows)
    assert result == assign_locked(list(reversed(rows)))
    assert len(set(result.values())) == 1


def test_source_group_cannot_bypass_split_group_lock():
    base = row(1, "split-a", "test") | {"source_group_id": "event-a"}
    registry = make_registry([base])
    new = row(2, "split-b") | {"source_group_id": "event-a"}
    assert assign_locked([new], registry, new_split="train") == {"split-b": "test"}


def test_repackaged_hpwren_event_is_not_a_new_source_group():
    base = row(1, "split-a", "test") | {"source_group_id": "hpwren-figlib:20180504_FIRE_rm-n-mobo-c"}
    new = row(2, "split-b") | {"source_group_id": "firebench:sequence:figlib/hpwren-figlib/20180504_fire_rm-n-mobo-c"}
    assert assign_locked([new], make_registry([base]), new_split="train") == {"split-b": "test"}


def test_coco_gate_checks_actual_bytes_and_original_source_groups(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "selection_manifest.jsonl").write_text("manifest")
    (dataset / "report.json").write_text("report")
    root = tmp_path / "coco"
    samples, splits = [], {}
    for index, split in enumerate(("train", "validation", "test")):
        folder = root / ("valid" if split == "validation" else split)
        folder.mkdir(parents=True)
        image = folder / "image.bin"
        image.write_bytes(bytes([index]))
        sha = digest(image)
        samples.append(row(index, f"split-{index}", split) | {"sha256": sha})
        coco = {"categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}], "annotations": [],
                "images": [{"id": index, "file_name": image.name, "fireviewer_sha256": sha,
                            "fireviewer_source_group_id": "shared-event"}]}
        annotation = folder / "_annotations.coco.json"
        annotation.write_text(json.dumps(coco))
        splits[split] = {"images": 1, "annotations": 0, "annotation_sha256": digest(annotation)}
    (root / "coco_view_receipt.json").write_text(json.dumps({"status": "ready", "dataset_root": str(dataset),
        "selection_manifest_sha256": digest(dataset / "selection_manifest.jsonl"),
        "dataset_report_sha256": digest(dataset / "report.json"), "splits": splits}))
    registry = make_registry(samples)
    result = verify_coco(root, registry)
    assert result["status"] == "identity_verified_source_group_conflicts"
    assert result["source_group_conflicts"]["shared-event"] == ["test", "train", "validation"]
    (root / "test" / "image.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image identity mismatch"):
        verify_coco(root, registry)
