import hashlib
import json

import pytest

from training.pointing_dataset_v8 import acquire_fasdd as module


def raw(**changes):
    return {"source_id": "fasdd_v9", "split": "train", "source_record_id": "CV:train:fire_123.jpg",
            "license": "CC-BY-SA-4.0", "sha256": "a" * 64, "sample_id": "sample",
            "width": 1280, "height": 720, "phash": "a" * 16,
            "split_group": "fasdd-v9:cv:fire:0012",
            "annotations": [{"class_name": "fire", "bbox_xywh": [500, 300, 20, 30]}], **changes}


@pytest.mark.parametrize("change", [
    {"split": "test"}, {"split": "validation"}, {"source_record_id": "CV:test:fire_123.jpg"},
    {"source_record_id": "RS:train:fire.tif"}, {"near_duplicate_of": "other"},
    {"license": "unknown"}, {"annotations": []}, {"width": 200},
])
def test_source_split_and_permission_filters_precede_images(change):
    assert module.source_candidate(raw(**change), 7) is None


def test_fire_and_smoke_labels_preserve_source_geometry_without_admission():
    row = module.source_candidate(raw(annotations=[
        {"class_name": "flame_visible", "class_id": 1, "bbox_xywh": [500, 300, 20, 30]},
        {"class_name": "smoke_visible", "class_id": 0, "bbox_xywh": [490, 240, 40, 80]},
    ]), 37)
    assert row["row_index"] == 37
    assert row["objects"]["category"] == [0, 1]
    assert row["selection_hint_small_fire"] and row["selection_hint_small_smoke"]
    assert row["objects"]["bbox"][1] == [490, 240, 40, 80]
    assert "v8_corpus_admitted" not in row


def test_invalid_boxes_are_not_sanitized_into_acceptance():
    with pytest.raises(ValueError, match="bbox"):
        module.source_candidate(raw(annotations=[{"class_name": "smoke", "bbox_xywh": [-4, 0, 20, 30]}]), 0)


def test_selection_excludes_known_images_records_and_holdout_groups():
    rows = [module.source_candidate(raw(sha256=str(n) * 64, source_record_id=f"CV:train:fire_{n}.jpg", split_group=f"group{n}"), n) for n in range(5)]
    result, excluded = module.select(rows, {rows[0]["sha256"]}, {rows[1]["source_record_id"]}, {"group2"}, 100)
    assert {r["row_index"] for r in result} == {3, 4}
    assert excluded == {"already_known_image_or_review": 2, "historical_holdout_group": 1}


def test_large_fire_front_with_tiny_secondary_box_is_not_priority():
    row = module.source_candidate(raw(annotations=[
        {"class_name": "fire", "bbox_xywh": [500, 300, 20, 30]},
        {"class_name": "fire", "bbox_xywh": [100, 100, 400, 300]},
    ]), 0)
    assert row["selection_hint_small_fire"]
    selected, exclusions = module.select([row], set(), set(), set(), 100)
    assert selected == []
    assert exclusions["not_an_entirely_small_fire_scene_hint"] == 1


def test_queue_plan_uses_original_identity_for_viewer_representation():
    from training.pointing_dataset_v8.prepare_extension_review import filter_selection_plan
    original = {"sha256": "original"}
    candidate = {"sha256": "jpeg", "source_original_sha256": "original", "v8_corpus_admitted": False}
    assert filter_selection_plan([candidate, {"sha256": "unplanned"}], [original]) == [candidate]


def test_metadata_stream_hash_and_row_positions_include_non_candidate_train_rows(tmp_path, monkeypatch):
    rows = [raw(split="validation"), raw(source_record_id="RS:train:fire.tif"), raw(), raw(split="test")]
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    monkeypatch.setattr(module, "MANIFEST_BYTES", len(payload))
    monkeypatch.setattr(module, "MANIFEST_SHA256", hashlib.sha256(payload).hexdigest())

    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size):
            yield payload[:33]
            yield payload[33:]

    class Session:
        def get(self, *args, **kwargs): return Response()

    monkeypatch.setattr(module, "session", Session)
    result = module.metadata(tmp_path)
    assert len(result) == 1 and result[0]["row_index"] == 1
    assert module.metadata(tmp_path) == result
    assert {p.name for p in tmp_path.iterdir()} == {"eligible_metadata.jsonl", "metadata_receipt.json"}
    (tmp_path / "eligible_metadata.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="identity"):
        module.metadata(tmp_path)


def test_viewer_mapping_is_bound_and_cached_without_persisting_signed_urls(monkeypatch):
    row = module.source_candidate(raw(), 7)
    calls = []
    def page(params, revision):
        calls.append((params, revision))
        return {"rows": [{"row_idx": 7, "row": {"source_name": "fasdd", "sha256": row["sha256"],
            "source_record_id": row["source_record_id"], "license": module.LICENSE,
            "annotations_json": json.dumps(raw()["annotations"]),
            "image": {"src": f"https://datasets-server.huggingface.co/cached-assets/{module.REPO}/--/{module.REVISION}/--/default/train/7/image/image.jpg?temporary=1", "width": 1280, "height": 720}}}]}
    monkeypatch.setattr(module, "api_rows", page)
    viewer = module.Viewer()
    assert viewer.image_url(row) == viewer.image_url(row)
    assert len(calls) == 1 and calls[0][1] == module.REVISION
    with pytest.raises(ValueError, match="immutable"):
        viewer.image_url(row | {"sha256": "b" * 64})
