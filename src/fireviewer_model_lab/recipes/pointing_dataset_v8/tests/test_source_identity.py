import copy
import json

import pytest

from training.pointing_dataset_v8.source_identity import frozen_camera_views, pyro_camera_view, pyro_camera_views, retire_camera_leaks
from training.pointing_dataset_v8.source_identity import bind_camera_reviews, camera_views, load_camera_reviews
from training.pointing_dataset_v8.source_identity import separate_validation_from_test
from training.pointing_dataset_v7.split_registry import digest


@pytest.mark.parametrize("group", ["pyrosdis:sdis-07:brison-200:20240220T155338",
                                  "pyro_sdis_a1e553e:sdis-07:brison-200",
                                  "Pyro-SDIS:sdis-07:brison-200:2024-04-01"])
def test_legacy_prefix_and_date_do_not_change_camera_identity(group):
    assert pyro_camera_view({"source_group_id": group}) == "pyro-camera:sdis-07:brison-200"


def test_original_filename_and_explicit_metadata_agree():
    row = {"source_record_id": "force-06_courmettes-212_2024-01-25T14-14-42.jpg",
           "partner": "force-06", "camera": "courmettes-212"}
    assert pyro_camera_view(row) == "pyro-camera:force-06:courmettes-212"
    with pytest.raises(ValueError, match="Conflicting"):
        pyro_camera_view(row | {"camera": "courmettes-275"})


def test_historical_registry_alone_can_freeze_a_retired_camera():
    history = {"source_groups": {"pyrosdis:force-06:ferion-142:20240109T111551": ["test"]}}
    assert frozen_camera_views([], history) == {"pyro-camera:force-06:ferion-142"}


def test_visually_merged_orientations_propagate_the_holdout_lock():
    heldout = {"split": "test", "source_group_id": "pyrosdis:sdis-07:brison-20:20240220T155338"}
    merged = {"split": "train", "source_record_id": "sdis-07_brison-39_2024-09-21T16-46-32.jpg",
              "scene_group_id": "Pyro-SDIS:sdis-07:brison-20", "scene_group_verified": True,
              "scene_group_evidence": "Inspected overlapping landscape, grouped together."}
    assert pyro_camera_view(merged) == "pyro-camera:sdis-07:brison-39"
    assert pyro_camera_views(merged) == {"pyro-camera:sdis-07:brison-20", "pyro-camera:sdis-07:brison-39"}
    assert frozen_camera_views([heldout, merged]) == pyro_camera_views(merged)


def test_train_reference_retirement_preserves_holdouts_and_original_data():
    heldout = {"sha256": "a", "split": "test", "source_group_id": "pyrosdis:sdis-07:brison-200:20240120T160320"}
    train = {"sha256": "b", "split": "train", "source_group_id": "Pyro-SDIS:sdis-07:brison-200:2024-08-01"}
    unrelated = {"sha256": "c", "split": "train", "source_group_id": "Pyro-SDIS:sdis-77:croix-augas-316:2024-08-01"}
    rows = [heldout, train, unrelated]
    before = copy.deepcopy(rows)
    retained, exclusions = retire_camera_leaks(rows, [heldout], {})
    assert rows == before
    assert retained == [heldout, unrelated]
    assert exclusions[0]["sha256"] == "b" and exclusions[0]["source_deleted"] is False


def test_disjoint_active_validation_preserves_test_train_and_original_records():
    shared = "pyrosdis:sdis-07:brison-200:"
    test = {"sha256": "a", "split": "test", "source_group_id": shared + "20240101", "objects": {"bbox": [[1, 2, 3, 4]]}}
    validation = {"sha256": "b", "split": "validation", "source_group_id": shared + "20240801", "objects": {"bbox": [[4, 3, 2, 1]]}}
    separate = {"sha256": "c", "split": "validation", "source_group_id": "pyrosdis:force-06:ferion-142:20240101"}
    train = {"sha256": "d", "split": "train", "source_group_id": "other"}
    rows = [test, validation, separate, train]
    before = copy.deepcopy(rows)
    retained, reserved = separate_validation_from_test(rows)
    assert rows == before and retained == [test, separate, train] and reserved == [validation]
    assert separate_validation_from_test(retained) == (retained, [])


def test_disjoint_validation_propagates_explicit_camera_aliases():
    test = {"split": "test", "reviewed_camera_view_ids": ["camera:a"], "reviewed_camera_evidence": ["inspected"]}
    alias = {"split": "validation", "reviewed_camera_view_ids": ["camera:a", "camera:b"], "reviewed_camera_evidence": ["inspected"]}
    other_date = {"split": "validation", "reviewed_camera_view_ids": ["camera:b"], "reviewed_camera_evidence": ["inspected"]}
    unknown = {"split": "validation", "reviewed_camera_view_ids": ["camera:a"]}
    retained, reserved = separate_validation_from_test([test, other_date, alias, unknown])
    assert retained == [test, unknown] and reserved == [other_date, alias]


def test_explicit_camera_review_protects_non_pyro_images_across_filename_blocks():
    heldout = {"sha256": "a", "split": "validation", "source_group_id": "dfire:aof:block-0040"}
    train = {"sha256": "b", "split": "train", "source_group_id": "dfire:aof:block-0044"}
    binding = {"reviewed_camera_view_ids": ["dfire-camera:camera1"],
               "reviewed_camera_evidence": [{"observation": "Explicit same-landscape inspection"}]}
    bound = bind_camera_reviews([heldout, train], {"a": binding, "b": binding})
    assert "reviewed_camera_view_ids" not in heldout
    retained, excluded = retire_camera_leaks(bound, bound, {})
    assert retained == [bound[0]] and excluded[0]["sha256"] == "b"
    assert camera_views({"reviewed_camera_view_ids": ["unreviewed-hint"]}) == set()
    assert frozen_camera_views([], {"camera_views": {"dfire-camera:camera1": ["train", "validation"]}}) == {"dfire-camera:camera1"}


def test_identity_review_binds_bytes_and_does_not_admit(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"original")
    row = {"sha256": digest(image), "source_image": str(image), "candidate_id": "sample", "review_index": 1}
    manifest = tmp_path / "review_manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    pages = tmp_path / "pages"
    pages.mkdir()
    page = pages / "review-0001.jpg"
    page.write_bytes(b"inspected page")
    registry = tmp_path / "packets.json"
    registry.write_text(json.dumps([{"packet": page.name, "sha256": digest(page),
                                    "candidate_ids": ["sample"], "review_indices": [1]}]))
    review = {"review_root": str(tmp_path), "manifest": manifest.name,
              "manifest_sha256": digest(manifest), "packets_sha256": digest(registry),
              "camera_view_id": "camera:reviewed", "review_indices": [1], "evidence": "Compared whole scene",
              "review_date": "2026-08-28"}
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps([review]))
    bindings = load_camera_reviews(path)
    assert bindings[row["sha256"]]["reviewed_camera_view_ids"] == ["camera:reviewed"]
    assert "v8_corpus_admitted" not in bind_camera_reviews([row], bindings)[0]
    page.write_bytes(b"changed page")
    with pytest.raises(ValueError, match="packet changed"):
        load_camera_reviews(path)
