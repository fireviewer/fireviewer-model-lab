import copy
import hashlib
import json

import pytest
from PIL import Image

from training.pointing_dataset_v7.split_registry import digest
from training.pointing_dataset_v8.assemble_extension import (
    apply_condition_reviews, export_and_reload, retire_unsuitable_condition_reviews, select_additions,
)
from training.pointing_dataset_v8.audit_coverage import POLICY


POLICY_DATA = json.loads(POLICY.read_text())


def row(i, **overrides):
    return {"sha256": f"{i:064x}", "phash": f"{i*0xabc123456789def:016x}", "phash_flipped": f"{i*0xbcdeabcdeabcd:016x}",
            "image_id": i, "split": "train", "source_family": "test", "source_dataset": "test",
            "source_group_id": "source-" + str(i), "scene_group_id": "scene-" + str(i), "scene_group_verified": True,
            "split_group_id": "source-" + str(i), "source_revision": "r", "license": "CC0", "license_evidence": "primary",
            "v8_corpus_admitted": True, "review_status": "accepted_v8_review", "width": 100, "height": 100,
            "synthetic": False, "objects": {"bbox": [[20, 20, 10, 10]], "category": [1]}, **overrides}


def select(reviews, **kwargs):
    return select_additions(kwargs.get("base", []), reviews, kwargs.get("baseline", []), kwargs.get("history", {}),
                            kwargs.get("fingerprints", []), kwargs.get("policy", POLICY_DATA))


def test_rejected_image_cannot_reenter_from_other_review():
    a = row(1)
    rejected = dict(a, v8_corpus_admitted=False, review_status="reject")
    accepted, excluded = select([a, rejected])
    assert not accepted
    assert excluded[0]["reason"] == "explicit_rejection_in_another_review"


def test_mirrored_dfire_is_not_a_new_source_or_safe_new_group():
    accepted, excluded = select([row(1, source_family="WUI-Fire-Detection", source_dataset="WUI-Fire-Detection",
                                     source_record_id="/test/WEB00001.jpg")])
    assert not accepted
    assert excluded[0]["reason"] == "dfire_mirror_without_original_group_binding"


def test_historical_holdouts_and_their_groups_are_protected():
    a = row(1)
    accepted, excluded = select([a], history={"source_groups": {"source-1": ["validation"]}})
    assert not accepted
    assert excluded[0]["reason"] == "historical_holdout_group"


def test_historical_camera_is_protected_across_dates_and_dataset_aliases():
    candidate = row(1, source_family="Pyro-SDIS", source_dataset="Pyro-SDIS",
                    source_group_id="Pyro-SDIS:sdis-07:brison-200:2024-08-01")
    history = {"source_groups": {"pyrosdis:sdis-07:brison-200:20240120T160320": ["test"]}}
    accepted, excluded = select([candidate], history=history)
    assert not accepted
    assert excluded[0]["reason"] == "historical_holdout_camera_view"


def test_global_near_duplicate_across_sources_is_excluded():
    a, b = row(1), row(2, source_family="other", phash=row(1)["phash"], phash_flipped=row(1)["phash_flipped"])
    accepted, excluded = select([a, b])
    assert len(accepted) == 1
    assert excluded[0]["reason"] == "global_mirrored_phash_le4"


def test_cap_is_by_reviewed_scene_not_renamed_source_days():
    rows = [row(i, scene_group_id="same_camera") for i in range(1, 11)]
    accepted, excluded = select(rows)
    assert len(accepted) == 8
    assert {r["reason"] for r in excluded} == {"reviewed_scene_cap"}
    assert select(list(reversed(rows)))[0] == accepted


def test_baseline_images_cannot_be_counted_new():
    a = row(1)
    accepted, excluded = select([a], baseline=[a], fingerprints=[a])
    assert not accepted
    assert excluded[0]["reason"] == "not_new_historical_v7_image"
    with pytest.raises(ValueError, match="omits historical"):
        select([a], baseline=[a])


def test_completion_checks_near_duplicates_against_previous_v8_additions():
    previous = row(1)
    later = row(2, phash=previous["phash"], phash_flipped=previous["phash_flipped"])
    accepted, excluded = select([later], base=[previous])
    assert not accepted
    assert excluded[0]["reason"] == "global_mirrored_phash_le4"
    assert excluded[0]["matching_sha256"] == previous["sha256"]


def test_historical_base_without_inline_fingerprints_uses_bound_registry():
    registered = row(1)
    base = {k: v for k, v in registered.items() if k not in {"phash", "phash_flipped"}}
    later = row(2, phash=registered["phash"], phash_flipped=registered["phash_flipped"])
    accepted, excluded = select([later], base=[base], baseline=[base], fingerprints=[registered])
    assert not accepted
    assert excluded[0]["reason"] == "global_mirrored_phash_le4"


@pytest.mark.parametrize("group_key,reason", [
    ("scene_group_id", "reviewed_scene_cap"),
    ("source_group_id", "declared_source_group_cap"),
    ("physical_event_id", "verified_event_cap"),
])
def test_completion_cannot_reset_previously_consumed_group_caps(group_key, reason):
    previous = [row(i, **{group_key: "same-context", "physical_event_verified": True}) for i in range(1, 9)]
    later = row(9, **{group_key: "same-context", "physical_event_verified": True})
    accepted, excluded = select([later], base=previous)
    assert not accepted
    assert excluded[0]["reason"] == reason


def test_new_group_caps_do_not_count_historical_v7_frames_again():
    previous = [row(i, scene_group_id="same-context") for i in range(1, 9)]
    later = row(9, scene_group_id="same-context")
    accepted, excluded = select([later], base=previous, baseline=previous, fingerprints=previous)
    assert accepted == [later]
    assert not excluded


def test_reload_is_real_idempotent_and_zero_copy(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (100, 100), "red").save(source)
    a = row(1, source_image=str(source), sha256=digest(source))
    root = tmp_path / "coco"
    before = copy.deepcopy(a)
    result = export_and_reload([a], root)
    assert result["status"] == "passed" and result["image_count"] == 1
    assert result["additional_image_payload_bytes"] == 0
    assert a == before
    assert export_and_reload([a], root) == result
    file = root / "train" / "images" / (a["sha256"] + ".png")
    assert source.samefile(file)
    assert json.loads((root / "train" / "_annotations.coco.json").read_text())["info"]["status"] == "review_only_not_training_ready"


def test_reload_refuses_modified_image_or_id_collision(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (100, 100), "blue").save(source)
    a = row(1, source_image=str(source), sha256="0" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        export_and_reload([a], tmp_path / "coco")
    with pytest.raises(ValueError, match="id collision"):
        export_and_reload([a, a], tmp_path / "coco")


def test_refresh_retires_only_obsolete_links_without_deleting_sources(tmp_path):
    rows = []
    for number, color in enumerate(("red", "blue"), 1):
        source = tmp_path / f"source-{number}.png"
        Image.new("RGB", (100, 100), color).save(source)
        rows.append(row(number, source_image=str(source), sha256=digest(source)))
    root = tmp_path / "coco"
    export_and_reload(rows, root)
    report = export_and_reload(rows[1:], root, known_sources=rows)
    assert report["directories_match_coco_exactly"] is True
    assert len(report["retired_export_links"]) == 1
    old = rows[0]
    retired = tmp_path / "retired_export_links" / "train" / (old["sha256"] + ".png")
    source = tmp_path / "source-1.png"
    assert retired.samefile(source) and digest(source) == old["sha256"]
    assert len(list((root / "train" / "images").iterdir())) == 1


def test_unknown_export_file_is_not_moved_or_deleted(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 100), "red").save(source)
    a = row(1, source_image=str(source), sha256=digest(source))
    root = tmp_path / "coco"
    export_and_reload([a], root)
    foreign = root / "train" / "images" / "user-owned.png"
    foreign.write_bytes(b"unrelated-user-file")
    with pytest.raises(ValueError, match="not a verified hardlink"):
        export_and_reload([a], root)
    assert foreign.read_bytes() == b"unrelated-user-file" and source.is_file()


@pytest.fixture
def condition_review(tmp_path):
    source, packet = tmp_path / "source.png", tmp_path / "packet.png"
    Image.new("RGB", (100, 100), "blue").save(source)
    Image.new("RGB", (200, 100), "white").save(packet)
    legacy = row(1, source_image=str(source), sha256=digest(source), review_status="accepted_existing_review")
    decision = {"sha256": legacy["sha256"],
                "annotation_sha256": hashlib.sha256(json.dumps(legacy["objects"], sort_keys=True).encode()).hexdigest(),
                "packet": str(packet), "packet_sha256": digest(packet), "review_date": "2026-08-28",
                "reason": "Explicit per-image scene observation", "framing_review": "well_framed",
                "obstruction_review": "clear", "visibility_review": "low_visibility", "lighting_review": "daylight"}
    ledger = tmp_path / "conditions_decisions.jsonl"
    ledger.write_text(json.dumps(decision) + "\n")
    (tmp_path / "packets.json").write_text(json.dumps([
        {"packet": str(packet), "sha256": digest(packet), "image_sha256s": [legacy["sha256"]]}]))
    return legacy, decision, ledger, packet


def test_legacy_condition_review_preserves_images_annotations_and_admission(condition_review):
    legacy, decision, ledger, _ = condition_review
    before = copy.deepcopy(legacy)
    result, report = apply_condition_reviews([legacy], [ledger])
    assert legacy == before
    assert result[0]["objects"] == before["objects"]
    assert result[0]["review_status"] == "accepted_existing_review"
    assert result[0]["v8_corpus_admitted"] == before["v8_corpus_admitted"]
    assert result[0]["lighting_review"] == "daylight"
    assert report["reviewed_images"] == report["packets_verified"] == 1
    assert set(before) <= set(result[0])


def test_assembly_verifies_legacy_ledger_before_new_camera_retirement(condition_review, tmp_path, monkeypatch):
    import sys
    from training.pointing_dataset_v8 import assemble_extension as assembly

    legacy, decision, ledger, _ = condition_review
    legacy.update(reviewed_camera_view_ids=['camera-newly-frozen'], reviewed_camera_evidence=[{'review': 'bound'}])
    holdout = row(2, split='test', reviewed_camera_view_ids=['camera-newly-frozen'],
                  reviewed_camera_evidence=[{'review': 'bound'}])
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'selection_manifest.jsonl').write_text(json.dumps(legacy)+'\n')
    baseline, history = tmp_path / 'baseline.jsonl', tmp_path / 'history.json'
    baseline.write_text(json.dumps(holdout)+'\n')
    history.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['assemble', '--base', str(base), '--baseline', str(baseline),
        '--history', str(history), '--review-root', str(tmp_path / 'unused'),
        '--conditions-review', str(ledger), '--output', str(tmp_path / 'output')])
    monkeypatch.setattr(assembly, 'verified_reviews', lambda root: [])
    retire = assembly.retire_camera_leaks

    class VerifiedRetirement(Exception):
        pass

    def check_retirement(rows, baseline_rows, registry):
        assert rows[0]['legacy_conditions_review']['decision'] == decision
        retained, excluded = retire(rows, baseline_rows, registry)
        assert not retained and [r['sha256'] for r in excluded] == [legacy['sha256']]
        assert digest(ledger.parent / 'source.png') == legacy['sha256']
        raise VerifiedRetirement

    monkeypatch.setattr(assembly, 'retire_camera_leaks', check_retirement)
    with pytest.raises(VerifiedRetirement):
        assembly.main()


def test_partial_and_unsuitable_conditions_are_not_manufactured_or_hidden(condition_review):
    legacy, decision, ledger, _ = condition_review
    decision.update(lighting_review=None, visibility_review="unknown", framing_review="unusable")
    ledger.write_text(json.dumps(decision) + "\n")
    result, report = apply_condition_reviews([legacy], [ledger])
    assert len(result) == 1 and result[0]["framing_review"] == "unusable"
    assert "lighting_review" not in result[0] and "visibility_review" not in result[0]
    assert report["unknown_fields"] == {"visibility_review": 1, "lighting_review": 1}
    kept, excluded = retire_unsuitable_condition_reviews(result)
    assert not kept
    assert excluded[0]["legacy_conditions_review"]["decision"] == decision
    assert excluded[0]["source_image_deleted"] is False
    assert digest(ledger.parent / "source.png") == legacy["sha256"]


def test_condition_retirement_cannot_remove_holdouts_unreviewed_or_partial_rows():
    rows = [row(1, framing_review="unusable"),
            row(2, split="test", framing_review="unusable", legacy_conditions_review={"ledger": "proof"}),
            row(3, lighting_review=None, legacy_conditions_review={"ledger": "proof"})]
    kept, excluded = retire_unsuitable_condition_reviews(rows)
    assert kept == rows and not excluded


@pytest.mark.parametrize("change,error", [
    ("annotation", "binding changed"), ("image", "binding changed"),
    ("packet", "packet changed"), ("membership", "not bound"),
    ("holdout", "legacy train"), ("new_admission", "legacy train"),
    ("missing_field", "explicitly resolved"), ("positive_not_applicable", "not-applicable"),
    ("conflicting", "Conflicting"),
])
def test_condition_review_rejects_unbound_or_invalid_evidence(condition_review, change, error):
    legacy, decision, ledger, packet = condition_review
    if change == "annotation":
        decision["annotation_sha256"] = "0" * 64
    elif change == "image":
        Image.new("RGB", (100, 100), "red").save(legacy["source_image"])
    elif change == "packet":
        packet.write_bytes(b"changed")
    elif change == "membership":
        manifests = json.loads((ledger.parent / "packets.json").read_text())
        manifests[0]["image_sha256s"] = []
        (ledger.parent / "packets.json").write_text(json.dumps(manifests))
    elif change == "holdout":
        legacy["split"] = "test"
    elif change == "new_admission":
        legacy["review_status"] = "accepted_v8_review"
    elif change == "missing_field":
        decision.pop("lighting_review")
    elif change == "positive_not_applicable":
        decision["visibility_review"] = "not_applicable"
    elif change == "conflicting":
        legacy["lighting_review"] = "night"
    ledger.write_text(json.dumps(decision) + "\n")
    with pytest.raises(ValueError, match=error):
        apply_condition_reviews([legacy], [ledger])


def test_condition_review_rejects_overlapping_review_lots(condition_review):
    legacy, _, ledger, _ = condition_review
    with pytest.raises(ValueError, match="Duplicate condition review"):
        apply_condition_reviews([legacy], [ledger, ledger])


@pytest.mark.parametrize("field", ["exclude_from_real_corpus_reason", "exclude_from_training_reason"])
def test_explicit_inspected_exclusion_preserves_source_and_evidence(condition_review, field):
    legacy, decision, ledger, _ = condition_review
    decision[field] = "Inspected scene violates corpus admission criteria"
    ledger.write_text(json.dumps(decision) + "\n")
    reviewed, _ = apply_condition_reviews([legacy], [ledger])
    kept, excluded = retire_unsuitable_condition_reviews(reviewed)
    assert not kept and len(excluded) == 1
    assert excluded[0]["legacy_conditions_review"]["decision"][field] == decision[field]
    assert digest(ledger.parent / "source.png") == legacy["sha256"]


def test_person_uncertainty_is_reserved_without_claiming_person_presence(condition_review):
    legacy, decision, ledger, _ = condition_review
    decision.update(requires_person_review=True, person_presence_review="unknown")
    ledger.write_text(json.dumps(decision) + "\n")
    reviewed, _ = apply_condition_reviews([legacy], [ledger])
    kept, excluded = retire_unsuitable_condition_reviews(reviewed)
    assert not kept and excluded[0]["reason"] == "legacy_person_presence_unresolved"
    assert excluded[0]["legacy_conditions_review"]["decision"]["person_presence_review"] == "unknown"
