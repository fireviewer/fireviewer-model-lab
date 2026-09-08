from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fireviewer_model_lab.training.dinov3_corpus_identity import (
    DECODED_PIXEL_HASH_ALGORITHM,
    PERCEPTUAL_HASH_ALGORITHM,
    deterministic_source_family_id,
)
from fireviewer_model_lab.training.dinov3_multitask_compose import compose_multitask


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _denylist(path: Path, *, raw_hashes: list[str] | None = None) -> Path:
    raw_hashes = sorted(set(raw_hashes or []))
    value = {
        "schema_version": 1,
        "kind": "fireviewer-independent-benchmark-hash-denylist",
        "hash_only": True,
        "provenance_guard_sha256": "d" * 64,
        "provenance_guard_rows": 1,
        "raw_image_sha256": raw_hashes,
        "decoded_pixel_sha256": [],
        "phash64_imagehash_v1": [],
    }
    _json(path, value)
    return path


def _registry(path: Path) -> dict:
    denylist = _denylist(path.with_name("benchmark-denylist.json"))
    registry = {
        "schema_version": 2,
        "campaign_id": "test-composition",
        "detection_base": {
            "repository": "owner/detection",
            "revision": "a" * 40,
            "split_counts": {"train": 1, "validation": 1},
            "manifest_paths": {"base": "manifests/base.jsonl"},
        },
        "overlay_sources": [
            {
                "name": "boreal",
                "source_id": "boreal-segmentation",
                "source_revision": "boreal-revision",
                "manifest_filename": "boreal.jsonl",
                "report_filename": "boreal-report.json",
                "strict_rows_min": 1,
                "strict_rows_max": 1,
                "task_profile": "smoke_segmentation_and_ground_point",
            },
            {
                "name": "camp-swift",
                "source_id": "camp-swift",
                "source_revision": "camp-revision",
                "manifest_filename": "camp.jsonl",
                "report_filename": "camp-report.json",
                "strict_rows_min": 1,
                "strict_rows_max": 1,
                "task_profile": "fire_segmentation_and_ground_point",
            },
            {
                "name": "kit",
                "source_id": "kit",
                "source_revision": "kit-revision",
                "manifest_filename": "kit.jsonl",
                "report_filename": "kit-report.json",
                "strict_rows_min": 1,
                "strict_rows_max": 1,
                "task_profile": "industrial_segmentation_presence_abstention_no_point",
            },
        ],
        "hard_gates": {
            "detection_rows_exact": 2,
            "detection_positive_rows_exact": 1,
            "detection_explicit_negative_rows_exact": 1,
            "bbox_derived_point_rows_max": 0,
            "weak_or_teacher_generated_rows_max": 0,
            "independent_benchmark_rows_max": 0,
            "benchmark_hash_matches_max": 0,
            "unknown_rights_rows_max": 0,
        },
        "quality_gates": {
            "pilot_point_positive_rows_min": 2,
            "professional_point_positive_rows_min": 3,
            "professional_fire_base_rows_min": 1,
            "professional_smoke_column_base_rows_min": 1,
            "professional_point_validation_rows_min": 1,
            "professional_point_test_rows_min": 1,
            "point_source_families_min": 2,
            "largest_point_source_share_max": 0.75,
            "top_three_point_source_share_max": 1.0,
            "presence_supervised_rows_min": 4,
            "explicit_negative_rows_min": 1,
        },
        "benchmark_boundary": {
            "independent_benchmark_must_remain_separate": True,
            "forbidden_references": ["independent-benchmark"],
            "denylist_schema_version": 1,
            "denylist_sha256": _sha(denylist.read_bytes()),
            "provenance_guard_sha256": "d" * 64,
            "provenance_guard_rows": 1,
            "denylist_entry_counts": {
                "raw_image_sha256": 0,
                "decoded_pixel_sha256": 0,
                "phash64_imagehash_v1": 0,
            },
            "decoded_pixel_hash_algorithm": DECODED_PIXEL_HASH_ALGORITHM,
            "perceptual_hash_algorithm": PERCEPTUAL_HASH_ALGORITHM,
            "phash_hamming_distance_max": 3,
        },
    }
    policies = {
        "boreal": {
            "allowed_annotation_strengths": ["strong"],
            "allowed_annotation_provenances": ["strict_human_mask"],
            "allowed_mask_qualities": ["strict_human_mask"],
            "allowed_point_derivations": ["strict_mask_lower_envelope"],
        },
        "camp-swift": {
            "allowed_annotation_strengths": ["strong"],
            "allowed_annotation_provenances": ["strict_human_mask"],
            "allowed_mask_qualities": ["strict_human_mask"],
            "allowed_point_derivations": ["strict_mask_lower_envelope"],
        },
        "kit": {
            "allowed_annotation_strengths": ["strong"],
            "allowed_annotation_provenances": ["strict_human_mask"],
            "allowed_mask_qualities": ["strict_human_mask"],
            "allowed_point_derivations": ["none"],
        },
    }
    for source in registry["overlay_sources"]:
        lineage_root = f"test-dataset:{source['name']}"
        family_id = deterministic_source_family_id([lineage_root])
        source.update(
            {
                "source_family": source["source_id"],
                "source_family_id": family_id,
                "license": "permissive-test-license",
                "redistribution_allowed": True,
                "allowed_validation_profiles": ["fireviewer_pointing_strict_automated_v1"],
                **policies[source["name"]],
            }
        )
    registry["source_identity_contract"] = {
        "schema_version": 1,
        "family_id_algorithm": "source-family-sha256-v1",
        "event_id_algorithm": "canonical-event-sha256-v1",
        "families": [
            {
                "source_family_id": source["source_family_id"],
                "lineage_root_ids": [f"test-dataset:{source['name']}"],
                "bindings": [
                    {
                        "kind": "overlay",
                        "name": source["name"],
                        "event_key_field": "split_group",
                    }
                ],
            }
            for source in registry["overlay_sources"]
        ],
        "event_aliases": [],
    }
    _json(path, registry)
    return registry


def _detection_manifest(path: Path, positive_sha: str) -> None:
    common = {
        "sample_validation_status": "strict_automated_validated",
        "validation_profile": "fireviewer_detection_strict_automated_v1",
        "license": "permissive-test-license",
        "consent_basis": {"kind": "published_dataset", "reference": "fixture"},
        "validation_run_id": "test-run",
    }
    _jsonl(
        path,
        [
            {
                **common,
                "sample_id": "detection:positive",
                "source_id": "boreal-detection",
                "source_record_id": "positive",
                "split": "train",
                "split_group": "positive-group",
                "sha256": positive_sha,
                "image_relpath": "positive.jpg",
                "annotations": [{"class_name": "smoke_visible", "bbox": [1, 2, 3, 4]}],
            },
            {
                **common,
                "sample_id": "detection:negative",
                "source_id": "negative-source",
                "source_record_id": "negative",
                "split": "validation",
                "split_group": "negative-group",
                "sha256": _sha(b"detection-negative"),
                "image_relpath": "negative.jpg",
                "annotations": [],
                "negative_tags": ["no_target_visible"],
            },
        ],
    )


def _overlay(
    root: Path,
    *,
    manifest_name: str,
    report_name: str,
    sample_id: str,
    source_id: str,
    source_revision: str,
    image_payload: bytes,
    split: str,
    point_kind: str | None,
) -> dict:
    image = root / "payload" / f"{sample_id}-image.bin"
    mask = root / "payload" / f"{sample_id}-mask.bin"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(image_payload)
    mask.write_bytes(f"mask:{sample_id}".encode())
    points = [] if point_kind is None else [{"kind": point_kind, "x": 0.4, "y": 0.8}]
    row = {
        "sample_id": sample_id,
        "source_id": source_id,
        "source_revision": source_revision,
        "source_record_id": sample_id,
        "sample_validation_status": "strict_automated_validated",
        "strict_keep": True,
        "training_eligible": True,
        "annotation_strength": "strong",
        "annotation_provenance": "strict_human_mask",
        "mask_quality": "strict_human_mask",
        "license": "permissive-test-license",
        "redistribution_allowed": True,
        "reviews_admitted": False,
        "validation_profile": "fireviewer_pointing_strict_automated_v1",
        "image_sha256": _sha(image_payload),
        "image_relpath": image.relative_to(root).as_posix(),
        "image_s3_uri": f"s3://test/{source_id}/image",
        "mask_sha256": _sha(mask.read_bytes()),
        "mask_relpath": mask.relative_to(root).as_posix(),
        "mask_s3_uri": f"s3://test/{source_id}/mask",
        "split": split,
        "split_group": f"{source_id}-group",
        "anchor_points": points,
        "point_derivation": "none" if point_kind is None else "strict_mask_lower_envelope",
        "ground_point_eligible": point_kind is not None,
        "visual_abstention_reason": (
            "industrial_flame_has_no_ground_contact_semantics" if point_kind is None else None
        ),
    }
    _jsonl(root / manifest_name, [row])
    _json(
        root / report_name,
        {
            "strict_automated_validated_rows": 1,
            "reviews_admitted": False,
            "publication_allowed": False,
            "source_gate_passed": True,
            "gate_errors": [],
            "decode_or_payload_errors": [],
            "split_group_leakage": [],
        },
    )
    return row


def _inputs(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[str, Path], Path]:
    registry_path = tmp_path / "registry.json"
    _registry(registry_path)
    positive_payload = b"boreal-positive-image"
    detection_path = tmp_path / "detection.jsonl"
    _detection_manifest(detection_path, _sha(positive_payload))
    boreal = tmp_path / "boreal"
    camp = tmp_path / "camp"
    kit = tmp_path / "kit"
    _overlay(
        boreal,
        manifest_name="boreal.jsonl",
        report_name="boreal-report.json",
        sample_id="boreal:one",
        source_id="boreal-segmentation",
        source_revision="boreal-revision",
        image_payload=positive_payload,
        split="train",
        point_kind="smoke_column_base",
    )
    _overlay(
        camp,
        manifest_name="camp.jsonl",
        report_name="camp-report.json",
        sample_id="camp:one",
        source_id="camp-swift",
        source_revision="camp-revision",
        image_payload=b"camp-positive-image",
        split="test",
        point_kind="fire_base",
    )
    _overlay(
        kit,
        manifest_name="kit.jsonl",
        report_name="kit-report.json",
        sample_id="kit:one",
        source_id="kit",
        source_revision="kit-revision",
        image_payload=b"kit-positive-image",
        split="validation",
        point_kind=None,
    )
    overlays = {
        "boreal": boreal,
        "camp-swift": camp,
        "kit": kit,
    }
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for contract in registry["overlay_sources"]:
        root = overlays[contract["name"]]
        contract["manifest_sha256"] = _sha((root / contract["manifest_filename"]).read_bytes())
        contract["report_sha256"] = _sha((root / contract["report_filename"]).read_bytes())
    _json(registry_path, registry)
    return (
        registry_path,
        {"base": detection_path},
        overlays,
        tmp_path / "benchmark-denylist.json",
    )


def _detection_receipts(detection: dict[str, Path]) -> list[dict]:
    return [
        {
            "name": name,
            "repository_path": "manifests/base.jsonl",
            "revision": "a" * 40,
            "bytes": path.stat().st_size,
            "sha256": _sha(path.read_bytes()),
        }
        for name, path in detection.items()
    ]


def _refresh_overlay_manifest_hash(registry_path: Path, name: str, manifest: Path) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    contract = next(source for source in registry["overlay_sources"] if source["name"] == name)
    contract["manifest_sha256"] = _sha(manifest.read_bytes())
    _json(registry_path, registry)


def test_composition_merges_strong_overlays_without_using_detection_boxes(tmp_path: Path) -> None:
    registry, detection, overlays, denylist = _inputs(tmp_path)

    report = compose_multitask(
        registry_path=registry,
        detection_manifests=detection,
        detection_receipts=_detection_receipts(detection),
        overlay_roots=overlays,
        output_dir=tmp_path / "output",
        benchmark_denylist_path=denylist,
    )

    assert report["integrity_gates_passed"] is True
    assert report["pilot_corpus_ready"] is True
    assert report["professional_corpus_ready"] is False
    assert report["composition_rows"] == 4
    assert report["overlay_rows_merged_into_detection_base"] == 1
    assert report["overlay_rows_added"] == 2
    assert report["bbox_derived_point_rows"] == 0
    assert report["task_counts"]["point_positive_rows"] == 2
    expected_families = sorted(
        deterministic_source_family_id([f"test-dataset:{name}"])
        for name in ("boreal", "camp-swift")
    )
    assert report["point_source_families"] == expected_families
    assert report["quality_gate_deficits"]["professional_point_positive_rows"] == {
        "actual": 2,
        "minimum": 3,
    }
    manifest = [
        json.loads(line)
        for line in (tmp_path / "output" / report["manifest"]).read_text().splitlines()
    ]
    assert len({row["image_sha256"] for row in manifest}) == len(manifest)
    merged = next(row for row in manifest if row["sample_id"] == "detection:positive")
    assert merged["point_supervised"] is True
    assert merged["anchor_points"][0]["kind"] == "smoke_column_base"
    assert "detection_annotation_count" in merged
    assert merged["overlay_sources"][0]["consent_basis"]["kind"] == "source_license"
    kit = next(row for row in manifest if row["sample_id"] == "kit:one")
    assert kit["segmentation_supervised"] is True
    assert kit["point_supervised"] is False
    assert kit["anchor_points"] == []
    assert kit["consent_basis"]["kind"] == "source_license"
    assert kit["redistribution_allowed"] is True
    assert report["unknown_rights_rows"] == 0
    assert report["training_ready"] is False
    assert report["publication_allowed"] is False


def test_composition_derives_pinned_s3_uris_for_materialized_overlay(tmp_path: Path) -> None:
    registry_path, detection, overlays, denylist = _inputs(tmp_path)
    camp_manifest = overlays["camp-swift"] / "camp.jsonl"
    camp_row = json.loads(camp_manifest.read_text(encoding="utf-8"))
    camp_row.pop("image_s3_uri")
    camp_row.pop("mask_s3_uri")
    _jsonl(camp_manifest, [camp_row])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    contract = next(
        source for source in registry["overlay_sources"] if source["name"] == "camp-swift"
    )
    contract["artifact_s3_prefix"] = "s3://bucket/materialized/camp"
    contract["manifest_sha256"] = _sha(camp_manifest.read_bytes())
    _json(registry_path, registry)

    report = compose_multitask(
        registry_path=registry_path,
        detection_manifests=detection,
        detection_receipts=_detection_receipts(detection),
        overlay_roots=overlays,
        output_dir=tmp_path / "output-derived-uri",
        benchmark_denylist_path=denylist,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "output-derived-uri" / report["manifest"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    camp = next(row for row in rows if row["sample_id"] == "camp:one")
    assert camp["image_locator"]["uri"] == (
        "s3://bucket/materialized/camp/payload/camp:one-image.bin"
    )
    assert camp["mask_locator"]["uri"] == (
        "s3://bucket/materialized/camp/payload/camp:one-mask.bin"
    )


def test_negative_control_is_verified_against_detection_without_append(tmp_path: Path) -> None:
    registry_path, detection, overlays, denylist = _inputs(tmp_path)
    registry = json.loads(registry_path.read_text())
    registry["control_sources"] = [
        {
            "name": "pyro-sdis-negative-qa",
            "source_id": "pyro-control",
            "source_revision": "pyro-revision",
            "detection_source_id": "negative-source",
            "control_record_field": "source_image_name",
            "manifest_filename": "pyro.jsonl",
            "report_filename": "pyro-report.json",
            "strict_rows_min": 1,
            "strict_rows_max": 1,
            "strict_base_matches_min": 1,
            "not_admitted_by_detection_base_max": 0,
            "detection_cleaning_evidence": {
                "source_rows": 1,
                "source_exact_duplicates_removed": 0,
                "post_source_dedup_rows": 1,
                "final_strict_kept_rows": 1,
                "final_strict_not_admitted_rows": 0,
            },
        }
    ]
    _json(registry_path, registry)
    control = tmp_path / "pyro-control"
    image = control / "payload" / "negative.jpg"
    mask = control / "payload" / "negative.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"detection-negative-reencoded")
    mask.write_bytes(b"zero-mask")
    _jsonl(
        control / "pyro.jsonl",
        [
            {
                "sample_id": "pyro:negative",
                "source_id": "pyro-control",
                "source_revision": "pyro-revision",
                "sample_validation_status": "strict_automated_validated",
                "strict_keep": True,
                "training_eligible": True,
                "negative": True,
                "source_annotations_exactly_empty": True,
                "reviews_admitted": False,
                "source_image_name": "negative",
                "image_sha256": _sha(image.read_bytes()),
                "image_relpath": image.relative_to(control).as_posix(),
                "image_s3_uri": "s3://test/pyro/image",
                "mask_sha256": _sha(mask.read_bytes()),
                "mask_relpath": mask.relative_to(control).as_posix(),
                "mask_s3_uri": "s3://test/pyro/mask",
            }
        ],
    )
    _json(
        control / "pyro-report.json",
        {
            "strict_automated_validated_rows": 1,
            "reviews_admitted": False,
            "publication_allowed": False,
            "source_gate_passed": True,
            "gate_errors": [],
            "split_group_leakage": [],
        },
    )
    registry["control_sources"][0]["manifest_sha256"] = _sha((control / "pyro.jsonl").read_bytes())
    registry["control_sources"][0]["report_sha256"] = _sha(
        (control / "pyro-report.json").read_bytes()
    )
    _json(registry_path, registry)

    report = compose_multitask(
        registry_path=registry_path,
        detection_manifests=detection,
        detection_receipts=_detection_receipts(detection),
        overlay_roots=overlays,
        control_roots={"pyro-sdis-negative-qa": control},
        output_dir=tmp_path / "output",
        benchmark_denylist_path=denylist,
    )

    assert report["composition_rows"] == 4
    assert report["control_rows_verified_without_append"] == 1
    assert report["control_receipts"][0]["all_rows_accounted_without_append"] is True
    assert report["control_receipts"][0]["exact_sha_matches"] == 0
    assert report["control_receipts"][0]["source_record_identity_matches"] == 1


def test_composition_rejects_bbox_derived_overlay_points(tmp_path: Path) -> None:
    registry, detection, overlays, denylist = _inputs(tmp_path)
    manifest_path = overlays["camp-swift"] / "camp.jsonl"
    row = json.loads(manifest_path.read_text())
    row["point_derivation"] = "bbox_bottom_center"
    _jsonl(manifest_path, [row])
    _refresh_overlay_manifest_hash(registry, "camp-swift", manifest_path)

    with pytest.raises(ValueError, match="bbox-derived overlay point rejected"):
        compose_multitask(
            registry_path=registry,
            detection_manifests=detection,
            detection_receipts=_detection_receipts(detection),
            overlay_roots=overlays,
            output_dir=tmp_path / "output",
            benchmark_denylist_path=denylist,
        )


def test_composition_rejects_overlay_split_conflict_on_matching_image(tmp_path: Path) -> None:
    registry, detection, overlays, denylist = _inputs(tmp_path)
    manifest_path = overlays["boreal"] / "boreal.jsonl"
    row = json.loads(manifest_path.read_text())
    row["split"] = "validation"
    _jsonl(manifest_path, [row])
    _refresh_overlay_manifest_hash(registry, "boreal", manifest_path)

    with pytest.raises(ValueError, match="detection and overlay split conflict"):
        compose_multitask(
            registry_path=registry,
            detection_manifests=detection,
            detection_receipts=_detection_receipts(detection),
            overlay_roots=overlays,
            output_dir=tmp_path / "output",
            benchmark_denylist_path=denylist,
        )


def test_composition_blocks_raw_hash_from_pinned_benchmark_denylist(
    tmp_path: Path,
) -> None:
    registry_path, detection, overlays, denylist = _inputs(tmp_path)
    detection_row = json.loads(detection["base"].read_text().splitlines()[0])
    _denylist(denylist, raw_hashes=[detection_row["sha256"]])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["benchmark_boundary"]["denylist_sha256"] = _sha(denylist.read_bytes())
    registry["benchmark_boundary"]["denylist_entry_counts"]["raw_image_sha256"] = 1
    _json(registry_path, registry)

    report = compose_multitask(
        registry_path=registry_path,
        detection_manifests=detection,
        detection_receipts=_detection_receipts(detection),
        overlay_roots=overlays,
        output_dir=tmp_path / "output",
        benchmark_denylist_path=denylist,
    )

    assert report["integrity_gates_passed"] is False
    assert report["benchmark_hash_matches"] == 1
    assert "benchmark_hash_matches:1" in report["hard_gate_errors"]
    assert report["composition_integrity_receipt"] is None
