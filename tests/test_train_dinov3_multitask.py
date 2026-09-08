from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from fireviewer_model_lab.training.dinov3_adapter import build_training_contract, training_contract_sha256
from fireviewer_model_lab.training.dinov3_corpus_identity import (
    DECODED_PIXEL_HASH_ALGORITHM,
    PERCEPTUAL_HASH_ALGORITHM,
    decoded_pixel_sha256,
    deterministic_source_family_id,
    phash64_imagehash_v1,
    resolve_namespaced_source_identity,
    resolve_source_identity,
    validate_source_identity_contract,
)
from fireviewer_model_lab.training.train_dinov3_multitask import (
    TRAINABLE_SAMPLE_STATUSES,
    _validate_smoke_report,
    build_preflight_report,
)

TEST_MODEL_REVISION = "a" * 40
TEST_ROLE_TARGETS = {
    "positive": 0.35,
    "negative": 0.25,
    "presence": 0.25,
    "abstention": 0.15,
}


def test_strict_automated_pointing_rows_are_trainable() -> None:
    assert "strict_automated_validated" in TRAINABLE_SAMPLE_STATUSES
    assert "teacher_generated_weak" not in TRAINABLE_SAMPLE_STATUSES
    assert "sensor_generated_weak" not in TRAINABLE_SAMPLE_STATUSES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_composition(tmp_path: Path) -> Path:
    campaign_id = "test-dinov3-composition"
    registry = tmp_path / "composition-registry.json"
    denylist = tmp_path / "benchmark-denylist.json"
    denylist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "fireviewer-independent-benchmark-hash-denylist",
                "hash_only": True,
                "provenance_guard_sha256": "9" * 64,
                "provenance_guard_rows": 1,
                "raw_image_sha256": [],
                "decoded_pixel_sha256": [],
                "phash64_imagehash_v1": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    family_root = "test-dataset:boreal"
    family_id = deterministic_source_family_id([family_root])
    detection_revision = "d" * 40
    registry.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "campaign_id": campaign_id,
                "detection_base": {
                    "repository": "test/detection",
                    "revision": detection_revision,
                    "validation_profile": "fireviewer_detection_strict_automated_v1",
                    "validation_run_id": "test-run",
                    "rows": 2,
                    "positive_rows": 1,
                    "explicit_negative_rows": 1,
                    "split_counts": {"train": 1, "validation": 0, "test": 1},
                    "manifest_paths": {"test": "manifests/test/manifest.jsonl"},
                },
                "control_sources": [
                    {
                        "name": "pyro-control",
                        "source_id": "test-pyro-control",
                        "source_revision": "e" * 40,
                        "manifest_sha256": "c" * 64,
                        "report_sha256": "f" * 64,
                        "strict_rows_min": 1,
                        "strict_rows_max": 1,
                        "role": "detection_negative_overlap_control_only_never_append",
                    }
                ],
                "overlay_sources": [
                    {
                        "name": "boreal",
                        "source_id": "boreal",
                        "source_family": "test-boreal-family",
                        "source_family_id": family_id,
                        "source_revision": "b" * 40,
                        "manifest_sha256": "b" * 64,
                        "report_sha256": "a" * 64,
                        "strict_rows_min": 1,
                        "strict_rows_max": 1,
                        "license": "CC-BY-4.0",
                        "redistribution_allowed": True,
                        "task_profile": "smoke_segmentation_and_ground_point",
                    }
                ],
                "hard_gates": {
                    "detection_rows_exact": 2,
                    "detection_positive_rows_exact": 1,
                    "detection_explicit_negative_rows_exact": 1,
                    "one_logical_row_per_image_sha256": True,
                    "bbox_derived_point_rows_max": 0,
                    "weak_or_teacher_generated_rows_max": 0,
                    "payload_hash_mismatches_max": 0,
                    "split_group_leaks_max": 0,
                    "perceptual_near_duplicate_pairs_max": 0,
                    "independent_benchmark_rows_max": 0,
                    "benchmark_hash_matches_max": 0,
                    "unknown_rights_rows_max": 0,
                },
                "quality_gates": {
                    "pilot_point_positive_rows_min": 1,
                    "professional_point_positive_rows_min": 1,
                    "professional_fire_base_rows_min": 0,
                    "professional_smoke_column_base_rows_min": 1,
                    "professional_point_validation_rows_min": 1,
                    "professional_point_test_rows_min": 0,
                    "point_source_families_min": 1,
                    "largest_point_source_share_max": 1.0,
                    "top_three_point_source_share_max": 1.0,
                    "presence_supervised_rows_min": 3,
                    "explicit_negative_rows_min": 1,
                },
                "benchmark_boundary": {
                    "independent_benchmark_must_remain_separate": True,
                    "forbidden_references": ["benchdata", "independent-benchmark"],
                    "denylist_schema_version": 1,
                    "denylist_sha256": _sha256(denylist),
                    "provenance_guard_sha256": "9" * 64,
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
                "source_identity_contract": {
                    "schema_version": 1,
                    "family_id_algorithm": "source-family-sha256-v1",
                    "event_id_algorithm": "canonical-event-sha256-v1",
                    "families": [
                        {
                            "source_family_id": family_id,
                            "lineage_root_ids": [family_root],
                            "bindings": [
                                {
                                    "kind": "overlay",
                                    "name": "boreal",
                                    "event_key_field": "split_group",
                                }
                            ],
                        }
                    ],
                    "event_aliases": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry_sha256 = _sha256(registry)
    registry_value = json.loads(registry.read_text(encoding="utf-8"))
    identities = validate_source_identity_contract(registry_value)
    images = []
    for index in range(3):
        image = tmp_path / f"image-{index}.jpg"
        Image.new("RGB", (32, 32), (40 + index * 50, 80, 120)).save(image, format="JPEG")
        images.append(image)
    mask = tmp_path / "mask.png"
    mask_image = Image.new("L", (32, 32), 0)
    for y in range(20, 30):
        for x in range(10, 23):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask, format="PNG")
    detection_presence_identity = resolve_namespaced_source_identity(
        lineage_root_id="hf-dataset:test/detection",
        source_event_key="fireviewer-detection-base:group-train",
    )
    detection_presence_identity["source_split_group"] = "group-train"
    overlay_identity = resolve_source_identity(
        {"split_group": "group-validation"},
        binding_kind="overlay",
        binding_name="boreal",
        identities=identities,
    )
    detection_negative_identity = resolve_namespaced_source_identity(
        lineage_root_id="hf-dataset:test/detection",
        source_event_key="pyronear-pyro-sdis:group-test",
    )
    detection_negative_identity["source_split_group"] = "group-test"

    def image_hashes(path: Path) -> dict[str, str]:
        with Image.open(path) as image:
            return {
                "decoded_pixel_sha256": decoded_pixel_sha256(image),
                "phash64_imagehash_v1": f"{phash64_imagehash_v1(image):016x}",
            }

    common = {
        "schema_version": 2,
        "sample_weight": 1.0,
        "sample_validation_status": "strict_automated_validated",
        "validation_profile": "fireviewer_multitask_composition_v1",
        "hydration_status": "sha256_and_semantic_validation_passed",
        "campaign_id": campaign_id,
        "composition_registry_sha256": registry_sha256,
        "anchor_points": [],
        "visual_abstention_reason": None,
        "overlay_sources": [],
    }
    rows = [
        {
            **common,
            "sample_id": "detection-presence",
            "split": "train",
            "split_group": f"event:{detection_presence_identity['canonical_event_id']}",
            "source_id": "fireviewer-detection-base",
            "source_manifest_path": "manifests/test/manifest.jsonl",
            "detection_validation_profile": "fireviewer_detection_strict_automated_v1",
            "detection_validation_run_id": "test-run",
            "image_relpath": images[0].name,
            "image_sha256": _sha256(images[0]),
            "image_locator": {
                "kind": "hf_dataset_row",
                "repository": "test/detection",
                "revision": detection_revision,
                "split": "train",
                "sample_id": "detection-presence",
                "sha256": _sha256(images[0]),
            },
            "annotation_strength": "strong",
            "segmentation_supervised": False,
            "point_supervised": False,
            "presence_supervised": True,
            "abstention_supervised": False,
            "presence_targets": {"flame_visible": True, "smoke_visible": False},
            "presence_provenance": "strict_detection_annotations_json",
            **detection_presence_identity,
            **image_hashes(images[0]),
        },
        {
            **common,
            "sample_id": "strong-point",
            "split": "validation",
            "split_group": f"event:{overlay_identity['canonical_event_id']}",
            "source_id": "boreal",
            "overlay_sources": [
                {
                    "name": "boreal",
                    "source_id": "boreal",
                    "source_family": "test-boreal-family",
                    "source_family_id": family_id,
                    "source_revision": "b" * 40,
                    "license": "CC-BY-4.0",
                    "manifest_sha256": "b" * 64,
                    **overlay_identity,
                }
            ],
            "image_relpath": images[1].name,
            "image_sha256": _sha256(images[1]),
            "image_locator": {
                "kind": "s3_object",
                "uri": "s3://test/boreal/image.jpg",
                "sha256": _sha256(images[1]),
                "extension": ".jpg",
            },
            "mask_relpath": mask.name,
            "mask_sha256": _sha256(mask),
            "mask_quality": "human_source_mask",
            "anchor_points": [{"kind": "smoke_column_base", "x": 0.5, "y": 0.8}],
            "annotation_strength": "strong",
            "segmentation_supervised": True,
            "point_supervised": True,
            "presence_supervised": True,
            "abstention_supervised": True,
            "presence_targets": {"flame_visible": False, "smoke_visible": True},
            "presence_provenance": "human_source_mask",
            "point_derivation": "human_mask_ground_base",
            **overlay_identity,
            **image_hashes(images[1]),
        },
        {
            **common,
            "sample_id": "explicit-negative",
            "split": "test",
            "split_group": f"event:{detection_negative_identity['canonical_event_id']}",
            "source_id": "pyronear-pyro-sdis",
            "source_manifest_path": "manifests/test/manifest.jsonl",
            "detection_validation_profile": "fireviewer_detection_strict_automated_v1",
            "detection_validation_run_id": "test-run",
            "image_relpath": images[2].name,
            "image_sha256": _sha256(images[2]),
            "image_locator": {
                "kind": "hf_dataset_row",
                "repository": "test/detection",
                "revision": detection_revision,
                "split": "test",
                "sample_id": "explicit-negative",
                "sha256": _sha256(images[2]),
            },
            "mask_encoding": "implicit_zero_from_explicit_negative",
            "mask_quality": "explicit_negative_zero",
            "annotation_strength": "negative",
            "segmentation_supervised": True,
            "point_supervised": True,
            "presence_supervised": True,
            "abstention_supervised": True,
            "visual_abstention_reason": "no_target_visible",
            "presence_targets": {"flame_visible": False, "smoke_visible": False},
            "presence_provenance": "strict_empty_annotation",
            **detection_negative_identity,
            **image_hashes(images[2]),
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def _write_hydration_report(tmp_path: Path, manifest: Path, *, professional: bool = True) -> Path:
    registry = tmp_path / "composition-registry.json"
    registry_data = json.loads(registry.read_text(encoding="utf-8"))
    registry_data["quality_gates"]["professional_point_positive_rows_min"] = (
        1 if professional else 2
    )
    registry.write_text(json.dumps(registry_data) + "\n", encoding="utf-8")
    registry_sha = _sha256(registry)
    identities = validate_source_identity_contract(registry_data)
    denylist_sha = _sha256(tmp_path / "benchmark-denylist.json")
    campaign_id = registry_data["campaign_id"]
    hydrated_rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    for row in hydrated_rows:
        row["composition_registry_sha256"] = registry_sha
        row["hydration_status"] = "sha256_and_semantic_validation_passed"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in hydrated_rows), encoding="utf-8")
    composition_rows = []
    for row in hydrated_rows:
        source = dict(row)
        source.pop("hydration_status", None)
        source.pop("decoded_pixel_sha256", None)
        source.pop("phash64_imagehash_v1", None)
        composition_rows.append(source)
    composition_manifest = tmp_path / "composition-manifest.jsonl"
    composition_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in composition_rows),
        encoding="utf-8",
    )
    deficits = (
        {} if professional else {"professional_point_positive_rows": {"actual": 1, "minimum": 2}}
    )
    task_counts = {
        "abstention_supervised_rows": 2,
        "explicit_negative_rows": 1,
        "point_positive_rows": 1,
        "point_positive_validation_rows": 1,
        "point_supervised_rows": 2,
        "presence_supervised_rows": 3,
        "segmentation_supervised_rows": 2,
        "smoke_column_base_rows": 1,
    }
    detection_receipts = [
        {
            "name": "test",
            "repository": "test/detection",
            "revision": "d" * 40,
            "repository_path": "manifests/test/manifest.jsonl",
            "sha256": "d" * 64,
            "bytes": 1,
        }
    ]
    overlay_receipts = [
        {
            "name": "boreal",
            "manifest_sha256": "b" * 64,
            "report_sha256": "a" * 64,
            "rows": 1,
            "source_gate_passed": True,
        }
    ]
    control_receipts = [
        {
            "name": "pyro-control",
            "manifest_sha256": "c" * 64,
            "report_sha256": "f" * 64,
            "rows": 1,
            "source_gate_passed": True,
            "all_rows_accounted_without_append": True,
        }
    ]
    integrity = tmp_path / "composition-integrity.json"
    integrity.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "campaign_id": campaign_id,
                "composition_registry_sha256": registry_sha,
                "source_identity_contract_sha256": identities.contract_sha256,
                "benchmark_denylist_sha256": denylist_sha,
                "manifest_sha256": _sha256(composition_manifest),
                "composition_rows": 3,
                "detection_revision": "d" * 40,
                "detection_manifest_sha256": {"test": "d" * 64},
                "overlay_manifest_sha256": {"boreal": "b" * 64},
                "control_manifest_sha256": {"pyro-control": "c" * 64},
                "integrity_gates_passed": True,
                "pilot_corpus_ready": True,
                "professional_corpus_ready": professional,
                "publication_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    composition_report = tmp_path / "composition-report.json"
    composition_report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "campaign_id": campaign_id,
                "composition_registry_sha256": registry_sha,
                "source_identity_contract_sha256": identities.contract_sha256,
                "benchmark_denylist_sha256": denylist_sha,
                "detection_repository": "test/detection",
                "detection_revision": "d" * 40,
                "detection_rows": 2,
                "detection_positive_rows": 1,
                "detection_explicit_negative_rows": 1,
                "detection_split_counts": {"test": 1, "train": 1, "validation": 0},
                "composition_rows": 3,
                "task_counts": task_counts,
                "point_source_family_counts": {
                    registry_data["overlay_sources"][0]["source_family_id"]: 1
                },
                "hard_gates": registry_data["hard_gates"],
                "quality_gates": registry_data["quality_gates"],
                "hard_gate_errors": [],
                "integrity_gates_passed": True,
                "quality_gate_deficits": deficits,
                "pilot_corpus_ready": True,
                "professional_corpus_ready": professional,
                "publication_allowed": False,
                "hf_replacement_allowed": False,
                "reviews_admitted": False,
                "manifest_sha256": _sha256(composition_manifest),
                "composition_integrity_receipt_sha256": _sha256(integrity),
                "split_group_leakage": [],
                "canonical_event_leakage": [],
                "bbox_derived_point_rows": 0,
                "weak_or_teacher_generated_rows": 0,
                "independent_benchmark_rows": 0,
                "benchmark_hash_matches": 0,
                "detection_manifest_receipts": detection_receipts,
                "overlay_receipts": overlay_receipts,
                "control_receipts": control_receipts,
            }
        ),
        encoding="utf-8",
    )
    hydration_integrity = tmp_path / "hydration-integrity.json"
    hydration_integrity.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_sha256": _sha256(manifest),
                "rows": 3,
                "campaign_id": campaign_id,
                "composition_registry_sha256": registry_sha,
                "source_identity_contract_sha256": identities.contract_sha256,
                "benchmark_denylist_sha256": denylist_sha,
                "composition_manifest_sha256": _sha256(composition_manifest),
                "composition_report_sha256": _sha256(composition_report),
                "composition_integrity_receipt_sha256": _sha256(integrity),
                "hydration_integrity_passed": True,
                "pilot_corpus_ready": True,
                "professional_corpus_ready": professional,
                "quality_gate_deficits": deficits,
                "training_ready": False,
                "publication_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "hydration-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "composition_manifest_sha256": _sha256(composition_manifest),
                "composition_report_sha256": _sha256(composition_report),
                "composition_integrity_receipt_sha256": _sha256(integrity),
                "rows": 3,
                "verified_image_rows": 3,
                "manifest_sha256": _sha256(manifest),
                "campaign_id": campaign_id,
                "composition_registry_sha256": registry_sha,
                "source_identity_contract_sha256": identities.contract_sha256,
                "benchmark_denylist_sha256": denylist_sha,
                "hydration_integrity_passed": True,
                "quality_gate_deficits": deficits,
                "pilot_corpus_ready": True,
                "professional_corpus_ready": professional,
                "ready_for_gpu_finite_loss_smoke": True,
                "ready_for_full_training": professional,
                "publication_allowed": False,
                "hard_gate_errors": [],
                "validation_errors": [],
                "decoded_pixel_duplicate_groups": 0,
                "split_group_leakage": [],
                "canonical_event_leakage": [],
                "benchmark_raw_hash_matches": 0,
                "benchmark_decoded_hash_matches": 0,
                "benchmark_phash_matches": 0,
                "perceptual_near_duplicates": {
                    "cross_split_pairs": 0,
                    "within_split_pairs": 0,
                },
                "hydration_integrity_receipt_sha256": _sha256(hydration_integrity),
            }
        ),
        encoding="utf-8",
    )
    return report


def _chain_kwargs(tmp_path: Path) -> dict[str, Path]:
    return {
        "composition_manifest": tmp_path / "composition-manifest.jsonl",
        "composition_report": tmp_path / "composition-report.json",
        "composition_integrity_receipt": tmp_path / "composition-integrity.json",
        "hydration_report": tmp_path / "hydration-report.json",
        "hydration_integrity_receipt": tmp_path / "hydration-integrity.json",
        "benchmark_denylist_path": tmp_path / "benchmark-denylist.json",
    }


def test_preflight_accepts_explicit_partial_supervision(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)
    _write_hydration_report(tmp_path, manifest)

    report = build_preflight_report(
        pointing_root=tmp_path / "separate-pointing-not-used",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is True
    assert report["training_errors"] == []
    assert report["verified_artifacts"] == 4
    assert report["verified_decodable_artifacts"] == 4
    assert report["implicit_zero_masks"] == 1
    assert report["supervision_counts"] == {
        "abstention_supervised": 2,
        "point_supervised": 2,
        "presence_supervised": 3,
        "segmentation_supervised": 2,
    }
    assert report["promotion_errors"] == [
        "independent_benchmark_missing",
        "ground_truth_acceptance_gate_pending",
    ]


def test_composed_manifest_requires_matching_hydration_receipt(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)

    missing = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        benchmark_denylist_path=tmp_path / "benchmark-denylist.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
    )
    assert "composition_hydration_report_missing" in missing["training_errors"]

    _write_hydration_report(tmp_path, manifest)
    accepted = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )
    assert accepted["training_ready"] is True


def test_minimal_registry_cannot_self_declare_professional_readiness(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)
    _write_hydration_report(tmp_path, manifest)
    (tmp_path / "composition-registry.json").write_text(
        json.dumps({"schema_version": 2, "campaign_id": "test-dinov3-composition"}),
        encoding="utf-8",
    )

    report = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is False
    assert "composition_registry_sections_missing" in report["training_errors"]


def test_forged_professional_boolean_cannot_override_recomputed_deficit(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_composition(tmp_path)
    _write_hydration_report(tmp_path, manifest, professional=False)
    hydration_path = tmp_path / "hydration-report.json"
    hydration = json.loads(hydration_path.read_text(encoding="utf-8"))
    hydration["professional_corpus_ready"] = True
    hydration["ready_for_full_training"] = True
    hydration["quality_gate_deficits"] = {}
    hydration_path.write_text(json.dumps(hydration), encoding="utf-8")

    report = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["professional_corpus_ready"] is False
    assert report["training_ready"] is False
    assert "professional_corpus_not_ready" in report["training_errors"]
    assert any(
        error.startswith("composition_hydration_report_gate_failed:")
        for error in report["training_errors"]
    )


def test_professional_gate_blocks_full_train_but_allows_pilot_smoke(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)
    _write_hydration_report(tmp_path, manifest, professional=False)

    full = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
        require_professional_corpus=True,
    )
    smoke = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
        require_professional_corpus=False,
    )

    assert "professional_corpus_not_ready" in full["training_errors"]
    assert full["training_ready"] is False
    assert smoke["training_ready"] is True
    assert smoke["ready_for_gpu_finite_loss_smoke"] is True


def test_preflight_recomputes_source_identity_instead_of_trusting_manifest(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_composition(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[1]["source_family_id"] = "sf1_" + "0" * 64
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_hydration_report(tmp_path, manifest)

    report = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is False
    assert any(
        error.startswith("composition_source_identity_invalid:")
        for error in report["training_errors"]
    )


def test_preflight_rehashes_image_against_pinned_benchmark_phash(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    guarded_phash = rows[0]["phash64_imagehash_v1"]
    denylist_path = tmp_path / "benchmark-denylist.json"
    denylist = json.loads(denylist_path.read_text(encoding="utf-8"))
    denylist["phash64_imagehash_v1"] = [guarded_phash]
    denylist_path.write_text(json.dumps(denylist) + "\n", encoding="utf-8")
    registry_path = tmp_path / "composition-registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["benchmark_boundary"]["denylist_sha256"] = _sha256(denylist_path)
    registry["benchmark_boundary"]["denylist_entry_counts"]["phash64_imagehash_v1"] = 1
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    _write_hydration_report(tmp_path, manifest)

    report = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=registry_path,
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is False
    assert report["benchmark_phash_matches"] >= 1
    assert any(
        error.startswith("benchmark_phash_forbidden:") for error in report["training_errors"]
    )


def test_full_train_smoke_receipt_is_bound_to_manifest_and_model(tmp_path: Path) -> None:
    manifest = tmp_path / "smoke-manifest.jsonl"
    smoke_rows = [
        {
            "sample_id": "positive",
            "split": "train",
            "source_id": "positive-source",
            "annotation_strength": "strong",
            "visual_abstention_reason": None,
        },
        {
            "sample_id": "negative",
            "split": "train",
            "source_id": "negative-source",
            "annotation_strength": "negative",
            "visual_abstention_reason": "no_target",
        },
        {
            "sample_id": "presence",
            "split": "train",
            "source_id": "presence-source",
            "annotation_strength": "strong_presence_only",
            "presence_supervised": True,
            "point_supervised": False,
            "visual_abstention_reason": None,
        },
        {
            "sample_id": "abstention",
            "split": "train",
            "source_id": "abstention-source",
            "annotation_strength": "strong",
            "visual_abstention_reason": "not_localizable",
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in smoke_rows), encoding="utf-8")
    contract = build_training_contract(
        manifest=manifest,
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        epochs=50,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        seed=42,
        image_size=448,
        num_workers=8,
        early_stopping_patience=8,
        balanced_sampling=True,
        role_targets=TEST_ROLE_TARGETS,
        pyro_share=0.33,
        samples_per_epoch=16,
        provenance={"composition_registry_sha256": "a" * 64},
    )
    smoke = tmp_path / "smoke-report.json"
    report = {
        "schema_version": 3,
        "model_schema_version": 4,
        "presence_labels": ["flame_visible", "smoke_visible"],
        "passed": True,
        "all_gradients_finite": True,
        "model_id": "facebook/dinov3",
        "model_revision": TEST_MODEL_REVISION,
        "manifest_sha256": _sha256(manifest),
        "training_contract": contract,
        "training_contract_sha256": training_contract_sha256(contract),
        "device": "cuda",
        "image_size": 448,
        "batch_size": 1,
        "smoke_steps": 4,
        "optimizer_steps": 1,
        "initialization": "immutable_base_pretrained",
        "initial_weights_sha256": None,
        "backbone_config_sha256": None,
        "sample_ids": ["positive", "negative", "presence", "abstention"],
        "observed_roles": {
            "positive": 1,
            "negative": 1,
            "presence": 1,
            "abstention": 1,
        },
        "observed_sources": {
            "abstention-source": 1,
            "negative-source": 1,
            "positive-source": 1,
            "presence-source": 1,
        },
        "sampling": {
            "target_role_shares": TEST_ROLE_TARGETS,
            "pyro_max_share": 0.33,
            "samples_per_epoch": 16,
        },
        "gradient_tensors": 4,
        "gradient_heads": {
            head: {"tensors": 1, "nonzero_tensors": 1, "max_abs": 0.1}
            for head in (
                "segmentation_head",
                "point_head",
                "abstention_head",
                "presence_head",
            )
        },
        "peak_vram_bytes": 1024,
        "trainable_parameters": 100,
        "loss_history": [
            {
                "loss": 1.0,
                "segmentation_loss": 0.2,
                "point_loss": 0.2,
                "abstention_loss": 0.2,
                "presence_loss": 0.2,
            }
            for _ in range(4)
        ],
    }
    smoke.write_text(json.dumps(report), encoding="utf-8")

    accepted = _validate_smoke_report(
        smoke,
        manifest=manifest,
        training_contract=contract,
        smoke_steps=4,
    )
    assert accepted["passed"] is True
    report["sample_ids"][0] = "invented-id"
    smoke.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="GPU smoke report contract failed"):
        _validate_smoke_report(
            smoke,
            manifest=manifest,
            training_contract=contract,
            smoke_steps=4,
        )


def test_preflight_rejects_bbox_derived_point_supervision(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["point_supervised"] = True
    rows[0]["anchor_points"] = [{"kind": "fire", "x": 0.5, "y": 0.5}]
    rows[0]["point_derivation"] = "bbox_center"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_hydration_report(tmp_path, manifest)

    report = build_preflight_report(
        pointing_root=tmp_path / "separate-pointing-not-used",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is False
    assert "bbox_derived_point_forbidden:1" in report["training_errors"]


def test_preflight_rejects_hidden_bbox_derivation_marker(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["point_supervised"] = True
    rows[0]["anchor_points"] = [{"kind": "fire", "x": 0.5, "y": 0.5}]
    rows[0]["point_derivation"] = "detector_bounding_box_center_projected"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_hydration_report(tmp_path, manifest)

    report = build_preflight_report(
        pointing_root=tmp_path / "separate-pointing-not-used",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is False
    assert "bbox_derived_point_forbidden:1" in report["training_errors"]


def test_preflight_rejects_negative_without_abstention_reason(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[2]["visual_abstention_reason"] = None
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_hydration_report(tmp_path, manifest)

    report = build_preflight_report(
        pointing_root=tmp_path / "separate-pointing-not-used",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is False
    assert "negative_abstention_reason_missing:3" in report["training_errors"]


def test_preflight_rejects_empty_or_contradictory_abstention(tmp_path: Path) -> None:
    manifest = _write_valid_composition(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[1]["visual_abstention_reason"] = "ambiguous_despite_point"
    rows[2]["visual_abstention_reason"] = ""
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_hydration_report(tmp_path, manifest)

    report = build_preflight_report(
        pointing_root=tmp_path / "separate-pointing-not-used",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is False
    assert "abstention_conflicts_with_point:2" in report["training_errors"]
    assert "visual_abstention_label_invalid:3" in report["training_errors"]


def test_preflight_rejects_non_composition_profile_and_undecodable_payload(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_composition(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["validation_profile"] = "legacy_unbound_profile"
    bad_image = tmp_path / rows[0]["image_relpath"]
    bad_image.write_bytes(b"not-an-image")
    rows[0]["image_sha256"] = _sha256(bad_image)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_hydration_report(tmp_path, manifest)

    report = build_preflight_report(
        pointing_root=tmp_path / "pointing",
        multitask_manifest=manifest,
        data_root=tmp_path,
        composition_registry=tmp_path / "composition-registry.json",
        model_id="facebook/dinov3",
        model_revision=TEST_MODEL_REVISION,
        **_chain_kwargs(tmp_path),
    )

    assert report["training_ready"] is False
    assert "composition_validation_profile_invalid:1" in report["training_errors"]
    assert any(
        error.startswith("artifact_decode_failed:1:image_relpath")
        for error in report["training_errors"]
    )
