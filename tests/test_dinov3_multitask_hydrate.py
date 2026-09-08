from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from fireviewer_model_lab.training.dinov3_corpus_identity import (
    DECODED_PIXEL_HASH_ALGORITHM,
    PERCEPTUAL_HASH_ALGORITHM,
    deterministic_source_family_id,
    phash64_imagehash_v1,
    resolve_namespaced_source_identity,
    resolve_source_identity,
    validate_source_identity_contract,
)
from fireviewer_model_lab.training.dinov3_multitask_hydrate import (
    _strict_binary_array,
    _validate_row,
    hydrate_composition,
)


def _png(array: np.ndarray, *, compress_level: int = 6) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=compress_level)
    return buffer.getvalue()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_denylist(
    root: Path, *, raw_hashes: list[str] | None = None, phashes: list[str] | None = None
) -> Path:
    path = root / "benchmark-denylist.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "fireviewer-independent-benchmark-hash-denylist",
                "hash_only": True,
                "provenance_guard_sha256": "d" * 64,
                "provenance_guard_rows": 1,
                "raw_image_sha256": sorted(set(raw_hashes or [])),
                "decoded_pixel_sha256": [],
                "phash64_imagehash_v1": sorted(set(phashes or [])),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_composition(
    root: Path,
    rows: list[dict],
    *,
    professional: bool = False,
    benchmark_raw_hashes: list[str] | None = None,
    benchmark_phashes: list[str] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    campaign_id = "test-dinov3-composition"
    denylist = _write_denylist(root, raw_hashes=benchmark_raw_hashes, phashes=benchmark_phashes)
    family_root = "test-dataset:camp-swift"
    family_id = deterministic_source_family_id([family_root])
    boundary = {
        "independent_benchmark_must_remain_separate": True,
        "forbidden_references": ["independent-benchmark"],
        "denylist_schema_version": 1,
        "denylist_sha256": _sha(denylist.read_bytes()),
        "provenance_guard_sha256": "d" * 64,
        "provenance_guard_rows": 1,
        "denylist_entry_counts": {
            "raw_image_sha256": len(set(benchmark_raw_hashes or [])),
            "decoded_pixel_sha256": 0,
            "phash64_imagehash_v1": len(set(benchmark_phashes or [])),
        },
        "decoded_pixel_hash_algorithm": DECODED_PIXEL_HASH_ALGORITHM,
        "perceptual_hash_algorithm": PERCEPTUAL_HASH_ALGORITHM,
        "phash_hamming_distance_max": 3,
    }
    registry_value = {
        "schema_version": 2,
        "campaign_id": campaign_id,
        "detection_base": {
            "repository": "owner/detection",
            "revision": "immutable-revision",
        },
        "overlay_sources": [
            {
                "name": "camp-swift",
                "source_id": "camp-swift",
                "source_revision": "camp-revision",
                "manifest_sha256": "b" * 64,
                "source_family_id": family_id,
            }
        ],
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
                            "name": "camp-swift",
                            "event_key_field": "split_group",
                        }
                    ],
                }
            ],
            "event_aliases": [],
        },
        "benchmark_boundary": boundary,
    }
    registry = root / "composition-registry.json"
    registry.write_text(
        json.dumps(registry_value) + "\n",
        encoding="utf-8",
    )
    registry_sha256 = _sha(registry.read_bytes())
    identities = validate_source_identity_contract(registry_value)
    bound_rows: list[dict] = []
    for source in rows:
        row = dict(source)
        source_group = str(row["split_group"])
        if row["image_locator"]["kind"] == "hf_dataset_row":
            identity = resolve_namespaced_source_identity(
                lineage_root_id="hf-dataset:owner/detection",
                source_event_key=f"{row['source_id']}:{source_group}",
            )
            identity["source_split_group"] = source_group
            overlay_sources: list[dict] = []
        else:
            identity = resolve_source_identity(
                {"split_group": source_group},
                binding_kind="overlay",
                binding_name="camp-swift",
                identities=identities,
            )
            overlay_sources = [
                {
                    "name": "camp-swift",
                    "source_id": "camp-swift",
                    "source_revision": "camp-revision",
                    "manifest_sha256": "b" * 64,
                    **identity,
                }
            ]
        bound_rows.append(
            {
                **row,
                "schema_version": 2,
                "split_group": f"event:{identity['canonical_event_id']}",
                "overlay_sources": overlay_sources,
                **identity,
                "sample_validation_status": "strict_automated_validated",
                "validation_profile": "fireviewer_multitask_composition_v1",
                "campaign_id": campaign_id,
                "composition_registry_sha256": registry_sha256,
            }
        )
    rows = bound_rows
    manifest = root / "composition_candidate_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    integrity_receipt = root / "composition_integrity_receipt.json"
    integrity_receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "campaign_id": campaign_id,
                "composition_registry_sha256": registry_sha256,
                "source_identity_contract_sha256": identities.contract_sha256,
                "benchmark_denylist_sha256": _sha(denylist.read_bytes()),
                "manifest_sha256": _sha(manifest.read_bytes()),
                "composition_rows": len(rows),
                "detection_revision": "a" * 40,
                "integrity_gates_passed": True,
                "pilot_corpus_ready": True,
                "professional_corpus_ready": professional,
                "publication_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    report = root / "composition_report.json"
    hard_gates = {"perceptual_near_duplicate_pairs_max": 0}
    quality_gates = {"pilot_point_positive_rows_min": 1}
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "integrity_gates_passed": True,
                "campaign_id": campaign_id,
                "composition_registry_sha256": registry_sha256,
                "source_identity_contract_sha256": identities.contract_sha256,
                "benchmark_denylist_sha256": _sha(denylist.read_bytes()),
                "composition_integrity_receipt_sha256": _sha(integrity_receipt.read_bytes()),
                "detection_revision": "a" * 40,
                "hard_gates": hard_gates,
                "quality_gates": quality_gates,
                "pilot_corpus_ready": True,
                "professional_corpus_ready": professional,
                "quality_gate_deficits": (
                    {} if professional else {"points": {"actual": 3, "minimum": 2000}}
                ),
                "publication_allowed": False,
                "composition_rows": len(rows),
                "manifest_sha256": _sha(manifest.read_bytes()),
            }
        ),
        encoding="utf-8",
    )
    return manifest, report, integrity_receipt, registry, denylist


def _hf_row(sample_id: str, split: str, payload: bytes, *, negative: bool) -> dict:
    digest = _sha(payload)
    row = {
        "sample_id": sample_id,
        "source_id": "detection",
        "split": split,
        "split_group": f"group:{sample_id}",
        "image_sha256": digest,
        "image_extension": ".png",
        "image_locator": {
            "kind": "hf_dataset_row",
            "repository": "owner/detection",
            "revision": "immutable-revision",
            "split": split,
            "sample_id": sample_id,
            "sha256": digest,
        },
        "anchor_points": [],
        "presence_supervised": True,
        "presence_targets": {"flame_visible": not negative, "smoke_visible": False},
        "presence_provenance": "strict_detection",
        "sample_weight": 1.0,
    }
    if negative:
        row.update(
            {
                "annotation_strength": "negative",
                "segmentation_supervised": True,
                "point_supervised": True,
                "abstention_supervised": True,
                "mask_encoding": "implicit_zero_from_explicit_negative",
                "mask_quality": "strict_zero",
                "visual_abstention_reason": "no_target_visible",
            }
        )
    else:
        row.update(
            {
                "annotation_strength": "strong_presence_only",
                "segmentation_supervised": False,
                "point_supervised": False,
                "abstention_supervised": False,
                "visual_abstention_reason": None,
            }
        )
    return row


def test_hydration_streams_hf_bytes_and_s3_overlays_with_global_gates(tmp_path: Path) -> None:
    train_array = np.zeros((40, 48, 3), dtype=np.uint8)
    train_array[:, :, 0] = np.arange(48, dtype=np.uint8)
    validation_array = np.zeros((40, 48, 3), dtype=np.uint8)
    validation_array[:, :, 1] = np.arange(40, dtype=np.uint8)[:, None]
    overlay_array = np.full((40, 48, 3), (220, 80, 20), dtype=np.uint8)
    train_payload = _png(train_array)
    validation_payload = _png(validation_array)
    overlay_payload = _png(overlay_array)
    mask_array = np.zeros((40, 48), dtype=np.uint8)
    mask_array[12:34, 14:38] = 255
    mask_payload = _png(mask_array)
    train = _hf_row("train-positive", "train", train_payload, negative=False)
    validation = _hf_row("validation-negative", "validation", validation_payload, negative=True)
    overlay = {
        "sample_id": "camp-overlay",
        "source_id": "camp-swift",
        "split": "test",
        "split_group": "camp:block",
        "image_sha256": _sha(overlay_payload),
        "image_extension": ".png",
        "image_locator": {
            "kind": "s3_object",
            "uri": "s3://bucket/overlay-image.png",
            "sha256": _sha(overlay_payload),
            "extension": ".png",
        },
        "mask_locator": {
            "kind": "s3_object",
            "uri": "s3://bucket/overlay-mask.png",
            "sha256": _sha(mask_payload),
            "extension": ".png",
        },
        "annotation_strength": "strong",
        "segmentation_supervised": True,
        "point_supervised": True,
        "presence_supervised": True,
        "abstention_supervised": True,
        "presence_targets": {"flame_visible": True, "smoke_visible": False},
        "presence_provenance": "strict_mask",
        "mask_quality": "strict_mask",
        "anchor_points": [
            {
                "kind": "fire_base",
                "x": round(25.5 / 47, 8),
                "y": round(33 / 39, 8),
            }
        ],
        "point_derivation": "sensor_mask_bottom_band_median",
        "visual_abstention_reason": None,
        "sample_weight": 2.0,
    }
    manifest, source_report, integrity_receipt, registry, denylist = _write_composition(
        tmp_path, [train, validation, overlay]
    )
    hf_payloads = {
        ("train", "train-positive"): train_payload,
        ("validation", "validation-negative"): validation_payload,
    }

    def hf_loader(_repository: str, _revision: str, split: str) -> list[dict]:
        return [
            {
                "sample_id": sample_id,
                "sha256": _sha(payload),
                "image": {"bytes": payload, "path": f"{sample_id}.png"},
            }
            for (payload_split, sample_id), payload in hf_payloads.items()
            if payload_split == split
        ]

    s3_payloads = {
        "s3://bucket/overlay-image.png": overlay_payload,
        "s3://bucket/overlay-mask.png": mask_payload,
    }

    report = hydrate_composition(
        composition_manifest=manifest,
        composition_report=source_report,
        composition_integrity_receipt=integrity_receipt,
        composition_registry=registry,
        benchmark_denylist_path=denylist,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "hydrated",
        hf_loader=hf_loader,
        s3_fetcher=lambda locator: s3_payloads[locator["uri"]],
        validation_workers=2,
    )

    assert report["hydration_integrity_passed"] is True
    assert report["ready_for_gpu_finite_loss_smoke"] is True
    assert report["training_ready"] is False
    assert report["verified_image_rows"] == 3
    assert report["hf_materialization"]["downloaded_files"] == 2
    assert report["s3_materialization"]["downloaded_files"] == 2
    assert report["perceptual_near_duplicates"]["cross_split_pairs"] == 0
    assert (tmp_path / "hydrated" / "hydration_integrity_receipt.json").is_file()
    hydrated = [
        json.loads(line)
        for line in (tmp_path / "hydrated" / report["manifest"]).read_text().splitlines()
    ]
    assert all((tmp_path / "data" / row["image_relpath"]).is_file() for row in hydrated)
    assert next(row for row in hydrated if row["sample_id"] == "camp-overlay")[
        "mask_relpath"
    ].startswith("masks/")


def test_hydration_blocks_decoded_duplicates_across_splits(tmp_path: Path) -> None:
    pixels = np.full((36, 36, 3), (30, 90, 180), dtype=np.uint8)
    train_payload = _png(pixels, compress_level=0)
    test_payload = _png(pixels, compress_level=9)
    rows = [
        _hf_row("same-train", "train", train_payload, negative=False),
        _hf_row("same-test", "test", test_payload, negative=False),
    ]
    manifest, source_report, integrity_receipt, registry, denylist = _write_composition(
        tmp_path, rows
    )
    payloads = {"same-train": train_payload, "same-test": test_payload}

    def hf_loader(_repository: str, _revision: str, split: str) -> list[dict]:
        sample_id = "same-train" if split == "train" else "same-test"
        payload = payloads[sample_id]
        return [{"sample_id": sample_id, "sha256": _sha(payload), "image": {"bytes": payload}}]

    report = hydrate_composition(
        composition_manifest=manifest,
        composition_report=source_report,
        composition_integrity_receipt=integrity_receipt,
        composition_registry=registry,
        benchmark_denylist_path=denylist,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "hydrated",
        hf_loader=hf_loader,
        s3_fetcher=lambda _locator: b"",
        validation_workers=2,
    )

    assert report["hydration_integrity_passed"] is False
    assert report["decoded_pixel_duplicate_groups"] == 1
    assert report["perceptual_near_duplicates"]["cross_split_pairs"] == 1
    assert not (tmp_path / "hydrated" / "hydration_integrity_receipt.json").exists()


def test_float_mask_values_cannot_pass_binary_validation() -> None:
    mask = np.array([[0.0, 0.9], [0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="floating dtype"):
        _strict_binary_array(mask, label="supervised mask", sample_id="float-mask")


def test_declared_bottom_band_median_point_must_match_exact_mask_geometry(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((40, 48, 3), 80, dtype=np.uint8)).save(image_path)
    mask = np.zeros((40, 48), dtype=np.uint8)
    mask[12:34, 14:38] = 255
    Image.fromarray(mask).save(mask_path)
    row = {
        "sample_id": "wrong-lateral-point",
        "split": "train",
        "image_relpath": image_path.name,
        "image_sha256": _sha(image_path.read_bytes()),
        "mask_relpath": mask_path.name,
        "mask_sha256": _sha(mask_path.read_bytes()),
        "annotation_strength": "strong",
        "presence_targets": {"flame_visible": True, "smoke_visible": False},
        "anchor_points": [{"kind": "fire_base", "x": round(14 / 47, 8), "y": round(33 / 39, 8)}],
        "point_derivation": "sensor_mask_bottom_band_median",
    }

    with pytest.raises(ValueError, match="does not match mask"):
        _validate_row(row, tmp_path)


def test_hydration_blocks_near_duplicates_within_one_split(tmp_path: Path) -> None:
    first_pixels = np.full((36, 36, 3), 80, dtype=np.uint8)
    second_pixels = np.full((36, 36, 3), 81, dtype=np.uint8)
    first_payload = _png(first_pixels)
    second_payload = _png(second_pixels)
    rows = [
        _hf_row("near-one", "train", first_payload, negative=False),
        _hf_row("near-two", "train", second_payload, negative=False),
    ]
    manifest, source_report, integrity_receipt, registry, denylist = _write_composition(
        tmp_path, rows
    )
    payloads = {"near-one": first_payload, "near-two": second_payload}

    def hf_loader(_repository: str, _revision: str, _split: str) -> list[dict]:
        return [
            {
                "sample_id": sample_id,
                "sha256": _sha(payload),
                "image": {"bytes": payload},
            }
            for sample_id, payload in payloads.items()
        ]

    report = hydrate_composition(
        composition_manifest=manifest,
        composition_report=source_report,
        composition_integrity_receipt=integrity_receipt,
        composition_registry=registry,
        benchmark_denylist_path=denylist,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "hydrated",
        hf_loader=hf_loader,
        s3_fetcher=lambda _locator: b"",
        validation_workers=2,
    )

    assert report["hydration_integrity_passed"] is False
    assert report["decoded_pixel_duplicate_groups"] == 0
    assert report["perceptual_near_duplicates"]["within_split_pairs"] == 1
    assert "within_split_near_duplicate_pairs:1" in report["hard_gate_errors"]


def test_hydration_blocks_guard_phash_after_payload_rehash(tmp_path: Path) -> None:
    pixels = np.zeros((40, 48, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(48, dtype=np.uint8)
    pixels[:, :, 1] = np.arange(40, dtype=np.uint8)[:, None]
    payload = _png(pixels)
    with Image.open(BytesIO(payload)) as image:
        guarded_phash = f"{phash64_imagehash_v1(image):016x}"
    row = _hf_row("guarded-phash", "train", payload, negative=False)
    manifest, source_report, integrity_receipt, registry, denylist = _write_composition(
        tmp_path,
        [row],
        benchmark_phashes=[guarded_phash],
    )

    report = hydrate_composition(
        composition_manifest=manifest,
        composition_report=source_report,
        composition_integrity_receipt=integrity_receipt,
        composition_registry=registry,
        benchmark_denylist_path=denylist,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "hydrated",
        hf_loader=lambda _repository, _revision, _split: [
            {
                "sample_id": "guarded-phash",
                "sha256": _sha(payload),
                "image": {"bytes": payload},
            }
        ],
        s3_fetcher=lambda _locator: b"",
    )

    assert report["hydration_integrity_passed"] is False
    assert report["benchmark_phash_matches"] == 1
    assert "benchmark_phash_matches:1" in report["hard_gate_errors"]
