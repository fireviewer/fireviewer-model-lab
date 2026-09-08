from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from fireviewer_model_lab.training.pointing_global_assemble import _difference_hash, assemble


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _random_image(seed: int) -> Image.Image:
    payload = b"".join(hashlib.sha256(f"{seed}:{index}".encode()).digest() for index in range(32))
    return Image.frombytes("L", (32, 32), payload).convert("RGB")


def _row(
    source_root: Path,
    *,
    sample_id: str,
    source_id: str,
    source_family: str,
    split: str,
    seed: int,
    kind: str | None,
) -> dict[str, Any]:
    image_relpath = f"images/{sample_id}.png"
    mask_relpath = f"masks/{sample_id}.png"
    image_path = source_root / image_relpath
    mask_path = source_root / mask_relpath
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = _random_image(seed)
    image.save(image_path, format="PNG")
    mask = Image.new("L", image.size, 0)
    if kind is not None:
        for y_value in range(24, 30):
            for x_value in range(10, 17):
                mask.putpixel((x_value, y_value), 255)
    mask.save(mask_path, format="PNG")
    is_negative = kind is None
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_id": source_id,
        "source_family": source_family,
        "source_revision": f"revision-{source_id}",
        "split": split,
        "final_split": split,
        "split_group": f"group:{sample_id}",
        "image_relpath": image_relpath,
        "mask_relpath": mask_relpath,
        "source_image_sha256": _sha256(image_path),
        "image_sha256": _sha256(image_path),
        "mask_sha256": _sha256(mask_path),
        "width": image.width,
        "height": image.height,
        "dhash": _difference_hash(image),
        "anchor_points": (
            [
                {
                    "kind": kind,
                    "x": 0.4,
                    "y": 0.8,
                    "origin": "source_mask_bottom_band_median",
                }
            ]
            if kind is not None
            else []
        ),
        "annotation_strength": "negative" if is_negative else "strong",
        "annotation_provenance": (
            "source_explicit_empty_mask" if is_negative else "source_pixel_mask"
        ),
        "mask_to_point_conversion": (
            "none" if is_negative else "deterministic_binary_mask_bottom_band_median"
        ),
        "mask_quality": "source_provided_strong",
        "visual_abstention_reason": None,
        "negative": is_negative,
        "negative_evidence": "source mask is explicitly empty" if is_negative else None,
        "sample_weight": 1.0,
        "variant": "clean",
        "media_license": "CC-BY-4.0",
        "mask_license": "MIT",
        "redistribution_allowed": True,
        "reviews_admitted": False,
        "validation_profile": "strict-test",
        "corpus_disposition": "eligible_genuinely_new_pool",
        "exclusion_reasons": [],
        "sample_validation_status": "strict_automated_validated",
        "training_eligible": True,
        "strict_keep": True,
    }


def _contract(manifest: Path, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_name": manifest.name,
        "manifest_sha256": _sha256(manifest),
        "source_id": row["source_id"],
        "source_family": row["source_family"],
        "source_revision": row["source_revision"],
        "validation_profile": "strict-test",
        "viewpoint": "ground_or_oblique",
        "allowed_licenses": ["CC-BY-4.0", "MIT"],
        "required_license_fields": ["media_license", "mask_license"],
        "allowed_annotation_provenances": [
            "source_pixel_mask",
            "source_explicit_empty_mask",
        ],
        "allowed_positive_annotation_strengths": ["strong"],
        "allowed_negative_annotation_strengths": ["negative"],
        "allowed_mask_to_point_conversions": [
            "deterministic_binary_mask_bottom_band_median",
            "none",
        ],
        "allowed_point_origins": ["source_mask_bottom_band_median"],
        "canonical_image_hash_field": "source_image_sha256",
        "artifacts": [
            {
                "role": "image",
                "path_field": "image_relpath",
                "sha256_field": "source_image_sha256",
            },
            {
                "role": "mask",
                "path_field": "mask_relpath",
                "sha256_field": "mask_sha256",
            },
        ],
        "redistribution_allowed": True,
        "reviews_admitted": False,
        "quarantines_admitted": False,
    }


def _quality(**overrides: int | float) -> dict[str, int | float]:
    values: dict[str, int | float] = {
        "unique_source_images_min": 1,
        "strict_automated_validated_point_images_min": 1,
        "fire_base_points_min": 0,
        "smoke_column_base_points_min": 0,
        "explicit_negative_images_min": 0,
        "source_families_min": 1,
        "largest_source_share_max": 1.0,
        "top_three_source_share_max": 1.0,
        "low_light_positive_images_min": 0,
        "small_or_faint_positive_images_min": 0,
        "validation_images_min": 0,
        "test_images_min": 0,
        "materialized_augmentation_fraction_max": 0.0,
        "unknown_semantics_max": 0,
        "invalid_geometry_max": 0,
        "exact_cross_split_duplicates_max": 0,
        "split_group_leaks_max": 0,
        "unknown_or_incompatible_rights_max": 0,
    }
    values.update(overrides)
    return values


def _registry(contracts: list[dict[str, Any]], **quality: int | float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": "test",
        "corpus_boundary": {
            "forbidden_references": [
                "fire-smoke-detection-corpus-v1",
                "benchdata",
                "independent-benchmark",
            ],
            "forbidden_source_families": ["FireBench"],
            "box_to_point_policy": "never_promote_box_bottom_center_to_ground_truth",
        },
        "strict_automation": {
            "validation_profile": "strict-test",
            "required_strict_manifest_names": [item["manifest_name"] for item in contracts],
            "strict_source_contracts": contracts,
            "reviews_admitted": False,
            "quarantines_admitted": False,
            "unknown_rights_admitted": False,
            "missing_point_annotations_admitted": False,
            "cross_split_overlaps_admitted": False,
            "ambiguous_semantics_admitted": False,
        },
        "quality_gates": _quality(**quality),
    }


def _write_registry(path: Path, registry: dict[str, Any]) -> Path:
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def test_global_assembly_binds_payloads_and_emits_hashed_publish_receipt(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    first_root = inputs / "source-a"
    second_root = inputs / "source-b"
    fire = _row(
        first_root,
        sample_id="fire-1",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=11,
        kind="fire_base",
    )
    smoke = _row(
        second_root,
        sample_id="smoke-1",
        source_id="source-b",
        source_family="family-b",
        split="validation",
        seed=29,
        kind="smoke_column_base",
    )
    assert (int(fire["dhash"], 16) ^ int(smoke["dhash"], 16)).bit_count() > 4
    first_manifest = _write_jsonl(first_root / "source_a_strict_validated_manifest.jsonl", [fire])
    second_manifest = _write_jsonl(
        second_root / "source_b_strict_validated_manifest.jsonl", [smoke]
    )
    registry_path = _write_registry(
        tmp_path / "registry.json",
        _registry(
            [_contract(first_manifest, fire), _contract(second_manifest, smoke)],
            unique_source_images_min=2,
            strict_automated_validated_point_images_min=2,
            fire_base_points_min=1,
            smoke_column_base_points_min=1,
            source_families_min=2,
            largest_source_share_max=0.5,
            validation_images_min=1,
        ),
    )
    output = tmp_path / "output"

    report = assemble(input_root=inputs, registry_path=registry_path, output_dir=output)

    assert report["publication_allowed"] is True
    assert report["payload_artifacts_verified"] == 4
    assert report["source_family_counts"] == {"family-a": 1, "family-b": 1}
    assert (output / "candidate_combined_manifest.jsonl").is_file()
    assert (output / "publish_ready_manifest.jsonl").is_file()
    assert (output / "strict_combined_manifest.jsonl").is_file()
    receipt = json.loads((output / "PUBLISH_READY_RECEIPT.json").read_text())
    assert receipt["manifest_sha256"] == _sha256(output / "publish_ready_manifest.jsonl")
    assert receipt["manifest_sha256"] == report["candidate_manifest_sha256"]


def test_red_quality_gate_emits_candidate_only_and_removes_stale_publish_files(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    source_root = inputs / "source-a"
    row = _row(
        source_root,
        sample_id="fire-1",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=11,
        kind="fire_base",
    )
    manifest = _write_jsonl(source_root / "source_a_strict_validated_manifest.jsonl", [row])
    registry_path = _write_registry(
        tmp_path / "registry.json",
        _registry([_contract(manifest, row)], unique_source_images_min=2),
    )
    output = tmp_path / "output"
    output.mkdir()
    for stale_name in (
        "publish_ready_manifest.jsonl",
        "strict_combined_manifest.jsonl",
        "PUBLISH_READY_RECEIPT.json",
    ):
        (output / stale_name).write_text("stale", encoding="utf-8")

    report = assemble(input_root=inputs, registry_path=registry_path, output_dir=output)

    assert report["publication_allowed"] is False
    assert report["blocking_gates"] == ["unique_source_images_min"]
    assert (output / "candidate_combined_manifest.jsonl").is_file()
    assert not (output / "publish_ready_manifest.jsonl").exists()
    assert not (output / "strict_combined_manifest.jsonl").exists()
    assert not (output / "PUBLISH_READY_RECEIPT.json").exists()


def test_global_assembly_requires_immutable_source_contracts(tmp_path: Path) -> None:
    registry = _registry([])
    registry["strict_automation"].pop("strict_source_contracts")
    registry_path = _write_registry(tmp_path / "registry.json", registry)

    with pytest.raises(ValueError, match="requires strict_source_contracts"):
        assemble(
            input_root=tmp_path / "inputs",
            registry_path=registry_path,
            output_dir=tmp_path / "out",
        )


def test_global_assembly_rejects_manifest_digest_mismatch(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    source_root = inputs / "source-a"
    row = _row(
        source_root,
        sample_id="fire-1",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=11,
        kind="fire_base",
    )
    manifest = _write_jsonl(source_root / "source_a_strict_validated_manifest.jsonl", [row])
    contract = _contract(manifest, row)
    contract["manifest_sha256"] = "0" * 64
    registry_path = _write_registry(tmp_path / "registry.json", _registry([contract]))

    with pytest.raises(ValueError, match="strict manifest sha256 mismatch"):
        assemble(input_root=inputs, registry_path=registry_path, output_dir=tmp_path / "out")


def test_global_assembly_rejects_payload_checksum_mismatch(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    source_root = inputs / "source-a"
    row = _row(
        source_root,
        sample_id="fire-1",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=11,
        kind="fire_base",
    )
    manifest = _write_jsonl(source_root / "source_a_strict_validated_manifest.jsonl", [row])
    contract = _contract(manifest, row)
    (source_root / row["image_relpath"]).write_bytes(b"corrupt")
    registry_path = _write_registry(tmp_path / "registry.json", _registry([contract]))

    with pytest.raises(ValueError, match="payload checksum mismatch"):
        assemble(input_root=inputs, registry_path=registry_path, output_dir=tmp_path / "out")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(annotation_strength="teacher_generated_weak"), "not contracted"),
        (lambda row: row.update(objects={"bbox": [[1, 2, 3, 4]]}), "box annotation"),
        (lambda row: row.update(ground_view=False), "non-ground view"),
        (
            lambda row: row.update(image_s3_uri="s3://bucket/benchdata/image.png"),
            "detection or independent-benchmark",
        ),
        (lambda row: row.update(reviews_admitted=True), "review-derived"),
        (lambda row: row.update(redistribution_allowed=False), "redistribution"),
        (lambda row: row.update(media_license="unknown"), "licence"),
        (lambda row: row.update(dhash=""), "dhash is missing"),
        (lambda row: row.update(split="quarantine", final_split="quarantine"), "unsupported split"),
        (lambda row: row.update(split_group=""), "split_group is missing"),
        (lambda row: row.update(corpus_disposition="quarantine"), "not eligible"),
    ],
)
def test_global_assembly_rejects_non_strict_rows(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    inputs = tmp_path / "inputs"
    source_root = inputs / "source-a"
    row = _row(
        source_root,
        sample_id="fire-1",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=11,
        kind="fire_base",
    )
    mutate(row)
    manifest = _write_jsonl(source_root / "source_a_strict_validated_manifest.jsonl", [row])
    registry_path = _write_registry(
        tmp_path / "registry.json", _registry([_contract(manifest, row)])
    )

    with pytest.raises(ValueError, match=message):
        assemble(input_root=inputs, registry_path=registry_path, output_dir=tmp_path / "out")


@pytest.mark.parametrize(
    ("abstention", "message"),
    [
        (None, "negative row also contains positive anchor points"),
        ("not visible", "abstentions cannot enter"),
    ],
)
def test_negative_must_be_exclusive_explicit_and_non_abstaining(
    tmp_path: Path, abstention: str | None, message: str
) -> None:
    inputs = tmp_path / "inputs"
    source_root = inputs / "source-a"
    row = _row(
        source_root,
        sample_id="negative-1",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=11,
        kind=None,
    )
    row["anchor_points"] = [
        {
            "kind": "fire_base",
            "x": 0.4,
            "y": 0.8,
            "origin": "source_mask_bottom_band_median",
        }
    ]
    row["visual_abstention_reason"] = abstention
    manifest = _write_jsonl(source_root / "source_a_strict_validated_manifest.jsonl", [row])
    registry_path = _write_registry(
        tmp_path / "registry.json", _registry([_contract(manifest, row)])
    )

    with pytest.raises(ValueError, match=message):
        assemble(input_root=inputs, registry_path=registry_path, output_dir=tmp_path / "out")


def test_valid_explicit_negative_is_counted_once(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    source_root = inputs / "source-a"
    row = _row(
        source_root,
        sample_id="negative-1",
        source_id="source-a",
        source_family="family-a",
        split="test",
        seed=11,
        kind=None,
    )
    manifest = _write_jsonl(source_root / "source_a_strict_validated_manifest.jsonl", [row])
    registry_path = _write_registry(
        tmp_path / "registry.json",
        _registry([_contract(manifest, row)], explicit_negative_images_min=1, test_images_min=1),
    )

    report = assemble(input_root=inputs, registry_path=registry_path, output_dir=tmp_path / "out")

    assert report["publication_allowed"] is True
    assert report["explicit_negative_images"] == 1
    assert report["fire_base_points"] == 0
    assert report["smoke_column_base_points"] == 0


def test_global_near_duplicate_detection_includes_same_split(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    source_root = inputs / "source-a"
    first = _row(
        source_root,
        sample_id="fire-1",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=11,
        kind="fire_base",
    )
    second = _row(
        source_root,
        sample_id="fire-2",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=29,
        kind="fire_base",
    )
    first_image = Image.open(source_root / first["image_relpath"]).copy()
    first_image.putpixel((0, 0), (0, 0, 0))
    second_image_path = source_root / second["image_relpath"]
    first_image.save(second_image_path, format="PNG")
    second["source_image_sha256"] = _sha256(second_image_path)
    second["image_sha256"] = second["source_image_sha256"]
    second["dhash"] = _difference_hash(first_image)
    assert second["source_image_sha256"] != first["source_image_sha256"]
    assert (int(second["dhash"], 16) ^ int(first["dhash"], 16)).bit_count() <= 4
    manifest = _write_jsonl(
        source_root / "source_a_strict_validated_manifest.jsonl", [first, second]
    )
    registry_path = _write_registry(
        tmp_path / "registry.json", _registry([_contract(manifest, first)])
    )

    with pytest.raises(ValueError, match="including within-split pairs"):
        assemble(input_root=inputs, registry_path=registry_path, output_dir=tmp_path / "out")


def test_source_family_gate_uses_contract_family_not_source_id(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    contracts: list[dict[str, Any]] = []
    for index, seed in enumerate((11, 29), 1):
        root = inputs / f"source-{index}"
        row = _row(
            root,
            sample_id=f"fire-{index}",
            source_id=f"source-{index}",
            source_family="same-real-family",
            split="train",
            seed=seed,
            kind="fire_base",
        )
        manifest = _write_jsonl(root / f"source_{index}_strict_validated_manifest.jsonl", [row])
        contracts.append(_contract(manifest, row))
    registry_path = _write_registry(
        tmp_path / "registry.json", _registry(contracts, source_families_min=2)
    )

    report = assemble(input_root=inputs, registry_path=registry_path, output_dir=tmp_path / "out")

    assert report["source_counts"] == {"source-1": 1, "source-2": 1}
    assert report["source_family_counts"] == {"same-real-family": 2}
    assert report["blocking_gates"] == ["source_families_min"]
    assert report["publication_allowed"] is False


def test_global_assembly_rejects_cross_split_group_leakage(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    source_root = inputs / "source-a"
    first = _row(
        source_root,
        sample_id="fire-1",
        source_id="source-a",
        source_family="family-a",
        split="train",
        seed=11,
        kind="fire_base",
    )
    second = _row(
        source_root,
        sample_id="fire-2",
        source_id="source-a",
        source_family="family-a",
        split="validation",
        seed=29,
        kind="fire_base",
    )
    second["split_group"] = first["split_group"]
    manifest = _write_jsonl(
        source_root / "source_a_strict_validated_manifest.jsonl", [first, second]
    )
    registry_path = _write_registry(
        tmp_path / "registry.json", _registry([_contract(manifest, first)])
    )

    with pytest.raises(ValueError, match="split-group leakage"):
        assemble(input_root=inputs, registry_path=registry_path, output_dir=tmp_path / "out")
