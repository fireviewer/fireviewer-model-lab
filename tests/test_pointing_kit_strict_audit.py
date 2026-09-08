from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from fireviewer_model_lab.training.pointing_kit_strict_audit import ARCHIVE_FILENAME, audit_kit


def _tiff(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="TIFF")
    return output.getvalue()


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _build_archive(root: Path) -> Path:
    archive_path = root / ARCHIVE_FILENAME
    payloads: dict[str, tuple[bytes, bytes]] = {}
    for index in range(1, 4):
        rng = np.random.default_rng(index)
        image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4 + index : 12 + index, 8:20] = 255
        payloads[f"{index:06d}"] = (_tiff(image), _tiff(mask))
    locations = {
        "DataA": {"train": ["000003"], "test": ["000001", "000002"]},
        "DataB": {"train": ["000001", "000002"], "test": ["000003"]},
    }
    with tarfile.open(archive_path, "w") as archive:
        for dataset, splits in locations.items():
            for source_split, identifiers in splits.items():
                for identifier in identifiers:
                    image, mask = payloads[identifier]
                    prefix = f"Unnamed entity/Dataset/{dataset}/{source_split}"
                    _add(archive, f"{prefix}/images/{identifier}.tif", image)
                    _add(archive, f"{prefix}/masks/{identifier}.tif.tif", mask)
    return archive_path


def test_kit_strict_audit_keeps_human_masks_but_forbids_fake_points(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    archive = _build_archive(inventory)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "strict_combined_manifest.jsonl").write_text("", encoding="utf-8")
    output = tmp_path / "output"

    report = audit_kit(
        inventory_root=inventory,
        baseline_root=baseline,
        output_dir=output,
        output_s3_prefix="s3://bucket/kit-strict/job",
        expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        expected_train_rows=2,
        expected_test_rows=1,
        validation_rows=1,
        expected_image_size=32,
    )

    assert report["source_gate_passed"] is True
    assert report["strict_automated_validated_rows"] == 3
    assert report["validated_split_counts"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }
    assert report["data_a_b_exact_copy_matches"] == 3
    assert report["point_supervised_rows"] == 0
    rows = [
        json.loads(line)
        for line in (output / "kit_strict_validated_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 3
    assert all(row["point_supervised"] is False for row in rows)
    assert all(row["anchor_points"] == [] for row in rows)
    assert all(row["segmentation_supervised"] is True for row in rows)
    assert all(row["presence_targets"]["flame_visible"] is True for row in rows)
    assert all(row["visual_abstention_reason"] for row in rows)
    for row in rows:
        assert (output / row["image_relpath"]).is_file()
        assert (output / row["mask_relpath"]).is_file()

