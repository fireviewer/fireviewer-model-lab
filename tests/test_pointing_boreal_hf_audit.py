from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
from fireviewer_model_lab.training.pointing_boreal_hf_audit import SOURCE_ROOT, audit_boreal_archive


def _image_bytes(mode: str, color: int | tuple[int, int, int], suffix: str) -> bytes:
    image = Image.new(mode, (24, 16), color)
    if mode == "L":
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 3, 18, 14), fill=255)
    output = io.BytesIO()
    image.save(output, format="PNG" if suffix == ".png" else "JPEG")
    return output.getvalue()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_row(sample_id: str, split: str, artifact_path: str, artifact: bytes) -> dict:
    return {
        "sample_id": sample_id,
        "source_id": "boreal-forest-fire-segmentation-v1",
        "source_record_id": sample_id,
        "split": split,
        "split_group": f"site:{split}",
        "license": "CC-BY-4.0",
        "artifact": {"path": artifact_path, "sha256": _sha(artifact)},
    }


def test_boreal_admits_only_strong_source_masks_without_new_review(tmp_path: Path) -> None:
    archive_path = tmp_path / "boreal.zip"
    image = _image_bytes("RGB", (180, 50, 20), ".jpg")
    mask = _image_bytes("L", 0, ".png")
    source_rows = []
    members: dict[str, bytes] = {}
    for index, strength in enumerate(("strong", "weak"), 1):
        sample_id = f"boreal:{index}"
        image_relpath = f"payload/image-{index}.jpg"
        mask_relpath = f"payload/mask-{index}.png"
        artifact_relpath = f"samples/sample-{index}.json"
        sample = {
            "sample_id": sample_id,
            "source_id": "boreal-forest-fire-segmentation-v1",
            "image": {"path": image_relpath, "sha256": _sha(image)},
            "annotation": {"path": mask_relpath, "sha256": _sha(mask)},
            "annotation_strength": strength,
            "annotation_provenance": (
                "human_pixel_mask" if strength == "strong" else "sam_generated_from_manual_box"
            ),
        }
        artifact = json.dumps(sample).encode()
        source_rows.append(_source_row(sample_id, "train", artifact_relpath, artifact))
        members[SOURCE_ROOT + artifact_relpath] = artifact
        members[SOURCE_ROOT + image_relpath] = image
        members[SOURCE_ROOT + mask_relpath] = mask
    members[SOURCE_ROOT + "manifest.jsonl"] = "".join(
        json.dumps(row) + "\n" for row in source_rows
    ).encode()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "pixel_inventory.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "seed:1",
                "status": "ok",
                "sha256": "f" * 64,
                "dhash": "f" * 16,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    report = audit_boreal_archive(
        archive_path=archive_path,
        baseline_root=baseline,
        output_dir=output,
        output_s3_prefix="s3://bucket/pointing-corpus-v2/boreal/run",
    )

    validated = [
        json.loads(line)
        for line in (output / "boreal_strict_validated_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    dispositions = [
        json.loads(line)
        for line in (output / "boreal_automatic_dispositions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert report["strict_automated_validated_rows"] == 1
    assert report["reviews_admitted"] is False
    assert len(validated) == 1
    assert validated[0]["anchor_points"][0]["kind"] == "smoke_column_base"
    assert validated[0]["sample_validation_status"] == "strict_automated_validated"
    weak = next(row for row in dispositions if row["annotation_strength"] == "weak")
    assert "weak_sam_mask_not_strict_ground_truth" in weak["exclusion_reasons"]
    assert not weak["training_eligible"]
    assert (output / validated[0]["image_relpath"]).is_file()
    assert (output / validated[0]["mask_relpath"]).is_file()
