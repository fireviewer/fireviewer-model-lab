import io
import zipfile

from PIL import Image

from fireviewer_model_lab.training.multinatsmoke_payload_audit import run_audit


def _png(mode, size, value):
    handle = io.BytesIO()
    Image.new(mode, size, value).save(handle, format="PNG")
    return handle.getvalue()


def test_payload_audit_is_non_promoting_and_detects_cross_split_duplicate(tmp_path):
    archive_path = tmp_path / "sample.zip"
    image = _png("RGB", (16, 12), (80, 120, 90))
    mask = Image.new("L", (16, 12), 0)
    for x in range(5, 10):
        for y in range(3, 11):
            mask.putpixel((x, y), 255)
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")
    with zipfile.ZipFile(archive_path, "w") as archive:
        for split, name in (("Train", "a"), ("Test", "b")):
            root = f"MultiNatSmokeDataset/{split}/D-Fire"
            archive.writestr(f"{root}/images/{name}.png", image)
            archive.writestr(f"{root}/masks/{name}.png", mask_bytes.getvalue())

    report = run_audit(
        source=archive_path,
        output_dir=tmp_path / "out",
        allowed_sources={"D-Fire"},
    )

    assert report["rows"] == 2
    assert report["rows_decoded"] == 2
    assert report["training_eligible_rows"] == 0
    assert report["cross_upstream_split_exact_groups"] == 1
    assert report["upstream_split_accepted"] is False
    assert report["publication_allowed"] is False
