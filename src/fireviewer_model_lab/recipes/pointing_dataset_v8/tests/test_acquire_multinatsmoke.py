import io
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image

from training.pointing_dataset_v8 import acquire_multinatsmoke as acquire


def test_exact_source_and_split_pairing():
    names = [f"root/{s}/{f}/{kind}/a.{ext}" for s in ("Train", "Test", "TestSmall")
             for f in ("AusSmoke", "FireSpot", "BorealForestFire") for kind, ext in (("images", "jpg"), ("masks", "png"))]
    pairs, incomplete = acquire.build_pairs([SimpleNamespace(filename=n, file_size=100) for n in names])
    assert len(pairs) == 4
    assert incomplete == []
    assert {r["family"] for r in pairs} == {"AusSmoke", "FireSpot"}
    assert all("TestSmall" not in r["image_member"] for r in pairs)


def test_duplicate_member_refused():
    member = SimpleNamespace(filename="root/Train/FireSpot/images/a.jpg", file_size=100)
    with pytest.raises(ValueError, match="Duplicate"):
        acquire.build_pairs([member, member])


def test_release_alias_and_apple_metadata():
    assert acquire.member_role("root/Train/AuSmoke/images/a.jpg") == ("AusSmoke", "train", "image")
    assert acquire.member_role("__MACOSX/root/Train/AuSmoke/images/._a.jpg") is None


def test_mask_envelope_not_components_and_not_negative():
    mask = {"unique_gray_value_count": 2, "unique_gray_values": [0, 255],
            "foreground_bbox_xyxy_nonzero": [10, 20, 30, 50]}
    assert acquire.mask_envelope({"mask": mask}) == {"bbox": [[10, 20, 20, 30]], "category": [1], "area": [600]}
    with pytest.raises(ValueError, match="Empty"):
        acquire.mask_envelope({"mask": mask | {"foreground_bbox_xyxy_nonzero": None}})
    with pytest.raises(ValueError, match="not binary"):
        acquire.mask_envelope({"mask": mask | {"unique_gray_values": [0, 127]}})


def test_budgets_refuse_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire.pilot.RecordingRemoteFetcher, "fetch", lambda *a, **k: pytest.fail("network contacted"))
    fetcher = acquire.CachedRangeFetcher("https://example.invalid/archive.zip", request_log=[],
        cache_root=tmp_path, max_total_bytes=20)
    with pytest.raises(ValueError, match="byte budget"):
        fetcher.fetch((0, 20))
    with pytest.raises(ValueError, match="full-archive"):
        fetcher.fetch((0, acquire.MAX_RANGE_BYTES))
    with pytest.raises(ValueError, match="Unbounded"):
        fetcher.fetch((0, None))


def test_index_cache_validated_and_never_reused_for_media(tmp_path, monkeypatch):
    cached = tmp_path / "10-19.bin"
    cached.write_bytes(b"0123456789")
    acquire.write_json(cached.with_suffix(".json"), {"revision": acquire.pilot.HF_REVISION,
        "sha256": acquire.pilot.sha256_bytes(cached.read_bytes())})
    fetcher = acquire.CachedRangeFetcher("https://example.invalid/archive.zip", request_log=[],
        cache_root=tmp_path, max_total_bytes=100)
    monkeypatch.setattr(acquire.pilot.RecordingRemoteFetcher, "fetch", lambda *a, **k: pytest.fail("network contacted"))
    assert fetcher.fetch((10, 19), stream=False).read() == b"0123456789"
    cached.write_bytes(b"abcdefghij")
    with pytest.raises(ValueError, match="index changed"):
        fetcher.fetch((10, 19), stream=False)


def test_source_test_cannot_enter_train(tmp_path):
    with pytest.raises(ValueError, match="Source test"):
        acquire.acquire_pair(None, {"source_split": "test"}, tmp_path)


def test_selection_is_bounded_and_excludes_aerial_artificial_test():
    rows = [{"family": "AusSmoke", "source_split": "train", "stem": f"clip_{i:06d}-+{i:06d}",
             "image_bytes": 100, "mask_bytes": 10} for i in range(100)]
    rows += [rows[0] | {"stem": "dji_000123-+000100"}, rows[0] | {"stem": "smokemachine_000123-+000100"},
             rows[0] | {"source_split": "test"}]
    chosen, report = acquire.select_pairs(rows)
    assert len(chosen) == 8
    assert len({r["stem"] for r in chosen}) == 8
    assert {r["source_group_id"] for r in chosen} == {"AusSmoke:clip"}
    assert report["exclusions"]["upstream_test_untouched"] == 1
    assert report["exclusions"]["aerial_obstructed_or_artificial_provenance_cue"] == 2


def test_firespot_compression_levels_are_not_positive_noise():
    pixels = np.array([[0, 2, 7, 0], [1, 249, 255, 3], [0, 0, 2, 0]], dtype=np.uint8)
    image = Image.fromarray(pixels)
    inspection = {"mask": {"unique_gray_value_count": 7, "unique_gray_values": [0, 1, 2, 3, 7, 249, 255]}}
    objects, processing = acquire.source_mask_proposal(image, inspection, "FireSpot")
    assert objects["bbox"] == [[1, 1, 2, 1]]
    assert processing["foreground_pixels"] == 2
    assert np.array_equal(np.asarray(image), pixels)
    with pytest.raises(ValueError, match="unresolved"):
        acquire.source_mask_proposal(image, inspection, "AusSmoke")
    pixels[1, 1] = 128
    with pytest.raises(ValueError, match="unresolved"):
        acquire.source_mask_proposal(Image.fromarray(pixels), inspection, "FireSpot")


def test_shared_budget_is_cumulative():
    budget = acquire.RangeBudget(100, 50)
    budget.reserve(30)
    budget.reserve(20)
    with pytest.raises(ValueError, match="byte budget"):
        budget.reserve(1)
