import pytest

from training.pointing_dataset_v8.acquire_pyrosdis import capture_key, parse_annotations, select


def test_single_smoke_class_mapping_and_geometry():
    obj = parse_annotations("1 .5 .5 .1 .2\n0 .2 .2 .1 .1", 1000, 500)
    assert obj["category"] == [1, 1]
    assert obj["bbox"][0] == pytest.approx([450, 200, 100, 100])


@pytest.mark.parametrize("text", ["2 .5 .5 .1 .1", "0 nan .5 .1 .1", "1 0 .5 .5 .1", "1 .5 .5 0 .1"])
def test_invalid_labels_are_not_negatives(text):
    with pytest.raises(ValueError):
        parse_annotations(text, 100, 100)


def test_camera_freezes_sibling_frames_across_all_dates():
    def row(date):
        return {"image_name": f"sdis-07_camera-a_{date}.jpg", "annotations": "1 .5 .5 .01 .01", "width": 1280, "height": 720}
    frozen = row("2024-01-15T12-00-00") | {"split": "test"}
    other = row("2024-02-15T12-00-00")
    rows, _ = select([row("2024-01-15T14-00-00"), other], [frozen], 10)
    assert rows == []
    assert capture_key(frozen) == "Pyro-SDIS:sdis-07:camera-a:2024-01-15"


def test_real_camera_heading_is_frozen_across_dates_and_aliases():
    heldout = {"source_record_id": "sdis-07_brison-200_2024-01-15T12-00-00.jpg", "split": "test"}
    other_day = {"image_name": "sdis-07_brison-200_2024-05-16T12-00-00.jpg",
                 "annotations": "1 .5 .5 .01 .01", "width": 1280, "height": 720}
    rows, excluded = select([other_day], [heldout], 10)
    assert rows == [] and excluded["known_or_frozen_capture"] == 1


def test_http_429_is_not_retried_or_delayed_by_retry_after():
    from training.pointing_dataset_v8.acquire_pyrosdis import session
    with session() as client:
        retry = client.get_adapter("https://example.test").max_retries
        assert not retry.is_retry("GET", 429, has_retry_after=True)
        assert retry.total == 2 and retry.respect_retry_after_header is False
