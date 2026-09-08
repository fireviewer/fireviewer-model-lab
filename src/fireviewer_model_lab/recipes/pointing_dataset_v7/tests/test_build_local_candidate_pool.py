from training.pointing_dataset_v7.build_local_candidate_pool import annotation_stats, bucket_for, hamming


def test_hamming() -> None:
    assert hamming("0000000000000000", "0000000000000003") == 2


def test_small_smoke_bucket_and_pixel_gate_stat() -> None:
    row = {
        "width": 1280,
        "height": 720,
        "annotations": [{"bbox_xywh": [100, 100, 20, 20], "class_name": "smoke_visible"}],
    }
    stats = annotation_stats(row)
    assert bucket_for(stats) == "small_distant_smoke"
    assert stats["min_target_side_at_704"] > 4


def test_negative_bucket() -> None:
    stats = annotation_stats({"width": 640, "height": 480, "annotations": []})
    assert bucket_for(stats) == "hard_negative"
