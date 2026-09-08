from fireviewer_model_lab.tools.materialize_cross_view_transient_masks import _collect_assets


def test_collect_assets_deduplicates_images_by_sha256() -> None:
    rows = [
        {
            "source_view": {"sha256": "a", "image_relpath": "one.jpg"},
            "map_view": {"sha256": "b", "image_relpath": "two.jpg"},
        },
        {
            "source_view": {"sha256": "a", "image_relpath": "one.jpg"},
            "map_view": {"sha256": "c", "image_relpath": "three.jpg"},
        },
    ]
    assert _collect_assets(rows) == [
        {"sha256": "a", "relpath": "one.jpg"},
        {"sha256": "b", "relpath": "two.jpg"},
        {"sha256": "c", "relpath": "three.jpg"},
    ]
