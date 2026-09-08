from fireviewer_model_lab.training.streetview_context_corpus import context_labels, selected_rows


def test_context_labels_keep_settings_and_field_mentions_separate() -> None:
    assert context_labels({"setting": "mountain", "scene_description": "A wooded ridge."}) == (
        "mountain",
    )
    assert context_labels(
        {"setting": "urban", "scene_description": "A paved road beside crop fields."}
    ) == ("field",)
    assert context_labels({"setting": "suburban", "scene_description": "Houses."}) == ()


def test_selected_rows_excludes_non_contextual_rows() -> None:
    rows = [
        {"id": "kept", "setting": "forest", "scene_description": "Trees"},
        {"id": "skipped", "setting": "urban", "scene_description": "Houses"},
    ]
    assert [(row["id"], labels) for row, labels in selected_rows(rows)] == [("kept", ("forest",))]
