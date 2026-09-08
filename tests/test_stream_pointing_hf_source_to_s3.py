from __future__ import annotations

import json

import pytest
from fireviewer_model_lab.tools.stream_pointing_hf_source_to_s3 import _safe_path, select_source_rows


def test_select_source_rows_is_exact_and_rejects_path_escape() -> None:
    manifest = "\n".join(
        [
            json.dumps({"sample_id": "a", "source_id": "wanted"}),
            json.dumps({"sample_id": "b", "source_id": "other"}),
        ]
    )
    assert [row["sample_id"] for row in select_source_rows(manifest, "wanted")] == ["a"]
    with pytest.raises(ValueError, match="unsafe HF source path"):
        _safe_path("../outside.jpg")
