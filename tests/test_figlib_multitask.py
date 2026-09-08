from __future__ import annotations

from fireviewer_model_lab.training.figlib_multitask import parse_sequence_page, select_offsets, visibility_role


def test_parse_and_select_figlib_offsets() -> None:
    html = """
    <a href=1000_-00600.jpg>negative</a>
    <a href=1600_%2B00000.jpg>onset</a>
    <a href=1660_%2B00060.jpg>positive</a>
    <a href=1720_%2B00120.jpg>positive</a>
    """

    available = parse_sequence_page(html, sequence_url="https://example.test/sequence/")
    selected = select_offsets(available, (-600, 0, 60, 120))

    assert [row["offset_seconds"] for row in selected] == [-600, 0, 60, 120]
    assert selected[2]["url"] == "https://example.test/sequence/1660_%2B00060.jpg"


def test_visibility_roles_are_conservative() -> None:
    assert visibility_role(-300) == "pre_onset_negative"
    assert visibility_role(-60) == "onset_ambiguous"
    assert visibility_role(0) == "onset_ambiguous"
    assert visibility_role(60) == "post_onset_positive_candidate"
