from __future__ import annotations

from fireviewer_model_lab.training.figlib_index import parse_index_html, parse_sequence_name


def test_parse_classic_sequence_name() -> None:
    row = parse_sequence_name("20200905_ValleyFire_pi-w-mobo-c")

    assert row["event_key"] == "20200905:valleyfire"
    assert row["camera_id"] == "pi-w-mobo-c"


def test_parse_timestamped_hyphen_sequence_name() -> None:
    row = parse_sequence_name("20230727.144836-BonnyFire-hp-e-mobo-c")

    assert row["event_key"] == "20230727:bonnyfire"
    assert row["camera_id"] == "hp-e-mobo-c"


def test_index_groups_same_incident_across_cameras() -> None:
    html = """
    <a href="20200905_ValleyFire_pi-w-mobo-c/index.html">first</a>
    <a href="20200905_ValleyFire_sm-e-mobo-c/index.html">second</a>
    <a href="20200906-BobcatFire-wilson-e-mobo-c/">third</a>
    <a href="Tar/">archives</a>
    """

    rows = parse_index_html(html)

    valley = [row for row in rows if row["event_name"] == "ValleyFire"]
    assert len(valley) == 2
    assert len({row["split_group"] for row in valley}) == 1
    assert len(rows) == 3
