from __future__ import annotations

import struct

from fireviewer_model_lab.training.remote_zip import RemoteZip


def test_zip64_extra_replaces_overflow_fields() -> None:
    values = (123, 456, 789)
    payload = b"".join(struct.pack("<Q", value) for value in values)
    extra = struct.pack("<HH", 0x0001, len(payload)) + payload

    resolved = RemoteZip._zip64_values(
        extra,
        uncompressed_size=0xFFFFFFFF,
        compressed_size=0xFFFFFFFF,
        local_header_offset=0xFFFFFFFF,
    )

    assert resolved == values


def test_zip64_extra_leaves_normal_fields_unchanged() -> None:
    assert RemoteZip._zip64_values(
        b"", uncompressed_size=1, compressed_size=2, local_header_offset=3
    ) == (1, 2, 3)
