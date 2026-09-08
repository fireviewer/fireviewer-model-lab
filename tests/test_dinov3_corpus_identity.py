from __future__ import annotations

import copy
import json
from pathlib import Path

import imagehash
import pytest
from PIL import Image
from fireviewer_model_lab.training.dinov3_corpus_identity import (
    benchmark_matches,
    deterministic_canonical_event_id,
    deterministic_source_family_id,
    load_benchmark_denylist,
    phash64_imagehash_v1,
    resolve_source_identity,
    source_event_key_sha256,
    validate_canonical_event_splits,
    validate_source_identity_contract,
)

REGISTRY = Path(__import__("fireviewer_model_lab.training", fromlist=["_"]).__file__).parent / Path("registries/dinov3-multitask-composition-v1.json")
DENYLIST = Path(__import__("fireviewer_model_lab.training", fromlist=["_"]).__file__).parent / Path("registries/dinov3-independent-benchmark-denylist-v1.json")


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_registry_v2_has_three_deterministic_families_and_no_alias_claims() -> None:
    registry = _registry()

    identities = validate_source_identity_contract(registry)

    assert registry["schema_version"] == 2
    assert len(identities.bindings) == 3
    assert registry["source_identity_contract"]["event_aliases"] == []
    for family in registry["source_identity_contract"]["families"]:
        assert family["source_family_id"] == deterministic_source_family_id(
            family["lineage_root_ids"]
        )


def test_source_family_contract_rejects_artificial_lineage_slicing() -> None:
    registry = _registry()
    first = registry["source_identity_contract"]["families"][0]
    roots = sorted([first["lineage_root_ids"][0], "synthetic:slice"])
    registry["source_identity_contract"]["families"].append(
        {
            "source_family_id": deterministic_source_family_id(roots),
            "lineage_root_ids": roots,
            "bindings": [
                {
                    "kind": "overlay",
                    "name": "synthetic-slice",
                    "event_key_field": "split_group",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match=r"lineage root.*multiple"):
        validate_source_identity_contract(registry)


def test_source_family_id_is_order_independent_and_rejects_duplicate_roots() -> None:
    roots = ["archive:z", "dataset:a"]

    assert deterministic_source_family_id(roots) == deterministic_source_family_id(
        list(reversed(roots))
    )
    with pytest.raises(ValueError, match="lineage roots are invalid"):
        deterministic_source_family_id([roots[0], roots[0]])


def test_cross_source_alias_resolves_one_event_and_exposes_split_leakage() -> None:
    registry = copy.deepcopy(_registry())
    contract = registry["source_identity_contract"]
    boreal = registry["overlay_sources"][0]
    camp = registry["overlay_sources"][1]
    members = [
        {
            "source_family_id": boreal["source_family_id"],
            "source_event_key_sha256": source_event_key_sha256("boreal-event"),
        },
        {
            "source_family_id": camp["source_family_id"],
            "source_event_key_sha256": source_event_key_sha256("camp-event"),
        },
    ]
    event_id = deterministic_canonical_event_id(members)
    contract["event_aliases"] = [{"canonical_event_id": event_id, "members": members}]
    identities = validate_source_identity_contract(registry)

    boreal_identity = resolve_source_identity(
        {"split_group": "boreal-event"},
        binding_kind="overlay",
        binding_name="boreal",
        identities=identities,
    )
    camp_identity = resolve_source_identity(
        {"split_group": "camp-event"},
        binding_kind="overlay",
        binding_name="camp-swift",
        identities=identities,
    )

    assert boreal_identity["canonical_event_id"] == event_id
    assert camp_identity["canonical_event_id"] == event_id
    assert validate_canonical_event_splits(
        [
            {"sample_id": "boreal", "split": "train", **boreal_identity},
            {"sample_id": "camp", "split": "test", **camp_identity},
        ]
    ) == [event_id]


def test_cross_source_alias_cannot_alias_two_events_inside_one_family() -> None:
    registry = copy.deepcopy(_registry())
    family_id = registry["overlay_sources"][0]["source_family_id"]
    members = [
        {
            "source_family_id": family_id,
            "source_event_key_sha256": source_event_key_sha256(event),
        }
        for event in ("event-a", "event-b")
    ]
    registry["source_identity_contract"]["event_aliases"] = [
        {
            "canonical_event_id": deterministic_canonical_event_id(members),
            "members": members,
        }
    ]

    with pytest.raises(ValueError, match="must span source families"):
        validate_source_identity_contract(registry)


def test_pinned_benchmark_denylist_contains_hashes_only_and_matches_guard_phash() -> None:
    registry = _registry()
    denylist_json = json.loads(DENYLIST.read_text(encoding="utf-8"))
    denylist = load_benchmark_denylist(DENYLIST, registry["benchmark_boundary"])

    assert denylist_json["provenance_guard_rows"] == 200
    assert len(denylist.raw_image_sha256) == 200
    assert len(denylist.phash64) == 392
    assert "sample_id" not in DENYLIST.read_text(encoding="utf-8")
    assert "path" not in DENYLIST.read_text(encoding="utf-8")
    guarded = next(iter(denylist.raw_image_sha256))
    assert benchmark_matches(raw_sha256=guarded, denylist=denylist)["raw_sha256"]
    assert benchmark_matches(raw_sha256=guarded.upper(), denylist=denylist)["raw_sha256"]

    image = Image.new("RGB", (32, 32), (40, 90, 130))
    assert f"{phash64_imagehash_v1(image):016x}" == str(
        imagehash.phash(image, hash_size=8, highfreq_factor=4)
    )


def test_benchmark_denylist_tampering_fails_its_registry_pin(tmp_path: Path) -> None:
    registry = _registry()
    tampered = tmp_path / DENYLIST.name
    tampered.write_bytes(DENYLIST.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 drift"):
        load_benchmark_denylist(tampered, registry["benchmark_boundary"])
