"""Shared fail-closed identity and benchmark-boundary contracts for DINOv3 data."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

FAMILY_ID_ALGORITHM = "source-family-sha256-v1"
EVENT_ID_ALGORITHM = "canonical-event-sha256-v1"
DECODED_PIXEL_HASH_ALGORITHM = "rgb8-wh-be32-sha256-v1"
PERCEPTUAL_HASH_ALGORITHM = "imagehash-phash64-hash-size-8-v1"
BENCHMARK_DENYLIST_KIND = "fireviewer-independent-benchmark-hash-denylist"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_hex(value: Any, length: int) -> bool:
    text = str(value or "")
    return (
        len(text) == length
        and text == text.casefold()
        and all(character in "0123456789abcdef" for character in text)
    )


def canonical_domain_sha256(domain: str, value: Any) -> str:
    """Hash canonical JSON in a domain-separated namespace."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def deterministic_source_family_id(lineage_root_ids: Sequence[str]) -> str:
    roots = list(lineage_root_ids)
    if (
        not roots
        or any(not isinstance(root, str) or not root or root != root.strip() for root in roots)
        or len(set(roots)) != len(roots)
    ):
        raise ValueError("source family lineage roots are invalid")
    roots.sort()
    return "sf1_" + canonical_domain_sha256(
        "fireviewer.source-family.v1",
        {"lineage_root_ids": roots},
    )


def source_event_key_sha256(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source event key is missing")
    normalized = unicodedata.normalize("NFC", value.strip())
    return canonical_domain_sha256("fireviewer.source-event-key.v1", normalized)


def _event_member(source_family_id: str, event_key_sha256: str) -> dict[str, str]:
    return {
        "source_event_key_sha256": event_key_sha256,
        "source_family_id": source_family_id,
    }


def deterministic_canonical_event_id(members: Sequence[Mapping[str, str]]) -> str:
    normalized = sorted(
        (
            {
                "source_event_key_sha256": str(member["source_event_key_sha256"]),
                "source_family_id": str(member["source_family_id"]),
            }
            for member in members
        ),
        key=lambda item: (item["source_family_id"], item["source_event_key_sha256"]),
    )
    return "evt1_" + canonical_domain_sha256(
        "fireviewer.canonical-event.v1",
        {"members": normalized},
    )


@dataclass(frozen=True)
class SourceBinding:
    kind: str
    name: str
    source_family_id: str
    event_key_field: str


@dataclass(frozen=True)
class SourceIdentityIndex:
    contract_sha256: str
    bindings: dict[tuple[str, str], SourceBinding]
    alias_by_member: dict[tuple[str, str], str]


def validate_source_identity_contract(registry: Mapping[str, Any]) -> SourceIdentityIndex:
    """Validate and index deterministic source-family and event-alias contracts."""

    contract = registry.get("source_identity_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("source identity contract is missing or unsupported")
    if contract.get("family_id_algorithm") != FAMILY_ID_ALGORITHM:
        raise ValueError("source family ID algorithm drift")
    if contract.get("event_id_algorithm") != EVENT_ID_ALGORITHM:
        raise ValueError("canonical event ID algorithm drift")
    families = contract.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("source identity contract has no families")

    bindings: dict[tuple[str, str], SourceBinding] = {}
    root_owner: dict[str, str] = {}
    family_ids: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("source family contract is not an object")
        roots = family.get("lineage_root_ids")
        if (
            not isinstance(roots, list)
            or not roots
            or any(not isinstance(root, str) or not root or root != root.strip() for root in roots)
            or roots != sorted(set(roots))
        ):
            raise ValueError("source family lineage roots are not canonical")
        family_id = str(family.get("source_family_id") or "")
        if family_id != deterministic_source_family_id(roots) or family_id in family_ids:
            raise ValueError("source family ID is not deterministic and unique")
        family_ids.add(family_id)
        for root in roots:
            owner = root_owner.setdefault(root, family_id)
            if owner != family_id:
                raise ValueError("one lineage root is assigned to multiple source families")
        family_bindings = family.get("bindings")
        if not isinstance(family_bindings, list) or not family_bindings:
            raise ValueError("source family has no bindings")
        for raw_binding in family_bindings:
            if not isinstance(raw_binding, dict):
                raise ValueError("source family binding is not an object")
            kind = str(raw_binding.get("kind") or "")
            name = str(raw_binding.get("name") or "")
            event_key_field = str(raw_binding.get("event_key_field") or "")
            if kind != "overlay" or not name or not event_key_field:
                raise ValueError("source family binding contract is invalid")
            key = (kind, name)
            if key in bindings:
                raise ValueError("source binding is assigned to multiple families")
            bindings[key] = SourceBinding(kind, name, family_id, event_key_field)

    overlays = registry.get("overlay_sources")
    if not isinstance(overlays, list):
        raise ValueError("overlay source contracts are missing")
    overlay_by_name = {
        str(item.get("name") or ""): item for item in overlays if isinstance(item, dict)
    }
    expected_binding_names = {name for kind, name in bindings if kind == "overlay"}
    if set(overlay_by_name) != expected_binding_names:
        raise ValueError("source family bindings do not cover the overlay contracts exactly")
    origin_owner: dict[tuple[str, str], str] = {}
    manifest_owner: dict[str, str] = {}
    for name, overlay in overlay_by_name.items():
        binding = bindings[("overlay", name)]
        if overlay.get("source_family_id") != binding.source_family_id:
            raise ValueError(f"overlay source family binding drift: {name}")
        origin = (str(overlay.get("source_id") or ""), str(overlay.get("source_revision") or ""))
        if not all(origin):
            raise ValueError(f"overlay source origin is incomplete: {name}")
        owner = origin_owner.setdefault(origin, binding.source_family_id)
        if owner != binding.source_family_id:
            raise ValueError("one source origin is assigned to multiple source families")
        manifest_sha256 = str(overlay.get("manifest_sha256") or "")
        if not _valid_hex(manifest_sha256, 64):
            raise ValueError(f"overlay manifest SHA-256 is invalid: {name}")
        manifest_family = manifest_owner.setdefault(manifest_sha256, binding.source_family_id)
        if manifest_family != binding.source_family_id:
            raise ValueError("one source manifest is assigned to multiple source families")

    aliases = contract.get("event_aliases")
    if not isinstance(aliases, list):
        raise ValueError("event aliases must be a list")
    alias_by_member: dict[tuple[str, str], str] = {}
    for alias in aliases:
        if not isinstance(alias, dict):
            raise ValueError("event alias is not an object")
        members = alias.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("cross-source event alias needs at least two members")
        normalized_members: list[dict[str, str]] = []
        seen_members: set[tuple[str, str]] = set()
        member_family_ids: set[str] = set()
        for member in members:
            if not isinstance(member, dict):
                raise ValueError("event alias member is not an object")
            family_id = str(member.get("source_family_id") or "")
            event_key_sha256 = str(member.get("source_event_key_sha256") or "")
            key = (family_id, event_key_sha256)
            if family_id not in family_ids or not _valid_hex(event_key_sha256, 64):
                raise ValueError("event alias member contract is invalid")
            if key in seen_members or key in alias_by_member:
                raise ValueError("event alias member is assigned more than once")
            seen_members.add(key)
            member_family_ids.add(family_id)
            normalized_members.append(_event_member(family_id, event_key_sha256))
        if len(member_family_ids) < 2:
            raise ValueError("cross-source event alias must span source families")
        expected_event_id = deterministic_canonical_event_id(normalized_members)
        if alias.get("canonical_event_id") != expected_event_id:
            raise ValueError("canonical event alias ID is not deterministic")
        for key in seen_members:
            alias_by_member[key] = expected_event_id

    return SourceIdentityIndex(
        contract_sha256=canonical_domain_sha256("fireviewer.source-identity-contract.v1", contract),
        bindings=bindings,
        alias_by_member=alias_by_member,
    )


def resolve_source_identity(
    row: Mapping[str, Any],
    *,
    binding_kind: str,
    binding_name: str,
    identities: SourceIdentityIndex,
) -> dict[str, str]:
    binding = identities.bindings.get((binding_kind, binding_name))
    if binding is None:
        raise ValueError(f"source identity binding is missing: {binding_kind}:{binding_name}")
    event_key_sha256 = source_event_key_sha256(row.get(binding.event_key_field))
    member = (binding.source_family_id, event_key_sha256)
    aliased = identities.alias_by_member.get(member)
    canonical_event_id = aliased or deterministic_canonical_event_id(
        [_event_member(binding.source_family_id, event_key_sha256)]
    )
    return {
        "source_family_id": binding.source_family_id,
        "source_event_key_sha256": event_key_sha256,
        "canonical_event_id": canonical_event_id,
        "canonical_event_basis": "cross_source_alias" if aliased else "family_event_key",
        "source_split_group": str(row[binding.event_key_field]).strip(),
    }


def resolve_namespaced_source_identity(
    *, lineage_root_id: str, source_event_key: str
) -> dict[str, str]:
    """Resolve an unaliased source identity from an immutable lineage namespace."""

    if not lineage_root_id or lineage_root_id != lineage_root_id.strip():
        raise ValueError("source lineage root ID is missing")
    source_family_id = deterministic_source_family_id([lineage_root_id])
    event_key_sha256 = source_event_key_sha256(source_event_key)
    return {
        "source_family_id": source_family_id,
        "source_event_key_sha256": event_key_sha256,
        "canonical_event_id": deterministic_canonical_event_id(
            [_event_member(source_family_id, event_key_sha256)]
        ),
        "canonical_event_basis": "family_event_key",
        "source_split_group": source_event_key.strip(),
    }


IDENTITY_FIELDS = (
    "source_family_id",
    "source_event_key_sha256",
    "canonical_event_id",
    "canonical_event_basis",
    "source_split_group",
)


def _identity_drift(observed: Mapping[str, Any], expected: Mapping[str, str]) -> list[str]:
    return [field for field in IDENTITY_FIELDS if observed.get(field) != expected[field]]


def validate_composed_row_identities(
    rows: Iterable[Mapping[str, Any]],
    *,
    registry: Mapping[str, Any],
    identities: SourceIdentityIndex,
) -> None:
    """Recompute every row/reference identity rather than trusting serialized IDs."""

    detection = registry.get("detection_base")
    if not isinstance(detection, dict):
        raise ValueError("detection identity namespace is missing")
    repository = str(detection.get("repository") or "")
    revision = str(detection.get("revision") or "")
    if not repository or not revision:
        raise ValueError("detection identity namespace is incomplete")
    lineage_root_id = f"hf-dataset:{repository}"

    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        references = row.get("overlay_sources")
        if not isinstance(references, list):
            raise ValueError(f"overlay source references are invalid: {sample_id}")
        validated_references: list[Mapping[str, Any]] = []
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError(f"overlay source reference is invalid: {sample_id}")
            name = str(reference.get("name") or "")
            binding = identities.bindings.get(("overlay", name))
            if binding is None:
                raise ValueError(f"overlay source identity is unregistered: {sample_id}:{name}")
            source_split_group = reference.get("source_split_group")
            expected = resolve_source_identity(
                {binding.event_key_field: source_split_group},
                binding_kind="overlay",
                binding_name=name,
                identities=identities,
            )
            drift = _identity_drift(reference, expected)
            if drift:
                raise ValueError(
                    f"overlay source identity drift: {sample_id}:{name}:{','.join(drift)}"
                )
            validated_references.append(reference)

        locator = row.get("image_locator")
        if not isinstance(locator, dict):
            raise ValueError(f"image locator is invalid: {sample_id}")
        if locator.get("kind") == "hf_dataset_row":
            if locator.get("repository") != repository or locator.get("revision") != revision:
                raise ValueError(f"detection lineage binding drift: {sample_id}")
            source_split_group = str(row.get("source_split_group") or "")
            source_id = str(row.get("source_id") or "")
            expected = resolve_namespaced_source_identity(
                lineage_root_id=lineage_root_id,
                source_event_key=f"{source_id}:{source_split_group}",
            )
            # source_split_group is intentionally kept human-auditable on the row;
            # only its source-qualified form is hashed into the event identity.
            expected["source_split_group"] = source_split_group
        else:
            if len(validated_references) != 1:
                raise ValueError(
                    f"overlay-only row does not have one identity reference: {sample_id}"
                )
            expected = {field: str(validated_references[0][field]) for field in IDENTITY_FIELDS}
        drift = _identity_drift(row, expected)
        if drift:
            raise ValueError(f"row source identity drift: {sample_id}:{','.join(drift)}")
        if row.get("split_group") != f"event:{expected['canonical_event_id']}":
            raise ValueError(f"canonical event is not the split group: {sample_id}")


def validate_canonical_event_splits(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    event_splits: dict[str, set[str]] = {}
    for row in rows:
        event_id = str(row.get("canonical_event_id") or "")
        if not event_id.startswith("evt1_") or not _valid_hex(event_id.removeprefix("evt1_"), 64):
            raise ValueError(f"canonical event ID is invalid: {row.get('sample_id')}")
        event_splits.setdefault(event_id, set()).add(str(row.get("split") or ""))
        overlay_sources = row.get("overlay_sources")
        if isinstance(overlay_sources, list):
            for reference in overlay_sources:
                if not isinstance(reference, dict):
                    continue
                overlay_event = str(reference.get("canonical_event_id") or "")
                if overlay_event:
                    if not overlay_event.startswith("evt1_") or not _valid_hex(
                        overlay_event.removeprefix("evt1_"), 64
                    ):
                        raise ValueError(
                            f"overlay canonical event ID is invalid: {row.get('sample_id')}"
                        )
                    event_splits.setdefault(overlay_event, set()).add(str(row.get("split") or ""))
    return sorted(event_id for event_id, splits in event_splits.items() if len(splits) > 1)


@dataclass(frozen=True)
class BenchmarkDenylist:
    sha256: str
    raw_image_sha256: frozenset[str]
    decoded_pixel_sha256: frozenset[str]
    phash64: frozenset[int]
    phash_hamming_distance_max: int


def _hash_list(value: Any, *, length: int, label: str) -> list[str]:
    if not isinstance(value, list) or any(not _valid_hex(item, length) for item in value):
        raise ValueError(f"benchmark denylist {label} is invalid")
    if value != sorted(set(value)):
        raise ValueError(f"benchmark denylist {label} must be sorted and unique")
    return value


def load_benchmark_denylist(path: Path, boundary: Mapping[str, Any]) -> BenchmarkDenylist:
    expected_sha256 = str(boundary.get("denylist_sha256") or "")
    if not path.is_file() or not _valid_hex(expected_sha256, 64):
        raise ValueError("benchmark denylist artifact is missing from its registry contract")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("benchmark denylist SHA-256 drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "decoded_pixel_sha256",
        "hash_only",
        "kind",
        "phash64_imagehash_v1",
        "provenance_guard_rows",
        "provenance_guard_sha256",
        "raw_image_sha256",
        "schema_version",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema_version") != boundary.get("denylist_schema_version")
        or value.get("kind") != BENCHMARK_DENYLIST_KIND
        or value.get("hash_only") is not True
        or boundary.get("decoded_pixel_hash_algorithm") != DECODED_PIXEL_HASH_ALGORITHM
        or boundary.get("perceptual_hash_algorithm") != PERCEPTUAL_HASH_ALGORITHM
        or value.get("provenance_guard_sha256") != boundary.get("provenance_guard_sha256")
        or value.get("provenance_guard_rows") != boundary.get("provenance_guard_rows")
    ):
        raise ValueError("benchmark denylist contract is invalid")
    raw_hashes = _hash_list(value.get("raw_image_sha256"), length=64, label="raw hashes")
    decoded_hashes = _hash_list(
        value.get("decoded_pixel_sha256"), length=64, label="decoded hashes"
    )
    phashes = _hash_list(value.get("phash64_imagehash_v1"), length=16, label="pHashes")
    expected_counts = boundary.get("denylist_entry_counts")
    actual_counts = {
        "decoded_pixel_sha256": len(decoded_hashes),
        "phash64_imagehash_v1": len(phashes),
        "raw_image_sha256": len(raw_hashes),
    }
    if not isinstance(expected_counts, dict) or expected_counts != actual_counts:
        raise ValueError("benchmark denylist entry counts drift")
    maximum = boundary.get("phash_hamming_distance_max")
    if not isinstance(maximum, int) or not 0 <= maximum <= 8:
        raise ValueError("benchmark pHash Hamming threshold is invalid")
    return BenchmarkDenylist(
        sha256=actual_sha256,
        raw_image_sha256=frozenset(raw_hashes),
        decoded_pixel_sha256=frozenset(decoded_hashes),
        phash64=frozenset(int(value, 16) for value in phashes),
        phash_hamming_distance_max=maximum,
    )


def decoded_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    digest = hashlib.sha256()
    digest.update(width.to_bytes(4, "big"))
    digest.update(height.to_bytes(4, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def phash64_imagehash_v1(image: Image.Image) -> int:
    """Match imagehash.phash(image, hash_size=8, highfreq_factor=4)."""

    import imagehash

    return int(str(imagehash.phash(image, hash_size=8, highfreq_factor=4)), 16)


def benchmark_matches(
    *,
    raw_sha256: str,
    denylist: BenchmarkDenylist,
    decoded_sha256: str | None = None,
    phash64: int | None = None,
) -> dict[str, bool]:
    normalized_raw_sha256 = str(raw_sha256).casefold()
    normalized_decoded_sha256 = (
        str(decoded_sha256).casefold() if decoded_sha256 is not None else None
    )
    return {
        "raw_sha256": normalized_raw_sha256 in denylist.raw_image_sha256,
        "decoded_sha256": (
            normalized_decoded_sha256 in denylist.decoded_pixel_sha256
            if normalized_decoded_sha256 is not None
            else False
        ),
        "phash": (
            any(
                (phash64 ^ denied).bit_count() <= denylist.phash_hamming_distance_max
                for denied in denylist.phash64
            )
            if phash64 is not None
            else False
        ),
    }
