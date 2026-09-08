"""Conservative fixed-camera identities from Pyro aliases and explicit reviews.

An orientation is a camera view, not an independent physical fire. Changing a
date or dataset prefix must not make a historical held-out view eligible again.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from training.pointing_dataset_v7.split_registry import digest, read_rows


CAMERA = re.compile(r"(?:^|[:/\\])((?:force|sdis)-\d+)[_:]([a-z][a-z0-9-]*)(?=[:_./\\]|$)", re.I)


def pyro_camera_view(row: dict) -> str | None:
    identities = set()
    if row.get("partner") and row.get("camera"):
        match = CAMERA.search(str(row["partner"]) + ":" + str(row["camera"]))
        if match:
            identities.add("pyro-camera:" + ":".join(match.groups()).casefold())
    for field in ("source_record_id", "image_name", "source_group_id", "split_group_id"):
        match = CAMERA.search(str(row.get(field, "")))
        if match:
            identities.add("pyro-camera:" + ":".join(match.groups()).casefold())
    if len(identities) > 1:
        raise ValueError("Conflicting Pyro-SDIS camera identities in one image record")
    return next(iter(identities), None)


def pyro_camera_views(row: dict) -> set[str]:
    """A reviewed broader scene can deliberately join overlapping orientations."""
    views = {pyro_camera_view(row)}
    if row.get("scene_group_verified") is True and row.get("scene_group_evidence"):
        views.add(pyro_camera_view({"source_group_id": row.get("scene_group_id", "")}))
    return views - {None}


def camera_views(row: dict) -> set[str]:
    """Include only explicitly image-bound visual identities for other sources."""
    result = pyro_camera_views(row)
    if row.get("reviewed_camera_evidence"):
        result.update(row.get("reviewed_camera_view_ids", []))
    return result


def load_camera_reviews(path: Path) -> dict:
    """Resolve explicit inspected indices; never infer identity from filename blocks.

    This proves a grouping review, not annotation quality or corpus admission.
    Both the index manifest and the inspected page registry are immutable inputs.
    """
    result = {}
    for review in json.loads(path.read_text(encoding="utf-8")):
        root = Path(review["review_root"])
        manifest = root / review["manifest"]
        if digest(manifest) != review["manifest_sha256"] or digest(root / "packets.json") != review["packets_sha256"]:
            raise ValueError("Camera identity review inputs changed")
        if not review.get("evidence") or not review.get("review_date") or not review.get("camera_view_id"):
            raise ValueError("Explicit camera review evidence is required")
        source = read_rows(manifest) if manifest.suffix == ".jsonl" else json.loads(manifest.read_text())
        indexed = {r["review_index"]: r for r in source}
        packets = json.loads((root / "packets.json").read_text())
        packet_by_index = {i: p for p in packets for i in p["review_indices"]}
        indices = review["review_indices"]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("Camera review needs explicit nonduplicated indices")
        for index in indices:
            row, packet = indexed[index], packet_by_index[index]
            if row["candidate_id"] not in packet["candidate_ids"] or digest(root / "pages" / packet["packet"]) != packet["sha256"]:
                raise ValueError("Inspected camera identity packet changed")
            if digest(Path(row["source_image"])) != row["sha256"]:
                raise ValueError("Camera identity image changed")
            binding = result.setdefault(row["sha256"], {"reviewed_camera_view_ids": [], "reviewed_camera_evidence": []})
            if review["camera_view_id"] not in binding["reviewed_camera_view_ids"]:
                binding["reviewed_camera_view_ids"].append(review["camera_view_id"])
            binding["reviewed_camera_evidence"].append({"review_file_sha256": digest(path),
                "review_date": review["review_date"], "evidence": review["evidence"],
                "packet_sha256": packet["sha256"], "image_sha256": row["sha256"]})
    return result


def bind_camera_reviews(rows: list[dict], bindings: dict) -> list[dict]:
    return [copy.deepcopy(row) | copy.deepcopy(bindings.get(row["sha256"], {})) for row in rows]


def frozen_camera_views(baseline: list[dict], history: dict | None = None) -> set[str]:
    views = {view for row in baseline if row.get("split") in {"validation", "test"} for view in camera_views(row)}
    views.update(view for view, splits in (history or {}).get("camera_views", {}).items()
                 if any(split in {"validation", "test"} for split in splits))
    for group, split in (history or {}).get("groups", {}).items():
        if split in {"validation", "test"}:
            views.add(pyro_camera_view({"source_group_id": group}))
    for group, splits in (history or {}).get("source_groups", {}).items():
        if any(split in {"validation", "test"} for split in splits):
            views.add(pyro_camera_view({"source_group_id": group}))
    views.discard(None)
    # Propagate explicit visual alias bindings even if the binding image itself
    # is later excluded. Another frame from that orientation cannot re-enter.
    bindings = [camera_views(row) for row in baseline]
    changed = True
    while changed:
        before = len(views)
        for binding in bindings:
            if binding & views:
                views.update(binding)
        changed = len(views) != before
    return views


def retire_camera_leaks(base: list[dict], baseline: list[dict], history: dict) -> tuple[list[dict], list[dict]]:
    """Exclude only V8 train references; preserve every source and frozen holdout."""
    frozen = frozen_camera_views(baseline, history)
    retained, excluded = [], []
    for row in base:
        cameras = camera_views(row)
        if row.get("split") == "train" and cameras & frozen:
            excluded.append({"sha256": row["sha256"], "source_group_id": row.get("source_group_id"),
                             "camera_views": sorted(cameras), "reason": "historical_holdout_camera_view",
                             "source_deleted": False, "historical_split_changed": False})
        else:
            retained.append(row)
    return retained, excluded


def separate_validation_from_test(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Reserve overlapping validation views; never move a holdout into training.

    Test membership, every annotation, and the caller's rows stay unchanged.
    Explicit camera aliases are propagated transitively. This resolves known
    identities only, and does not assert independence for unidentified cameras.
    """
    test_views = {view for row in rows if row.get("split") == "test" for view in camera_views(row)}
    changed = True
    while changed:
        before = len(test_views)
        for row in rows:
            views = camera_views(row)
            if views & test_views:
                test_views.update(views)
        changed = len(test_views) != before
    retained, reserved = [], []
    for row in rows:
        if row.get("split") == "validation" and camera_views(row) & test_views:
            reserved.append(row)
        else:
            retained.append(row)
    return retained, reserved
