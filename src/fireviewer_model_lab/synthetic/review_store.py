"""Persistent, authenticated review inventory for generated synthetic cases."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fireviewer_model_lab.synthetic.real_world import (
    OUTPUT_RENDER_PROFILES,
    RENDER_REVISION,
)


CATEGORY_IDS = (
    "terrestrial_fire_points",
    "france_cross_view",
    "response_engagement",
    "france_incident_days",
)
CATEGORY_LABELS = {
    "terrestrial_fire_points": "Points feu et fumée",
    "france_cross_view": "Recalage France",
    "response_engagement": "Moyens engagés",
    "france_incident_days": "Journées A-à-Z",
}
TARGET_PER_CATEGORY = 4096
MAX_TARGET_PER_CATEGORY = 8192
PILOT_PER_CATEGORY = 8
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
DECISIONS = frozenset({"accepted", "rejected"})
COMMON_VISUAL_QUALITY_CHECKS = frozenset(
    {
        "terrain_and_scale_realistic",
        "fire_smoke_semantics_aligned",
        "camera_and_annotations_coherent",
        "occlusion_and_distance_plausible",
        "lighting_and_render_artifacts_acceptable",
    }
)
REQUIRED_QUALITY_CHECKS = {
    "terrestrial_fire_points": COMMON_VISUAL_QUALITY_CHECKS,
    "france_cross_view": COMMON_VISUAL_QUALITY_CHECKS
    | {"orthophoto_mnt_photo_coherent"},
    "response_engagement": COMMON_VISUAL_QUALITY_CHECKS
    | {
        "actor_identity_and_engagement_credible",
        "actor_visual_fidelity_and_materials_acceptable",
        "box_tight_and_object_visible",
    },
    "france_incident_days": frozenset(
        {
            "sources_traceable",
            "accepted_and_rejected_facts_justified",
            "contradictions_explicit",
            "fire_zone_overlay_coherent",
        }
    ),
}
PRODUCTION_STAGES = frozenset({"pilot", "bulk", "replacement"})
RESPONSE_CLASSES = frozenset(
    {
        "sdis_vehicle",
        "canadair",
        "dash",
        "securite_civile_helicopter",
        "hard_negative_construction_truck",
        "hard_negative_crop_duster",
        "hard_negative_utility_helicopter",
    }
)
REQUIRED_POINT_LABELS = frozenset(
    {"active_fire_point", "visible_fire_front_point", "smoke_column_base"}
)
VISUAL_BACKGROUND_SOURCES = frozenset(
    {
        "new_real_world_capture_nurec_reconstruction",
        "new_omniverse_synthetic_french_reference_scene",
    }
)
PROGRESSION_PHASES = frozenset(
    {
        "initial_growth",
        "advancing_flame_zone",
        "front_split",
        "partial_suppression",
        "reignition",
        "multi_front_spread",
        "decay",
    }
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _identifier(value: object, *, field: str) -> str:
    candidate = str(value or "").strip()
    if not IDENTIFIER.fullmatch(candidate):
        raise ValueError(f"invalid {field}")
    return candidate


def _within(root: Path, relative: object, *, field: str) -> Path:
    value = str(relative or "").strip().replace("\\", "/")
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise ValueError(f"invalid {field}")
    resolved = (root / value).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{field} escapes the production volume")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_coordinate(value: object, *, field: str) -> float:
    coordinate = float(value)
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
        raise ValueError(f"{field} must be a finite normalized coordinate")
    return coordinate


def _validate_overlays(overlays: object) -> list[dict[str, Any]]:
    if not isinstance(overlays, list) or len(overlays) > 128:
        raise ValueError("case overlays must be a list containing at most 128 entries")
    normalized: list[dict[str, Any]] = []
    for index, overlay in enumerate(overlays):
        if not isinstance(overlay, dict):
            raise ValueError("case overlays must be objects")
        item = dict(overlay)
        item["label"] = _identifier(item.get("label"), field=f"overlays[{index}].label")
        kind = str(item.get("kind", "")).strip()
        if kind == "point":
            item["x_normalized"] = _normalized_coordinate(
                item.get("x_normalized"), field=f"overlays[{index}].x_normalized"
            )
            item["y_normalized"] = _normalized_coordinate(
                item.get("y_normalized"), field=f"overlays[{index}].y_normalized"
            )
        elif kind == "box":
            x_min = _normalized_coordinate(
                item.get("x_min"), field=f"overlays[{index}].x_min"
            )
            y_min = _normalized_coordinate(
                item.get("y_min"), field=f"overlays[{index}].y_min"
            )
            x_max = _normalized_coordinate(
                item.get("x_max"), field=f"overlays[{index}].x_max"
            )
            y_max = _normalized_coordinate(
                item.get("y_max"), field=f"overlays[{index}].y_max"
            )
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("case box overlays must have positive width and height")
            item.update(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        else:
            raise ValueError("case overlays only support point and box entries")
        item["kind"] = kind
        normalized.append(item)
    return normalized


def _require_artifact_kinds(artifacts: list[dict[str, Any]], required: set[str]) -> None:
    present = {str(artifact.get("kind", "")) for artifact in artifacts}
    missing = required - present
    if missing:
        raise ValueError(f"case artifacts are missing required kinds: {sorted(missing)}")


def _validate_performance(
    payload: object,
    *,
    category: str,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("case requires observed performance measurements")
    if payload.get("measurement") != "observed_pilot_case_v1":
        raise ValueError("case performance measurement contract is unsupported")
    elapsed_seconds = float(payload.get("elapsed_seconds", math.nan))
    if not math.isfinite(elapsed_seconds) or not 0.0 < elapsed_seconds <= 86_400.0:
        raise ValueError("case elapsed_seconds must be a finite observed duration")
    artifact_bytes = int(payload.get("artifact_bytes", -1))
    actual_artifact_bytes = sum(int(item["bytes"]) for item in artifacts)
    if artifact_bytes != actual_artifact_bytes or artifact_bytes <= 0:
        raise ValueError("case artifact byte measurement does not match the files")
    record_bytes = int(payload.get("record_bytes", -1))
    case_output_bytes = int(payload.get("case_output_bytes", -1))
    if (
        record_bytes <= 0
        or case_output_bytes != artifact_bytes + record_bytes
    ):
        raise ValueError("case output byte measurement is inconsistent")
    vram_measurement = str(payload.get("vram_measurement", ""))
    baseline = int(payload.get("vram_baseline_bytes", -1))
    peak = int(payload.get("vram_peak_bytes", -1))
    delta = int(payload.get("vram_delta_peak_bytes", -1))
    samples = int(payload.get("vram_sample_count", -1))
    if category == "france_incident_days":
        if (
            vram_measurement != "not_applicable_non_gpu_case"
            or any(value != 0 for value in (baseline, peak, delta, samples))
        ):
            raise ValueError("incident-day cases must mark VRAM as not applicable")
    elif (
        vram_measurement != "nvidia_smi_device_total_memory"
        or baseline <= 0
        or peak < baseline
        or delta != peak - baseline
        or samples < 2
    ):
        raise ValueError("visual cases require real sampled NVIDIA VRAM measurements")
    return {
        **payload,
        "elapsed_seconds": elapsed_seconds,
        "artifact_bytes": artifact_bytes,
        "record_bytes": record_bytes,
        "case_output_bytes": case_output_bytes,
        "vram_baseline_bytes": baseline,
        "vram_peak_bytes": peak,
        "vram_delta_peak_bytes": delta,
        "vram_sample_count": samples,
    }


def _validate_nurec_provenance(record: dict[str, Any], truth: dict[str, Any]) -> None:
    if truth.get("background_source") not in VISUAL_BACKGROUND_SOURCES:
        raise ValueError("visual cases require a validated new Omniverse background")
    for field in ("capture_manifest_sha256", "scene_asset_sha256"):
        if not SHA256.fullmatch(str(truth.get(field, ""))):
            raise ValueError(f"visual case truth requires {field}")
    if truth.get("human_review_required") is not True:
        raise ValueError("visual cases must require human review")
    if truth.get("usable_for_training") is not False:
        raise ValueError("visual cases stay training-locked until accepted review")
    if not str(truth.get("event_id", "")).strip():
        raise ValueError("visual cases require a distinct fire event_id")
    if not 1 <= int(truth.get("fire_duration_days", 0)) <= 15:
        raise ValueError("visual cases require fire_duration_days between 1 and 15")
    if not str(truth.get("landscape_profile", "")).strip():
        raise ValueError("visual cases require a French landscape_profile")
    progression = truth.get("progression")
    if (
        not isinstance(progression, dict)
        or progression.get("phase") not in PROGRESSION_PHASES
    ):
        raise ValueError("visual cases require validated fire progression metadata")
    render = record.get("render")
    if not isinstance(render, dict):
        raise ValueError("visual cases require render metadata")
    profile = str(render.get("profile", ""))
    expected_render = OUTPUT_RENDER_PROFILES.get(profile)
    if expected_render is None:
        raise ValueError(
            f"visual cases require one of {sorted(OUTPUT_RENDER_PROFILES)}"
        )
    if render.get("revision") != expected_render["revision"]:
        raise ValueError(
            f"visual cases require render revision {expected_render['revision']} "
            f"for profile {profile}"
        )
    if int(render.get("rt_subframes", 0)) < 8:
        raise ValueError("visual cases require at least eight RTX subframes")
    if int(render.get("warmup_steps", 0)) < 16:
        raise ValueError("visual cases require at least sixteen Flow warmup steps")
    if not str(render.get("camera_pose_id", "")).strip():
        raise ValueError("visual cases require a validated camera_pose_id")
    for field in ("variation_id", "lighting_variant_id", "flow_state_id", "time_of_day"):
        if not str(render.get(field, "")).strip():
            raise ValueError(f"visual cases require {field}")
    if not SHA256.fullmatch(str(render.get("diversity_signature", ""))):
        raise ValueError("visual cases require a diversity_signature SHA-256")
    viewpoint = render.get("viewpoint")
    if not isinstance(viewpoint, dict):
        raise ValueError("visual cases require viewpoint validation metadata")
    if viewpoint.get("reference_validation") not in {
        "reference_render_human_approved",
        "pending_console_review",
    }:
        raise ValueError("viewpoint must be prevalidated or pending console review")
    camera = record.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("visual cases require a camera contract")
    for field in ("position", "axis", "intrinsics"):
        if field not in camera:
            raise ValueError(f"visual case camera is missing {field}")


def _validate_category_contract(record: dict[str, Any]) -> None:
    category = str(record["category"])
    overlays = record["overlays"]
    artifacts = record["artifacts"]
    truth = record.get("truth")
    if not isinstance(truth, dict) or truth.get("synthetic") is not True:
        raise ValueError("case truth must explicitly identify synthetic ground truth")
    if truth.get("real_world_claim") is not False:
        raise ValueError("synthetic cases must explicitly disable real-world claims")

    if category == "terrestrial_fire_points":
        _validate_nurec_provenance(record, truth)
        labels = {
            str(overlay["label"])
            for overlay in overlays
            if overlay["kind"] == "point"
        }
        points = [overlay for overlay in overlays if overlay["kind"] == "point"]
        if labels != REQUIRED_POINT_LABELS or len(points) != 3:
            raise ValueError("terrestrial fire cases require exactly all three point labels")
        _require_artifact_kinds(artifacts, {"ground_photo", "point_annotations"})
    elif category == "france_cross_view":
        _validate_nurec_provenance(record, truth)
        _require_artifact_kinds(
            artifacts,
            {"ground_photo", "orthophoto", "mnt", "site_manifest"},
        )
        if not str(truth.get("site_code", "")).strip():
            raise ValueError("cross-view cases require a stable site_code")
        if truth.get("fire_position_verified_from_generator") is not True:
            raise ValueError("cross-view fire position must be verified from generator state")
    elif category == "response_engagement":
        _validate_nurec_provenance(record, truth)
        boxes = [overlay for overlay in overlays if overlay["kind"] == "box"]
        if len(boxes) != 1:
            raise ValueError("response cases require exactly one validated box")
        if truth.get("object_class") not in RESPONSE_CLASSES:
            raise ValueError("response case object_class is unsupported")
        if truth.get("engagement_label") not in {
            "simulated_engagement",
            "not_engaged_hard_negative",
        }:
            raise ValueError("response cases require an explicit synthetic engagement label")
        if truth.get("operational_truth") != "synthetic_only":
            raise ValueError("response cases cannot claim real operational engagement")
        if not SHA256.fullmatch(str(truth.get("actor_asset_sha256", ""))):
            raise ValueError("response cases require the validated actor asset SHA-256")
        if truth.get("target_actor_isolated") is not True:
            raise ValueError("response cases must hide every non-target actor")
        if boxes[0]["label"] != truth.get("object_class"):
            raise ValueError("response box label must match object_class")
        _require_artifact_kinds(artifacts, {"ground_photo", "box_annotations"})
    elif category == "france_incident_days":
        if truth.get("fixture_kind") != "fictional_synthetic_incident_day":
            raise ValueError("incident-day cases must remain explicit fictional fixtures")
        _require_artifact_kinds(
            artifacts,
            {
                "source_packet",
                "research_log",
                "fact_ledger",
                "contradiction_log",
                "fire_zone_overlay",
            },
        )


def validate_case_record(record: dict[str, Any], volume_root: Path) -> dict[str, Any]:
    if record.get("schema_version") != 1:
        raise ValueError("unsupported case schema_version")
    category = _identifier(record.get("category"), field="category")
    if category not in CATEGORY_IDS:
        raise ValueError("unsupported case category")
    case_id = _identifier(record.get("case_id"), field="case_id")
    if record.get("data_origin") != "new_synthetic_generation":
        raise ValueError("case data_origin must identify new synthetic generation")
    seed = int(record.get("seed", -1))
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("case seed is outside the supported range")
    preview = _within(volume_root, record.get("preview_relpath"), field="preview_relpath")
    if not preview.is_file():
        raise ValueError(f"case preview is absent: {preview}")
    production_stage = str(record.get("production_stage", "")).strip()
    if production_stage not in PRODUCTION_STAGES:
        raise ValueError("case production_stage is unsupported")
    overlays = _validate_overlays(record.get("overlays"))
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("case artifacts must be a non-empty list")
    normalized_artifacts: list[dict[str, Any]] = []
    artifact_paths: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError("case artifacts must be objects")
        path = _within(
            volume_root, artifact.get("relpath"), field=f"artifacts[{index}].relpath"
        )
        if not path.is_file():
            raise ValueError(f"case artifact is absent: {path}")
        relpath = path.relative_to(volume_root).as_posix()
        if relpath in artifact_paths:
            raise ValueError("case artifact paths must be unique")
        artifact_paths.add(relpath)
        digest = str(artifact.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("case artifact requires a lowercase SHA-256")
        actual_digest = _sha256(path)
        if not hmac.compare_digest(digest, actual_digest):
            raise ValueError(f"case artifact SHA-256 mismatch: {relpath}")
        item = dict(artifact)
        item["relpath"] = relpath
        item["sha256"] = actual_digest
        item["bytes"] = path.stat().st_size
        normalized_artifacts.append(item)
    normalized = dict(record)
    normalized["category"] = category
    normalized["case_id"] = case_id
    normalized["seed"] = seed
    normalized["production_stage"] = production_stage
    normalized["overlays"] = overlays
    normalized["artifacts"] = normalized_artifacts
    normalized["performance"] = _validate_performance(
        record.get("performance"),
        category=category,
        artifacts=normalized_artifacts,
    )
    _validate_category_contract(normalized)
    return normalized


class CaseStore:
    """Own the review index without moving generated payloads off the pod."""

    def __init__(
        self,
        volume_root: Path,
        *,
        target_per_category: int = TARGET_PER_CATEGORY,
        active_categories: tuple[str, ...] = CATEGORY_IDS,
        render_revision: str = RENDER_REVISION,
    ) -> None:
        if target_per_category not in {TARGET_PER_CATEGORY, MAX_TARGET_PER_CATEGORY}:
            raise ValueError("target_per_category must be 4096 or 8192")
        if (
            not active_categories
            or len(set(active_categories)) != len(active_categories)
            or any(category not in CATEGORY_IDS for category in active_categories)
        ):
            raise ValueError("active_categories must be a non-empty supported subset")
        known_revisions = {
            str(profile["revision"]) for profile in OUTPUT_RENDER_PROFILES.values()
        }
        if render_revision not in known_revisions:
            raise ValueError("render_revision must belong to a supported render profile")
        self.volume_root = volume_root.resolve()
        self.target_per_category = target_per_category
        self.categories = tuple(active_categories)
        self.render_revision = render_revision
        self.production_root = self.volume_root / "production"
        self.index_root = self.production_root / "indexes"
        self.review_root = self.production_root / "reviews"
        self.event_log = self.volume_root / "logs" / "production-events.jsonl"
        self._lock = threading.RLock()
        self._index_cache: dict[str, list[dict[str, str]]] = {}
        self._review_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.review_root.mkdir(parents=True, exist_ok=True)
        self.event_log.parent.mkdir(parents=True, exist_ok=True)

    def _index_path(self, category: str) -> Path:
        if category not in self.categories:
            raise ValueError("unsupported case category")
        return self.index_root / f"{category}.jsonl"

    def _case_path(self, category: str, case_id: str) -> Path:
        _identifier(case_id, field="case_id")
        return self.production_root / "cases" / category / case_id / "case.json"

    def _review_path(self, category: str, case_id: str) -> Path:
        _identifier(case_id, field="case_id")
        return self.review_root / category / f"{case_id}.json"

    def _entries(self, category: str) -> list[dict[str, str]]:
        if category in self._index_cache:
            return list(self._index_cache[category])
        path = self._index_path(category)
        if not path.is_file():
            self._index_cache[category] = []
            return []
        entries: list[dict[str, str]] = []
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            case_id = _identifier(item.get("case_id"), field="case_id")
            if case_id in seen:
                raise RuntimeError(f"duplicate case index entry: {category}/{case_id}")
            seen.add(case_id)
            stage = str(item.get("production_stage", ""))
            case_path = self._case_path(category, case_id)
            if category != "france_incident_days" and case_path.is_file():
                record = json.loads(case_path.read_text(encoding="utf-8"))
                if record.get("render", {}).get("revision") != self.render_revision:
                    # Keep stale unreviewed pilots on disk for audit, but never
                    # expose or count them as current deliverables.
                    continue
            if stage not in PRODUCTION_STAGES:
                if not case_path.is_file():
                    raise RuntimeError(f"indexed case record is absent: {category}/{case_id}")
                record = json.loads(case_path.read_text(encoding="utf-8"))
                stage = str(record.get("production_stage", ""))
            if stage not in PRODUCTION_STAGES:
                raise RuntimeError(f"indexed case stage is invalid: {category}/{case_id}")
            entries.append({"case_id": case_id, "production_stage": stage})
        self._index_cache[category] = entries
        return list(entries)

    def _ids(self, category: str) -> list[str]:
        return [entry["case_id"] for entry in self._entries(category)]

    def _append_index(self, *, category: str, case_id: str, stage: str) -> None:
        entry = {"case_id": case_id, "production_stage": stage}
        index_path = self._index_path(category)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._index_cache.setdefault(category, []).append(entry)

    def register(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_case_record(record, self.volume_root)
        category = normalized["category"]
        case_id = normalized["case_id"]
        if category not in self.categories:
            raise ValueError(f"case category is outside the active campaign: {category}")
        with self._lock:
            case_path = self._case_path(category, case_id)
            if case_path.is_file():
                current = json.loads(case_path.read_text(encoding="utf-8"))
                if current != normalized:
                    review = self.get_review(category, case_id)
                    replaceable = (
                        review is None
                        and current.get("production_stage") == "pilot"
                        and normalized.get("production_stage") == "pilot"
                        and current.get("seed") == normalized.get("seed")
                        and current.get("category") == normalized.get("category")
                        and current.get("render", {}).get("revision")
                        != self.render_revision
                    )
                    if not replaceable:
                        raise RuntimeError(f"case id collision: {category}/{case_id}")
                    _write_json(case_path, normalized)
                    self._index_cache.pop(category, None)
                    if case_id not in self._ids(category):
                        self._append_index(
                            category=category,
                            case_id=case_id,
                            stage=normalized["production_stage"],
                        )
                    self.log_event(
                        "case_replaced_after_quality_revision",
                        category=category,
                        case_id=case_id,
                        revision=self.render_revision,
                    )
                    return normalized
                if case_id not in self._ids(category):
                    self._append_index(
                        category=category,
                        case_id=case_id,
                        stage=normalized["production_stage"],
                    )
                    self.log_event(
                        "case_index_recovered", category=category, case_id=case_id
                    )
                return current
            _write_json(case_path, normalized)
            self._append_index(
                category=category,
                case_id=case_id,
                stage=normalized["production_stage"],
            )
            self.log_event(
                "case_produced",
                category=category,
                case_id=case_id,
                seed=normalized["seed"],
            )
        return normalized

    def get(self, category: str, case_id: str) -> dict[str, Any]:
        if category not in self.categories:
            raise KeyError("unknown category")
        path = self._case_path(category, case_id)
        if not path.is_file():
            raise KeyError("unknown case")
        record = json.loads(path.read_text(encoding="utf-8"))
        review = self.get_review(category, case_id)
        return {**record, "review": review}

    def get_review(self, category: str, case_id: str) -> dict[str, Any] | None:
        cache_key = (category, case_id)
        if cache_key in self._review_cache:
            return self._review_cache[cache_key]
        path = self._review_path(category, case_id)
        if not path.is_file():
            self._review_cache[cache_key] = None
            return None
        review = json.loads(path.read_text(encoding="utf-8"))
        self._review_cache[cache_key] = review
        return review

    def list(
        self,
        *,
        category: str,
        offset: int,
        limit: int,
        review_state: str | None = None,
    ) -> dict[str, Any]:
        if not 0 <= offset <= 1_000_000:
            raise ValueError("offset is outside the supported range")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if review_state not in {None, "unreviewed", "accepted", "rejected"}:
            raise ValueError("unsupported review_state")
        with self._lock:
            identifiers = self._ids(category)
            filtered: list[str] = []
            for case_id in identifiers:
                review = self.get_review(category, case_id)
                state = "unreviewed" if review is None else str(review["decision"])
                if review_state is None or review_state == state:
                    filtered.append(case_id)
            selected = filtered[offset : offset + limit]
            items = [self.get(category, case_id) for case_id in selected]
        return {
            "category": category,
            "offset": offset,
            "limit": limit,
            "total": len(filtered),
            "items": items,
        }

    def review(
        self,
        *,
        category: str,
        case_id: str,
        decision: str,
        reviewer: str,
        notes: str,
        quality_checks: object | None = None,
    ) -> dict[str, Any]:
        if decision not in DECISIONS:
            raise ValueError("decision must be accepted or rejected")
        reviewer_id = _identifier(reviewer, field="reviewer")
        normalized_notes = str(notes).strip()
        if len(normalized_notes) > 1000:
            raise ValueError("review notes cannot exceed 1000 characters")
        self.get(category, case_id)
        required_checks = REQUIRED_QUALITY_CHECKS[category]
        if not isinstance(quality_checks, dict):
            normalized_checks: dict[str, bool] = {}
        else:
            normalized_checks = {
                str(key): value is True for key, value in quality_checks.items()
            }
        unexpected = set(normalized_checks) - required_checks
        if unexpected:
            raise ValueError(f"unsupported quality checks: {sorted(unexpected)}")
        normalized_checks = {
            key: normalized_checks.get(key, False) for key in sorted(required_checks)
        }
        if decision == "accepted" and not all(normalized_checks.values()):
            missing = [key for key, value in normalized_checks.items() if not value]
            raise ValueError(
                f"accepted review is missing mandatory quality checks: {missing}"
            )
        if decision == "rejected" and not normalized_notes:
            raise ValueError("rejected review requires notes describing the defect")
        review = {
            "schema_version": 1,
            "category": category,
            "case_id": case_id,
            "decision": decision,
            "reviewer": reviewer_id,
            "notes": normalized_notes,
            "quality_checks": normalized_checks,
            "reviewed_at": _utc_now(),
        }
        with self._lock:
            _write_json(self._review_path(category, case_id), review)
            self._review_cache[(category, case_id)] = review
            history_path = self.review_root / "review-events.jsonl"
            with history_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(review, sort_keys=True, separators=(",", ":")) + "\n")
            self.log_event(
                "case_reviewed",
                category=category,
                case_id=case_id,
                decision=decision,
                reviewer=reviewer_id,
            )
        return review

    def status(self) -> dict[str, Any]:
        categories: list[dict[str, Any]] = []
        all_ready = True
        all_pilots_ready = True
        with self._lock:
            for category in self.categories:
                entries = self._entries(category)
                identifiers = [entry["case_id"] for entry in entries]
                accepted = 0
                rejected = 0
                pilot_produced = 0
                pilot_accepted = 0
                pilot_rejected = 0
                pilot_elapsed: list[float] = []
                pilot_vram: list[int] = []
                pilot_bytes: list[int] = []
                for entry in entries:
                    case_id = entry["case_id"]
                    review = self.get_review(category, case_id)
                    if review and review["decision"] == "accepted":
                        accepted += 1
                    elif review and review["decision"] == "rejected":
                        rejected += 1
                    if entry["production_stage"] == "pilot":
                        pilot_produced += 1
                        record = json.loads(
                            self._case_path(category, case_id).read_text(
                                encoding="utf-8"
                            )
                        )
                        performance = record.get("performance", {})
                        if isinstance(performance, dict):
                            pilot_elapsed.append(
                                float(performance.get("elapsed_seconds", 0.0))
                            )
                            pilot_vram.append(
                                int(performance.get("vram_peak_bytes", 0))
                            )
                            pilot_bytes.append(
                                int(performance.get("case_output_bytes", 0))
                            )
                        if review and review["decision"] == "accepted":
                            pilot_accepted += 1
                        elif review and review["decision"] == "rejected":
                            pilot_rejected += 1
                produced = len(identifiers)
                reviewed = accepted + rejected
                ready = accepted >= self.target_per_category
                pilot_ready = (
                    pilot_produced >= PILOT_PER_CATEGORY
                    and pilot_accepted == pilot_produced
                    and pilot_rejected == 0
                )
                all_ready = all_ready and ready
                all_pilots_ready = all_pilots_ready and pilot_ready
                categories.append(
                    {
                        "category": category,
                        "label": CATEGORY_LABELS[category],
                        "target": self.target_per_category,
                        "produced": produced,
                        "reviewed": reviewed,
                        "accepted": accepted,
                        "rejected": rejected,
                        "export_ready": ready,
                        "pilot_target": PILOT_PER_CATEGORY,
                        "pilot_produced": pilot_produced,
                        "pilot_accepted": pilot_accepted,
                        "pilot_rejected": pilot_rejected,
                        "pilot_ready": pilot_ready,
                        "pilot_measurements": {
                            "observed_case_count": len(pilot_elapsed),
                            "reviewed_case_count": pilot_accepted
                            + pilot_rejected,
                            "rejection_rate_reviewed": (
                                round(
                                    pilot_rejected
                                    / (pilot_accepted + pilot_rejected),
                                    6,
                                )
                                if pilot_accepted + pilot_rejected
                                else None
                            ),
                            "elapsed_seconds_total": round(
                                sum(pilot_elapsed), 6
                            ),
                            "elapsed_seconds_mean": (
                                round(
                                    sum(pilot_elapsed) / len(pilot_elapsed),
                                    6,
                                )
                                if pilot_elapsed
                                else None
                            ),
                            "vram_peak_bytes_max": max(pilot_vram, default=0),
                            "vram_peak_bytes_mean": (
                                round(sum(pilot_vram) / len(pilot_vram))
                                if pilot_vram
                                else None
                            ),
                            "case_output_bytes_total": sum(pilot_bytes),
                            "case_output_bytes_mean": (
                                round(sum(pilot_bytes) / len(pilot_bytes))
                                if pilot_bytes
                                else None
                            ),
                        },
                    }
                )
        return {
            "schema_version": 1,
            "data_origin": "new_synthetic_generation",
            "target_total": self.target_per_category * len(self.categories),
            "categories": categories,
            "pilot_ready": all_pilots_ready,
            "pilot_guard": (
                "ready_for_bulk"
                if all_pilots_ready
                else f"locked_until_{PILOT_PER_CATEGORY}_pilots_accepted_per_category"
            ),
            "export_ready": all_ready,
            "export_guard": (
                "ready"
                if all_ready
                else f"locked_until_{self.target_per_category}_accepted_per_category"
            ),
            "training_ready_for_integrity_audit": all_ready,
            "training_guard": (
                "eligible_for_full_integrity_audit"
                if all_ready
                else f"locked_until_{self.target_per_category}_human_accepted_per_category"
            ),
        }

    def preview(self, category: str, case_id: str) -> Path:
        record = self.get(category, case_id)
        path = _within(self.volume_root, record["preview_relpath"], field="preview_relpath")
        if not path.is_file():
            raise KeyError("case preview is absent")
        return path

    def log_event(self, event: str, **fields: Any) -> None:
        payload = {"at": _utc_now(), "event": event, **fields}
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def tail_events(self, count: int) -> list[dict[str, Any]]:
        if not 1 <= count <= 500:
            raise ValueError("log tail count must be between 1 and 500")
        if not self.event_log.is_file():
            return []
        lines: deque[str] = deque(maxlen=count)
        with self.event_log.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    lines.append(line)
        return [json.loads(line) for line in lines]
