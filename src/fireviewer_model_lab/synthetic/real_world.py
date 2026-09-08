"""Fail-closed contracts for high-resolution Omniverse real-world scenes."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from pathlib import Path
from typing import Any


PIPELINE_ID = "nvidia_omniverse_nurec_3dgut"
PIPELINE_IDS = frozenset(
    {PIPELINE_ID, "nvidia_omniverse_simready_flow"}
)
RENDER_PROFILE = "omniverse_realworld_hd_v1"
RENDER_REVISION = "realworld-hd-composite-gate-v20"
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
LOCAL_RENDER_PROFILE = "omniverse_realworld_local_720p_v1"
LOCAL_RENDER_REVISION = "realworld-local-720p-composite-gate-v2"
LOCAL_RENDER_WIDTH = 1280
LOCAL_RENDER_HEIGHT = 720
OUTPUT_RENDER_PROFILES = {
    RENDER_PROFILE: {
        "resolution": [RENDER_WIDTH, RENDER_HEIGHT],
        "revision": RENDER_REVISION,
    },
    LOCAL_RENDER_PROFILE: {
        "resolution": [LOCAL_RENDER_WIDTH, LOCAL_RENDER_HEIGHT],
        "revision": LOCAL_RENDER_REVISION,
    },
}
MIN_CAPTURE_IMAGES = 100
MIN_REGISTERED_RATIO = 0.95
MIN_SOURCE_WIDTH = 3840
MIN_SOURCE_HEIGHT = 2160
MIN_PSNR = 25.0
MIN_SSIM = 0.90
MIN_CAMERA_POSES = 2
MAX_CAMERA_POSES = 8
MIN_ACTOR_CAMERA_POSES = 1
MIN_LIGHTING_VARIANTS = 4
MIN_EVENT_STATES = 4
MIN_EVENT_VARIATIONS = 4
MIN_FIRE_DURATION_DAYS = 1
MAX_FIRE_DURATION_DAYS = 15
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
REQUIRED_CAMPAIGN_PROGRESSION_PHASES = frozenset(
    {"advancing_flame_zone", "front_split", "reignition"}
)
DISTANCE_BANDS = frozenset({"near", "medium", "far", "very_far"})
OCCLUSION_CLASSES = frozenset(
    {"clear", "partial_building", "partial_mountain"}
)
TIME_OF_DAY_CLASSES = frozenset({"day", "night", "dawn", "dusk"})
FRENCH_LANDSCAPE_PROFILES = frozenset(
    {
        "rural",
        "mountain",
        "agricultural",
        "rural_mountain",
        "rural_agricultural",
        "mountain_agricultural",
    }
)
SCENE_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
REQUIRED_ANCHORS = frozenset(
    {"active_fire_point", "visible_fire_front_point", "smoke_column_base"}
)
REQUIRED_ACTOR_CLASSES = frozenset(
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
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PENDING_REVIEW = "pending_console_review"
APPROVED_REFERENCE = "reference_render_human_approved"


def validate_output_render_profile(
    profile: object, resolution: object
) -> dict[str, object]:
    identifier = str(profile or "").strip()
    expected = OUTPUT_RENDER_PROFILES.get(identifier)
    if expected is None:
        raise ValueError(
            "render_profile must be one of "
            f"{sorted(OUTPUT_RENDER_PROFILES)}"
        )
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("resolution must contain exactly two integers")
    normalized = [int(value) for value in resolution]
    if normalized != expected["resolution"]:
        width, height = expected["resolution"]
        raise ValueError(
            f"render profile {identifier} requires exactly {width}x{height}"
        )
    return {
        "profile": identifier,
        "resolution": normalized,
        "revision": str(expected["revision"]),
    }


def _vector(value: object, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain three numbers")
    result = [float(component) for component in value]
    if any(not math.isfinite(component) for component in result):
        raise ValueError(f"{field} must contain finite numbers")
    return result


def _resolve_file(
    value: object,
    *,
    field: str,
    contract_root: Path,
    volume_root: Path,
    suffixes: frozenset[str] | None = None,
) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{field} is required")
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else contract_root / candidate).resolve()
    if resolved != volume_root and volume_root not in resolved.parents:
        raise ValueError(f"{field} must remain inside the production volume")
    if not resolved.is_file():
        raise ValueError(f"{field} is absent: {resolved}")
    if suffixes is not None and resolved.suffix.lower() not in suffixes:
        raise ValueError(f"{field} has an unsupported file type")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(
    payload: dict[str, Any],
    *,
    path_field: str,
    digest_field: str,
    contract_root: Path,
    volume_root: Path,
    suffixes: frozenset[str] | None = None,
) -> Path:
    path = _resolve_file(
        payload.get(path_field),
        field=path_field,
        contract_root=contract_root,
        volume_root=volume_root,
        suffixes=suffixes,
    )
    expected = str(payload.get(digest_field, "")).strip()
    if not SHA256.fullmatch(expected):
        raise ValueError(f"{digest_field} must be a lowercase SHA-256")
    actual = sha256_file(path)
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"{digest_field} does not match {path_field}")
    return path


def _validate_capture(
    payload: object, *, contract_root: Path, volume_root: Path
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("capture must be an object")
    capture = dict(payload)
    source = str(capture.get("source", "")).strip()
    if source not in {"new_real_world_capture", "new_synthetic_french_reference"}:
        raise ValueError("capture.source is unsupported")
    manifest = _verified_file(
        capture,
        path_field="capture_manifest",
        digest_field="capture_manifest_sha256",
        contract_root=contract_root,
        volume_root=volume_root,
    )
    if source == "new_synthetic_french_reference":
        reference_count = int(
            capture.get(
                "reference_render_slot_count",
                capture.get("reference_image_count", 0),
            )
        )
        resolution = capture.get("minimum_source_resolution")
        if reference_count < 12:
            raise ValueError(
                "synthetic French scenes require at least twelve review render slots"
            )
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or int(resolution[0]) < MIN_SOURCE_WIDTH
            or int(resolution[1]) < MIN_SOURCE_HEIGHT
        ):
            raise ValueError("scene references must be at least 3840x2160")
        for field in ("terrain_scale_validated", "orthophoto_mnt_coherence_validated"):
            if capture.get(field) is not True:
                raise ValueError(f"capture.{field} must be explicitly true")
        if capture.get("materials_validation") not in {
            APPROVED_REFERENCE,
            PENDING_REVIEW,
        }:
            raise ValueError(
                "capture.materials_validation requires an approved reference or console review"
            )
        if capture.get("coordinate_convention") != "usd_z_up_meters_lambert93":
            raise ValueError(
                "synthetic scene coordinate_convention must be usd_z_up_meters_lambert93"
            )
        return {
            **capture,
            "source": source,
            "capture_manifest": manifest,
            "reference_render_slot_count": reference_count,
            "minimum_source_resolution": [int(resolution[0]), int(resolution[1])],
        }
    image_count = int(capture.get("image_count", 0))
    registered = int(capture.get("registered_image_count", 0))
    if image_count < MIN_CAPTURE_IMAGES:
        raise ValueError(f"capture requires at least {MIN_CAPTURE_IMAGES} images")
    if registered < math.ceil(image_count * MIN_REGISTERED_RATIO):
        raise ValueError("COLMAP must register at least 95 percent of capture images")
    resolution = capture.get("minimum_source_resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or int(resolution[0]) < MIN_SOURCE_WIDTH
        or int(resolution[1]) < MIN_SOURCE_HEIGHT
    ):
        raise ValueError("capture source resolution must be at least 3840x2160")
    reprojection_error = float(capture.get("mean_reprojection_error_px", math.inf))
    if not math.isfinite(reprojection_error) or not 0.0 <= reprojection_error <= 1.0:
        raise ValueError("COLMAP mean reprojection error must be at most one pixel")
    if capture.get("overlap_validated") is not True:
        raise ValueError("capture overlap must be explicitly validated")
    for field in ("intrinsics_validated", "extrinsics_validated", "timestamps_validated"):
        if capture.get(field) is not True:
            raise ValueError(f"capture.{field} must be explicitly true")
    if capture.get("coordinate_convention") != "ncore_rig_and_camera_v4":
        raise ValueError("capture.coordinate_convention must be ncore_rig_and_camera_v4")
    capture.update(
        capture_manifest=manifest,
        image_count=image_count,
        registered_image_count=registered,
        minimum_source_resolution=[int(resolution[0]), int(resolution[1])],
        mean_reprojection_error_px=reprojection_error,
    )
    return capture


def _validate_camera_poses(payload: object) -> list[dict[str, Any]]:
    if (
        not isinstance(payload, list)
        or not MIN_CAMERA_POSES <= len(payload) <= MAX_CAMERA_POSES
    ):
        raise ValueError(
            "composition requires between "
            f"{MIN_CAMERA_POSES} and {MAX_CAMERA_POSES} fixed validated camera stations"
        )
    identifiers: set[str] = set()
    geometries: set[tuple[float, ...]] = set()
    poses: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("camera poses must be objects")
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError("camera pose ids must be non-empty and unique")
        identifiers.add(identifier)
        position = _vector(item.get("position"), field=f"camera_poses[{index}].position")
        look_at = _vector(item.get("look_at"), field=f"camera_poses[{index}].look_at")
        geometry = tuple(position + look_at)
        if geometry in geometries:
            raise ValueError("camera poses must have unique position/look_at geometry")
        geometries.add(geometry)
        viewpoint = item.get("viewpoint")
        if not isinstance(viewpoint, dict):
            raise ValueError("every camera pose requires viewpoint metadata")
        distance_band = str(viewpoint.get("distance_band", "")).strip()
        occlusion = str(viewpoint.get("occlusion", "")).strip()
        if distance_band not in DISTANCE_BANDS:
            raise ValueError("camera viewpoint distance_band is unsupported")
        if occlusion not in OCCLUSION_CLASSES:
            raise ValueError("camera viewpoint occlusion is unsupported")
        azimuth_deg = float(viewpoint.get("azimuth_deg", math.nan))
        elevation_deg = float(viewpoint.get("elevation_deg", math.nan))
        occlusion_fraction = float(
            viewpoint.get("occlusion_fraction", math.nan)
        )
        if not math.isfinite(azimuth_deg) or not 0.0 <= azimuth_deg < 360.0:
            raise ValueError("camera viewpoint azimuth_deg must be in [0, 360)")
        if not math.isfinite(elevation_deg) or not -45.0 <= elevation_deg <= 60.0:
            raise ValueError("camera viewpoint elevation_deg is not operationally plausible")
        if (
            not math.isfinite(occlusion_fraction)
            or not 0.0 <= occlusion_fraction <= 0.8
        ):
            raise ValueError("camera viewpoint occlusion_fraction must be in [0, 0.8]")
        occluder_prim_path = str(viewpoint.get("occluder_prim_path", "")).strip()
        if occlusion == "clear" and occlusion_fraction != 0.0:
            raise ValueError("clear viewpoints cannot declare an occlusion fraction")
        if occlusion != "clear" and not occluder_prim_path.startswith("/World/"):
            raise ValueError("occluded viewpoints require an authored USD occluder")
        line_of_sight_validation = str(
            viewpoint.get("line_of_sight_validation", "")
        ).strip()
        if line_of_sight_validation not in {
            "usd_raycast_and_reference_render_passed",
            PENDING_REVIEW,
        }:
            raise ValueError(
                "camera viewpoint requires raycast/reference validation or console review"
            )
        anchors_visibility = viewpoint.get("required_anchors_visible")
        if anchors_visibility not in {True, PENDING_REVIEW}:
            raise ValueError(
                "camera viewpoint anchor visibility must be proved or pending console review"
            )
        reference_validation = str(
            viewpoint.get("reference_validation", "")
        ).strip()
        if reference_validation not in {
            APPROVED_REFERENCE,
            PENDING_REVIEW,
        }:
            raise ValueError(
                "camera viewpoint requires prevalidated reference or console review"
            )
        poses.append(
            {
                "id": identifier,
                "position": position,
                "look_at": look_at,
                "validation": str(item.get("validation", "")),
                "viewpoint": {
                    **viewpoint,
                    "distance_band": distance_band,
                    "occlusion": occlusion,
                    "azimuth_deg": azimuth_deg,
                    "elevation_deg": elevation_deg,
                    "occlusion_fraction": occlusion_fraction,
                    "occluder_prim_path": occluder_prim_path,
                    "reference_validation": reference_validation,
                },
            }
        )
        if poses[-1]["validation"] not in {
            "ncore_project_to_image_passed",
            "calibrated_project_to_image_passed",
            PENDING_REVIEW,
        }:
            raise ValueError(
                "every camera pose requires calibrated projection or console review"
            )
    return poses


def _validate_actors(
    payload: object,
    *,
    contract_root: Path,
    volume_root: Path,
    camera_pose_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("composition.actors must be a list")
    actors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("composition actors must be objects")
        actor = dict(item)
        class_id = str(actor.get("class_id", "")).strip()
        if class_id not in REQUIRED_ACTOR_CLASSES or class_id in seen:
            raise ValueError("composition actor classes must be complete and unique")
        seen.add(class_id)
        asset = _verified_file(
            actor,
            path_field="asset",
            digest_field="asset_sha256",
            contract_root=contract_root,
            volume_root=volume_root,
            suffixes=SCENE_SUFFIXES,
        )
        center = _vector(actor.get("center_world_m"), field=f"actors[{index}].center_world_m")
        minimum = _vector(actor.get("aabb_min_world_m"), field=f"actors[{index}].aabb_min_world_m")
        maximum = _vector(actor.get("aabb_max_world_m"), field=f"actors[{index}].aabb_max_world_m")
        translation = _vector(
            actor.get("translation_world_m"), field=f"actors[{index}].translation_world_m"
        )
        rotation = _vector(
            actor.get("rotation_xyz_deg"), field=f"actors[{index}].rotation_xyz_deg"
        )
        scale = _vector(actor.get("scale_xyz"), field=f"actors[{index}].scale_xyz")
        if any(component <= 0.0 for component in scale):
            raise ValueError("actor scale_xyz values must be positive")
        pose_ids = actor.get("camera_pose_ids")
        if (
            not isinstance(pose_ids, list)
            or len(pose_ids) < MIN_ACTOR_CAMERA_POSES
            or len({str(value) for value in pose_ids}) != len(pose_ids)
            or any(str(value) not in camera_pose_ids for value in pose_ids)
        ):
            raise ValueError(
                f"each actor requires at least {MIN_ACTOR_CAMERA_POSES} validated camera_pose_ids"
            )
        if any(minimum[axis] >= maximum[axis] for axis in range(3)):
            raise ValueError("actor AABBs must have positive extent")
        expected_context = (
            "hard_negative_not_engaged"
            if class_id.startswith("hard_negative")
            else "wildfire_response_engaged"
        )
        if actor.get("engagement_context") != expected_context:
            raise ValueError(f"actor {class_id} requires engagement_context={expected_context}")
        if actor.get("quality_validation") not in {
            "simready_asset_human_approved",
            PENDING_REVIEW,
        }:
            raise ValueError(
                f"actor {class_id} requires a SimReady quality approval or console review"
            )
        actors.append(
            {
                **actor,
                "class_id": class_id,
                "asset": asset,
                "center_world_m": center,
                "aabb_min_world_m": minimum,
                "aabb_max_world_m": maximum,
                "translation_world_m": translation,
                "rotation_xyz_deg": rotation,
                "scale_xyz": scale,
                "camera_pose_ids": [str(value) for value in pose_ids],
                "positive": not class_id.startswith("hard_negative"),
            }
        )
    if seen != REQUIRED_ACTOR_CLASSES:
        raise ValueError(f"composition actors are missing classes: {sorted(REQUIRED_ACTOR_CLASSES - seen)}")
    return actors


def _validate_lighting_variants(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or len(payload) < MIN_LIGHTING_VARIANTS:
        raise ValueError(f"composition requires at least {MIN_LIGHTING_VARIANTS} lighting variants")
    variants: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    selections: set[tuple[str, str, str]] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("lighting variants must be objects")
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError("lighting variant ids must be non-empty and unique")
        identifiers.add(identifier)
        prim_path = str(item.get("prim_path", "")).strip()
        if not prim_path.startswith("/World/"):
            raise ValueError("lighting variant prim_path must remain under /World")
        variant_set = str(item.get("variant_set", "")).strip()
        selection = str(item.get("selection", "")).strip()
        if not variant_set or not selection:
            raise ValueError("lighting variants require variant_set and selection")
        authored = (prim_path, variant_set, selection)
        if authored in selections:
            raise ValueError("lighting variants must select unique authored USD variants")
        selections.add(authored)
        if item.get("validation") not in {APPROVED_REFERENCE, PENDING_REVIEW}:
            raise ValueError(
                "every lighting variant requires an approved reference or console review"
            )
        time_of_day = str(item.get("time_of_day", "")).strip()
        if time_of_day not in TIME_OF_DAY_CLASSES:
            raise ValueError("lighting variant time_of_day is unsupported")
        variants.append(
            {
                **item,
                "id": identifier,
                "prim_path": prim_path,
                "variant_set": variant_set,
                "selection": selection,
                "time_of_day": time_of_day,
            }
        )
    return variants


def _validate_flow_states(
    payload: object, *, lighting_variant_ids: set[str], duration_days: int
) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or len(payload) < MIN_EVENT_STATES:
        raise ValueError(
            f"composition requires at least {MIN_EVENT_STATES} baked progression states"
        )
    states: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    previous_time = -1.0
    previous_burned_area = -1.0
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("Flow states must be objects")
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError("Flow state ids must be non-empty and unique")
        identifiers.add(identifier)
        time_seconds = float(item.get("time_seconds", math.nan))
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("Flow state time_seconds must be finite and non-negative")
        if time_seconds <= previous_time:
            raise ValueError("Flow states must use strictly increasing baked time_seconds")
        previous_time = time_seconds
        event_day = int(item.get("event_day", 0))
        if not 1 <= event_day <= duration_days:
            raise ValueError("Flow state event_day exceeds the fire duration")
        lighting_variant_id = str(item.get("lighting_variant_id", "")).strip()
        if lighting_variant_id not in lighting_variant_ids:
            raise ValueError(
                "every Flow state must reference an approved lighting variant"
            )
        anchors_payload = item.get("anchors_world_m")
        if not isinstance(anchors_payload, dict) or set(anchors_payload) != REQUIRED_ANCHORS:
            raise ValueError("every Flow state requires exactly the three fire/smoke anchors")
        if item.get("validation") not in {
            "flow_reference_render_human_approved",
            PENDING_REVIEW,
        }:
            raise ValueError(
                "every Flow state requires an approved reference or console review"
            )
        progression = item.get("progression")
        if not isinstance(progression, dict):
            raise ValueError("every Flow state requires progression metadata")
        phase = str(progression.get("phase", "")).strip()
        if phase not in PROGRESSION_PHASES:
            raise ValueError("Flow state progression phase is unsupported")
        front_ids = progression.get("front_ids")
        if (
            not isinstance(front_ids, list)
            or not front_ids
            or len({str(value) for value in front_ids}) != len(front_ids)
            or any(not str(value).strip() for value in front_ids)
        ):
            raise ValueError("Flow state progression requires unique active front_ids")
        burned_area_m2 = float(progression.get("burned_area_m2", math.nan))
        active_flame_area_m2 = float(
            progression.get("active_flame_area_m2", math.nan)
        )
        if (
            not math.isfinite(burned_area_m2)
            or burned_area_m2 < previous_burned_area
            or not math.isfinite(active_flame_area_m2)
            or active_flame_area_m2 <= 0.0
        ):
            raise ValueError(
                "progression areas must be finite, positive, and burned area non-decreasing"
            )
        previous_burned_area = burned_area_m2
        advancing_zone_ids = progression.get("advancing_zone_ids", [])
        parent_front_ids = progression.get("parent_front_ids", [])
        reignited_zone_ids = progression.get("reignited_zone_ids", [])
        if phase == "advancing_flame_zone" and not advancing_zone_ids:
            raise ValueError("advancing_flame_zone requires advancing_zone_ids")
        if phase == "front_split" and (len(front_ids) < 2 or not parent_front_ids):
            raise ValueError("front_split requires multiple fronts and parent_front_ids")
        if phase == "reignition" and not reignited_zone_ids:
            raise ValueError("reignition requires reignited_zone_ids")
        states.append(
            {
                **item,
                "id": identifier,
                "time_seconds": time_seconds,
                "event_day": event_day,
                "lighting_variant_id": lighting_variant_id,
                "progression": {
                    **progression,
                    "phase": phase,
                    "front_ids": [str(value) for value in front_ids],
                    "burned_area_m2": burned_area_m2,
                    "active_flame_area_m2": active_flame_area_m2,
                },
                "anchors_world_m": {
                    label: _vector(value, field=f"flow_states[{index}].{label}")
                    for label, value in anchors_payload.items()
                },
            }
        )
    return states


def load_real_world_contract(path: Path, *, volume_root: Path) -> dict[str, Any]:
    contract_path = path.resolve()
    volume = volume_root.resolve()
    if contract_path != volume and volume not in contract_path.parents:
        raise ValueError("real-world contract must remain inside the production volume")
    if not contract_path.is_file():
        raise ValueError(f"real-world contract is absent: {contract_path}")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported real-world contract schema_version")
    if payload.get("pipeline") not in PIPELINE_IDS:
        raise ValueError(f"real-world contract pipeline must be one of {sorted(PIPELINE_IDS)}")
    if payload.get("render_profile") != RENDER_PROFILE:
        raise ValueError(f"real-world contract render_profile must be {RENDER_PROFILE}")
    site_id = str(payload.get("site_id", "")).strip()
    if not site_id:
        raise ValueError("real-world contract site_id is required")
    event_id = str(payload.get("event_id", "")).strip()
    if not event_id:
        raise ValueError("real-world contract event_id is required")
    duration_days = int(payload.get("duration_days", 0))
    if not MIN_FIRE_DURATION_DAYS <= duration_days <= MAX_FIRE_DURATION_DAYS:
        raise ValueError("real-world contract duration_days must be between 1 and 15")
    contract_root = contract_path.parent
    capture = _validate_capture(
        payload.get("capture"), contract_root=contract_root, volume_root=volume
    )

    reconstruction = payload.get("reconstruction")
    if not isinstance(reconstruction, dict):
        raise ValueError("reconstruction must be an object")
    reconstruction = dict(reconstruction)
    synthetic_scene = capture["source"] == "new_synthetic_french_reference"
    if synthetic_scene:
        if reconstruction.get("trainer") != "fireviewer/omniverse_usd_terrain":
            raise ValueError(
                "synthetic scene trainer must be fireviewer/omniverse_usd_terrain"
            )
        if reconstruction.get("format") != "review_gated_usd":
            raise ValueError("synthetic scene format must be review_gated_usd")
    else:
        if reconstruction.get("trainer") not in {"nv-tlabs/3dgrut", "nvidia/nre"}:
            raise ValueError("reconstruction trainer must be an NVIDIA NuRec implementation")
        if reconstruction.get("format") not in {"particle_field", "nurec_usdz"}:
            raise ValueError("reconstruction format must be particle_field or nurec_usdz")
    scene_asset = _verified_file(
        reconstruction,
        path_field="asset",
        digest_field="asset_sha256",
        contract_root=contract_root,
        volume_root=volume,
        suffixes=SCENE_SUFFIXES,
    )
    metrics = reconstruction.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("reconstruction metrics are required")
    if synthetic_scene:
        quality_review = str(metrics.get("quality_review", "")).strip()
        if quality_review not in {
            "reference_render_human_approved",
            "pending_console_review",
        }:
            raise ValueError("SimReady scene requires reference or console quality review")
        normalized_metrics = {"quality_review": quality_review}
    else:
        psnr = float(metrics.get("psnr", -math.inf))
        ssim = float(metrics.get("ssim", -math.inf))
        if not math.isfinite(psnr) or psnr < MIN_PSNR:
            raise ValueError(f"reconstruction PSNR must be at least {MIN_PSNR}")
        if not math.isfinite(ssim) or ssim < MIN_SSIM:
            raise ValueError(f"reconstruction SSIM must be at least {MIN_SSIM}")
        if metrics.get("held_out_evaluation") is not True:
            raise ValueError("reconstruction metrics must come from held-out evaluation")
        held_out_view_count = int(metrics.get("held_out_view_count", 0))
        if held_out_view_count < 10:
            raise ValueError("reconstruction requires at least ten held-out evaluation views")
        normalized_metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "held_out_evaluation": True,
            "held_out_view_count": held_out_view_count,
        }

    composition = payload.get("composition")
    if not isinstance(composition, dict):
        raise ValueError("composition must be an object")
    composition = dict(composition)
    flow_asset = _verified_file(
        composition,
        path_field="flow_asset",
        digest_field="flow_asset_sha256",
        contract_root=contract_root,
        volume_root=volume,
        suffixes=SCENE_SUFFIXES,
    )
    flow_validation = composition.get("flow_validation")
    if not isinstance(flow_validation, dict):
        raise ValueError("composition.flow_validation is required")
    if flow_validation.get("preset_rendered_and_anchor_verified") not in {
        True,
        PENDING_REVIEW,
    }:
        raise ValueError(
            "Flow preset and anchors must be visually verified or pending console review"
        )
    camera_poses = _validate_camera_poses(composition.get("camera_poses"))
    lighting_variants = _validate_lighting_variants(composition.get("lighting_variants"))
    flow_states = _validate_flow_states(
        composition.get("flow_states"),
        lighting_variant_ids={variant["id"] for variant in lighting_variants},
        duration_days=duration_days,
    )
    if int(flow_validation.get("simulated_frame_count", 0)) < len(flow_states):
        raise ValueError(
            "Flow validation must cover every coupled event state"
        )
    diversity = composition.get("diversity")
    if not isinstance(diversity, dict):
        raise ValueError("composition.diversity is required")
    if diversity.get("selector") != "operational_viewpoint_progression_v1":
        raise ValueError("unsupported composition diversity selector")
    diversity_capacity = len(camera_poses) * len(flow_states)
    if int(diversity.get("capacity_per_category", 0)) != diversity_capacity:
        raise ValueError("declared diversity capacity does not match validated axes")
    if diversity_capacity < MIN_EVENT_VARIATIONS:
        raise ValueError(
            f"event diversity must support at least {MIN_EVENT_VARIATIONS} viewpoint/progression pairs"
        )
    scope = payload.get("scope", {})
    if not isinstance(scope, dict):
        raise ValueError("real-world contract scope must be an object")
    response_engagement_in_scope = scope.get("response_engagement", True)
    if not isinstance(response_engagement_in_scope, bool):
        raise ValueError("scope.response_engagement must be a boolean")
    if response_engagement_in_scope:
        actors = _validate_actors(
            composition.get("actors"),
            contract_root=contract_root,
            volume_root=volume,
            camera_pose_ids={pose["id"] for pose in camera_poses},
        )
    else:
        if composition.get("actors") != []:
            raise ValueError(
                "out-of-scope response engagement contracts must contain no actors"
            )
        actors = []

    geospatial = payload.get("geospatial")
    if not isinstance(geospatial, dict):
        raise ValueError("geospatial must be an object")
    geospatial = dict(geospatial)
    if geospatial.get("crs") != "EPSG:2154":
        raise ValueError("geospatial CRS must be EPSG:2154")
    if geospatial.get("country_profile") != "FR":
        raise ValueError("geospatial country_profile must be FR")
    landscape_profile = str(geospatial.get("landscape_profile", "")).strip()
    if landscape_profile not in FRENCH_LANDSCAPE_PROFILES:
        raise ValueError("geospatial landscape_profile is not a supported French terrain")
    landscape_origin = str(geospatial.get("landscape_origin", "")).strip()
    if landscape_origin not in {"real_french_capture", "synthetic_french_reference"}:
        raise ValueError("geospatial landscape_origin is unsupported")
    expected_origin = (
        "synthetic_french_reference" if synthetic_scene else "real_french_capture"
    )
    if landscape_origin != expected_origin:
        raise ValueError("geospatial landscape_origin conflicts with scene provenance")
    site_context_validation = str(
        geospatial.get("site_context_validation", "")
    ).strip()
    if site_context_validation not in {
        "reference_render_human_approved",
        "pending_console_review",
    }:
        raise ValueError("French landscape context requires reference or console review")
    if synthetic_scene and geospatial.get("real_world_claim") is not False:
        raise ValueError("synthetic French terrain must disable real-world claims")
    if geospatial.get("world_axes_aligned_lambert93") is not True:
        raise ValueError("scene axes must be explicitly aligned to Lambert-93")
    origin = _vector(geospatial.get("world_origin_lambert93_m"), field="world_origin_lambert93_m")
    orthophoto = _verified_file(
        geospatial,
        path_field="orthophoto",
        digest_field="orthophoto_sha256",
        contract_root=contract_root,
        volume_root=volume,
    )
    mnt = _verified_file(
        geospatial,
        path_field="mnt",
        digest_field="mnt_sha256",
        contract_root=contract_root,
        volume_root=volume,
    )
    mnt_preview = _verified_file(
        geospatial,
        path_field="mnt_preview",
        digest_field="mnt_preview_sha256",
        contract_root=contract_root,
        volume_root=volume,
    )

    return {
        **payload,
        "contract_path": contract_path,
        "site_id": site_id,
        "event_id": event_id,
        "duration_days": duration_days,
        "scope": {
            **scope,
            "response_engagement": response_engagement_in_scope,
        },
        "capture": capture,
        "reconstruction": {
            **reconstruction,
            "asset": scene_asset,
            "metrics": normalized_metrics,
        },
        "composition": {
            **composition,
            "flow_asset": flow_asset,
            "camera_poses": camera_poses,
            "lighting_variants": lighting_variants,
            "flow_states": flow_states,
            "diversity": {
                **diversity,
                "capacity_per_category": diversity_capacity,
            },
            "actors": actors,
        },
        "geospatial": {
            **geospatial,
            "world_origin_lambert93_m": origin,
            "country_profile": "FR",
            "landscape_profile": landscape_profile,
            "landscape_origin": landscape_origin,
            "orthophoto": orthophoto,
            "mnt": mnt,
            "mnt_preview": mnt_preview,
        },
    }


def select_camera_pose(contract: dict[str, Any], seed: int) -> dict[str, Any]:
    poses = contract["composition"]["camera_poses"]
    return dict(poses[seed % len(poses)])


def select_actor_camera_pose(
    contract: dict[str, Any], class_id: str, seed: int
) -> dict[str, Any]:
    poses = {
        pose["id"]: pose for pose in contract["composition"]["camera_poses"]
    }
    actor = next(
        item
        for item in contract["composition"]["actors"]
        if item["class_id"] == class_id
    )
    identifiers = actor["camera_pose_ids"]
    return dict(poses[identifiers[seed % len(identifiers)]])


def _signature(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _operational_pairs(pose_count: int, state_count: int) -> list[tuple[int, int]]:
    """Front-load marginal coverage, then enumerate every unique valid pair."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for diagonal in range(state_count):
        for pose_index in range(pose_count):
            pair = (pose_index, (pose_index + diagonal) % state_count)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    if len(pairs) != pose_count * state_count:
        raise RuntimeError("operational variation pair enumeration is incomplete")
    return pairs


def select_case_variation(contract: dict[str, Any], case_index: int) -> dict[str, Any]:
    if case_index < 0:
        raise ValueError("case_index must be non-negative")
    composition = contract["composition"]
    poses = composition["camera_poses"]
    lighting = {
        item["id"]: item for item in composition["lighting_variants"]
    }
    flows = composition["flow_states"]
    pairs = _operational_pairs(len(poses), len(flows))
    capacity = len(pairs)
    if case_index >= capacity:
        raise ValueError("case_index exceeds the validated diversity capacity")
    pose_index, state_index = pairs[case_index]
    pose = poses[pose_index]
    flow_state = flows[state_index]
    lighting_variant = lighting[flow_state["lighting_variant_id"]]
    return {
        "id": f"variation-{case_index:06d}",
        "camera_pose": dict(pose),
        "lighting": dict(lighting_variant),
        "flow": dict(flow_state),
        "diversity_signature": _signature(
            contract["event_id"], pose["id"], lighting_variant["id"], flow_state["id"]
        ),
    }


def select_actor_variation(
    contract: dict[str, Any], class_id: str, class_ordinal: int
) -> dict[str, Any]:
    if class_ordinal < 0:
        raise ValueError("class_ordinal must be non-negative")
    composition = contract["composition"]
    actor = next(item for item in composition["actors"] if item["class_id"] == class_id)
    pose_ids = actor["camera_pose_ids"]
    lighting = {
        item["id"]: item for item in composition["lighting_variants"]
    }
    flows = composition["flow_states"]
    pairs = _operational_pairs(len(pose_ids), len(flows))
    capacity = len(pairs)
    if class_ordinal >= capacity:
        raise ValueError("response class ordinal exceeds validated diversity capacity")
    poses = {pose["id"]: pose for pose in composition["camera_poses"]}
    pose_index, state_index = pairs[class_ordinal]
    pose = poses[pose_ids[pose_index]]
    flow_state = flows[state_index]
    lighting_variant = lighting[flow_state["lighting_variant_id"]]
    return {
        "id": f"{class_id}-variation-{class_ordinal:06d}",
        "camera_pose": dict(pose),
        "lighting": dict(lighting_variant),
        "flow": dict(flow_state),
        "diversity_signature": _signature(
            contract["event_id"], class_id, pose["id"], lighting_variant["id"], flow_state["id"]
        ),
    }


def apply_case_variation(
    stage: Any, application: Any, variation: dict[str, Any]
) -> None:
    """Apply only pre-authored and reference-render-approved variation axes."""
    import omni.timeline

    lighting = variation["lighting"]
    prim = stage.GetPrimAtPath(lighting["prim_path"])
    if not prim or not prim.IsValid():
        raise RuntimeError(f"lighting variant prim is absent: {lighting['prim_path']}")
    variant_set = prim.GetVariantSets().GetVariantSet(lighting["variant_set"])
    if lighting["selection"] not in set(variant_set.GetVariantNames()):
        raise RuntimeError(f"lighting variant selection is absent: {lighting['id']}")
    if not variant_set.SetVariantSelection(lighting["selection"]):
        raise RuntimeError(f"lighting variant could not be selected: {lighting['id']}")
    timeline = omni.timeline.get_timeline_interface()
    _set_flow_emitter_radius(
        stage,
        progression=variation["flow"]["progression"],
    )
    timeline.set_current_time(variation["flow"]["time_seconds"])
    application.update()


def _flow_emitter_radius_m(active_flame_area_m2: float) -> float:
    """Map authored fire-front area to a bounded Flow emitter radius in metres."""
    if not math.isfinite(active_flame_area_m2) or active_flame_area_m2 <= 0.0:
        raise ValueError("active flame area must be a positive finite value")
    # The emitter is only the energetic core, not the whole authored flame area.
    return max(0.8, min(2.5, math.sqrt(active_flame_area_m2 / math.pi) * 0.22))


def _set_flow_emitter_radius(
    stage: Any, *, progression: dict[str, Any]
) -> dict[str, Any]:
    """Bound the hero preset and drive it from the authored fire progression."""
    from pxr import Gf, Usd

    root = stage.GetPrimAtPath("/World/FireAndSmoke")
    if not root or not root.IsValid():
        raise RuntimeError("Flow root is absent from the composed stage")
    active_flame_area_m2 = float(progression["active_flame_area_m2"])
    radius_m = _flow_emitter_radius_m(active_flame_area_m2)
    wind_speed_mps = float(progression.get("wind_speed_mps", 0.0))
    wind_heading_deg = float(progression.get("wind_heading_deg", 0.0))
    if (
        not math.isfinite(wind_speed_mps)
        or not 0.0 <= wind_speed_mps <= 20.0
        or not math.isfinite(wind_heading_deg)
    ):
        raise ValueError("Flow progression wind must be finite and operationally bounded")
    heading = math.radians(wind_heading_deg)
    emitter_velocity = Gf.Vec3f(
        math.cos(heading) * wind_speed_mps,
        math.sin(heading) * wind_speed_mps,
        max(0.6, min(2.0, 0.35 + radius_m * 0.45)),
    )
    radius_attributes: list[str] = []
    world_space_attributes: list[str] = []
    density_cell_attributes: list[str] = []
    bounded_attributes: list[dict[str, Any]] = []
    velocity_attributes: list[str] = []

    def cap(attribute: Any, maximum: float) -> None:
        current = attribute.Get()
        if isinstance(current, (int, float)) and math.isfinite(float(current)):
            value = min(float(current), maximum)
            attribute.Set(value)
            bounded_attributes.append(
                {
                    "path": str(attribute.GetPath()),
                    "previous": float(current),
                    "value": value,
                }
            )

    for prim in Usd.PrimRange(root):
        prim_name = prim.GetName().lower()
        for attribute in prim.GetAttributes():
            base_name = str(attribute.GetBaseName())
            attribute_path = str(attribute.GetPath())
            if base_name == "radius" and "emitter" in prim_name:
                attribute.Set(radius_m)
                radius_attributes.append(attribute_path)
            elif base_name == "radiusIsWorldSpace" and "emitter" in prim_name:
                # NVIDIA documents this switch as the way to keep the radius in
                # world units. Author the exact metric radius instead of relying
                # on an Xform scale that the preset may intentionally ignore.
                attribute.Set(True)
                world_space_attributes.append(attribute_path)
            elif base_name == "densityCellSize" and "simulate" in prim_name:
                attribute.Set(0.35)
                density_cell_attributes.append(attribute_path)
            elif "emitter" in prim_name and base_name == "smoke":
                cap(attribute, 0.35)
            elif "emitter" in prim_name and base_name == "smokeScale":
                cap(attribute, 1.0)
            elif "emitter" in prim_name and base_name == "coupleRateSmoke":
                cap(attribute, 30.0)
            elif "emitter" in prim_name and base_name == "velocity":
                attribute.Set(emitter_velocity)
                velocity_attributes.append(attribute_path)
            elif "emitter" in prim_name and base_name == "velocityIsWorldSpace":
                attribute.Set(True)
                world_space_attributes.append(attribute_path)
            elif "raymarch" in prim_name and base_name == "attenuation":
                cap(attribute, 1.0)
            elif "advection" in prim_name and base_name == "smokePerBurn":
                cap(attribute, 0.35)
            elif "advection" in prim_name and base_name == "divergencePerBurn":
                cap(attribute, 0.4)
    if not radius_attributes:
        raise RuntimeError("bundled Flow preset exposes no calibratable emitter radius")
    if not velocity_attributes:
        raise RuntimeError("bundled Flow preset exposes no wind-driven emitter velocity")
    return {
        "active_flame_area_m2": round(active_flame_area_m2, 4),
        "emitter_radius_m": round(radius_m, 4),
        "wind_heading_deg": round(wind_heading_deg % 360.0, 4),
        "wind_speed_mps": round(wind_speed_mps, 4),
        "emitter_velocity_mps": [round(float(value), 4) for value in emitter_velocity],
        "radius_attributes": radius_attributes,
        "world_space_attributes": world_space_attributes,
        "density_cell_attributes": density_cell_attributes,
        "velocity_attributes": velocity_attributes,
        "bounded_attributes": bounded_attributes,
    }


def calibrate_flow_for_wildfire(stage: Any, contract: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the NVIDIA Flow payload accepts metric calibration."""
    progression = contract["composition"]["flow_states"][0]["progression"]
    return _set_flow_emitter_radius(
        stage,
        progression=progression,
    )


def _composition_provenance(contract: dict[str, Any]) -> dict[str, Any]:
    synthetic_scene = contract["capture"]["source"] == "new_synthetic_french_reference"
    provenance = {
        "kind": (
            "review_gated_omniverse_usd_composition"
            if synthetic_scene
            else "nurec_real_world_composition"
        ),
        "pipeline": contract["pipeline"],
        "render_profile": contract["render_profile"],
        "site_id": contract["site_id"],
        "scene_format": contract["reconstruction"]["format"],
        "scene_sha256": contract["reconstruction"]["asset_sha256"],
        "capture_manifest_sha256": contract["capture"]["capture_manifest_sha256"],
        "diversity_capacity_per_category": contract["composition"]["diversity"]["capacity_per_category"],
    }
    metrics = contract["reconstruction"]["metrics"]
    if synthetic_scene:
        provenance["quality_review"] = metrics["quality_review"]
    else:
        provenance["psnr"] = metrics["psnr"]
        provenance["ssim"] = metrics["ssim"]
    return provenance


def compose_omniverse_stage(stage: Any, contract: dict[str, Any]) -> dict[str, Any]:
    """Reference NuRec and Flow into a writable USD root stage."""
    from pxr import Gf, UsdGeom

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene = stage.DefinePrim("/World/RealWorldScene", "Xform")
    if not scene.GetReferences().AddReference(str(contract["reconstruction"]["asset"])):
        raise RuntimeError("NuRec scene reference could not be authored")
    flow = stage.DefinePrim("/World/FireAndSmoke", "Xform")
    if not flow.GetPayloads().AddPayload(str(contract["composition"]["flow_asset"])):
        raise RuntimeError("Flow payload could not be authored")
    active_fire = contract["composition"]["flow_states"][0]["anchors_world_m"]["active_fire_point"]
    flow_xform = UsdGeom.XformCommonAPI(flow)
    flow_xform.SetTranslate(Gf.Vec3d(*active_fire))
    # The bundled preset is calibrated after payload loading by authoring its
    # emitter radius directly in metres. Keep this parent transform unscaled:
    # Flow emitters may explicitly ignore parent scale when radiusIsWorldSpace
    # is enabled.
    flow_xform.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))
    UsdGeom.Scope.Define(stage, "/World/Cameras")

    actors_root = UsdGeom.Scope.Define(stage, "/World/Actors")
    del actors_root
    for index, actor in enumerate(contract["composition"]["actors"]):
        prim = stage.DefinePrim(f"/World/Actors/Actor{index:02d}", "Xform")
        if not prim.GetReferences().AddReference(str(actor["asset"])):
            raise RuntimeError(f"actor reference could not be authored: {actor['class_id']}")
        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(Gf.Vec3d(*actor["translation_world_m"]))
        xform.SetRotate(Gf.Vec3f(*actor["rotation_xyz_deg"]))
        xform.SetScale(Gf.Vec3f(*actor["scale_xyz"]))
        prim.SetCustomDataByKey("fireviewer_class_id", actor["class_id"])
        UsdGeom.Imageable(prim).MakeInvisible()

    return _composition_provenance(contract)
