"""Provision and validate the pinned AerialExtreMatch-RoMa registration path.

There is intentionally no training command.  The official checkpoint must first pass the held-out
cross-view benchmark and the future double-validated critical lot.  Qwen training remains a
separate, locked fire-pointing concern.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fireviewer_contracts.contracts import CameraMetadataV2
from fireviewer_geolocation.roma_registration import (
    ROMA_ASSETS,
    ROMA_LICENSE,
    ROMA_SOURCE_REVISION,
    RomaAssetError,
    load_roma_model,
    match_pair,
    provision_roma_assets,
    verify_roma_assets,
)
from fireviewer_geolocation.spatial_geometry import (
    CameraPoseSolution,
    SpatialGeometryError,
    crop_georeferenced_map,
    cross_view_search_radii,
    select_consistent_cross_view_pose,
    solve_pnp_pose,
)

CORPUS_RELATIVE = Path("corpus/cross-view-registration-v0.1.0")
MANIFEST_NAME = "manifest.jsonl"
REPORT_NAME = "build-report.json"
OUTPUT_RELATIVE = Path("evaluation/aerialextrematch-roma-v1")
DENIED_TOKENS = (
    "fireviewer-operational-reference",
    "operational-reference-a",
    "operational-reference-a",
)
VRAM_LIMIT_BYTES = 14 * 1024**3
RAM_LIMIT_BYTES = 10 * 1024**3
EXPECTED_SOURCE_IDS = {
    "aerialextrematch_localization",
    "odm_sance_mountain",
    "odm_seneca_rural",
}
EXPECTED_AEM_DSM_SHA256 = "319aa4bac96171693763e6b45d1074812b0020bc96c13eed9a0b99d653e5e74a"


class RegistrationSetupError(RuntimeError):
    """Raised when the registration benchmark cannot safely run."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrationSetupError(f"invalid JSON report: {path}") from exc
    if not isinstance(value, dict):
        raise RegistrationSetupError(f"JSON report is not an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        source = path.open(encoding="utf-8")
    except OSError as exc:
        raise RegistrationSetupError(f"missing registration manifest: {path}") from exc
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegistrationSetupError(
                    f"invalid registration row at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise RegistrationSetupError(
                    f"registration row is not an object at {path}:{line_number}"
                )
            yield value


def _resolve_media(dataset_root: Path, relpath: str, expected_sha256: str) -> Path:
    root = dataset_root.resolve()
    candidate = (root / Path(relpath)).resolve()
    if candidate != root and root not in candidate.parents:
        raise RegistrationSetupError(f"media path escapes dataset root: {relpath}")
    if any(token in str(candidate).lower() for token in DENIED_TOKENS):
        raise RegistrationSetupError(f"operational incident media denied: {candidate}")
    if not candidate.is_file():
        raise RegistrationSetupError(f"missing registration media: {candidate}")
    if _sha256_file(candidate) != expected_sha256:
        raise RegistrationSetupError(f"registration media SHA-256 mismatch: {candidate}")
    return candidate


def preflight(
    dataset_root: Path,
    *,
    roma_root: Path | None = None,
    require_assets: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    corpus_root = dataset_root / CORPUS_RELATIVE
    report = _read_json(corpus_root / REPORT_NAME)
    manifest_path = corpus_root / MANIFEST_NAME
    expected_manifest_sha256 = str(report.get("manifest_sha256", ""))
    if len(expected_manifest_sha256) != 64:
        raise RegistrationSetupError("registration report has no pinned manifest SHA-256")
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise RegistrationSetupError("registration manifest SHA-256 differs from build report")
    gates = report.get("gates", {})
    if not isinstance(gates, dict) or gates.get("training_ready") is not True:
        raise RegistrationSetupError("cross-view bootstrap training gate is false")
    if gates.get("deployment_ready") is True:
        raise RegistrationSetupError("unexpected deployment-ready claim before critical validation")
    rows = list(_iter_jsonl(manifest_path))
    if len(rows) != report.get("rows"):
        raise RegistrationSetupError("registration row count differs from build report")
    source_ids = {str(row.get("source_id")) for row in rows}
    if not EXPECTED_SOURCE_IDS.issubset(source_ids):
        raise RegistrationSetupError("rural and mountain cross-view domains are incomplete")
    split_groups: dict[str, set[str]] = {}
    for row in rows:
        if row.get("operational_incident") is not False:
            raise RegistrationSetupError(
                f"operational registration row denied: {row.get('sample_id')}"
            )
        combined = json.dumps(row, ensure_ascii=False).lower()
        if any(token in combined for token in DENIED_TOKENS):
            raise RegistrationSetupError(
                f"operational incident token denied: {row.get('sample_id')}"
            )
        split_groups.setdefault(str(row["split_group"]), set()).add(str(row["split"]))
    if any(len(splits) != 1 for splits in split_groups.values()):
        raise RegistrationSetupError("registration split-group leak")
    assets_verified = False
    if require_assets:
        if roma_root is None:
            raise RegistrationSetupError("--roma-root is required with --require-assets")
        try:
            verify_roma_assets(roma_root)
        except RomaAssetError as exc:
            raise RegistrationSetupError(str(exc)) from exc
        assets_verified = True
    return {
        "assets_verified": assets_verified,
        "corpus_manifest_sha256": expected_manifest_sha256,
        "critical_lot_included": False,
        "deployment_ready": False,
        "model": "AerialExtreMatch-RoMa",
        "model_license": ROMA_LICENSE,
        "rows": len(rows),
        "source_ids": sorted(source_ids),
        "source_revision": ROMA_SOURCE_REVISION,
        "split_counts": report.get("split_counts", {}),
        "split_group_leaks": 0,
        "training_command_available": False,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def provision(dataset_root: Path, roma_root: Path) -> dict[str, Any]:
    preflight(dataset_root)
    try:
        manifest = provision_roma_assets(roma_root)
    except RomaAssetError as exc:
        raise RegistrationSetupError(str(exc)) from exc
    report = {
        **manifest,
        "asset_root": str(roma_root.resolve()),
        "dataset_in_model_volume": False,
    }
    _write_json(dataset_root.resolve() / OUTPUT_RELATIVE / "provision-report.json", report)
    return report


def _select_probe_row(dataset_root: Path) -> dict[str, Any]:
    manifest = dataset_root.resolve() / CORPUS_RELATIVE / MANIFEST_NAME
    candidates = [
        row
        for row in _iter_jsonl(manifest)
        if row.get("split") == "validation"
        and row.get("source_id") == "aerialextrematch_localization"
        and row.get("validation_status") == "source_pose_provided"
    ]
    if not candidates:
        raise RegistrationSetupError("no held-out pose-provided AerialExtreMatch row")
    return min(candidates, key=lambda row: str(row["sample_id"]))


class _RasterTerrainSurface:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.crs = str(dataset.crs)

    @property
    def resolution_m(self) -> float:
        return max(abs(float(value)) for value in self.dataset.res)

    def sample_many(self, eastings: Any, northings: Any) -> Any:
        import numpy as np

        east = np.asarray(eastings, dtype=np.float64).reshape(-1)
        north = np.asarray(northings, dtype=np.float64).reshape(-1)
        values = np.asarray(
            [value[0] for value in self.dataset.sample(zip(east, north, strict=True))],
            dtype=np.float64,
        )
        if self.dataset.nodata is not None:
            values[values == self.dataset.nodata] = np.nan
        return values

    def sample(self, east_m: float, north_m: float) -> float | None:
        import numpy as np

        value = float(self.sample_many([east_m], [north_m])[0])
        return value if np.isfinite(value) else None


def _benchmark_camera(row: dict[str, Any]) -> CameraMetadataV2:
    source = row["source_view"]
    intrinsics = source["intrinsics"]
    width = int(source["width"])
    horizontal_fov = math.degrees(2.0 * math.atan(width / (2.0 * float(intrinsics["fx"]))))
    wgs84 = row["ground_truth"]["wgs84_derived"]
    return CameraMetadataV2(
        longitude=float(wgs84["longitude"]),
        latitude=float(wgs84["latitude"]),
        orthometric_height_m=float(row["ground_truth"]["camera_center_xyz"][2]),
        horizontal_accuracy_m=125.0,
        horizontal_fov_deg=horizontal_fov,
        image_width_px=width,
        image_height_px=int(source["height"]),
        pose_origin="USER_DECLARED",
    )


def _map_bounds(dataset: Any) -> tuple[float, float, float, float]:
    bounds = dataset.bounds
    return (
        float(bounds.left),
        float(bounds.bottom),
        float(bounds.right),
        float(bounds.top),
    )


def _quaternion_rotation_matrix(quaternion_wxyz: Any) -> Any:
    """Return the held-out world-to-camera rotation used for benchmark diagnostics only."""

    import numpy as np

    values = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, abs_tol=1e-5):
        raise RegistrationSetupError("held-out pose quaternion is not normalized")
    w, x, y, z = values
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _held_out_match_diagnostics(
    *,
    source_pixels: Any,
    map_pixels: Any,
    certainties: Any,
    map_image_size: tuple[int, int],
    map_bounds: tuple[float, float, float, float],
    terrain: _RasterTerrainSurface,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Measure RoMa matches against truth without exposing truth to pose admission."""

    import numpy as np

    source = np.asarray(source_pixels, dtype=np.float64).reshape(-1, 2)
    mapped = np.asarray(map_pixels, dtype=np.float64).reshape(-1, 2)
    confidence = np.asarray(certainties, dtype=np.float64).reshape(-1)
    if not (len(source) == len(mapped) == len(confidence)):
        raise RegistrationSetupError("held-out diagnostic match arrays differ in length")
    finite = (
        np.isfinite(source).all(axis=1) & np.isfinite(mapped).all(axis=1) & np.isfinite(confidence)
    )
    source, mapped, confidence = source[finite], mapped[finite], confidence[finite]
    width, height = map_image_size
    left, bottom, right, top = map_bounds
    east = left + mapped[:, 0] / float(width) * (right - left)
    north = top - mapped[:, 1] / float(height) * (top - bottom)
    altitude = terrain.sample_many(east, north)
    valid_terrain = np.isfinite(altitude)
    source, east, north, altitude, confidence = (
        source[valid_terrain],
        east[valid_terrain],
        north[valid_terrain],
        altitude[valid_terrain],
        confidence[valid_terrain],
    )
    world = np.column_stack((east, north, altitude))
    truth = row["ground_truth"]
    rotation = _quaternion_rotation_matrix(truth["quaternion_wxyz_world_to_camera"])
    translation = np.asarray(truth["translation_xyz_world_to_camera"], dtype=np.float64).reshape(3)
    camera_points = (rotation @ world.T).T + translation
    in_front = camera_points[:, 2] > 1e-9
    source, camera_points, confidence = (
        source[in_front],
        camera_points[in_front],
        confidence[in_front],
    )
    intrinsics = row["source_view"]["intrinsics"]
    projected = np.column_stack(
        (
            float(intrinsics["fx"]) * camera_points[:, 0] / camera_points[:, 2]
            + float(intrinsics["cx"]),
            float(intrinsics["fy"]) * camera_points[:, 1] / camera_points[:, 2]
            + float(intrinsics["cy"]),
        )
    )
    errors = np.linalg.norm(projected - source, axis=1)
    source_width = int(row["source_view"]["width"])
    source_height = int(row["source_view"]["height"])
    inside = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < source_width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < source_height)
    )

    def summarize(mask: Any) -> dict[str, Any]:
        selected_errors = errors[mask]
        count = len(selected_errors)
        result: dict[str, Any] = {
            "count": count,
            "median_error_px": None,
            "p95_error_px": None,
            "within_px": {},
        }
        if count == 0:
            return result
        result["median_error_px"] = float(np.median(selected_errors))
        result["p95_error_px"] = float(np.quantile(selected_errors, 0.95))
        result["within_px"] = {
            str(limit): {
                "count": int(np.count_nonzero(selected_errors <= limit)),
                "ratio": float(np.mean(selected_errors <= limit)),
            }
            for limit in (4, 8, 16, 32)
        }
        return result

    confidence_threshold = (
        max(0.2, float(np.quantile(confidence, 0.5))) if len(confidence) else math.inf
    )
    return {
        "admission_input": False,
        "finite_match_count": int(np.count_nonzero(finite)),
        "terrain_valid_count": int(np.count_nonzero(valid_terrain)),
        "projected_in_front_count": len(errors),
        "projected_inside_image_count": int(np.count_nonzero(inside)),
        "all_in_front": summarize(np.ones(len(errors), dtype=bool)),
        "inside_source_image": summarize(inside),
        "production_confidence_threshold": confidence_threshold,
        "production_confidence_subset": summarize(confidence >= confidence_threshold),
    }


def probe(
    dataset_root: Path,
    roma_root: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    preflight(dataset_root, roma_root=roma_root, require_assets=True)
    row = _select_probe_row(dataset_root)
    source = row["source_view"]
    map_view = row["map_view"]
    source_path = _resolve_media(
        dataset_root,
        str(source["image_relpath"]),
        str(source["sha256"]),
    )
    dom_path = _resolve_media(
        dataset_root,
        str(map_view["reference_dom_relpath"]),
        str(map_view["reference_dom_sha256"]),
    )
    dsm_path = _resolve_media(
        dataset_root,
        str(Path(str(map_view["reference_dom_relpath"])).with_name("DSM.tif")),
        EXPECTED_AEM_DSM_SHA256,
    )
    try:
        import psutil
        import rasterio
        import torch
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - explicit setup failure
        raise RegistrationSetupError("install RoMa probe dependencies") from exc
    if not torch.cuda.is_available():
        raise RegistrationSetupError("CUDA is required for the real RoMa probe")
    process = psutil.Process()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = None
    map_image = None
    scale_attempts: list[dict[str, Any]] = []
    candidates: list[tuple[float, CameraPoseSolution]] = []
    selected_pose: CameraPoseSolution | None = None
    admission_error: str | None = None
    inference_completed = False
    camera = _benchmark_camera(row)
    expected_camera = tuple(float(value) for value in row["ground_truth"]["camera_center_xyz"])
    prior_limit_m = max(
        100.0,
        min(3_000.0, (camera.horizontal_accuracy_m or 250.0) * 3.0),
    )
    try:
        model = load_roma_model(roma_root, device="cuda")
        with rasterio.open(dom_path) as dom, rasterio.open(dsm_path) as dsm:
            if dom.crs != dsm.crs or dom.transform != dsm.transform:
                raise RegistrationSetupError("DOM and DSM grids differ")
            with Image.open(dom_path) as map_handle:
                map_image = map_handle.convert("RGB")
            map_bounds = _map_bounds(dom)
            terrain = _RasterTerrainSurface(dsm)
            centre_east, centre_north, centre_height = expected_camera
            for requested_radius_m in cross_view_search_radii(camera.horizontal_accuracy_m):
                crop = None
                attempt: dict[str, Any] = {
                    "requested_radius_m": requested_radius_m,
                    "status": "abstained",
                }
                try:
                    crop = crop_georeferenced_map(
                        map_image,
                        map_bounds=map_bounds,
                        centre_east_m=centre_east,
                        centre_north_m=centre_north,
                        radius_m=requested_radius_m,
                    )
                    matches = match_pair(model, source_path, crop.image)
                    inference_completed = True
                    attempt.update(
                        {
                            "effective_radius_m": crop.scale_radius_m,
                            "map_bounds": crop.bounds_m,
                            "match_count": len(matches.source_pixels),
                            "median_certainty": float(
                                __import__("numpy").median(matches.certainties)
                            ),
                        }
                    )
                    attempt["held_out_truth_diagnostics"] = _held_out_match_diagnostics(
                        source_pixels=matches.source_pixels,
                        map_pixels=matches.map_pixels,
                        certainties=matches.certainties,
                        map_image_size=crop.image.size,
                        map_bounds=crop.bounds_m,
                        terrain=terrain,
                        row=row,
                    )
                    pose = solve_pnp_pose(
                        source_pixels=matches.source_pixels,
                        map_pixels=matches.map_pixels,
                        certainties=matches.certainties,
                        map_image_size=crop.image.size,
                        map_bounds=crop.bounds_m,
                        terrain=terrain,
                        camera=camera,
                        prior_camera_center=(centre_east, centre_north, centre_height),
                        maximum_prior_distance_m=prior_limit_m,
                    )
                    candidates.append((crop.scale_radius_m, pose))
                    attempt.update(
                        {
                            "camera_center_xyz": pose.camera_center.tolist(),
                            "pnp_inlier_count": pose.inlier_count,
                            "pnp_inlier_ratio": pose.inlier_ratio,
                            "pnp_median_reprojection_error_px": (pose.median_reprojection_error_px),
                            "pnp_p95_reprojection_error_px": pose.p95_reprojection_error_px,
                            "status": "candidate",
                        }
                    )
                except SpatialGeometryError as exc:
                    attempt["reason"] = exc.code
                finally:
                    if crop is not None:
                        crop.image.close()
                    scale_attempts.append(attempt)
            try:
                selected_pose = select_consistent_cross_view_pose(
                    candidates,
                    horizontal_accuracy_m=camera.horizontal_accuracy_m,
                )
            except SpatialGeometryError as exc:
                admission_error = exc.code
        torch.cuda.synchronize()
        peak_vram = int(torch.cuda.max_memory_allocated())
        process_rss = int(process.memory_info().rss)
    finally:
        if map_image is not None:
            map_image.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    camera_position_error_m = (
        math.dist(selected_pose.camera_center.tolist(), expected_camera)
        if selected_pose is not None
        else None
    )
    resource_gate = peak_vram <= VRAM_LIMIT_BYTES and process_rss <= RAM_LIMIT_BYTES
    inference_gate = inference_completed
    quality_gate = camera_position_error_m is not None and camera_position_error_m <= 75.0
    report = {
        "expected_camera_center_xyz": expected_camera,
        "inference_gate": inference_gate,
        "model": "AerialExtreMatch-RoMa",
        "model_assets": [spec.sha256 for spec in ROMA_ASSETS],
        "observed": {
            "camera_position_error_m": camera_position_error_m,
            "admission_error": admission_error,
            "candidate_count": len(candidates),
            "peak_vram_bytes": peak_vram,
            "predicted_camera_center_xyz": (
                selected_pose.camera_center.tolist() if selected_pose is not None else None
            ),
            "process_rss_bytes": process_rss,
            "scale_attempts": scale_attempts,
        },
        "probe_succeeded": inference_gate and resource_gate,
        "quality_gate": quality_gate,
        "quality_claim": (
            "single_sample_smoke_pass_not_deployment_validation"
            if quality_gate
            else "single_sample_quality_failed_not_deployment_ready"
        ),
        "registration_admitted": quality_gate,
        "resource_gate": resource_gate,
        "sample_id": row["sample_id"],
        "training_started": False,
    }
    _write_json(
        report_path.resolve()
        if report_path is not None
        else dataset_root.resolve() / OUTPUT_RELATIVE / "probe-report.json",
        report,
    )
    if not report["probe_succeeded"]:
        raise RegistrationSetupError("real RoMa probe failed an inference or resource gate")
    return report


def launch_plan(dataset_root: Path, roma_root: Path) -> dict[str, Any]:
    preflight(dataset_root, roma_root=roma_root, require_assets=True)
    report = {
        "decision": "benchmark_official_checkpoint_before_any_fine_tuning",
        "evaluation_sets": {
            "bootstrap_validation": 57,
            "held_out_test": 45,
            "production_critical": "blocked_until_double_validation",
        },
        "fine_tuning_allowed": False,
        "model": "AerialExtreMatch-RoMa",
        "qwen_registration_training": False,
        "resource_limits": {
            "host_ram_bytes": RAM_LIMIT_BYTES,
            "vram_bytes": VRAM_LIMIT_BYTES,
        },
        "training_started": False,
    }
    _write_json(dataset_root.resolve() / OUTPUT_RELATIVE / "launch-plan.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "provision", "probe", "launch-plan"):
        command = commands.add_parser(name)
        command.add_argument("--dataset-root", type=Path, required=True)
        if name != "preflight":
            command.add_argument("--roma-root", type=Path, required=True)
    preflight_parser = commands.choices["preflight"]
    preflight_parser.add_argument("--roma-root", type=Path)
    preflight_parser.add_argument("--require-assets", action="store_true")
    commands.choices["probe"].add_argument("--report-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "preflight":
        report = preflight(
            args.dataset_root,
            roma_root=args.roma_root,
            require_assets=args.require_assets,
        )
    elif args.command == "provision":
        report = provision(args.dataset_root, args.roma_root)
    elif args.command == "probe":
        report = probe(args.dataset_root, args.roma_root, report_path=args.report_path)
    elif args.command == "launch-plan":
        report = launch_plan(args.dataset_root, args.roma_root)
    else:  # pragma: no cover - argparse rejects unknown commands
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
