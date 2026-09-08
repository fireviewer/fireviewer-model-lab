"""Build leakage-safe cross-view pairs from the shared Gaussians on Fire export."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

GAUSSIANS_REVISION = "001ad0c2a06c24eff21bec326b4e0b9600a3fb04"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _project(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera = (rotation @ points.T).T + translation[None]
    depth = camera[:, 2]
    pixel = np.empty((len(points), 2), dtype=np.float64)
    pixel[:, 0] = k[0, 0] * camera[:, 0] / np.maximum(depth, 1e-9) + k[0, 2]
    pixel[:, 1] = k[1, 1] * camera[:, 1] / np.maximum(depth, 1e-9) + k[1, 2]
    return pixel, depth


def select_shared_landmark(
    *,
    points: np.ndarray,
    source_rotation: np.ndarray,
    source_translation: np.ndarray,
    source_k: np.ndarray,
    source_size: tuple[int, int],
    map_rotation: np.ndarray,
    map_translation: np.ndarray,
    map_k: np.ndarray,
    map_size: tuple[int, int],
) -> tuple[int, tuple[float, float]]:
    """Choose the reconstructed point nearest the source optical axis and visible in both views."""

    source_pixel, source_depth = _project(points, source_rotation, source_translation, source_k)
    map_pixel, map_depth = _project(points, map_rotation, map_translation, map_k)
    sw, sh = source_size
    mw, mh = map_size
    visible = (
        (source_depth > 0)
        & (map_depth > 0)
        & (source_pixel[:, 0] >= 0)
        & (source_pixel[:, 0] < sw)
        & (source_pixel[:, 1] >= 0)
        & (source_pixel[:, 1] < sh)
        & (map_pixel[:, 0] >= 0)
        & (map_pixel[:, 0] < mw)
        & (map_pixel[:, 1] >= 0)
        & (map_pixel[:, 1] < mh)
    )
    indices = np.flatnonzero(visible)
    if not len(indices):
        raise ValueError("camera pair has no reconstructed landmark visible in both views")
    source_normalized = np.column_stack(
        (
            source_pixel[indices, 0] / max(1, sw - 1),
            source_pixel[indices, 1] / max(1, sh - 1),
        )
    )
    map_normalized = np.column_stack(
        (
            map_pixel[indices, 0] / max(1, mw - 1),
            map_pixel[indices, 1] / max(1, mh - 1),
        )
    )
    score = ((source_normalized - 0.5) ** 2).sum(axis=1) + 0.1 * ((map_normalized - 0.5) ** 2).sum(
        axis=1
    )
    selected = int(indices[int(np.argmin(score))])
    target = map_pixel[selected]
    return selected, (float(target[0] / max(1, mw - 1)), float(target[1] / max(1, mh - 1)))


def _split(scene: int) -> str:
    if scene <= 12:
        return "train"
    if scene <= 14:
        return "validation"
    return "test"


def build_gaussians_cross_view_manifest(
    *,
    campaign_root: Path,
    shared_root: Path,
    source_root: Path,
    output_root: Path,
    frame_stride: int = 5,
) -> dict[str, Any]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    campaign_root = campaign_root.resolve()
    shared_root = shared_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    rows: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}
    for scene_root in sorted(path for path in shared_root.iterdir() if path.is_dir()):
        scene = int(scene_root.name)
        transforms = json.loads((scene_root / "transforms.json").read_text(encoding="utf-8"))
        frames_by_camera: dict[str, list[dict[str, Any]]] = {}
        for frame in transforms["frames"]:
            frames_by_camera.setdefault(str(frame["camera"]), []).append(frame)
        for frames in frames_by_camera.values():
            frames.sort(key=lambda row: str(row["file_path"]))
        cameras = sorted(frames_by_camera)
        meta = np.load(source_root / "scenes" / scene_root.name / "meta.npz")
        points = np.asarray(meta["points3D"], dtype=np.float64)
        geometry: dict[tuple[str, str], dict[str, Any]] = {}
        for source_name, map_name in permutations(cameras, 2):
            source_index, map_index = cameras.index(source_name), cameras.index(map_name)
            source_frame = frames_by_camera[source_name][0]
            map_frame = frames_by_camera[map_name][0]
            source_k = np.array(
                [
                    [source_frame["fl_x"], 0, source_frame["cx"]],
                    [0, source_frame["fl_y"], source_frame["cy"]],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            map_k = np.array(
                [
                    [map_frame["fl_x"], 0, map_frame["cx"]],
                    [0, map_frame["fl_y"], map_frame["cy"]],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            landmark, target = select_shared_landmark(
                points=points,
                source_rotation=np.asarray(meta["R"][source_index], dtype=np.float64),
                source_translation=np.asarray(meta["t"][source_index], dtype=np.float64),
                source_k=source_k,
                source_size=(int(source_frame["w"]), int(source_frame["h"])),
                map_rotation=np.asarray(meta["R"][map_index], dtype=np.float64),
                map_translation=np.asarray(meta["t"][map_index], dtype=np.float64),
                map_k=map_k,
                map_size=(int(map_frame["w"]), int(map_frame["h"])),
            )
            source_center = -np.asarray(meta["R"][source_index]).T @ np.asarray(
                meta["t"][source_index]
            )
            map_center = -np.asarray(meta["R"][map_index]).T @ np.asarray(meta["t"][map_index])
            source_axis = np.asarray(meta["R"][source_index]).T @ np.array([0.0, 0.0, 1.0])
            map_axis = np.asarray(meta["R"][map_index]).T @ np.array([0.0, 0.0, 1.0])
            cosine = float(np.clip(np.dot(source_axis, map_axis), -1.0, 1.0))
            geometry[(source_name, map_name)] = {
                "landmark_index": landmark,
                "landmark_xyz": points[landmark].tolist(),
                "target": target,
                "baseline_meters": float(np.linalg.norm(source_center - map_center)),
                "view_angle_degrees": float(math.degrees(math.acos(cosine))),
            }
        frame_count = min(len(value) for value in frames_by_camera.values())
        for position in range(0, frame_count, frame_stride):
            for source_name, map_name in permutations(cameras, 2):
                source_frame = frames_by_camera[source_name][position]
                map_frame = frames_by_camera[map_name][position]
                source_path = scene_root / str(source_frame["file_path"])
                map_path = scene_root / str(map_frame["file_path"])
                for path in (source_path, map_path):
                    if path not in hash_cache:
                        hash_cache[path] = _sha256(path)
                pair_geometry = geometry[(source_name, map_name)]
                rows.append(
                    {
                        "sample_id": (
                            f"gaussians:{scene_root.name}:{position:05d}:"
                            f"{source_name}-to-{map_name}"
                        ),
                        "family": "cross_view_registration",
                        "split": _split(scene),
                        "split_group": f"gaussians-scene:{scene_root.name}",
                        "source_id": "jna-358/fire_actioncam",
                        "source_revision": GAUSSIANS_REVISION,
                        "license": "CC-BY-4.0",
                        "consent_basis": "public_research_dataset_cc_by_4_0",
                        "operational_incident": False,
                        "dynamic_fire_scene": True,
                        "transient_mask_status": "required_before_full_train",
                        "source_view": {
                            "image_relpath": _relative(campaign_root, source_path),
                            "sha256": hash_cache[source_path],
                            "camera": source_name,
                            "time_us": float(source_frame["time_us"]),
                        },
                        "map_view": {
                            "image_relpath": _relative(campaign_root, map_path),
                            "sha256": hash_cache[map_path],
                            "camera": map_name,
                            "time_us": float(map_frame["time_us"]),
                            "optical_axis_ground_pixel_normalized": list(pair_geometry["target"]),
                        },
                        "geometry": {
                            "shared_landmark_index": pair_geometry["landmark_index"],
                            "shared_landmark_xyz": pair_geometry["landmark_xyz"],
                            "baseline_meters": pair_geometry["baseline_meters"],
                            "view_angle_degrees": pair_geometry["view_angle_degrees"],
                            "pose_source": "meta.npz_R_t_points3D",
                        },
                    }
                )
    rows.sort(key=lambda row: str(row["sample_id"]))
    manifest = output_root / "corpus" / "cross-view-registration-v0.1.0" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    split_counts = Counter(str(row["split"]) for row in rows)
    report = {
        "schema_version": 1,
        "dataset_family": "fireviewer-cross-view-v2",
        "manifest": _relative(output_root, manifest),
        "manifest_sha256": _sha256(manifest),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_images": len(hash_cache),
        "source_revision": GAUSSIANS_REVISION,
        "frame_stride": frame_stride,
        "consumers": ["dinov3_cross_view", "roma_pycolmap", "moge2"],
        "training_ready": False,
        "training_blockers": [
            "wildfire3data_source_unavailable",
            "camp_swift_pairs_not_ingested",
            "transient_fire_smoke_masks_not_generated",
        ],
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
