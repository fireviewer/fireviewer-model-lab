"""Build a sparse-SfM depth/FOV benchmark for MoGe-2 from Gaussians on Fire."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fireviewer_model_lab.training.gaussians_cross_view import GAUSSIANS_REVISION, _project, _split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sparse_depth_from_points(
    *,
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    pixels, depths = _project(points, rotation, translation, intrinsics)
    xs = np.rint(pixels[:, 0]).astype(np.int64)
    ys = np.rint(pixels[:, 1]).astype(np.int64)
    visible = (depths > 0) & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    flat = np.full(height * width, np.inf, dtype=np.float32)
    indices = ys[visible] * width + xs[visible]
    np.minimum.at(flat, indices, depths[visible].astype(np.float32))
    valid = np.isfinite(flat)
    depth = flat.reshape(height, width)
    depth[~valid.reshape(height, width)] = 0.0
    return depth, valid.reshape(height, width).astype(np.uint8) * 255


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def _atomic_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.png")
    Image.fromarray(array).save(temporary)
    os.replace(temporary, path)


def build_gaussians_moge_manifest(
    *,
    campaign_root: Path,
    shared_root: Path,
    source_root: Path,
    output_root: Path,
    frame_stride: int = 30,
) -> dict[str, Any]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    campaign_root = campaign_root.resolve()
    shared_root = shared_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    rows: list[dict[str, Any]] = []
    image_hashes: dict[Path, str] = {}
    depth_map_count = 0
    for scene_root in sorted(path for path in shared_root.iterdir() if path.is_dir()):
        scene = int(scene_root.name)
        transforms = json.loads((scene_root / "transforms.json").read_text(encoding="utf-8"))
        frames_by_camera: dict[str, list[dict[str, Any]]] = {}
        for frame in transforms["frames"]:
            frames_by_camera.setdefault(str(frame["camera"]), []).append(frame)
        for frames in frames_by_camera.values():
            frames.sort(key=lambda frame: str(frame["file_path"]))
        cameras = sorted(frames_by_camera)
        meta = np.load(source_root / "scenes" / scene_root.name / "meta.npz")
        points = np.asarray(meta["points3D"], dtype=np.float64)
        for camera_index, camera in enumerate(cameras):
            frames = frames_by_camera[camera]
            first = frames[0]
            width, height = int(first["w"]), int(first["h"])
            intrinsics = np.array(
                [
                    [first["fl_x"], 0.0, first["cx"]],
                    [0.0, first["fl_y"], first["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            depth, valid = sparse_depth_from_points(
                points=points,
                rotation=np.asarray(meta["R"][camera_index], dtype=np.float64),
                translation=np.asarray(meta["t"][camera_index], dtype=np.float64),
                intrinsics=intrinsics,
                width=width,
                height=height,
            )
            if int((valid > 0).sum()) < 20:
                raise ValueError(
                    f"insufficient sparse depth support: scene={scene_root.name} camera={camera}"
                )
            depth_path = output_root / "depth" / scene_root.name / f"{camera}.npy"
            valid_path = output_root / "depth-valid" / scene_root.name / f"{camera}.png"
            _atomic_npy(depth_path, depth)
            _atomic_png(valid_path, valid)
            depth_map_count += 1
            depth_sha = _sha256(depth_path)
            valid_sha = _sha256(valid_path)
            fov = math.degrees(2.0 * math.atan(width / (2.0 * float(first["fl_x"]))))
            for position in range(0, len(frames), frame_stride):
                frame = frames[position]
                image_path = scene_root / str(frame["file_path"])
                if image_path not in image_hashes:
                    image_hashes[image_path] = _sha256(image_path)
                rows.append(
                    {
                        "sample_id": f"gaussians-moge:{scene_root.name}:{camera}:{position:05d}",
                        "source_id": "jna-358/fire_actioncam",
                        "source_revision": GAUSSIANS_REVISION,
                        "split": _split(scene),
                        "split_group": f"gaussians-scene:{scene_root.name}",
                        "image_relpath": _relative(campaign_root, image_path),
                        "image_sha256": image_hashes[image_path],
                        "depth_relpath": _relative(campaign_root, depth_path),
                        "depth_sha256": depth_sha,
                        "depth_valid_relpath": _relative(campaign_root, valid_path),
                        "depth_valid_sha256": valid_sha,
                        "depth_kind": "sparse_sfm_points3D",
                        "depth_valid_pixels": int((valid > 0).sum()),
                        "depth_bounds_meters": {
                            "near": float(meta["near"][camera_index]),
                            "far": float(meta["far"][camera_index]),
                        },
                        "fov_ground_truth_deg": fov,
                        "intrinsics_ground_truth": {
                            "fx": float(first["fl_x"]),
                            "fy": float(first["fl_y"]),
                            "cx": float(first["cx"]),
                            "cy": float(first["cy"]),
                            "width": width,
                            "height": height,
                        },
                        "camera_pose_world_to_camera": {
                            "R": np.asarray(meta["R"][camera_index]).tolist(),
                            "t": np.asarray(meta["t"][camera_index]).tolist(),
                        },
                        "license": "CC-BY-4.0",
                        "operational_incident": False,
                    }
                )
    rows.sort(key=lambda row: str(row["sample_id"]))
    manifest = output_root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".jsonl.partial")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, manifest)
    split_counts = Counter(str(row["split"]) for row in rows)
    report = {
        "schema_version": 1,
        "dataset_family": "gaussians-on-fire-moge2-benchmark-v1",
        "manifest": _relative(campaign_root, manifest),
        "manifest_sha256": _sha256(manifest),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "depth_maps": depth_map_count,
        "unique_images": len(image_hashes),
        "depth_kind": "sparse_sfm_points3D",
        "frame_stride": frame_stride,
        "benchmark_ready": True,
        "training_ready": False,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
