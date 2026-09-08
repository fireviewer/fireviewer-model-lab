"""Benchmark pinned RoMaV2 matches through PyCOLMAP LO-RANSAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fireviewer_model_lab.training.romav2_offline import load_romav2_offline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relpath: str) -> Path:
    path = (root / relpath).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"manifest path escapes data root: {relpath}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _pair_assets(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    pair_id = str(row.get("pair_id") or row.get("sample_id"))
    if "source_view" in row:
        return (
            pair_id,
            str(row["source_view"]["image_relpath"]),
            str(row["map_view"]["image_relpath"]),
            str(row["source_transient_mask_relpath"]),
            str(row["map_transient_mask_relpath"]),
        )
    return (
        pair_id,
        str(row["source_image_relpath"]),
        str(row["map_image_relpath"]),
        str(row["source_transient_mask_relpath"]),
        str(row["map_transient_mask_relpath"]),
    )


def _static_match_mask(
    points_a: np.ndarray,
    points_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> np.ndarray:
    xa = np.clip(np.rint(points_a[:, 0]).astype(int), 0, mask_a.shape[1] - 1)
    ya = np.clip(np.rint(points_a[:, 1]).astype(int), 0, mask_a.shape[0] - 1)
    xb = np.clip(np.rint(points_b[:, 0]).astype(int), 0, mask_b.shape[1] - 1)
    yb = np.clip(np.rint(points_b[:, 1]).astype(int), 0, mask_b.shape[0] - 1)
    return ~(mask_a[ya, xa] | mask_b[yb, xb])


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import pycolmap
    import torch

    data_root = args.data_root.resolve()
    manifest = args.manifest.resolve()
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.one_per_split_group:
        selected: list[dict[str, Any]] = []
        seen_groups: set[str] = set()
        for row in rows:
            group = str(row.get("split_group"))
            if group not in seen_groups:
                seen_groups.add(group)
                selected.append(row)
        rows = selected
    if args.max_pairs is not None:
        rows = rows[: args.max_pairs]
    if not rows:
        raise ValueError("RoMaV2 benchmark manifest is empty")

    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    model, provenance = load_romav2_offline(
        romav2_source=args.romav2_source,
        dinov3_source=args.dinov3_source,
        weights=args.weights,
        setting=args.setting,
    )
    options = pycolmap.RANSACOptions()
    options.max_error = args.ransac_max_error
    options.confidence = args.ransac_confidence
    options.min_num_trials = args.ransac_min_trials
    options.max_num_trials = args.ransac_max_trials

    results: list[dict[str, Any]] = []
    for row in rows:
        pair_id, image_a_rel, image_b_rel, mask_a_rel, mask_b_rel = _pair_assets(row)
        image_a = _safe_path(data_root, image_a_rel)
        image_b = _safe_path(data_root, image_b_rel)
        mask_a = np.asarray(Image.open(_safe_path(data_root, mask_a_rel)).convert("L")) > 0
        mask_b = np.asarray(Image.open(_safe_path(data_root, mask_b_rel)).convert("L")) > 0
        with Image.open(image_a) as opened_a:
            width_a, height_a = opened_a.size
        with Image.open(image_b) as opened_b:
            width_b, height_b = opened_b.size
        started = time.perf_counter()
        predictions = model.match(image_a, image_b)
        matches, _overlap, _precision_ab, _precision_ba = model.sample(predictions, args.samples)
        points_a, points_b = model.to_pixel_coordinates(
            matches, height_a, width_a, height_b, width_b
        )
        points_a = points_a.detach().cpu().numpy().astype(np.float64)
        points_b = points_b.detach().cpu().numpy().astype(np.float64)
        static = _static_match_mask(points_a, points_b, mask_a, mask_b)
        points_a = points_a[static]
        points_b = points_b[static]
        estimate = (
            pycolmap.estimate_fundamental_matrix(points_a, points_b, options)
            if len(points_a) >= 8
            else None
        )
        inliers = int(np.asarray(estimate["inlier_mask"]).sum()) if estimate else 0
        static_count = int(static.sum())
        results.append(
            {
                "pair_id": pair_id,
                "sampled_matches": len(matches),
                "static_matches": static_count,
                "transient_matches_rejected": int(len(matches) - static_count),
                "lo_ransac_inliers": inliers,
                "lo_ransac_inlier_ratio": inliers / max(1, static_count),
                "elapsed_seconds": time.perf_counter() - started,
                "passed": inliers >= args.minimum_inliers
                and inliers / max(1, static_count) >= args.minimum_inlier_ratio,
            }
        )

    report = {
        "schema_version": 1,
        "benchmark": "romav2-v2.0.1-pycolmap-4.1.1-lo-ransac",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "pair_count": len(results),
        "passed_pairs": sum(int(row["passed"]) for row in results),
        "all_pairs_passed": all(row["passed"] for row in results),
        "settings": {
            "romav2": args.setting,
            "samples": args.samples,
            "ransac_max_error": args.ransac_max_error,
            "minimum_inliers": args.minimum_inliers,
            "minimum_inlier_ratio": args.minimum_inlier_ratio,
            "one_per_split_group": args.one_per_split_group,
        },
        "assets": provenance,
        "gpu_peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "pairs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--romav2-source", type=Path, required=True)
    parser.add_argument("--dinov3-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", choices=("turbo", "fast", "base", "precise"), default="turbo")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--one-per-split-group", action="store_true")
    parser.add_argument("--ransac-max-error", type=float, default=2.0)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-min-trials", type=int, default=1000)
    parser.add_argument("--ransac-max-trials", type=int, default=10000)
    parser.add_argument("--minimum-inliers", type=int, default=50)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.10)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(benchmark(parse_args()), ensure_ascii=False, indent=2))
