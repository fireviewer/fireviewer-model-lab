"""Prepare a reproducible MoGe-2 ViT-B depth/FOV benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_preflight(manifest: Path, data_root: Path | None = None) -> dict[str, Any]:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    groups: dict[str, set[str]] = defaultdict(set)
    verified_assets: set[Path] = set()
    verified_depth_maps: set[Path] = set()
    resolved_root = data_root.resolve() if data_root is not None else None
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            errors.append(f"row_not_object:{line_number}")
            continue
        for field in ("sample_id", "split", "split_group", "image_relpath", "image_sha256"):
            if not row.get(field):
                errors.append(f"missing:{field}:{line_number}")
        if row.get("split") not in {"train", "validation", "test"}:
            errors.append(f"unsupported_split:{line_number}")
        if not row.get("depth_relpath"):
            errors.append(f"missing:depth_relpath:{line_number}")
        if not row.get("fov_ground_truth_deg") and not row.get("intrinsics_ground_truth"):
            errors.append(f"missing:fov_or_intrinsics_ground_truth:{line_number}")
        if row.get("depth_kind") != "sparse_sfm_points3D":
            errors.append(f"unsupported_depth_kind:{line_number}")
        if resolved_root is not None:
            for rel_key, sha_key in (
                ("image_relpath", "image_sha256"),
                ("depth_relpath", "depth_sha256"),
                ("depth_valid_relpath", "depth_valid_sha256"),
            ):
                relpath, expected = row.get(rel_key), row.get(sha_key)
                if not relpath or not expected:
                    errors.append(f"missing:{rel_key}_or_{sha_key}:{line_number}")
                    continue
                path = (resolved_root / str(relpath)).resolve()
                if path != resolved_root and resolved_root not in path.parents:
                    errors.append(f"path_escape:{rel_key}:{line_number}")
                elif not path.is_file():
                    errors.append(f"asset_missing:{rel_key}:{line_number}")
                elif path not in verified_assets:
                    if _sha256(path) != str(expected).lower():
                        errors.append(f"asset_hash_mismatch:{rel_key}:{line_number}")
                    else:
                        verified_assets.add(path)
            depth_path = (resolved_root / str(row.get("depth_relpath", ""))).resolve()
            valid_path = (resolved_root / str(row.get("depth_valid_relpath", ""))).resolve()
            if (
                depth_path in verified_assets
                and valid_path in verified_assets
                and depth_path not in verified_depth_maps
            ):
                depth = np.load(depth_path)
                valid = np.asarray(Image.open(valid_path).convert("L")) > 0
                if depth.shape != valid.shape or not bool(valid.any()):
                    errors.append(f"invalid_sparse_depth_shape:{line_number}")
                elif not bool(np.isfinite(depth[valid]).all()) or not bool(
                    (depth[valid] > 0).all()
                ):
                    errors.append(f"invalid_sparse_depth_values:{line_number}")
                else:
                    verified_depth_maps.add(depth_path)
        rows.append(row)
        groups[str(row.get("split_group"))].add(str(row.get("split")))
    errors.extend(
        f"split_group_leakage:{group}" for group, splits in groups.items() if len(splits) > 1
    )
    split_counts = Counter(str(row.get("split")) for row in rows)
    errors.extend(
        f"missing_split:{split}"
        for split in ("train", "validation", "test")
        if not split_counts[split]
    )
    return {
        "schema_version": 1,
        "model_family": "MoGe-2 ViT-B",
        "role": "benchmark_auxiliary",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "training_ready": False,
        "benchmark_ready": not errors,
        "errors": errors,
        "verified_assets": len(verified_assets) if resolved_root is not None else None,
        "verified_depth_maps": (len(verified_depth_maps) if resolved_root is not None else None),
        "required_metrics": ["depth_abs_rel", "depth_rmse", "fov_abs_deg", "inlier_ratio"],
        "promotion": "never_authoritative_for_coordinates",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the MoGe-2 ViT-B benchmark")
    parser.add_argument("command", choices=("preflight", "plan"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_preflight(
        args.manifest.resolve(), args.data_root.resolve() if args.data_root else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.command == "plan":
        (args.output / "benchmark-plan.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_family": "MoGe-2 ViT-B",
                    "fallback": "MoGe-2 ViT-S",
                    "report": "preflight-report.json",
                    "device_policy": "GPU benchmark only",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["benchmark_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
