"""Decode and inspect every image in the isolated FireViewer pointing campaign."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

FORBIDDEN_URI_MARKERS = (
    "fire-smoke-detection-corpus-v1",
    "benchdata",
    "fireviewer_bench",
    "independent-benchmark",
)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    value = uri[5:]
    bucket, separator, key = value.partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"incomplete S3 URI: {uri}")
    if any(marker in uri.lower() for marker in FORBIDDEN_URI_MARKERS):
        raise ValueError(f"forbidden non-pointing image URI: {uri}")
    return bucket, key


def difference_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.tobytes())
    bits = 0
    for y_value in range(8):
        offset = y_value * 9
        for x_value in range(8):
            bits = (bits << 1) | int(pixels[offset + x_value] > pixels[offset + x_value + 1])
    return f"{bits:016x}"


def _image_metrics(image: Image.Image) -> dict[str, float]:
    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    stat = ImageStat.Stat(gray)
    brightness = float(stat.mean[0] / 255.0)
    contrast = float(stat.stddev[0] / 255.0)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_energy = float(edge_stat.mean[0] / 255.0)
    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_energy": edge_energy,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inspect_row(row: dict[str, Any], *, s3_client: Any) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    uri = str(row["image_s3_uri"])
    bucket, key = parse_s3_uri(uri)
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "image_s3_uri": uri,
        "split": str(row.get("split") or ""),
        "source_family": str(row.get("source_family") or "unknown"),
        "status": "ok",
        "errors": [],
    }
    try:
        payload = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        result["bytes"] = len(payload)
        expected_sha = str(row.get("source_image_sha256") or row.get("source_sha256") or "").lower()
        if expected_sha:
            observed_sha = hashlib.sha256(payload).hexdigest()
            result["sha256"] = observed_sha
            if observed_sha != expected_sha:
                result["errors"].append("sha256_mismatch")
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            image = opened.convert("RGB")
        result["format"] = str(opened.format or "")
        result["width"] = image.width
        result["height"] = image.height
        if image.width != int(row.get("width") or 0) or image.height != int(row.get("height") or 0):
            result["errors"].append("dimension_mismatch")
        result["dhash"] = difference_hash(image)
        result.update(_image_metrics(image))
    except Exception as exc:
        result["errors"].append(f"decode_or_fetch_error:{type(exc).__name__}:{exc}")
    if result["errors"]:
        result["status"] = "error"
    return result


def cross_split_visual_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        signature = str(row.get("dhash") or "")
        if signature:
            grouped[signature].append(row)
    results: list[dict[str, Any]] = []
    for signature, members in grouped.items():
        splits = sorted({str(member["split"]) for member in members})
        if len(splits) <= 1:
            continue
        results.append(
            {
                "dhash": signature,
                "splits": splits,
                "samples": sorted(str(member["sample_id"]) for member in members),
                "automatic_rejection": True,
                "exclusion_reason": "cross_split_visual_signature",
            }
        )
    return sorted(results, key=lambda row: (row["dhash"], row["samples"]))


def run_audit(*, input_dir: Path, output_dir: Path, workers: int) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    seed_rows = _read_jsonl(input_dir / "seed_points_pending_strict_validation.jsonl")
    reservoir_rows = _read_jsonl(input_dir / "box_reservoir_inventory.jsonl")
    all_rows = seed_rows + reservoir_rows
    for row in all_rows:
        parse_s3_uri(str(row["image_s3_uri"]))

    import boto3
    from botocore.config import Config

    s3_client = boto3.client(
        "s3",
        config=Config(max_pool_connections=max(16, workers), retries={"max_attempts": 6}),
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspections = list(
            executor.map(lambda row: inspect_row(row, s3_client=s3_client), all_rows)
        )
    inspections.sort(key=lambda row: row["sample_id"])
    visual_groups = cross_split_visual_groups(inspections)
    rejected_visual_samples = {
        sample_id for group in visual_groups for sample_id in group["samples"]
    }
    source_by_sample = {str(row["sample_id"]): row for row in all_rows}
    dispositions: list[dict[str, Any]] = []
    for inspection in inspections:
        sample_id = str(inspection["sample_id"])
        source = source_by_sample[sample_id]
        reasons = list(inspection["errors"])
        if sample_id in rejected_visual_samples:
            reasons.append("cross_split_visual_signature")
        reasons.extend(str(value) for value in source.get("exclusion_reasons", []))
        reasons = sorted(set(reasons))
        dispositions.append(
            {
                "sample_id": sample_id,
                "split": inspection["split"],
                "source_family": inspection["source_family"],
                "strict_keep": not reasons,
                "strict_validation_status": (
                    "strict_automated_validated" if not reasons else "excluded_unvalidated"
                ),
                "validation_profile": "fireviewer_pointing_strict_automated_v1",
                "exclusion_reasons": reasons,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "pixel_inventory.jsonl", inspections)
    _write_jsonl(output_dir / "cross_split_visual_exclusions.jsonl", visual_groups)
    _write_jsonl(output_dir / "strict_automatic_dispositions.jsonl", dispositions)
    error_codes = Counter(error.split(":", 1)[0] for row in inspections for error in row["errors"])
    brightness_values = [float(row["brightness"]) for row in inspections if "brightness" in row]
    report = {
        "schema_version": 1,
        "campaign_id": "fireviewer-pointing-corpus-v2",
        "images_requested": len(all_rows),
        "images_ok": sum(row["status"] == "ok" for row in inspections),
        "images_failed": sum(row["status"] != "ok" for row in inspections),
        "error_codes": dict(sorted(error_codes.items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in inspections).items())),
        "source_family_counts": dict(
            sorted(Counter(row["source_family"] for row in inspections).items())
        ),
        "cross_split_visual_signature_groups": len(visual_groups),
        "cross_split_visual_signatures_are_automatically_excluded": True,
        "strict_automated_validated_rows": sum(row["strict_keep"] for row in dispositions),
        "excluded_unvalidated_rows": sum(not row["strict_keep"] for row in dispositions),
        "reviews_admitted": False,
        "brightness": {
            "min": min(brightness_values) if brightness_values else math.nan,
            "max": max(brightness_values) if brightness_values else math.nan,
            "mean": (
                sum(brightness_values) / len(brightness_values) if brightness_values else math.nan
            ),
        },
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "box_derived_points": 0,
        "publication_allowed": False,
        "next_action": "add_new_point_label_sources_and_repeat_until_quality_gates_pass",
    }
    _write_json(output_dir / "pixel_audit_summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_audit(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
