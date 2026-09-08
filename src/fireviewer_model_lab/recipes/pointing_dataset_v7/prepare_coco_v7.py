#!/usr/bin/env python3
"""Create and verify a zero-copy COCO training view of the reviewed V7 corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V5_CONVERTER_DIR = Path(__file__).resolve().parents[1] / "pointing_dual_v5"
sys.path.insert(0, str(V5_CONVERTER_DIR))
from prepare_coco_v5 import (  # noqa: E402
    CATEGORIES,
    SPLITS,
    atomic_json,
    convert_split,
    sha256_file,
)


REPORT_SCHEMA = "fireviewer.pointing-v7-expanded-ready-local.v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if dataset_root == output_root or dataset_root in output_root.parents or output_root in dataset_root.parents:
        raise RuntimeError("COCO view and dataset roots must remain isolated")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output must be absent or empty: {output_root}")

    report_path = dataset_root / "report.json"
    validation_path = dataset_root / "reload_validation.json"
    selection_path = dataset_root / "selection_manifest.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "ready_for_training":
        raise RuntimeError("V7 report is not a ready-for-training artifact")
    if validation.get("status") != "passed" or validation.get("image_count") != report.get("selected_count"):
        raise RuntimeError("V7 reload validation did not pass or has the wrong cardinality")
    expected_selection_hash = report.get("artifact_hashes", {}).get("selection_manifest.jsonl")
    if sha256_file(selection_path) != expected_selection_hash:
        raise RuntimeError("V7 selection manifest hash mismatch")

    output_root.mkdir(parents=True, exist_ok=True)
    split_receipts = {
        source: convert_split(dataset_root, output_root, source, target)
        for source, target in SPLITS.items()
    }
    actual = {split: split_receipts[split]["images"] for split in SPLITS}
    expected = {split: int(report["counts"][split]) for split in SPLITS}
    if actual != expected:
        raise RuntimeError(f"COCO split cardinality mismatch: {actual} != {expected}")

    receipt = {
        "schema": "fireviewer.pointing-v7-coco-hardlink-view.v1",
        "status": "ready",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "dataset_report_sha256": sha256_file(report_path),
        "selection_manifest_sha256": expected_selection_hash,
        "categories": CATEGORIES,
        "splits": split_receipts,
        "storage_policy": {
            "image_materialization": "NTFS hardlinks",
            "additional_image_payload_bytes": 0,
            "retain_until_user_review": True,
            "automatic_cleanup": False,
        },
    }
    atomic_json(output_root / "coco_view_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
