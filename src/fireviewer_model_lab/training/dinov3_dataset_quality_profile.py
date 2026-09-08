"""Profile a composed DINOv3 multitask manifest for SageMaker dataset QA.

The job is deliberately metadata-only. It validates the admission contract and
emits Canvas/Data Wrangler-friendly aggregates plus an automated semantic-audit
queue. Human labeling and SageMaker Ground Truth are intentionally prohibited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_BENCHMARK_MARKERS = (
    "benchdata",
    "fireviewer_bench",
    "independent-benchmark",
    "benchmark-independent",
)
ALLOWED_SPLITS = {"train", "validation", "test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _discover_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def _point_kind(row: dict[str, Any]) -> str:
    points = row.get("anchor_points")
    if not isinstance(points, list) or not points:
        return "none"
    kinds = sorted({str(point.get("kind") or "unknown") for point in points if isinstance(point, dict)})
    return "+".join(kinds) if kinds else "invalid"


def profile_manifest(
    *,
    input_dir: Path,
    output_dir: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    manifest = _discover_one(input_dir, "composition_candidate_manifest.jsonl")
    composition_report_path = _discover_one(input_dir, "composition_report.json")
    actual_manifest_sha256 = _sha256(manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("composition manifest SHA-256 does not match the pinned input")

    composition_report = json.loads(composition_report_path.read_text(encoding="utf-8"))
    if composition_report.get("manifest_sha256") != actual_manifest_sha256:
        raise ValueError("composition report does not pin the input manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    findings: list[dict[str, Any]] = []
    automated_audit_queue: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    source_split_task: Counter[tuple[str, str, str, str]] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    point_kind_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    locator_kinds: Counter[str] = Counter()
    rows = 0
    unknown_rights_rows = 0
    benchmark_reference_rows = 0
    invalid_rows = 0

    def record_finding(code: str, sample_id: str, detail: str) -> None:
        nonlocal invalid_rows
        invalid_rows += 1
        findings.append({"severity": "error", "code": code, "sample_id": sample_id, "detail": detail})

    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            rows += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                record_finding("invalid_json", f"line:{line_number}", str(exc))
                continue
            sample_id = str(row.get("sample_id") or "")
            split = str(row.get("split") or "")
            source_id = str(row.get("source_id") or "")
            family_id = str(row.get("source_family_id") or "")
            group = str(row.get("split_group") or "")
            digest = str(row.get("image_sha256") or "").casefold()
            license_name = str(row.get("license") or "")
            consent = row.get("consent_basis")
            locator = row.get("image_locator")
            locator_kind = str(locator.get("kind") or "missing") if isinstance(locator, dict) else "missing"
            point_kind = _point_kind(row)
            point_positive = point_kind not in {"none", "invalid"}

            if not sample_id or sample_id in sample_ids:
                record_finding("duplicate_or_missing_sample_id", sample_id or f"line:{line_number}", "sample_id must be unique")
            else:
                sample_ids.add(sample_id)
            if not SHA256_RE.fullmatch(digest) or digest in image_hashes:
                record_finding("duplicate_or_invalid_image_sha256", sample_id, digest)
            else:
                image_hashes.add(digest)
            if split not in ALLOWED_SPLITS:
                record_finding("invalid_split", sample_id, split)
            if not source_id or not family_id or not group:
                record_finding("missing_identity", sample_id, "source_id/source_family_id/split_group required")
            if isinstance(locator, dict):
                locator_sha = str(locator.get("sha256") or "").casefold()
                if locator_sha != digest:
                    record_finding("image_locator_sha_mismatch", sample_id, locator_sha)
            else:
                record_finding("missing_image_locator", sample_id, "image_locator must be an object")

            consent_reference = str(consent.get("reference") or "") if isinstance(consent, dict) else ""
            if not license_name or not consent_reference:
                unknown_rights_rows += 1
                record_finding("unknown_rights", sample_id, "license and consent reference are required")

            benchmark_text = json.dumps(row, sort_keys=True).casefold()
            if any(marker in benchmark_text for marker in FORBIDDEN_BENCHMARK_MARKERS):
                benchmark_reference_rows += 1
                record_finding("independent_benchmark_reference", sample_id, "forbidden benchmark marker")

            points = row.get("anchor_points")
            if points is not None and not isinstance(points, list):
                record_finding("invalid_anchor_points", sample_id, "anchor_points must be a list")
            elif isinstance(points, list):
                for point in points:
                    try:
                        x, y = float(point["x"]), float(point["y"])
                    except (KeyError, TypeError, ValueError):
                        record_finding("invalid_anchor_point", sample_id, "point requires numeric x/y")
                        continue
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        record_finding("anchor_point_out_of_bounds", sample_id, f"x={x}, y={y}")
            if point_positive and row.get("point_supervised") is not True:
                record_finding("point_target_not_supervised", sample_id, point_kind)
            if row.get("bbox_to_point") is not False or "bbox" in str(row.get("point_derivation") or "").casefold():
                record_finding("bbox_derived_point", sample_id, str(row.get("point_derivation") or ""))

            group_splits[group].add(split)
            split_counts[split] += 1
            source_counts[source_id] += 1
            family_counts[family_id] += 1
            point_kind_counts[point_kind] += 1
            licenses[license_name or "unknown"] += 1
            locator_kinds[locator_kind] += 1
            for task in ("presence", "point", "segmentation", "abstention"):
                supervised = bool(row.get(f"{task}_supervised"))
                if supervised:
                    task_counts[f"{task}_supervised_rows"] += 1
            if point_positive:
                task_counts["point_positive_rows"] += 1
                queue_row = {
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "source_family_id": family_id,
                    "split": split,
                    "split_group": group,
                    "image_sha256": digest,
                    "image_locator": locator,
                    "anchor_points": points,
                    "point_kind": point_kind,
                    "annotation_provenance": row.get("annotation_provenance"),
                    "audit_decision": "pending_automated_geometry_and_overlap_checks",
                }
                automated_audit_queue.append(queue_row)
            source_split_task[(source_id, family_id, split, point_kind)] += 1

    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    for group in leaking_groups:
        findings.append(
            {
                "severity": "error",
                "code": "split_group_leakage",
                "sample_id": "",
                "detail": f"{group}: {sorted(group_splits[group])}",
            }
        )

    canvas_path = output_dir / "canvas_source_split_quality.csv"
    with canvas_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_id", "source_family_id", "split", "point_kind", "rows"),
        )
        writer.writeheader()
        for (source_id, family_id, split, point_kind), count in sorted(source_split_task.items()):
            writer.writerow(
                {
                    "source_id": source_id,
                    "source_family_id": family_id,
                    "split": split,
                    "point_kind": point_kind,
                    "rows": count,
                }
            )

    _write_jsonl(output_dir / "quality_findings.jsonl", findings)
    _write_jsonl(output_dir / "automated_point_audit_queue.jsonl", automated_audit_queue)
    summary = {
        "schema_version": 1,
        "kind": "fireviewer-dinov3-sagemaker-dataset-quality-profile",
        "manifest": manifest.name,
        "manifest_sha256": actual_manifest_sha256,
        "composition_report_sha256": _sha256(composition_report_path),
        "rows": rows,
        "unique_sample_ids": len(sample_ids),
        "unique_image_sha256": len(image_hashes),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(family_counts.items())),
        "license_counts": dict(sorted(licenses.items())),
        "image_locator_kind_counts": dict(sorted(locator_kinds.items())),
        "point_kind_counts": dict(sorted(point_kind_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "unknown_rights_rows": unknown_rights_rows,
        "benchmark_reference_rows": benchmark_reference_rows,
        "split_group_leakage": leaking_groups,
        "error_findings": len(findings),
        "automated_point_audit_queue_rows": len(automated_audit_queue),
        "ground_truth_import_rows": 0,
        "ground_truth_ready": False,
        "sagemaker_ground_truth_allowed": False,
        "canvas_data_wrangler_ready": True,
        "integrity_passed": not findings,
        "professional_corpus_ready": composition_report.get("professional_corpus_ready") is True and not findings,
        "publication_allowed": False,
        "training_ready": False,
        "quality_gate_deficits": composition_report.get("quality_gate_deficits", {}),
        "next_action": "ingest_published_native_mask_sources_then_run_automated_semantic_and_leakage_gates",
    }
    _write_json(output_dir / "dataset_quality_summary.json", summary)
    _write_json(
        output_dir / "human_annotation_policy.json",
        {
            "schema_version": 1,
            "allowed": False,
            "ground_truth_import_rows": 0,
            "automated_audit_rows": len(automated_audit_queue),
            "task": "published_native_mask_intake_only",
            "automatic_launch_allowed": False,
            "reason": "campaign policy prohibits human annotation and pseudo-label generation",
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = profile_manifest(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        expected_manifest_sha256=args.expected_manifest_sha256.casefold(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
