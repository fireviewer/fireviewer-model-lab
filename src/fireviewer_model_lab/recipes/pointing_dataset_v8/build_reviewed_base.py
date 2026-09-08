"""Reuse the audited, review-consistent V7 base without copying any image bytes.

This creates a V8 draft, never a training-ready COCO export. Historical holdouts
keep their role; unknown semantic dimensions remain unknown.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from training.pointing_dataset_v7.split_registry import canonical_source_group, digest, read_rows
from training.pointing_dataset_v8.audit_coverage import POLICY, audit, geometries


def select(reconciled, audited, history=None):
    reference = {r["sha256"]: r for r in audited}
    eligible, excluded = [], []
    for row in reconciled:
        current = reference.get(row["sha256"])
        heldout_group = bool(history) and row.get("split") == "train" and (
            history.get("sha256", {}).get(row["sha256"]) in {"validation", "test"}
            or history.get("groups", {}).get(row.get("split_group_id", "")) in {"validation", "test"}
            or any(s != "train" for s in history.get("source_groups", {}).get(canonical_source_group(row.get("source_group_id", "")), [])))
        reason = ("historical_evaluation_quarantine" if current is None else
                  "historically_mixed_split_group" if heldout_group else
                  "unresolved_review" if row.get("review_status") != "accepted_existing_review" else
                  "missing_annotation_or_person_clearance" if row.get("annotation_exploitable") is not True
                  or row.get("person_risk_reviewed_clear") is not True else None)
        if reason:
            excluded.append({"sha256": row["sha256"], "reason": reason, "split": row["split"]})
            continue
        if current["objects"] != row["objects"] or current["split"] != row["split"]:
            raise ValueError("Attempted silent annotation or split change during reuse")
        geometries(row)
        eligible.append(dict(row))
    return eligible, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--audited", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-generated-draft", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        previous = json.loads((args.output / "coverage_report.json").read_text())
        if not args.refresh_generated_draft or previous.get("draft_status") != "incomplete_do_not_train" or previous["inputs"]["reconciled_sha256"] != digest(args.reconciliation / "reconciled_manifest.jsonl"):
            raise FileExistsError("Refusing to overwrite an unrelated or finalized V8 draft")
    ledgers = read_rows(args.reconciliation / "review_sources.jsonl")
    for ledger in ledgers:
        if digest(Path(ledger["path"])) != ledger["sha256"]:
            raise ValueError("Existing visual review ledger changed")
    history = json.loads((args.audited / "historical_split_registry.json").read_text())
    selected, excluded = select(read_rows(args.reconciliation / "reconciled_manifest.jsonl"),
                                read_rows(args.audited / "selection_manifest.jsonl"), history)

    def verify(row):
        path = args.original / "data" / row["split"] / row["file_name"]
        if digest(path) != row["sha256"]:
            raise ValueError("Retained image bytes changed")
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("Retained image dimensions changed")
        row.update(source_image=str(path.resolve()), v8_corpus_admitted=True,
                   v8_training_admitted=row["split"] == "train",
                   v8_reuse_basis="unchanged_audited_image_annotation_and_bound_existing_visual_review",
                   synthetic=False)
        return row

    with ThreadPoolExecutor(max_workers=4) as executor:
        selected = list(executor.map(verify, selected))
    policy = json.loads(POLICY.read_text())
    report = audit(selected, read_rows(args.original / "selection_manifest.jsonl"), policy,
                   history)
    report["draft_status"] = "incomplete_do_not_train"
    report["storage"] = {"image_bytes_copied": 0, "mode": "references_to_retained_canonical_images"}
    report["reuse"] = {"images": len(selected), "splits": dict(Counter(r["split"] for r in selected)),
                        "excluded": dict(Counter(r["reason"] for r in excluded)),
                        "review_ledgers_verified": len(ledgers), "new_images_admitted": 0}
    report["inputs"] = {"reconciled_sha256": digest(args.reconciliation / "reconciled_manifest.jsonl"),
                        "audited_sha256": digest(args.audited / "selection_manifest.jsonl"),
                        "policy_sha256": digest(POLICY)}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("selection_manifest.jsonl", selected), ("excluded_from_reuse.jsonl", excluded)):
        (args.output / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (args.output / "coverage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["draft_status"], "reuse": report["reuse"], "ready": report["ready"],
                      "blocked_gates": report["blocked_gates"]}, indent=2))


if __name__ == "__main__":
    main()
