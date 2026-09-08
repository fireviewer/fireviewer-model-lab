"""Consolidate the unfinished V8 visual work without modifying an active corpus.

Acquisition, a model proposal, or inclusion in this queue is never admission.
All original images and previous review ledgers remain in their existing places.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.pointing_dataset_v7.split_registry import canonical_source_group, digest, read_rows
from training.pointing_dataset_v8.prepare_extension_review import render
from training.pointing_dataset_v8.source_identity import frozen_camera_views, pyro_camera_views


def collect_candidates(review_roots, base, history):
    known = {row["sha256"] for row in base}
    decisions = defaultdict(list)
    candidates = {}
    for root in review_roots:
        for row in read_rows(root / "curation_manifest.jsonl"):
            decisions[row["sha256"]].append({
                "root": str(root.resolve()),
                "decision": row["visual_review"]["decision"]["decision"],
                "reason": row["visual_review"]["decision"].get("reason", ""),
            })
        for row in read_rows(root / "review_manifest.jsonl"):
            candidates[row["sha256"]] = copy.deepcopy(row) | {
                "previous_review_root": str(root.resolve()),
                "previous_review_index": row["review_index"],
            }
    frozen = frozen_camera_views(base + list(candidates.values()), history)
    excluded = []
    kept = []
    for sha, row in candidates.items():
        prior = decisions[sha]
        states = {entry["decision"] for entry in prior}
        groups = {canonical_source_group(row.get(key, "")) for key in ("source_group_id", "split_group_id") if row.get(key)}
        reason = None
        if sha in known:
            reason = "already_in_active_corpus"
        elif "reject" in states:
            reason = "prior_explicit_rejection_not_overridden"
        elif any(state.startswith("accept_") for state in states):
            reason = "already_reviewed_acceptance_or_global_exclusion"
        elif pyro_camera_views(row) & frozen:
            reason = "historical_holdout_camera"
        elif any(any(split != "train" for split in history.get("source_groups", {}).get(group, [])) for group in groups):
            reason = "historical_holdout_group"
        elif row.get("source_family") == "AusSmoke" or not row.get("license") or row.get("license") in {"unknown", "unspecified"}:
            reason = "source_permission_not_established"
        if reason:
            excluded.append({"sha256": sha, "candidate_id": row["candidate_id"], "reason": reason})
            continue
        row.update({"previous_review_decisions": prior, "review_status": "pending_completion_visual_review",
                    "v8_corpus_admitted": False, "v8_training_admitted": False, "training_admitted": False})
        kept.append(row)

    def priority(row):
        hint = str(row.get("source_record_id", "")).lower()
        labels = row.get("objects", {}).get("category", [])
        return (not ("/night/" in hint and "/fire/" in hint),
                not ("/evening/" in hint and "/fire/" in hint),
                0 not in labels, row.get("source_family", ""), row["sha256"])

    kept.sort(key=priority)
    return kept, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite completion work")
    roots = sorted(root for root in args.artifact_root.glob("pointing-v8-review-*-20260827")
                   if (root / "curation_manifest.jsonl").is_file())
    proposal_root = args.artifact_root / "pointing-v8-review-wui-20260827/proposals"
    if (proposal_root / "curation_manifest.jsonl").is_file():
        roots.append(proposal_root)
    base = read_rows(args.base)
    history = json.loads(args.history.read_text(encoding="utf-8"))
    candidates, exclusions = collect_candidates(roots, base, history)
    for index, row in enumerate(candidates, 1):
        row["review_index"] = index
    args.output.mkdir(parents=True)
    for name, rows in (("review_manifest.jsonl", candidates), ("queue_exclusions.jsonl", exclusions)):
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output / "manual_decisions.jsonl").write_text("", encoding="utf-8")
    packets = render(candidates, args.output / "pages", 1)
    (args.output / "packets.json").write_text(json.dumps(packets, indent=2), encoding="utf-8")
    report = {"schema": "fireviewer.pointing-v8-completion-queue.v1", "status": "pending_visual_review_not_admitted",
              "base_manifest": str(args.base.resolve()), "base_manifest_sha256": digest(args.base),
              "history_sha256": digest(args.history), "source_review_roots": [str(root.resolve()) for root in roots],
              "queued_images": len(candidates), "source_counts": dict(Counter(row.get("source_family", "unknown") for row in candidates)),
              "exclusions": dict(Counter(row["reason"] for row in exclusions)), "image_bytes_downloaded": 0,
              "image_bytes_copied": 0, "admitted_images": 0, "active_training_inputs_modified": False}
    (args.output / "queue_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
