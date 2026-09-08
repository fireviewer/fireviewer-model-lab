"""Reconcile existing per-image review evidence without inventing visual clearance."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from training.pointing_dataset_v7.split_registry import digest, read_rows


def source_family(value: str, record_id: str = "") -> str:
    # D-Fire mirrors in WUI preserve WEB/AoF filenames. A different download
    # repository is not a new source family. Group/byte binding is still needed.
    filename = record_id.replace("\\", "/").rsplit("/", 1)[-1]
    if value == "WUI-Fire-Detection" and re.fullmatch(r"(?:WEB|AoF)\d+\.(?:jpg|jpeg|png)", filename, re.I):
        return "DFire"
    if "hpwren-figlib" in record_id.casefold():
        return "HPWREN"
    lower = value.casefold()
    if lower.replace("-", "").replace("_", "") == "dfire":
        return "DFire"
    for token, family in (("fasdd", "FASDD"), ("nemo", "NEMO"), ("fogfire", "FogFire"),
                          ("dfire", "DFire"), ("hpwren", "HPWREN"), ("pyro", "Pyro-SDIS"),
                          ("sainet", "SAINet"), ("cqu", "CQU"), ("firebench", "FireBench")):
        if token in lower:
            return family
    return value


def reconcile(row: dict, evidence: list[dict]) -> dict:
    accepted = [r for r in evidence if r["decision"] in {"accept", "accepted"}]
    # Duplicate/not-selected is not a visual finding that the original image is bad.
    rejected = [r for r in evidence if r["decision"] in {"reject", "rejected"}
                and not any(token in str(r.get("reason", "")) for token in ("duplicate_of_", "not_reviewed", "not_selected"))]
    status = "conflicting_reviews" if accepted and rejected else "accepted_existing_review" if accepted else "rejected" if rejected else "needs_review"
    out = dict(row)
    out["source_family"] = source_family(row["source_dataset"], row.get("source_record_id", ""))
    negative = not row["objects"]["bbox"]
    out["is_negative"] = negative
    out["negative_verified"] = negative and bool(accepted) and not rejected
    out["negative_review_status"] = "accepted_existing_review" if out["negative_verified"] else "not_applicable" if not negative else "needs_review"
    out["review_status"] = status
    out["review_evidence"] = evidence
    # Explicit zero-human evidence can resolve the stale false flag, never absence of evidence.
    human_evidence = [r for r in accepted if r.get("zero_human_foreground_confirmed") is True]
    if human_evidence and not rejected:
        out["person_risk_reviewed_clear"] = True
        out["person_clearance_basis"] = "hash_matched_existing_zero_human_review"
    out["framing_review_status"] = "legacy_flag_not_revalidated"
    out["obstruction_review_status"] = "not_quantified"
    out["v8_training_admitted"] = False  # annotation, framing and new-case review still pending
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refresh-generated-reports", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.refresh_generated_reports:
        raise FileExistsError(args.output)
    if args.output.exists():
        previous = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        if previous.get("schema") != "fireviewer.pointing-v8-review-reconciliation.v1" or previous["source_manifest_sha256"] != digest(args.dataset / "selection_manifest.jsonl"):
            raise ValueError("Refusing to replace reports for a different source or schema")
    evidence = defaultdict(list)
    ledger_refs = []

    def register(path, decisions, candidates=None, candidate_path=None):
        checksum = digest(path)
        candidate_checksum = digest(candidate_path or path.parent / "candidate_manifest.jsonl") if candidates is not None else None
        ledger_refs.append({"path": str(path.resolve()), "sha256": checksum, "rows": len(decisions)})
        for decision in decisions:
            candidate = (candidates or {}).get(decision.get("candidate_id"), {})
            sha = decision.get("sha256") or candidate.get("sha256")
            if not sha:
                continue
            evidence[sha].append({"decision": decision["decision"], "ledger": str(path.resolve()),
                                  "ledger_sha256": checksum,
                                  "review_id": decision.get("review_id", decision.get("candidate_id", sha)),
                                  "zero_human_foreground_confirmed": decision.get("zero_human_foreground_confirmed"),
                                  "reason": decision.get("reason", decision.get("reasons", decision.get("rejection_reasons", []))),
                                  "candidate_manifest_sha256": candidate_checksum})

    v4 = args.artifact_root / "pointing-dataset-v4-final-zero-human-merged-20260824"
    receipt = json.loads((v4 / "reviewer_receipts.json").read_text(encoding="utf-8"))
    if digest(v4 / "decision_ledger.jsonl") != receipt["decision_ledger_sha256"]:
        raise ValueError("V4 visual review ledger hash mismatch")
    register(v4 / "decision_ledger.jsonl", read_rows(v4 / "decision_ledger.jsonl"))
    hpwren = args.artifact_root / "pointing-dataset-v5-hpwren-distant-smoke-ready-20260824"
    hp_report = json.loads((hpwren / "report.json").read_text(encoding="utf-8"))
    if digest(hpwren / "review_decisions.jsonl") != hp_report["review_decisions_sha256"]:
        raise ValueError("HPWREN visual review ledger hash mismatch")
    register(hpwren / "review_decisions.jsonl", read_rows(hpwren / "review_decisions.jsonl"))
    v6 = args.artifact_root / "fireviewer-pointing-v6-ready-local-3163-20260825-r2"
    v6_report = json.loads((v6 / "report.json").read_text(encoding="utf-8"))
    v6_candidates = args.artifact_root / "pointing-v6-gap-extension-candidates-r2-20260825" / "candidate_manifest.jsonl"
    if digest(v6_candidates) != v6_report["candidate_manifest_sha256"] or digest(v6 / "visual_review_decisions.jsonl") != v6_report["visual_review_decisions_sha256"]:
        raise ValueError("V6 review/candidate identity mismatch")
    register(v6 / "visual_review_decisions.jsonl", read_rows(v6 / "visual_review_decisions.jsonl"),
             {r["candidate_id"]: r for r in read_rows(v6_candidates)}, v6_candidates)
    for path in sorted(args.artifact_root.glob("pointing-v7-*/review_decisions.jsonl")):
        candidate_path = path.parent / "candidate_manifest.jsonl"
        if candidate_path.is_file():
            candidates = {r["candidate_id"]: r for r in read_rows(candidate_path)}
            register(path, read_rows(path), candidates)
    # V7 first additions have review ledgers nested in the immutable built corpus.
    first_v7 = args.artifact_root / "fireviewer-pointing-v7-ready-local-20260825-r1"
    for label, directory in (("local", "pointing-v7-local-candidates-r2-20260825"),
                             ("sainet", "pointing-v7-sainet-candidates-r3-20260825")):
        candidates = {r["candidate_id"]: r for r in read_rows(args.artifact_root / directory / "candidate_manifest.jsonl")}
        path = first_v7 / "review" / f"{label}_candidate_decisions.jsonl"
        checksum = digest(path)
        candidate_sha = digest(args.artifact_root / directory / "candidate_manifest.jsonl")
        ledger_refs.append({"path": str(path.resolve()), "sha256": checksum})
        for decision in read_rows(path):
            if decision["candidate_id"] in candidates:
                candidate = candidates[decision["candidate_id"]]
                evidence[candidate["sha256"]].append({"decision": decision["decision"], "ledger": str(path.resolve()),
                    "ledger_sha256": checksum, "review_id": decision["candidate_id"], "reason": decision["reason"],
                    "candidate_manifest_sha256": candidate_sha})
    rows = read_rows(args.dataset / "selection_manifest.jsonl")
    quarantine = {r["sha256"] for r in read_rows(args.split_audit / "quarantine.jsonl")}
    output = []
    for row in rows:
        updated = reconcile(row, evidence[row["sha256"]])
        updated["v8_role"] = "retired_evaluation_quarantine" if row["sha256"] in quarantine else "frozen_evaluation" if row["split"] != "train" else "training_candidate"
        output.append(updated)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("reconciled_manifest.jsonl", output), ("review_sources.jsonl", ledger_refs)):
        (args.output / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in values), encoding="utf-8")
    report = {"schema": "fireviewer.pointing-v8-review-reconciliation.v1", "status": "needs_corpus_review",
              "source_manifest_sha256": digest(args.dataset / "selection_manifest.jsonl"),
              "rows": len(rows), "review_status": dict(Counter(r["review_status"] for r in output)),
              "train_review_status": dict(Counter(r["review_status"] for r in output if r["split"] == "train")),
              "roles": dict(Counter(r["v8_role"] for r in output)),
              "positive_negative_flag_corrected": sum(bool(r.get("negative_verified")) and bool(r["objects"]["bbox"]) for r in rows),
              "stale_person_flag_resolved_from_review": sum(not bool(old.get("person_risk_reviewed_clear")) and bool(new.get("person_risk_reviewed_clear")) for old, new in zip(rows, output)),
              "source_families": dict(Counter(r["source_family"] for r in output)),
              "blocking_gates": ["Resolve missing/conflicting per-image reviews",
                                 "Review framing/obstruction consistently", "Admit complementary distant/small-target cases"],
              "training_started": False, "source_images_changed": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
