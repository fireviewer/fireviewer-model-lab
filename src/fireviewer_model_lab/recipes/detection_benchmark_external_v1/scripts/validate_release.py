from __future__ import annotations

import argparse
import json
import sys

from .common import data_root, load_config, read_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = data_root(config)
    independence_path = root / "audit" / "independence_summary.json"
    independence = json.loads(independence_path.read_text(encoding="utf-8")) if independence_path.exists() else None
    still = read_jsonl(root / "audit" / "still_candidates.audited.jsonl")
    sequences = read_jsonl(root / "candidates" / "sequence_candidates.jsonl")
    still_pending = [row["sample_id"] for row in still if row.get("annotation_status") != "accepted_double_review"]
    sequence_pending = [row["sequence_id"] for row in sequences if row.get("annotation_status") != "accepted_double_review"]
    failures: list[str] = []
    if not independence or independence.get("status") != "pass":
        failures.append("independence audit is missing or failed")
    if still_pending:
        failures.append(f"{len(still_pending)} still candidates lack accepted double review")
    if sequence_pending:
        failures.append(f"{len(sequence_pending)} sequences lack accepted double review")
    status = {
        "benchmark_id": config["benchmark_id"],
        "ready": not failures,
        "failures": failures,
        "still_candidates": len(still),
        "still_pending": len(still_pending),
        "sequence_candidates": len(sequences),
        "sequence_pending": len(sequence_pending),
    }
    write_json(root / "release" / "validation_status.json", status)
    print(json.dumps(status, indent=2))
    return 0 if status["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
