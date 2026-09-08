"""Check the retained V7/V8 artifacts after the bounded local media cleanup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.pointing_dataset_v7.split_registry import digest, make_registry, read_rows, verify_coco


def verify(root: Path) -> dict:
    artifacts = root / "artifacts/local"
    ready = artifacts / "fireviewer-pointing-v7-ready-local-5000-20260825-r7"
    rows = read_rows(ready / "selection_manifest.jsonl")
    for row in rows:
        path = ready / "data" / row["split"] / row["file_name"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Canonical V7 image identity mismatch: {path}")
    original = verify_coco(
        artifacts / "fireviewer-pointing-v7-coco-hardlinks-5000-20260825-r1", make_registry(rows)
    )
    audited = artifacts / "fireviewer-pointing-v7-audited-20260827"
    audited_result = verify_coco(audited / "coco", json.loads((audited / "split_registry.json").read_text()))
    if audited_result["status"] != "passed":
        raise ValueError("Audited corpus failed verification")

    benchmark = artifacts / "pointing-v7-audited-benchmark-20260827"
    manifest = json.loads((benchmark / "artifact_manifest.json").read_text())
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Benchmark artifact changed: {path}")
    ledgers = read_rows(artifacts / "pointing-v8-corpus-reconciliation-20260827/review_sources.jsonl")
    for item in ledgers:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"V8 review ledger changed: {item['path']}")
    candidates = read_rows(artifacts / "pointing-v8-targeted-reuse-20260827/candidate_manifest.jsonl")
    for item in candidates:
        if digest(Path(item["source_image"])) != item["sha256"]:
            raise ValueError(f"V8 source image changed: {item['source_image']}")
        if not Path(item["source_annotation"]).is_file():
            raise ValueError(f"V8 source annotation missing: {item['source_annotation']}")
    checkpoint = artifacts / "pointing-deim-dfine-large-v7-704-resume-e4-b2a8-5000-20260825-r2/best_stg2.pth"
    if digest(checkpoint) != "7b26fbf4cf05723670c1fd01da4542ca98fb92aef1e2f0c14dfde4f86184dec5":
        raise ValueError("V7 checkpoint identity mismatch")
    return {
        "status": "passed", "canonical_v7_images_verified": len(rows),
        "original_coco": original, "audited_coco": audited_result,
        "benchmark_artifacts_verified": len(manifest["artifacts"]),
        "v8_review_ledgers_verified": len(ledgers), "v8_candidates_verified": len(candidates),
        "v7_checkpoint_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[2])
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
