"""Local CPU annotation proposals, never ground truth or automatic admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8.prepare_extension_review import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/local/hf-dfine-large-fire-smoke-v7-public-20260827/payload"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 6:
        raise ValueError("CPU thread budget exceeded")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = args.model.resolve()
    sys.path.insert(0, str(root))
    from fireviewer_inference import FireViewerDFINE
    model = FireViewerDFINE(root, device="cpu", precision="fp32")
    model_sha = digest(root/"model.safetensors")
    candidates = [r for r in read_rows(args.review_root/"review_manifest.jsonl") if "/FINAL_DATASET/1/" in r["source_record_id"]]
    output = args.review_root/"proposals"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, row in enumerate(candidates, 1):
        receipt = output/f"proposal-{row['review_index']:04d}.json"
        if receipt.exists():
            proposed = json.loads(receipt.read_text())
            if proposed["sha256"] != row["sha256"] or proposed["annotation_proposal_model_sha256"] != model_sha:
                raise ValueError("Proposal model or image changed")
        else:
            if digest(Path(row["source_image"])) != row["sha256"]:
                raise ValueError("Candidate bytes changed before annotation proposal")
            predicted = model.predict(row["source_image"], threshold=.3, max_detections=30)
            boxes, labels, scores = [], [], []
            for item in predicted:
                x1, y1, x2, y2 = item["bbox_xyxy"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(row["width"], x2), min(row["height"], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2-x1, y2-y1])
                labels.append(item["class_id"])
                scores.append(item["score"])
            proposed = row | {"objects": {"bbox": boxes, "category": labels, "area": [b[2]*b[3] for b in boxes]},
                "proposal_scores": scores, "annotation_state": "model_proposals_NOT_ground_truth",
                "annotation_proposal_model_sha256": model_sha,
                "annotation_proposal_settings": {"device": "cpu", "precision": "fp32", "score_threshold": .3},
                "annotation_review_complete": False, "v8_training_admitted": False, "v8_corpus_admitted": False}
            proposed["proposal_objects_sha256"] = hashlib.sha256(json.dumps(proposed["objects"], sort_keys=True).encode()).hexdigest()
            receipt.write_text(json.dumps(proposed), encoding="utf-8")
        rows.append(proposed)
        if n % 20 == 0:
            print(json.dumps({"proposed_images": n, "total": len(candidates), "admitted": 0}), flush=True)
    (output/"review_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    packets = render(rows, output/"pages", 1)
    (output/"packets.json").write_text(json.dumps(packets, indent=2), encoding="utf-8")
    print(json.dumps({"proposal_images": len(rows), "admitted": 0, "training_started": False, "device": "cpu"}), flush=True)


if __name__ == "__main__":
    main()
