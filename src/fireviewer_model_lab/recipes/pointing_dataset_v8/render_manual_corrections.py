"""Render explicitly supplied boxes for inspection. Does not admit any image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8.audit_coverage import geometries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--inspection-crops", action="store_true", help="Keep native-pixel padded target crops alongside the whole-scene overlay")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--corrections", type=Path)
    mode.add_argument("--source-indices", type=int, nargs="+", help="Render unchanged source boxes for additional all-target inspection")
    args = parser.parse_args()
    rows = {r["review_index"]: r for r in read_rows(args.review_root / "review_manifest.jsonl")}
    output = args.review_root / ("source_overlays" if args.source_indices else "corrected_overlays")
    output.mkdir(exist_ok=True)
    receipt_path = output / "receipts.jsonl"
    receipts = read_rows(receipt_path) if receipt_path.exists() else []
    previous = {r["review_index"]: r for r in receipts}
    if len(previous) != len(receipts):
        raise ValueError("Duplicate existing overlay receipt")
    corrections = (read_rows(args.corrections) if args.corrections else [
        {"review_index": i, "image_sha256": rows[i]["sha256"], "corrected_objects": rows[i]["objects"]}
        for i in args.source_indices])
    if len(corrections) != len({r["review_index"] for r in corrections}):
        raise ValueError("Duplicate requested correction")
    # A later annotation lot must neither erase earlier receipts nor silently
    # replace the pixels underlying an already recorded visual decision.
    for correction in corrections:
        old = previous.get(correction["review_index"])
        if old is not None:
            if (old["image_sha256"] != correction["image_sha256"]
                    or old["corrected_objects"] != correction["corrected_objects"]):
                raise ValueError("Refusing to overwrite an existing annotation proposal")
            if digest(Path(old["corrected_overlay"])) != old["corrected_overlay_sha256"]:
                raise ValueError("Existing overlay changed")
            if args.inspection_crops and "native_target_inspection_crops" not in old:
                raise ValueError("Existing proposal has no native crops; retain it and use a separate review lot")
    for correction in corrections:
        row = rows[correction["review_index"]]
        if row["sha256"] != correction["image_sha256"] or digest(Path(row["source_image"])) != row["sha256"]:
            raise ValueError("Manual boxes do not reference these image bytes")
        if correction["review_index"] in previous:
            continue
        updated = row | {"objects": correction["corrected_objects"], "annotation_state": "manual_proposal_not_reviewed"}
        boxes = geometries(updated)
        with Image.open(row["source_image"]) as im:
            canvas = im.convert("RGB")
        drawing = ImageDraw.Draw(canvas)
        for number, ((x, y, w, h), label) in enumerate(boxes, 1):
            color = "#00bfff" if label else "#ff6200"
            drawing.rectangle((x, y, x+w, y+h), outline=color, width=2)
            drawing.text((x+2, max(0, y-13)), f"{number} {'smoke' if label else 'fire'}", fill=color)
        path = output / f"manual-{row['review_index']:04d}.png"
        canvas.save(path)
        receipt = {"review_index": row["review_index"], "image_sha256": row["sha256"],
                         "corrected_overlay": str(path.resolve()), "corrected_overlay_sha256": digest(path),
                         "corrected_objects": correction["corrected_objects"], "admitted": False}
        if args.inspection_crops and boxes:
            x0 = min(b[0] for b, _ in boxes)
            y0 = min(b[1] for b, _ in boxes)
            x1 = max(b[0]+b[2] for b, _ in boxes)
            y1 = max(b[1]+b[3] for b, _ in boxes)
            pad = max(96, int(max(x1-x0, y1-y0)*.35))
            bounds = (max(0, int(x0)-pad), max(0, int(y0)-pad),
                      min(canvas.width, int(x1)+pad+1), min(canvas.height, int(y1)+pad+1))
            crop_path = output / f"crop-{row['review_index']:04d}.png"
            canvas.crop(bounds).save(crop_path)
            receipt.update({"inspection_crop": str(crop_path.resolve()),
                            "inspection_crop_sha256": digest(crop_path), "inspection_crop_bounds_xyxy": bounds,
                            "inspection_crop_scale": 1})
            target_crops = []
            for target, ((x, y, w, h), label) in enumerate(boxes, 1):
                target_pad = max(32, int(max(w, h) * .1))
                target_bounds = (max(0, int(x)-target_pad), max(0, int(y)-target_pad),
                                 min(canvas.width, int(x+w)+target_pad+1),
                                 min(canvas.height, int(y+h)+target_pad+1))
                target_path = output / f"target-{row['review_index']:04d}-{target:02d}.png"
                canvas.crop(target_bounds).save(target_path)
                target_crops.append({"target": target, "category": label,
                    "path": str(target_path.resolve()), "sha256": digest(target_path),
                    "bounds_xyxy": target_bounds, "scale": 1})
            receipt["native_target_inspection_crops"] = target_crops
        receipts.append(receipt)
    receipt_path.write_text("".join(json.dumps(r)+"\n" for r in receipts), encoding="utf-8")
    print(json.dumps({"rendered": len(receipts)-len(previous), "retained_receipts": len(receipts), "admitted": 0}))


if __name__ == "__main__":
    main()
