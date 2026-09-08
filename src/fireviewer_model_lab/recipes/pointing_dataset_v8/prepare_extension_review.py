"""Incremental, immutable review queue, with known-image exclusion and all-box views.

Rendering and near-duplicate screening do not constitute visual admission.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageOps

from training.pointing_dataset_v7.split_registry import digest, read_rows


def load_candidates(source):
    """Reuse either acquisition format without copying images or granting review."""
    receipts = sorted((source / "receipts").glob("*.json"))
    if receipts:
        return [json.loads(p.read_text(encoding="utf-8")) for p in receipts]
    manifest = source / "candidate_manifest.jsonl"
    if not manifest.is_file():
        raise ValueError("No acquisition receipts or candidate manifest found")
    return read_rows(manifest)


def filter_selection_plan(candidates, plan):
    """Only queue explicitly planned source identities; never renumber prior work.

    HF Viewer JPEGs keep a distinct original/source SHA. A representation change
    must not lose the binding to the acquisition selection or count it twice.
    """
    identities = {row["sha256"] for row in plan}
    return [row for row in candidates if row.get("source_original_sha256", row["sha256"]) in identities]


def known_fingerprints(fingerprints, external):
    result = read_rows(fingerprints)
    for row in read_rows(external):
        path = external.parent / "images" / row["image_relpath"]
        sha = row.get("image_sha256", row.get("sha256"))
        if digest(path) != sha:
            raise ValueError("External benchmark image changed")
        with Image.open(path) as image:
            result.append({"sha256": sha, "phash": str(imagehash.phash(image)),
                           "phash_flipped": str(imagehash.phash(ImageOps.mirror(image)))})
    return result


def deduplicate(candidates, known, previous):
    seen_ids = {r["candidate_id"] for r in previous}
    hashes = {r["sha256"] for r in known + previous}
    index = [(int(r["phash"], 16), int(r["phash_flipped"], 16), r["sha256"]) for r in known + previous]
    added, excluded = [], []
    for row in candidates:
        if row["candidate_id"] in seen_ids:
            continue
        match = row["sha256"] if row["sha256"] in hashes else None
        kind = "exact_overlap" if match else None
        if not match:
            a, af = int(row["phash"], 16), int(row["phash_flipped"], 16)
            for b, bf, sha in index:
                if min((a ^ b).bit_count(), (af ^ b).bit_count(), (a ^ bf).bit_count(), (af ^ bf).bit_count()) <= 4:
                    match, kind = sha, "perceptual_overlap_le4"
                    break
        if match:
            excluded.append({"candidate_id": row["candidate_id"], "sha256": row["sha256"],
                             "reason": kind, "matching_sha256": match})
            continue
        added.append(row)
        hashes.add(row["sha256"])
        index.append((int(row["phash"], 16), int(row["phash_flipped"], 16), row["sha256"]))
    return added, excluded


def render(rows, output, start, per_page=6):
    output.mkdir(parents=True, exist_ok=True)
    packets = []
    for offset in range(0, len(rows), per_page):
        page = rows[offset:offset+per_page]
        width, height = 1080, 505
        canvas = Image.new("RGB", (width*2, height*((len(page)+1)//2)), "#f4f4f4")
        for tile, row in enumerate(page):
            x0, y0 = tile % 2 * width, tile // 2 * height
            path = Path(row["source_image"])
            if digest(path) != row["sha256"]:
                raise ValueError("Review image differs from candidate receipt")
            with Image.open(path) as image:
                original = image.convert("RGB")
            context = ImageOps.contain(original, (768, 432))
            draw = ImageDraw.Draw(context)
            sx, sy = context.width/original.width, context.height/original.height
            boxes, labels = row["objects"]["bbox"], row["objects"]["category"]
            for n, (box, label) in enumerate(zip(boxes, labels)):
                x, y, w, h = box
                color = "#00bfff" if label else "#ff6200"
                draw.rectangle((x*sx, y*sy, (x+w)*sx, (y+h)*sy), outline=color, width=2)
                draw.text((x*sx, max(0, y*sy-12)), str(n+1), fill=color)
                if n < 3:
                    padding = max(25, max(w,h)*.7)
                    crop_box = [max(0, int(x-padding)), max(0, int(y-padding)),
                                min(original.width, int(x+w+padding+1)), min(original.height, int(y+h+padding+1))]
                    crop = original.crop(crop_box)
                    target = ImageOps.contain(crop, (296, min(420//len(boxes), 300)))
                    zx, zy = target.width/crop.width, target.height/crop.height
                    ImageDraw.Draw(target).rectangle(((x-crop_box[0])*zx, (y-crop_box[1])*zy,
                        (x+w-crop_box[0])*zx, (y+h-crop_box[1])*zy), outline=color, width=1)
                    canvas.paste(target, (x0+776, y0+n*(432//min(len(boxes), 3))))
                    ImageDraw.Draw(canvas).text((x0+776, y0+n*(432//min(len(boxes),3))), str(n+1), fill="#0088ff")
            canvas.paste(context, (x0, y0))
            title = f"{row['review_index']:04d} {row['candidate_id']} | {original.width}x{original.height} | {len(boxes)} boxes"
            sub = row.get("source_record_id", "")
            ImageDraw.Draw(canvas).text((x0+4, y0+436), title, fill="black")
            ImageDraw.Draw(canvas).text((x0+4, y0+454), sub[:120], fill="black")
            if not boxes:
                ImageDraw.Draw(canvas).text((x0+4, y0+474), "CLASSIFICATION ONLY: inspect and annotate every visible target", fill="#a00000")
        name = f"review-{start+offset//per_page:04d}.jpg"
        canvas.save(output/name, quality=94)
        packets.append({"packet": name, "sha256": digest(output/name),
                        "review_indices": [r["review_index"] for r in page], "candidate_ids": [r["candidate_id"] for r in page]})
    return packets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerender", action="store_true")
    parser.add_argument("--selection-plan", type=Path,
                        help="Optional acquisition identities to prioritize; preserves every already indexed review")
    parser.add_argument("--fingerprints", type=Path, default=Path("artifacts/local/pointing-v7-split-audit-groupwise-20260827/fingerprints.jsonl"))
    parser.add_argument("--external", type=Path, default=Path("fireviewer_bench/data/homefire-pointing-independent-v1-r3/samples.jsonl"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "review_manifest.jsonl"
    previous = read_rows(manifest) if manifest.exists() else []
    candidates = load_candidates(args.source)
    acquired_count = len(candidates)
    if args.selection_plan:
        candidates = filter_selection_plan(candidates, read_rows(args.selection_plan))
    known = known_fingerprints(args.fingerprints, args.external)
    added, excluded = deduplicate(candidates, known, previous)
    for index, row in enumerate(added, len(previous)+1):
        row["review_index"] = index
    packets_path = args.output / "packets.json"
    packets = json.loads(packets_path.read_text()) if packets_path.exists() else []
    if args.rerender:
        by_id = {r["candidate_id"]: r for r in previous}
        refreshed = []
        for old_packet in packets:
            number = int(Path(old_packet["packet"]).stem.rsplit("-", 1)[1])
            refreshed += render([by_id[i] for i in old_packet["candidate_ids"]], args.output / "pages", number)
        packets = refreshed
    packets += render(added, args.output / "pages", len(packets)+1)
    manifest.write_text("".join(json.dumps(r)+"\n" for r in previous+added), encoding="utf-8")
    (args.output / "duplicate_exclusions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in excluded), encoding="utf-8")
    packets_path.write_text(json.dumps(packets, indent=2), encoding="utf-8")
    report = {"acquired_candidates": acquired_count, "planned_acquired_candidates": len(candidates),
              "selection_plan_sha256": digest(args.selection_plan) if args.selection_plan else None,
              "queued": len(previous)+len(added), "added_to_queue": len(added),
              "duplicates": dict(Counter(r["reason"] for r in excluded)), "pages": len(packets), "admitted": 0}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
