"""Full-frame review packets with every annotation and target detail, no admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from training.pointing_dataset_v7.render_candidate_zooms import image_path, objects
from training.pointing_dataset_v7.split_registry import digest, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.artifact / "candidate_manifest.jsonl")
    rows.sort(key=lambda r: (not bool(objects(r)), r["candidate_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for start in range(0, len(rows), 2):
        page = rows[start:start + 2]
        sheet = Image.new("RGB", (1440, 620 * len(page)), "#f8f8f8")
        for position, row in enumerate(page):
            source = image_path(row, args.artifact)
            if digest(source) != row["sha256"]:
                raise ValueError("Candidate image changed before review")
            with Image.open(source) as image:
                original = image.convert("RGB")
            context = ImageOps.contain(original, (960, 540))
            drawing = ImageDraw.Draw(context)
            sx, sy = context.width / original.width, context.height / original.height
            targets = objects(row)
            for i, (box, label) in enumerate(targets):
                x, y, w, h = box
                color = "#007bff" if label == 1 else "#ff7300"
                drawing.rectangle((x * sx, y * sy, (x + w) * sx, (y + h) * sy), outline=color, width=2)
                drawing.text((x * sx, max(0, y * sy - 13)), str(i + 1), fill=color)
                if i < 3:
                    margin = max(30, .6 * max(w, h))
                    crop = original.crop((max(0, x - margin), max(0, y - margin),
                                          min(original.width, x + w + margin), min(original.height, y + h + margin)))
                    zoom = ImageOps.contain(crop, (440, 174))
                    sheet.paste(zoom, (986, position * 620 + i * 180))
            sheet.paste(context, (0, position * 620))
            text = f"{row['candidate_id']} | {original.width}x{original.height} | {len(targets)} annotation(s) | {row.get('source_video_id', '')}"
            if len(targets) > 3:
                text += " | EXTRA TARGETS: OPEN ORIGINAL"
            ImageDraw.Draw(sheet).text((8, position * 620 + 554), text, fill="black")
        path = args.output / f"review-{start // 2 + 1:03d}.jpg"
        sheet.save(path, quality=94)
        index.append({"packet": path.name, "sha256": digest(path), "candidate_ids": [r["candidate_id"] for r in page]})
    (args.output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps({"packets": len(index), "images": len(rows), "admissions": 0}))


if __name__ == "__main__":
    main()
