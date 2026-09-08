#!/usr/bin/env python3
"""Render context/target zoom sheets for manual V7 candidate review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def load_rows(path: Path, first: int, last: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if first <= int(row["candidate_id"].rsplit("-", 1)[-1]) <= last]


def image_path(row: dict, artifact: Path) -> Path:
    if row.get("source_image"):
        return Path(row["source_image"])
    return artifact / row["candidate_image"]


def objects(row: dict) -> list[tuple[list[float], int]]:
    if "annotations" in row:
        return [
            (
                ann.get("bbox", ann.get("bbox_xywh")),
                1 if "smoke" in ann.get("class_name", "").lower() else 0,
            )
            for ann in row["annotations"]
        ]
    obj = row.get("objects", {})
    return [(bbox, int(category)) for bbox, category in zip(obj.get("bbox", []), obj.get("category", []))]


def render(rows: list[dict], artifact: Path, output: Path, per_sheet: int = 12) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 840, 360
    for page_i in range(math.ceil(len(rows) / per_sheet)):
        page = rows[page_i * per_sheet : (page_i + 1) * per_sheet]
        sheet = Image.new("RGB", (tile_w, tile_h * len(page)), "white")
        for i, row in enumerate(page):
            with Image.open(image_path(row, artifact)) as raw:
                image = raw.convert("RGB")
            anns = objects(row)
            context = ImageOps.contain(image, (400, 320))
            overlay = context.copy()
            draw = ImageDraw.Draw(overlay)
            sx, sy = context.width / image.width, context.height / image.height
            if anns:
                x, y, w, h = anns[0][0]
                color = "#1683ff" if anns[0][1] == 1 else "#ff7a00"
                draw.rectangle((x * sx, y * sy, (x + w) * sx, (y + h) * sy), outline=color, width=3)
                margin = max(w, h) * 2.5
                crop = image.crop((max(0, x - margin), max(0, y - margin), min(image.width, x + w + margin), min(image.height, y + h + margin)))
                zoom = ImageOps.contain(crop, (400, 320))
            else:
                zoom = Image.new("RGB", (400, 320), "#eeeeee")
            y0 = i * tile_h
            sheet.paste(overlay, (0, y0))
            sheet.paste(zoom, (420, y0))
            label = f"{row['candidate_id']} {row.get('gap_bucket', '')} | context / target"
            ImageDraw.Draw(sheet).text((8, y0 + 326), label, fill="black")
        name = f"zoom-{page_i + 1:03d}-{page[0]['candidate_id']}-{page[-1]['candidate_id']}.jpg"
        sheet.save(output / name, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(load_rows(args.artifact / "candidate_manifest.jsonl", args.first, args.last), args.artifact, args.output)


if __name__ == "__main__":
    main()
