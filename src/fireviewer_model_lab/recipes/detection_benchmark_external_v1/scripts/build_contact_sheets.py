from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .common import data_root, ensure_layout, load_config, read_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=16)
    args = parser.parse_args()
    config = load_config(args.config)
    root = data_root(config)
    ensure_layout(root)
    audited = root / "audit" / "still_candidates.audited.jsonl"
    source = audited if audited.exists() else root / "candidates" / "still_candidates.jsonl"
    rows = read_jsonl(source)
    destination = root / "review" / "contact_sheets"
    tile_width, tile_height, caption_height = 480, 300, 72
    font = ImageFont.load_default(size=18)
    outputs: list[str] = []
    for page_index in range(math.ceil(len(rows) / args.page_size)):
        page_rows = rows[page_index * args.page_size : (page_index + 1) * args.page_size]
        columns = args.columns
        page_row_count = math.ceil(len(page_rows) / columns)
        sheet = Image.new("RGB", (columns * tile_width, page_row_count * (tile_height + caption_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(page_rows):
            x = (offset % columns) * tile_width
            y = (offset // columns) * (tile_height + caption_height)
            with Image.open(row["image_path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
                paste_x = x + (tile_width - image.width) // 2
                paste_y = y + (tile_height - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))
            passed = row.get("independence_audit", {}).get("passed")
            caption = f"{row['sample_id']}\nsource={row['source_media_id']} independent={passed}"
            draw.multiline_text((x + 6, y + tile_height + 4), caption, fill="black", font=font, spacing=4)
            draw.rectangle((x, y, x + tile_width - 1, y + tile_height + caption_height - 1), outline="#666666", width=1)
        output = destination / f"still-review-{page_index + 1:03d}.jpg"
        sheet.save(output, quality=92, subsampling=0)
        outputs.append(str(output))
    write_json(
        root / "review" / "contact_sheet_index.json",
        {"candidate_manifest": str(source), "count": len(rows), "sheets": outputs},
    )
    print(f"created {len(outputs)} contact sheets for {len(rows)} candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
