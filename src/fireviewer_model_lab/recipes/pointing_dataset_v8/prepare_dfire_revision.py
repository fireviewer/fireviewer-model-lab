"""Prepare a bounded D-Fire visual-review queue for a corpus revision.

The bulk V8P3 import deliberately kept upstream annotations without claiming a
manual review.  This module narrows that import before any new training view is
built: negatives are quarantined, positive sequences are capped, and rare small
targets are prioritized.  The output remains a review queue, never an admitted
training corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageOps
from pycocotools.coco import COCO

from training.pointing_dataset_v7.split_registry import digest, read_rows


CATEGORIES = [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}]
SMALL_RELATIVE_AREA = 0.005


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_small(row: dict, category: int) -> bool:
    image_area = row["width"] * row["height"]
    return any(
        cls == category and area / image_area <= SMALL_RELATIVE_AREA
        for cls, area in zip(row["objects"]["category"], row["objects"]["area"], strict=True)
    )


def scene_kind(row: dict) -> str:
    classes = set(row["objects"]["category"])
    if classes == {0, 1}:
        return "fire_and_smoke"
    if classes == {0}:
        return "fire_only"
    if classes == {1}:
        return "smoke_only"
    return "negative"


def priority(row: dict) -> tuple:
    """Prefer the measured V8 gaps, then stable identity for determinism."""
    classes = row["objects"]["category"]
    return (
        has_small(row, 1),
        1 in classes,
        has_small(row, 0),
        scene_kind(row) == "fire_and_smoke",
        min(len(classes), 6),
        row["width"] * row["height"],
        row["sha256"],
    )


def select_group_capped(rows: list[dict], max_per_group: int) -> tuple[list[dict], list[dict]]:
    if max_per_group <= 0:
        raise ValueError("max_per_group must be positive")
    groups: dict[str, list[dict]] = defaultdict(list)
    excluded: list[dict] = []
    for row in rows:
        if row.get("split") != "train" or row.get("source_family") != "D-Fire":
            raise ValueError("The revision queue accepts only D-Fire train rows")
        if not row["objects"]["bbox"]:
            excluded.append({
                "sha256": row["sha256"],
                "source_record_id": row["source_record_id"],
                "source_group_id": row["source_group_id"],
                "reason": "unreviewed_source_negative_quarantined",
            })
            continue
        groups[row["source_group_id"]].append(row)
    selected: list[dict] = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=priority, reverse=True)
        selected.extend(ordered[:max_per_group])
        excluded.extend({
            "sha256": row["sha256"],
            "source_record_id": row["source_record_id"],
            "source_group_id": group,
            "reason": "sequence_diversity_cap",
        } for row in ordered[max_per_group:])
    selected.sort(key=lambda row: (row["source_group_id"], priority(row)), reverse=False)
    selected = [copy.deepcopy(row) for row in selected]
    for index, row in enumerate(selected):
        row["revision_review_index"] = index
        row["review_status"] = "pending_v8p4_visual_and_annotation_review"
        row["v8_corpus_admitted"] = False
        row["v8_training_admitted"] = False
    return selected, excluded


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Existing review image differs: {target}")
    else:
        os.link(source, target)


def build_screening_coco(rows: list[dict], output: Path) -> Path:
    split = output / "screening-coco" / "valid"
    images, annotations = [], []
    annotation_id = 1
    for image_id, row in enumerate(rows, 1):
        filename = "images/" + row["sha256"] + ".jpg"
        source = Path(row["source_image"])
        if digest(source) != row["sha256"]:
            raise ValueError(f"Source identity changed: {source}")
        link(source, split / filename)
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": row["width"],
            "height": row["height"],
            "fireviewer_sha256": row["sha256"],
            "fireviewer_source_group_id": row["source_group_id"],
        })
        for box, category, area in zip(
            row["objects"]["bbox"], row["objects"]["category"], row["objects"]["area"], strict=True
        ):
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": category,
                                "bbox": box, "area": area, "iscrowd": 0})
            annotation_id += 1
    annotation_path = split / "_annotations.coco.json"
    write_json(annotation_path, {"images": images, "annotations": annotations, "categories": CATEGORIES})
    loaded = COCO(str(annotation_path))
    if len(loaded.imgs) != len(images) or len(loaded.anns) != len(annotations):
        raise ValueError("Screening COCO reload mismatch")
    return annotation_path


def _fit(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    contained = ImageOps.contain(image, size)
    canvas = Image.new("RGB", size, "#111111")
    ox, oy = (size[0] - contained.width) // 2, (size[1] - contained.height) // 2
    canvas.paste(contained, (ox, oy))
    return canvas, contained.width / image.width, ox, oy


def render_packets(rows: list[dict], output: Path, per_page: int = 8) -> list[dict]:
    if per_page != 8:
        raise ValueError("The evidence layout is fixed at eight images per page")
    page_dir = output / "review-pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    packets = []
    tile_w, tile_h, caption_h = 960, 540, 54
    for start in range(0, len(rows), per_page):
        page_rows = rows[start:start + per_page]
        sheet = Image.new("RGB", (tile_w * 2, (tile_h + caption_h) * 4), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(page_rows):
            with Image.open(row["source_image"]) as opened:
                original = opened.convert("RGB")
            frame, scale, ox, oy = _fit(original, (tile_w, tile_h))
            frame_draw = ImageDraw.Draw(frame)
            for ordinal, (box, category) in enumerate(zip(
                row["objects"]["bbox"], row["objects"]["category"], strict=True
            ), 1):
                x, y, width, height = box
                color = "#ff6200" if category == 0 else "#00bfff"
                xy = (ox + x * scale, oy + y * scale,
                      ox + (x + width) * scale, oy + (y + height) * scale)
                frame_draw.rectangle(xy, outline=color, width=4)
                frame_draw.text((max(2, xy[0]), max(2, xy[1] - 14)),
                                f"{ordinal}:{'F' if category == 0 else 'S'}", fill=color,
                                stroke_fill="black", stroke_width=2)
            col, line = slot % 2, slot // 2
            x0, y0 = col * tile_w, line * (tile_h + caption_h)
            sheet.paste(frame, (x0, y0))
            draw.text((x0 + 5, y0 + tile_h + 3),
                      f"{row['revision_review_index']:04d} | {row['source_record_id']} | "
                      f"{row['width']}x{row['height']} | {scene_kind(row)}", fill="black")
            flags = f"small_fire={has_small(row, 0)} small_smoke={has_small(row, 1)} group={row['source_group_id']}"
            draw.text((x0 + 5, y0 + tile_h + 27), flags, fill="black")
        name = f"review-{start // per_page:04d}.jpg"
        path = page_dir / name
        sheet.save(path, quality=94, subsampling=0)
        packets.append({"packet": name, "sha256": digest(path),
                        "review_indices": [row["revision_review_index"] for row in page_rows],
                        "source_record_ids": [row["source_record_id"] for row in page_rows]})
    return packets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-group", type=int, default=4)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    rows = read_rows(source)
    selected, excluded = select_group_capped(rows, args.max_per_group)
    annotation_path = build_screening_coco(selected, output)
    packets = render_packets(selected, output)
    write_rows(output / "candidate_manifest.jsonl", selected)
    write_rows(output / "excluded.jsonl", excluded)
    write_json(output / "packets.json", packets)
    kinds = Counter(scene_kind(row) for row in selected)
    report = {
        "schema": "fireviewer.pointing-v8p4-dfire-review-queue.v1",
        "status": "pending_visual_and_annotation_review_not_training_ready",
        "input": str(source),
        "input_sha256": digest(source),
        "input_rows": len(rows),
        "selected_positive_candidates": len(selected),
        "excluded": dict(Counter(row["reason"] for row in excluded)),
        "max_images_per_declared_group": max(Counter(row["source_group_id"] for row in selected).values(), default=0),
        "declared_groups": len({row["source_group_id"] for row in selected}),
        "scene_kinds": dict(kinds),
        "small_fire_images": sum(has_small(row, 0) for row in selected),
        "small_smoke_images": sum(has_small(row, 1) for row in selected),
        "screening_annotation": str(annotation_path),
        "screening_annotation_sha256": digest(annotation_path),
        "review_pages": len(packets),
        "training_ready": False,
        "limitations": [
            "Model disagreement may prioritize review but cannot validate source annotations.",
            "Every retained image and every target still requires visual review.",
            "The frozen validation and test sets are not read or changed by this queue builder.",
        ],
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
