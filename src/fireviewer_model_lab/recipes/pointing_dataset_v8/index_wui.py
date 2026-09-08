"""Read only the public WUI source catalogue; do not download its image corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import requests

ROOT = "13eOsOgSBPeeDJjR-8EenAvtf2L3X8c5l"
OFFICIAL = "https://github.com/les2feup/fire-wui-dataset"
FOLDER_MIME = "application/vnd.google-apps.folder"


def listing(session, identifier):
    response = session.get("https://drive.google.com/drive/folders/" + identifier, timeout=25)
    response.raise_for_status()
    match = re.search(r"window\['_DRIVE_ivd'\] = '(.*?)';", response.text)
    if not match:
        raise ValueError("Public Drive folder listing unavailable")
    raw = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), match[1]).replace("\\/", "/")
    data = json.loads(raw)
    return [{"id": x[0], "name": x[2], "mime": x[3]} for x in data[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    info = session.get("https://api.github.com/repos/les2feup/fire-wui-dataset/commits/main", timeout=20)
    info.raise_for_status()
    revision = info.json()["sha"]
    readme = session.get(f"https://raw.githubusercontent.com/les2feup/fire-wui-dataset/{revision}/README.md", timeout=20)
    readme.raise_for_status()
    if "dataset is licensed under the MIT License" not in readme.text:
        raise ValueError("Dataset-specific primary license declaration not confirmed")
    (args.output / "upstream-README.md").write_text(readme.text, encoding="utf-8")
    queue, entries, folders, excluded = [(ROOT, "")], [], [], []
    while queue:
        identifier, parent = queue.pop(0)
        if len(folders) >= 45:
            raise ValueError("Bounded catalogue inspection limit reached")
        children = listing(session, identifier)
        folders.append({"id": identifier, "path": parent, "visible_entries": len(children),
                        "may_be_paginated": len(children) >= 50})
        for item in children:
            path = parent + "/" + item["name"]
            if item["mime"] == FOLDER_MIME:
                if item["name"].casefold() == "perto":
                    excluded.append({"path": path, "reason": "upstream_close_range_category_outside_priority"})
                else:
                    queue.append((item["id"], path))
            else:
                entries.append(item | {"source_path": path, "source_dataset": "WUI-Fire-Detection",
                                       "source_revision": revision, "license": "MIT", "license_evidence": OFFICIAL,
                                       "annotation_state": "classification_only_not_detection_boxes",
                                       "review_status": "not_downloaded_not_reviewed", "v8_training_admitted": False})
        print(json.dumps({"folders_inspected": len(folders), "image_entries": len(entries)}), flush=True)
    (args.output / "source_catalogue.jsonl").write_text("".join(json.dumps(r) + "\n" for r in entries), encoding="utf-8")
    report = {"schema": "fireviewer.wui-v8-source-catalogue.v1", "status": "metadata_only_not_admitted",
              "official_repository": OFFICIAL, "repository_revision": revision,
              "readme_sha256": hashlib.sha256(readme.text.encode()).hexdigest(),
              "files_listed": len(entries), "folders": folders, "excluded_folders": excluded,
              "full_inventory_proven": not any(f["may_be_paginated"] for f in folders),
              "image_bytes_downloaded": 0, "admitted_images": 0,
              "limitations": ["Public-folder listing may be paginated; this is not a claim of full dataset size.",
                              "Upstream distance and lighting labels require image review.",
                              "Bounding boxes and original image provenance must be verified before admission.",
                              "A new aggregate dataset name does not establish new original sources or independent incidents."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files_listed": len(entries), "image_bytes_downloaded": 0}), flush=True)


if __name__ == "__main__":
    main()
