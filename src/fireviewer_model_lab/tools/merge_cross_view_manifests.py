"""Merge prepared cross-view manifests without copying their image assets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def merge_manifests(*, data_root: Path, manifests: list[Path], output: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    sources = Counter()
    for manifest in manifests:
        manifest = manifest.resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for _line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample id: {sample_id}")
            sample_ids.add(sample_id)
            split = str(row["split"])
            split_group = str(row["split_group"])
            group_splits[split_group].add(split)
            sources[str(row["source_id"])] += 1
            for view_name in ("source_view", "map_view"):
                image = (data_root / str(row[view_name]["image_relpath"])).resolve()
                if data_root not in image.parents or not image.is_file():
                    raise FileNotFoundError(
                        f"missing or unsafe {view_name} image: {row[view_name]['image_relpath']}"
                    )
            for mask_key in (
                "source_transient_mask_relpath",
                "map_transient_mask_relpath",
            ):
                mask = (data_root / str(row[mask_key])).resolve()
                if data_root not in mask.parents or not mask.is_file():
                    raise FileNotFoundError(f"missing or unsafe transient mask: {row[mask_key]}")
            rows.append(row)
    leakage = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leakage:
        raise ValueError(f"split-group leakage: {leakage}")
    if {str(row["split"]) for row in rows} != {"train", "validation", "test"}:
        raise ValueError("merged manifest must contain train, validation and test splits")
    rows.sort(key=lambda row: str(row["sample_id"]))
    output = output.resolve()
    if data_root not in output.parents:
        raise ValueError("merged manifest must be written inside data root")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "dataset_family": "fireviewer-cross-view-training-v1",
        "manifest": str(output),
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "source_counts": dict(sorted(sources.items())),
        "split_groups": len(group_splits),
        "all_rows_transient_masked": all(
            row.get("source_transient_mask_relpath") and row.get("map_transient_mask_relpath")
            for row in rows
        ),
        "assets_copied": False,
        "training_ready": bool(rows),
    }
    (output.parent / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            merge_manifests(
                data_root=args.data_root,
                manifests=args.manifest,
                output=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
