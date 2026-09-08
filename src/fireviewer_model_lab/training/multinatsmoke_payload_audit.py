"""Audit every image/mask payload for selected MultiNatSmoke source families.

The audit is deliberately non-promoting: it computes exact and perceptual
identity, verifies masks, and derives a *candidate* base point, but never marks
rows training eligible. Semantic and split gates remain separate.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


def _ensure(module: str, package: str) -> None:
    try:
        __import__(module)
    except ImportError:  # pragma: no cover - exercised in the SageMaker image
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


_ensure("fsspec", "fsspec>=2024.6,<2027")
_ensure("aiohttp", "aiohttp>=3.9,<4")
_ensure("requests", "requests>=2.31,<3")
_ensure("PIL", "Pillow>=10,<12")
_ensure("scipy", "scipy>=1.11,<2")

import fsspec
import numpy as np
from PIL import Image
from scipy import fftpack, ndimage


def _open_source(stack: contextlib.ExitStack, source: str | Path) -> BinaryIO:
    value = str(source)
    if value.startswith(("https://", "http://")):
        return stack.enter_context(
            fsspec.open(value, "rb", block_size=16 * 1024 * 1024, cache_type="blockcache")
        )
    return stack.enter_context(Path(value).open("rb"))


def _phash64(image: Image.Image) -> str:
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    transformed = fftpack.dct(fftpack.dct(pixels, axis=0), axis=1)
    low = transformed[:8, :8]
    median = float(np.median(low[1:, :]))
    bits = (low > median).reshape(-1)
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _candidate_base(mask: np.ndarray) -> dict[str, Any]:
    foreground = mask > 0
    fraction = float(foreground.mean())
    if not foreground.any() or foreground.all():
        return {"mask_fraction": fraction, "valid": False}
    labels, count = ndimage.label(foreground)
    sizes = np.bincount(labels.reshape(-1))[1:]
    component = labels == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(component)
    bottom = int(ys.max())
    band = max(1, int(round(mask.shape[0] * 0.02)))
    base_xs = xs[ys >= bottom - band + 1]
    x = float(np.median(base_xs))
    return {
        "mask_fraction": fraction,
        "valid": True,
        "component_count": int(count),
        "largest_component_fraction": float(component.mean()),
        "point_x": x / max(1, mask.shape[1] - 1),
        "point_y": bottom / max(1, mask.shape[0] - 1),
        "derivation": "largest_mask_component_bottom_band_median_x_candidate_only",
    }


def _pairs(archive: zipfile.ZipFile, allowed_sources: set[str]) -> list[tuple[str, str, str, zipfile.ZipInfo, zipfile.ZipInfo]]:
    grouped: dict[tuple[str, str, str], dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = [part for part in info.filename.replace("\\", "/").split("/") if part]
        if (
            len(parts) != 5
            or parts[0] != "MultiNatSmokeDataset"
            or parts[1] not in {"Train", "Test"}
            or parts[2] not in allowed_sources
            or parts[3] not in {"images", "masks"}
        ):
            continue
        key = (parts[1].casefold(), parts[2], PurePosixPath(parts[4]).stem)
        grouped[key][parts[3]] = info
    missing = [key for key, value in grouped.items() if set(value) != {"images", "masks"}]
    if missing:
        raise ValueError(f"missing image/mask pairs: {len(missing)}")
    return [(*key, value["images"], value["masks"]) for key, value in sorted(grouped.items())]


def run_audit(*, source: str | Path, output_dir: Path, allowed_sources: set[str]) -> dict[str, Any]:
    if not allowed_sources:
        raise ValueError("at least one source family is required")
    rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    with contextlib.ExitStack() as stack:
        handle = _open_source(stack, source)
        with zipfile.ZipFile(handle) as archive:
            pairs = _pairs(archive, allowed_sources)
            for split, family, record_id, image_info, mask_info in pairs:
                row: dict[str, Any] = {
                    "sample_id": hashlib.sha256(
                        f"{family}:{split}:{record_id}".encode("utf-8")
                    ).hexdigest(),
                    "source_family": family,
                    "source_record_id": record_id,
                    "upstream_split": split,
                    "training_eligible": False,
                    "admission_status": "pending_semantic_split_and_rights_gates",
                    "errors": [],
                }
                try:
                    image_payload = archive.read(image_info)
                    mask_payload = archive.read(mask_info)
                    with Image.open(io.BytesIO(image_payload)) as opened:
                        opened.load()
                        image = opened.convert("RGB")
                    with Image.open(io.BytesIO(mask_payload)) as opened_mask:
                        opened_mask.load()
                        mask_image = opened_mask.convert("L")
                    if image.size != mask_image.size:
                        row["errors"].append("dimension_mismatch")
                    row.update(
                        {
                            "width": image.width,
                            "height": image.height,
                            "image_bytes": len(image_payload),
                            "mask_bytes": len(mask_payload),
                            "image_sha256": hashlib.sha256(image_payload).hexdigest(),
                            "mask_sha256": hashlib.sha256(mask_payload).hexdigest(),
                            "image_phash64": _phash64(image),
                            "base_candidate": _candidate_base(np.asarray(mask_image)),
                        }
                    )
                    if not row["base_candidate"].get("valid"):
                        row["errors"].append("empty_or_full_mask")
                except Exception as exc:  # retain a complete failure inventory
                    row["errors"].append(f"decode_error:{type(exc).__name__}:{exc}")
                for error in row["errors"]:
                    failures[str(error).split(":", 1)[0]] += 1
                rows.append(row)

    image_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("image_sha256"):
            image_sha[str(row["image_sha256"])].append(row)
        if row.get("image_phash64"):
            phash[str(row["image_phash64"])].append(row)
    duplicate_sha = [members for members in image_sha.values() if len(members) > 1]
    duplicate_phash = [members for members in phash.values() if len(members) > 1]
    cross_split_sha = [m for m in duplicate_sha if len({r["upstream_split"] for r in m}) > 1]
    cross_split_phash = [m for m in duplicate_phash if len({r["upstream_split"] for r in m}) > 1]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "multinatsmoke_payload_audit.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "schema_version": 1,
        "kind": "multinatsmoke_full_payload_audit",
        "source_families": sorted(allowed_sources),
        "rows": len(rows),
        "rows_decoded": sum(not row["errors"] for row in rows),
        "rows_with_errors": sum(bool(row["errors"]) for row in rows),
        "error_counts": dict(sorted(failures.items())),
        "exact_duplicate_image_groups": len(duplicate_sha),
        "equal_phash_image_groups": len(duplicate_phash),
        "cross_upstream_split_exact_groups": len(cross_split_sha),
        "cross_upstream_split_equal_phash_groups": len(cross_split_phash),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "training_eligible_rows": 0,
        "upstream_split_accepted": False,
        "publication_allowed": False,
        "next_action": "semantic_base_gate_near_duplicate_clustering_and_event_grouped_resplit",
    }
    (output_dir / "multinatsmoke_payload_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-source", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(source=args.source, output_dir=args.output_dir, allowed_sources=set(args.allowed_source)), indent=2))


if __name__ == "__main__":
    main()
