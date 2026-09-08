from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fireviewer_model_lab.training.corpus_pipeline import CLASS_NAMES

CORPORA = {
    "fasdd_v9": Path("corpus/fasdd"),
    "pyro_sdis_v0_1_0": Path("corpus/pyro-sdis-v0.1.0"),
    "nasa_hls_burn_scars_v1": Path("corpus/hls-burn-scars-v1"),
    "wikimedia_candidates_v0_1_0": Path("corpus/wikimedia-candidates-v0.1.0"),
    "streetview_global_context_v1": Path("corpus/streetview-global-context-v1"),
}
TRANSIENT_SUFFIXES = {".part", ".zip"}
TRANSIENT_NAMES = {
    "manifest.finalizing.jsonl",
    "visual-fingerprints.cache.jsonl",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_within(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path escaped dataset root: {resolved_path}") from exc
    return resolved_path


def _corpus_inventory(
    root: Path,
    relative_dir: Path,
    *,
    digest_owners: dict[str, str],
    corpus_name: str,
) -> tuple[dict[str, Any], int]:
    corpus_dir = _ensure_within(root, root / relative_dir)
    manifest_path = _ensure_within(root, corpus_dir / "manifest.jsonl")
    rows = 0
    image_bytes = 0
    split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    annotation_counts: Counter[str] = Counter()
    negative_rows = 0
    geo_pair_rows = 0
    cross_corpus_duplicates = 0

    with manifest_path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            digest = str(record["sha256"])
            owner = digest_owners.setdefault(digest, corpus_name)
            if owner != corpus_name:
                cross_corpus_duplicates += 1
            split_counts[str(record["split"])] += 1
            role = str(record["corpus_role"])
            role_counts[role] += 1
            annotations = record.get("annotations", [])
            if role == "detector_training" and not annotations:
                negative_rows += 1
            for annotation in annotations:
                annotation_counts[str(annotation["class_name"])] += 1
            if record.get("location") is not None:
                geo_pair_rows += 1
            image_path = _ensure_within(
                corpus_dir,
                corpus_dir / str(record["image_relpath"]),
            )
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing image at manifest line {line_number}: {image_path}"
                )
            image_bytes += image_path.stat().st_size

    return (
        {
            "relative_path": relative_dir.as_posix(),
            "manifest_sha256": _sha256_file(manifest_path),
            "rows": rows,
            "image_bytes": image_bytes,
            "split_counts": dict(sorted(split_counts.items())),
            "role_counts": dict(sorted(role_counts.items())),
            "annotation_counts": dict(sorted(annotation_counts.items())),
            "negative_training_rows": negative_rows,
            "geo_pair_rows": geo_pair_rows,
        },
        cross_corpus_duplicates,
    )


def build_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.name.casefold() != "datasetfire":
        raise ValueError("Dataset root must be the dedicated datasetfire directory")
    if not root.is_dir():
        raise FileNotFoundError(root)

    transient_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix.casefold() in TRANSIENT_SUFFIXES or path.name in TRANSIENT_NAMES)
    )
    staging_root = root / "_staging"
    staging_files = (
        sorted(
            path.relative_to(root).as_posix() for path in staging_root.rglob("*") if path.is_file()
        )
        if staging_root.exists()
        else []
    )
    if transient_files or staging_files:
        raise ValueError(
            f"Dataset root still contains transient payloads: {transient_files + staging_files}"
        )

    digest_owners: dict[str, str] = {}
    corpora: dict[str, Any] = {}
    cross_corpus_duplicates = 0
    for name, relative_dir in CORPORA.items():
        corpus, duplicates = _corpus_inventory(
            root,
            relative_dir,
            digest_owners=digest_owners,
            corpus_name=name,
        )
        corpora[name] = corpus
        cross_corpus_duplicates += duplicates

    massif_dir = root / "sources" / "massifs"
    commune_csv = massif_dir / "processed" / "communes_massifs_cog2021.csv"
    summary_csv = massif_dir / "processed" / "massifs_summary_cog2021.csv"
    provenance_path = massif_dir / "processed" / "provenance.json"
    with commune_csv.open(encoding="utf-8-sig") as handle:
        commune_rows = sum(1 for _line in handle) - 1
    with summary_csv.open(encoding="utf-8-sig") as handle:
        massif_rows = sum(1 for _line in handle) - 1
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    payload_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"dataset-index.json", "README.md"}
    ]
    payload_bytes = sum(path.stat().st_size for path in payload_files)
    training_classes = set()
    for corpus in corpora.values():
        if "detector_training" in corpus["role_counts"]:
            training_classes.update(corpus["annotation_counts"])
    missing_training_classes = sorted(set(CLASS_NAMES.values()) - training_classes)

    return {
        "schema_version": 1,
        "dataset_id": "firewarning-datasetfire-local-v1",
        "root_name": root.name,
        "corpora": corpora,
        "totals": {
            "rows": sum(corpus["rows"] for corpus in corpora.values()),
            "unique_image_sha256": len(digest_owners),
            "image_bytes": sum(corpus["image_bytes"] for corpus in corpora.values()),
            "payload_files": len(payload_files),
            "payload_bytes": payload_bytes,
            "cross_corpus_exact_sha256_duplicates": cross_corpus_duplicates,
        },
        "massif_reference": {
            "source_id": "anct_massifs_cog2021",
            "commune_membership_rows": commune_rows,
            "canonical_massifs": massif_rows,
            "raw_resource_sha1": provenance["resource_sha1"],
            "geometry_included": False,
        },
        "storage_policy": {
            "source_archives_retained": False,
            "transient_files": [],
            "retained_raw_references": [
                "sources/massifs/raw/diffusion-zonages-massifs-cog2021.xls"
            ],
        },
        "training_gate": {
            "ready_for_four_class_training": not missing_training_classes,
            "missing_training_classes": missing_training_classes,
            "candidate_media_requires_annotation_and_double_validation": True,
        },
    }


def write_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory = build_inventory(root)
    index_path = _ensure_within(root, root / "dataset-index.json")
    index_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    totals = inventory["totals"]
    gate = inventory["training_gate"]
    readme = f"""# Dataset FireWarning local

Cette racine contient uniquement les médias et références retenus pour FireWarning. Elle ne doit pas
être copiée dans l'image Docker publique.

- {totals["rows"]:,} lignes de manifeste et {totals["unique_image_sha256"]:,} images uniques ;
- {totals["cross_corpus_exact_sha256_duplicates"]} doublon SHA-256 entre corpus ;
- FASDD, Pyro-SDIS, zones brûlées NASA, candidats Wikimedia et contexte StreetView
  centralisés sous `corpus/` ;
- {inventory["massif_reference"]["commune_membership_rows"]:,} appartenances commune→massif
  COG 2021 ;
- aucune archive source ou fichier temporaire conservé.

Le training quatre classes n'est pas encore autorisé. Classes manquantes dans les lots réellement
annotés pour l'entraînement : {", ".join(gate["missing_training_classes"])}. Les médias Wikimedia
restent des candidats à annoter et à faire double-valider avant promotion.

Le corpus StreetView Global est une supervision de contexte faible (forêt, campagne, champs,
montagne) sous CC-BY-SA-4.0 : ses descriptions et catégories source ne sont jamais promues en
vérité terrain pour le détecteur, le pointage ou le recalage sans revue humaine.

Le corpus NASA HLS Burn Scars est réservé à un entraînement de segmentation binaire de zones
brûlées ; il ne modifie pas les classes du détecteur RT-DETR et ne prouve pas un front de feu actif.

Le fichier `dataset-index.json` est l'inventaire machine de référence. Le XLS officiel des massifs
est conservé pour la provenance, mais il ne contient pas de géométries polygonales.
"""
    readme_path = _ensure_within(root, root / "README.md")
    readme_path.write_text(readme, encoding="utf-8", newline="\n")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory the external FireWarning dataset root")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_inventory(args.root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
