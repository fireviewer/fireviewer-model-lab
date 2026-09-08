from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import requests

BUFFER_SIZE = 4 * 1024 * 1024
CFDB_LANDING_PAGE = "https://cfdb.univ-corse.fr/"
CFDB_PROCEDURE_PAGE = "https://cfdb.univ-corse.fr/demarches_page_136_menu,3.htm"
CFDB_AGREEMENT_URL = (
    "https://cfdb.univ-corse.fr/catalog_repository/uploads/13/"
    "Accord_licence_Corsican_Fire_Database.pdf"
)
MCPED_ARTICLE_ID = 28_528_868
MCPED_API_URL = f"https://api.figshare.com/v2/articles/{MCPED_ARTICLE_ID}"
MCPED_LANDING_PAGE = (
    "https://springernature.figshare.com/articles/dataset/"
    "A_multi-modality_ground-to-air_cross-view_pose_estimation_dataset_for_field_robots/28528868"
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_corsican_request(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    agreement = output_root / "Accord_licence_Corsican_Fire_Database.pdf"
    response = requests.get(CFDB_AGREEMENT_URL, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF-"):
        raise ValueError("Corsican Fire Database agreement is not a PDF")
    agreement.write_bytes(response.content)
    status = {
        "schema_version": 1,
        "source_id": "corsican-fire-database-access-request-v1",
        "landing_page": CFDB_LANDING_PAGE,
        "procedure_page": CFDB_PROCEDURE_PAGE,
        "agreement_url": CFDB_AGREEMENT_URL,
        "agreement": {
            "path": agreement.name,
            "size_bytes": agreement.stat().st_size,
            "sha256": _sha256(agreement),
        },
        "status": "awaiting_human_identity_signature_and_submission",
        "submission_address": "lrossi@univ-corse.fr",
        "automatically_sent": False,
        "blocking_reasons": [
            "Applicant identity and institutional details are not provided",
            "The license agreement requires a human signature",
            "No authorization was given to send email on the applicant's behalf",
        ],
        "next_human_action": (
            "Complete and sign the official PDF, then send it to the address stated by the "
            "official procedure. Store the received credentials outside the dataset package."
        ),
    }
    (output_root / "REQUEST_STATUS.json").write_bytes(_canonical_json_bytes(status))
    return status


def _classify_mcped_file(name: str) -> tuple[str, str]:
    relative = PurePosixPath(name)
    if relative.name != name or ".." in relative.parts:
        raise ValueError(f"Unsafe McPed filename: {name}")
    lower = name.lower()
    if lower.endswith((".png", ".jpg", ".jpeg")):
        return (
            "blocked_from_republication",
            "standalone_satellite_or_undocumented_preview_requires_rights_audit",
        )
    if lower.endswith("image.zip"):
        return "candidate_private_audit", "ground_view_archive_per_official_description"
    if lower.endswith("lidar.zip"):
        return "candidate_private_audit", "ground_lidar_archive"
    if lower.endswith("npy.zip"):
        return "candidate_private_audit", "top_view_ground_array_requires_content_audit"
    return "candidate_private_audit", "labels_calibration_or_documentation"


def prepare_mcped_quarantine(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    response = requests.get(MCPED_API_URL, timeout=60)
    response.raise_for_status()
    article = response.json()
    if int(article.get("id")) != MCPED_ARTICLE_ID or not article.get("files"):
        raise ValueError("Unexpected McPed Figshare metadata")
    files = []
    counts: dict[str, int] = {}
    bytes_by_status: dict[str, int] = {}
    for row in article["files"]:
        status, reason = _classify_mcped_file(str(row["name"]))
        counts[status] = counts.get(status, 0) + 1
        bytes_by_status[status] = bytes_by_status.get(status, 0) + int(row["size"])
        files.append(
            {
                "figshare_file_id": int(row["id"]),
                "name": str(row["name"]),
                "size_bytes": int(row["size"]),
                "md5": str(row["computed_md5"]),
                "download_url": str(row["download_url"]),
                "quarantine_status": status,
                "reason": reason,
                "downloaded": False,
                "republish_allowed": False,
            }
        )
    inventory = {
        "schema_version": 1,
        "source_id": "mcped-technical-quarantine-v1",
        "title": str(article["title"]),
        "landing_page": MCPED_LANDING_PAGE,
        "figshare_api": MCPED_API_URL,
        "figshare_version": int(article["version"]),
        "doi": str(article["doi"]),
        "record_license": article["license"],
        "status": "technical_and_rights_quarantine",
        "policy": {
            "google_earth_or_satellite_views_must_not_be_republished": True,
            "standalone_png_jpg_previews_must_not_be_downloaded_for_public_bundle": True,
            "ground_archives_require_content_and_duplicate_audit_before_download": True,
            "no_file_is_approved_for_training_or_republication_yet": True,
        },
        "counts_by_status": dict(sorted(counts.items())),
        "bytes_by_status": dict(sorted(bytes_by_status.items())),
        "files": files,
    }
    (output_root / "QUARANTINE_MANIFEST.json").write_bytes(_canonical_json_bytes(inventory))
    (output_root / "FIGSHARE_METADATA.json").write_bytes(_canonical_json_bytes(article))
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare controlled-access and quarantined supplemental sources."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    corsican = subparsers.add_parser("corsican-request")
    corsican.add_argument("--output-root", type=Path, required=True)
    mcped = subparsers.add_parser("mcped-quarantine")
    mcped.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "corsican-request":
        report = prepare_corsican_request(args.output_root.resolve())
    elif args.command == "mcped-quarantine":
        report = prepare_mcped_quarantine(args.output_root.resolve())
    else:
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
