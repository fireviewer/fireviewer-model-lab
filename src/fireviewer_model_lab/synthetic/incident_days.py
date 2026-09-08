"""Deterministic fictional French incident-day fixtures for pipeline evaluation."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

from fireviewer_model_lab.synthetic.artifacts import artifact, finalize_case_record, write_json


def _zone(seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    center_x = rng.uniform(420.0, 580.0)
    center_y = rng.uniform(330.0, 470.0)
    points: list[list[float]] = []
    for index in range(12):
        angle = math.tau * index / 12
        radius = rng.uniform(55.0, 125.0)
        points.append(
            [
                round(center_x + math.cos(angle) * radius, 3),
                round(center_y + math.sin(angle) * radius, 3),
            ]
        )
    points.append(points[0])
    return points


def _preview(
    path: Path,
    *,
    case_id: str,
    site_code: str,
    polygon: list[list[float]],
    accepted_count: int,
    rejected_count: int,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - exercised in the pod runtime
        raise RuntimeError("Pillow is required to render incident-day previews") from exc

    image = Image.new("RGB", (1024, 768), "#101416")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, 1024, 70), fill="#171d20")
    draw.text((28, 24), f"JOURNEE A-Z SYNTHETIQUE  {case_id}", fill="#f0f2f3", font=font)
    draw.text((758, 24), "AUCUN FAIT REEL", fill="#e56a2f", font=font)

    map_box = (28, 100, 650, 735)
    draw.rectangle(map_box, fill="#20282a", outline="#465155", width=2)
    for offset in range(0, 623, 52):
        draw.line((28 + offset, 100, 28 + offset, 735), fill="#2c373a")
    for offset in range(0, 636, 53):
        draw.line((28, 100 + offset, 650, 100 + offset), fill="#2c373a")
    scaled = [
        (28 + point[0] * 0.622, 100 + point[1] * 0.794)
        for point in polygon
    ]
    draw.polygon(scaled, fill="#7f322a", outline="#f07a3c")
    draw.text((46, 118), f"Calque zone de feu — {site_code}", fill="#c9d0d2", font=font)

    draw.rectangle((680, 100, 996, 735), fill="#171d20", outline="#465155", width=2)
    rows = [
        ("SOURCES RECUES", "4 paquets synthetiques"),
        ("RECHERCHE", "3 pistes tracees"),
        ("FAITS ACCEPTES", str(accepted_count)),
        ("FAITS REJETES", str(rejected_count)),
        ("CONTRADICTIONS", "2 groupes resolus"),
        ("STATUT", "revue humaine requise"),
    ]
    y = 128
    for heading, value in rows:
        draw.text((704, y), heading, fill="#8f9a9f", font=font)
        draw.text((704, y + 23), value, fill="#f0f2f3", font=font)
        draw.line((704, y + 52, 972, y + 52), fill="#323d41")
        y += 88
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def generate_incident_day(
    *,
    volume_root: Path,
    batch_root: Path,
    case_index: int,
    seed: int,
    production_stage: str,
) -> Path:
    """Create one new, explicitly fictional A-to-Z dossier and its review card."""
    case_started = time.perf_counter()
    rng = random.Random(seed)
    case_id = f"fid-{case_index:06d}"
    site_code = f"fr-syn-{rng.randrange(100000, 999999)}"
    case_root = (
        volume_root
        / "production"
        / "generated"
        / "france_incident_days"
        / case_id
    )
    case_root.mkdir(parents=True, exist_ok=True)
    polygon = _zone(seed)

    sources: dict[str, Any] = {
        "schema_version": 1,
        "fixture_kind": "fictional_synthetic_incident_day",
        "site_code": site_code,
        "warning": "synthetic fixture; not a real incident or public fact",
        "received": [
            {
                "source_id": f"src-{index + 1}",
                "channel": channel,
                "received_minute": minute,
                "claim": claim,
                "synthetic": True,
            }
            for index, (channel, minute, claim) in enumerate(
                [
                    ("codis_fixture", 0, "smoke report in synthetic sector alpha"),
                    ("camera_fixture", 7, "visible plume on generated frame"),
                    ("air_support_fixture", 16, "simulated resource requested"),
                    ("field_team_fixture", 24, "synthetic perimeter observation"),
                ]
            )
        ],
    }
    research = {
        "schema_version": 1,
        "fixture_kind": sources["fixture_kind"],
        "queries": [
            {
                "query_id": f"qry-{index + 1}",
                "question": question,
                "consulted_source_ids": source_ids,
                "result": result,
            }
            for index, (question, source_ids, result) in enumerate(
                [
                    ("Where is the generated ignition?", ["src-1", "src-2"], "candidate localized"),
                    ("Is a response asset engaged?", ["src-3"], "synthetic request only"),
                    ("Which perimeter is defensible?", ["src-2", "src-4"], "generated overlay retained"),
                ]
            )
        ],
    }
    accepted = [
        {
            "fact_id": "fact-accepted-1",
            "statement": "A generated smoke cue exists in sector alpha.",
            "evidence": ["src-1", "src-2"],
        },
        {
            "fact_id": "fact-accepted-2",
            "statement": "The displayed fire zone is the generator polygon.",
            "evidence": ["src-2", "src-4"],
        },
    ]
    if rng.random() > 0.5:
        accepted.append(
            {
                "fact_id": "fact-accepted-3",
                "statement": "The synthetic field cue is internally consistent.",
                "evidence": ["src-4"],
            }
        )
    rejected = [
        {
            "fact_id": "fact-rejected-1",
            "statement": "A real Canadair was operationally engaged.",
            "reason": "unsupported real-world claim; fixture only",
        },
        {
            "fact_id": "fact-rejected-2",
            "statement": "The generated polygon is an official fire perimeter.",
            "reason": "synthetic geometry cannot be an official perimeter",
        },
    ]
    facts = {
        "schema_version": 1,
        "fixture_kind": sources["fixture_kind"],
        "accepted": accepted,
        "rejected": rejected,
        "human_review_required": True,
    }
    contradictions = {
        "schema_version": 1,
        "fixture_kind": sources["fixture_kind"],
        "groups": [
            {
                "contradiction_id": "contradiction-1",
                "claims": ["synthetic request", "real engagement"],
                "resolution": "retain synthetic request; reject real engagement",
            },
            {
                "contradiction_id": "contradiction-2",
                "claims": ["generator polygon", "official perimeter"],
                "resolution": "retain generator polygon as evaluation overlay only",
            },
        ],
    }
    zone = {
        "type": "FeatureCollection",
        "name": f"{site_code}-synthetic-fire-zone",
        "synthetic": True,
        "real_world_claim": False,
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "kind": "synthetic_fire_zone_overlay",
                    "crs_profile": "local_enu_fictional_france_fixture",
                },
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
            }
        ],
    }

    source_path = case_root / "sources-received.json"
    research_path = case_root / "research-log.json"
    facts_path = case_root / "fact-ledger.json"
    contradiction_path = case_root / "contradictions.json"
    zone_path = case_root / "fire-zone-overlay.geojson"
    preview_path = case_root / "preview.png"
    write_json(source_path, sources)
    write_json(research_path, research)
    write_json(facts_path, facts)
    write_json(contradiction_path, contradictions)
    write_json(zone_path, zone)
    _preview(
        preview_path,
        case_id=case_id,
        site_code=site_code,
        polygon=polygon,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
    )

    record = {
        "schema_version": 1,
        "category": "france_incident_days",
        "case_id": case_id,
        "data_origin": "new_synthetic_generation",
        "production_stage": production_stage,
        "seed": seed,
        "preview_relpath": preview_path.relative_to(volume_root).as_posix(),
        "overlays": [],
        "artifacts": [
            artifact(volume_root, source_path, kind="source_packet"),
            artifact(volume_root, research_path, kind="research_log"),
            artifact(volume_root, facts_path, kind="fact_ledger"),
            artifact(volume_root, contradiction_path, kind="contradiction_log"),
            artifact(volume_root, zone_path, kind="fire_zone_overlay"),
            artifact(volume_root, preview_path, kind="preview"),
        ],
        "truth": {
            "synthetic": True,
            "real_world_claim": False,
            "fixture_kind": "fictional_synthetic_incident_day",
            "site_code": site_code,
            "facts_accepted": len(accepted),
            "facts_rejected": len(rejected),
            "contradictions": len(contradictions["groups"]),
        },
        "camera": {},
    }
    return finalize_case_record(
        batch_root=batch_root,
        record=record,
        started_monotonic=case_started,
    )
