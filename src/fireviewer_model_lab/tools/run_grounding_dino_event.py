from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from fireviewer_contracts.mvp.contracts import DetectionResultV1, EvidenceMedia
from fireviewer_geolocation.mvp.localization import LocalEvidenceImageLoader
from fireviewer_orchestrator.mvp.orchestration import CorpusEventRuntimeInput
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth
from fireviewer_vision_runtime.mvp.vision import (
    EventVisionConfig,
    EventVisionRunner,
    GroundingDinoConfig,
    GroundingDinoVisionProvider,
    LocalGroundingDinoBundleManifest,
    LocalGroundingDinoModelLoader,
    vision_result_reference,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached Grounding DINO fire/smoke detection over one event."
    )
    parser.add_argument("--runtime-input", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-media", type=int, default=256)
    return parser.parse_args()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _serialized(model: object) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json", by_alias=True),  # type: ignore[attr-defined]
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


class _CachedVisionProvider:
    def __init__(self, provider: GroundingDinoVisionProvider, directory: Path) -> None:
        self.provider = provider
        self.directory = directory
        self.descriptor: ProviderDescriptor = provider.descriptor
        self.cache_hits = 0
        self.inferences = 0

    def healthcheck(self) -> ProviderHealth:
        return self.provider.healthcheck()

    def detect(self, media: EvidenceMedia) -> DetectionResultV1:
        reference = vision_result_reference(self, media.media_id, media.sha256)
        path = self.directory / f"{reference}.detection.json"
        legacy_path = self.directory / f"{media.media_id}.detection.json"
        for candidate in (path, legacy_path):
            if not candidate.is_file():
                continue
            cached = DetectionResultV1.model_validate_json(candidate.read_text(encoding="utf-8"))
            run = cached.provider_run
            if (
                cached.media_id == media.media_id
                and run.input_hash == media.sha256
                and run.provider_id == self.descriptor.provider_id
                and run.provider_version == self.descriptor.provider_version
                and run.model_id == self.descriptor.model_id
                and run.model_version == self.descriptor.model_version
                and run.config == self.descriptor.config
            ):
                if candidate != path:
                    _atomic_write(path, _serialized(cached))
                self.cache_hits += 1
                return cached
        result = self.provider.detect(media)
        _atomic_write(path, _serialized(result))
        self.inferences += 1
        return result


def main() -> int:
    args = _arguments()
    runtime_input = CorpusEventRuntimeInput.model_validate_json(
        args.runtime_input.read_text(encoding="utf-8")
    )
    manifest = LocalGroundingDinoBundleManifest.model_validate_json(
        args.model_manifest.read_text(encoding="utf-8")
    )
    provider = GroundingDinoVisionProvider(
        image_loader=LocalEvidenceImageLoader(
            root=args.repository_root,
            relative_paths_by_media_id=runtime_input.relative_paths_by_media_id,
        ),
        model_loader=LocalGroundingDinoModelLoader(
            directory=args.model_directory,
            manifest=manifest,
        ),
        model_version=manifest.revision,
        config=GroundingDinoConfig(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        ),
    )
    health = provider.healthcheck()
    if health.status != "healthy":
        raise RuntimeError(f"Grounding DINO provider is {health.status}: {health.reason_codes}")
    cached_provider = _CachedVisionProvider(provider, args.output / "detections")
    event_run = EventVisionRunner(
        provider=cached_provider,
        config=EventVisionConfig(max_media=args.max_media),
    ).run(runtime_input.evidence)
    event_path = args.output / f"{event_run.evidence.event_id}.event-evidence.json"
    _atomic_write(event_path, _serialized(event_run.evidence))
    statuses = Counter(artifact.result.status for artifact in event_run.artifacts)
    print(
        json.dumps(
            {
                "cache_hits": cached_provider.cache_hits,
                "detections": sum(
                    len(artifact.result.detections) for artifact in event_run.artifacts
                ),
                "event_id": event_run.evidence.event_id,
                "inferences": cached_provider.inferences,
                "media_processed": len(event_run.artifacts),
                "needs_human_review": event_run.evidence.needs_human_review,
                "output": str(event_path),
                "statuses": dict(sorted(statuses.items())),
                "uncertainty_codes": [item.code for item in event_run.evidence.uncertainties],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
