"""Prepare and launch the local FireWarning Qwen3-VL fire-pointing LoRA pilot.

The command deliberately separates preparation, a no-update GPU probe, and the real run.  The
real optimizer loop is impossible to start without ``--confirm-training``.  Media, model weights,
annotations, adapters, and checkpoints stay below the user-provided dataset root and are never
placed in the public worker image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_LICENSE = "Apache-2.0"
MODEL_SAFETENSOR_BYTES = 8_875_719_344
ATTENTION_IMPLEMENTATION = "eager"
TRAINING_PRECISION = "bfloat16"
VRAM_LIMIT_BYTES = 14 * 1024**3
RAM_LIMIT_BYTES = 10 * 1024**3
MIN_PIXELS = 64 * 32 * 32
MAX_PIXELS = 256 * 32 * 32
POINTING_MANIFEST = Path("corpus/fire-pointing-v0.1.0/manifest.jsonl")
POINTING_REPORT = Path("corpus/fire-pointing-v0.1.0/build-report.json")
POINTING_CRITICAL = Path("corpus/fire-pointing-critical-v0.1.0/manifest.jsonl")
OUTPUT_RELATIVE = Path("training/qwen3-vl-4b-spatial")
DENIED_TOKENS = (
    "fireviewer-operational-reference",
    "operational-reference-a",
    "operational-reference-a",
)
CHECKPOINT_FRACTIONS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
POINTING_PROMPTS = {
    "fire_base": "base of the directly visible flames",
    "smoke_column_base": "observable base of the directly visible smoke column",
    "fire_response_vehicle": "directly visible fire-response vehicle",
    "firefighting_aircraft": "directly visible firefighting aircraft",
}


class TrainingSetupError(RuntimeError):
    """Raised when a local training invariant is not satisfied."""


@dataclass(frozen=True)
class Profile:
    name: str
    epochs: int
    learning_rate: float
    gradient_accumulation: int
    lora_rank: int = 8
    lora_alpha: int = 16
    validation_samples: int = 128


PROFILES = {
    "fire-pointing": Profile(
        name="fire-pointing",
        epochs=1,
        learning_rate=1e-6,
        gradient_accumulation=16,
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(_json_line(value), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for value in values:
            output.write(_json_line(value))
            count += 1
    os.replace(temporary, path)
    return count, _sha256_file(path)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise TrainingSetupError(f"missing JSONL: {path}")
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingSetupError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise TrainingSetupError(f"JSONL row is not an object at {path}:{line_number}")
            yield value


def _resolve_media(dataset_root: Path, relative: str, *, expected_sha256: str) -> Path:
    candidate = (dataset_root / Path(relative)).resolve()
    root = dataset_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise TrainingSetupError(f"media path escapes dataset root: {relative}")
    lowered = str(candidate).lower()
    if any(token in lowered for token in DENIED_TOKENS):
        raise TrainingSetupError(f"operational incident media denied: {candidate}")
    if not candidate.is_file():
        raise TrainingSetupError(f"missing media: {candidate}")
    if len(expected_sha256) != 64:
        raise TrainingSetupError(f"invalid SHA-256 contract for {candidate}")
    return candidate


def _pointing_annotations(dataset_root: Path) -> Iterator[dict[str, Any]]:
    for row in _iter_jsonl(dataset_root / POINTING_MANIFEST):
        split = str(row.get("proposed_split", ""))
        if split not in {"train", "validation"}:
            continue
        if row.get("pointing_status") != "point_candidate":
            continue
        if row.get("training_eligibility") != "weak_supervision_only_until_point_validation":
            continue
        source = row["source"]
        image_path = _resolve_media(
            dataset_root,
            str(source["image_relpath"]),
            expected_sha256=str(source["sha256"]),
        )
        targets = row.get("targets")
        if not isinstance(targets, list) or not targets:
            raise TrainingSetupError(f"point candidate has no target: {row.get('sample_id')}")
        for target in targets:
            anchor = str(target["semantic_anchor"])
            prompt_anchor = POINTING_PROMPTS.get(anchor)
            if prompt_anchor is None:
                raise TrainingSetupError(f"unsupported pointing anchor: {anchor}")
            point = target["source_pixel_normalized"]
            if not (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, int | float) and 0 <= value <= 1 for value in point)
            ):
                raise TrainingSetupError(f"invalid normalized point: {row.get('sample_id')}")
            answer = {
                "semantic_anchor": anchor,
                "source_pixel_normalized": [round(float(point[0]), 6), round(float(point[1]), 6)],
                "status": "ground_point",
                "uncertainty_codes": ["weak_bbox_bottom_center_unvalidated"],
            }
            yield {
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nLocate the "
                            f"{prompt_anchor}. Do not infer a geographic coordinate. "
                            "Return exactly one compact JSON object with the keys status, "
                            "semantic_anchor, source_pixel_normalized, and uncertainty_codes."
                        ),
                    },
                    {"from": "gpt", "value": json.dumps(answer, separators=(",", ":"))},
                ],
                "firewarning": {
                    "family": "fire_pointing",
                    "label_strength": "weak_supervision",
                    "license": source["license"],
                    "media_sha256": [source["sha256"]],
                    "sample_id": row["sample_id"],
                    "split": split,
                    "split_group": row["split_group"],
                    "target_id": target["target_id"],
                },
                "id": f"{row['sample_id']}:{target['target_id']}",
                "image": str(image_path),
            }


def _split_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for record in records:
        split = record["firewarning"]["split"]
        if split == "train":
            train.append(record)
        elif split == "validation":
            validation.append(record)
        else:  # pragma: no cover - builders constrain this before the split
            raise TrainingSetupError(f"unexpected prepared split: {split}")
    return train, validation


def _dataset_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    anchors: Counter[str] = Counter()
    groups: set[str] = set()
    for record in records:
        metadata = record["firewarning"]
        groups.add(str(metadata["split_group"]))
        answer = json.loads(record["conversations"][1]["value"])
        anchors[str(answer["semantic_anchor"])] += 1
    return {
        "anchor_counts": dict(sorted(anchors.items())),
        "rows": len(records),
        "split_groups": len(groups),
    }


def prepare(dataset_root: Path, *, verify_hashes: bool) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    report_path = dataset_root / POINTING_REPORT
    if not report_path.is_file():
        raise TrainingSetupError(f"missing corpus report: {report_path}")
    corpus_report = json.loads(report_path.read_text(encoding="utf-8"))
    if corpus_report.get("gates", {}).get("training_ready") is not True:
        raise TrainingSetupError(f"corpus training gate is false: {report_path}")

    pointing_train, pointing_validation = _split_records(_pointing_annotations(dataset_root))
    if not pointing_train or not pointing_validation:
        raise TrainingSetupError("pointing train and validation annotations are required")

    if verify_hashes:
        unique_media: dict[str, Path] = {}
        for record in (*pointing_train, *pointing_validation):
            images = record["image"] if isinstance(record["image"], list) else [record["image"]]
            expected_hashes = record["firewarning"]["media_sha256"]
            if len(images) != len(expected_hashes):
                raise TrainingSetupError(f"prepared media hash count mismatch: {record['id']}")
            for expected, image in zip(expected_hashes, images, strict=True):
                unique_media[str(expected)] = Path(image)
        for expected_sha256, path in unique_media.items():
            if _sha256_file(path) != expected_sha256:
                raise TrainingSetupError(f"prepared media SHA-256 mismatch: {path}")

    output_root = dataset_root / OUTPUT_RELATIVE
    annotations_root = output_root / "annotations"
    files: dict[str, dict[str, Any]] = {}
    datasets = {
        "fire-pointing-train": pointing_train,
        "fire-pointing-validation": pointing_validation,
    }
    for name, records in datasets.items():
        path = annotations_root / f"{name}.jsonl"
        count, sha256 = _write_jsonl(path, records)
        files[name] = {
            "relpath": path.relative_to(dataset_root).as_posix(),
            "rows": count,
            "sha256": sha256,
        }

    report = {
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "critical_lots_included": False,
        "datasets": {
            "fire_pointing": {
                "label_strength": "weak_bbox_bottom_center_unvalidated",
                "production_ready": False,
                "train": _dataset_summary(pointing_train),
                "validation": _dataset_summary(pointing_validation),
            },
        },
        "files": files,
        "gates": {
            "bootstrap_training_ready": True,
            "critical_lots_excluded": True,
            "no_group_leak": True,
            "production_training_ready": False,
        },
        "model": {
            "id": MODEL_ID,
            "license": MODEL_LICENSE,
            "revision": MODEL_REVISION,
            "safetensor_bytes": MODEL_SAFETENSOR_BYTES,
        },
        "precision": TRAINING_PRECISION,
        "resource_limits": {
            "host_ram_bytes": RAM_LIMIT_BYTES,
            "vram_bytes": VRAM_LIMIT_BYTES,
        },
        "schema_version": 1,
        "storage_policy": "local_only_no_docker_image_no_docker_volume",
        "verify_hashes": verify_hashes,
    }
    assignments: dict[str, set[str]] = {}
    for record in (*pointing_train, *pointing_validation):
        metadata = record["firewarning"]
        assignments.setdefault(metadata["split_group"], set()).add(metadata["split"])
    if any(len(splits) != 1 for splits in assignments.values()):
        raise TrainingSetupError("prepared split-group leak")
    _write_json(output_root / "prepare-report.json", report)
    return report


def download_model(dataset_root: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - explicit environment failure
        raise TrainingSetupError("install huggingface-hub before downloading the model") from exc
    dataset_root = dataset_root.resolve()
    model_root = dataset_root / "models" / "huggingface-cache"
    required_free_bytes = MODEL_SAFETENSOR_BYTES + 5 * 1024**3
    if shutil.disk_usage(dataset_root).free < required_free_bytes:
        raise TrainingSetupError(
            "insufficient disk space for the pinned 4B snapshot and 5 GiB reserve"
        )
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise TrainingSetupError(f"model revision mismatch: {info.sha}")
    license_id = str(info.card_data.license) if info.card_data is not None else ""
    if license_id.lower() != MODEL_LICENSE.lower():
        raise TrainingSetupError(f"model license mismatch: {license_id}")
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=model_root,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.model",
                "*.py",
                "*.safetensors",
                "*.tiktoken",
                "*.txt",
            ],
        )
    )
    safetensors = sorted(snapshot.glob("*.safetensors"))
    safetensor_bytes = sum(path.stat().st_size for path in safetensors)
    if safetensor_bytes != MODEL_SAFETENSOR_BYTES:
        raise TrainingSetupError(
            f"unexpected safetensor bytes: {safetensor_bytes} != {MODEL_SAFETENSOR_BYTES}"
        )
    report = {
        "files": [path.name for path in safetensors],
        "license": MODEL_LICENSE,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "safetensor_bytes": safetensor_bytes,
        "snapshot": str(snapshot),
        "storage_policy": "local_only_no_docker_image_no_docker_volume",
    }
    _write_json(dataset_root / OUTPUT_RELATIVE / "model-download-report.json", report)
    return report


def _require_training_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import psutil
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - explicit local environment failure
        raise TrainingSetupError("install the spatial-train dependencies") from exc
    return torch, transformers, psutil, (LoraConfig, get_peft_model)


def hardware_preflight() -> dict[str, Any]:
    torch, transformers, psutil, _ = _require_training_runtime()
    if not torch.cuda.is_available():
        raise TrainingSetupError("CUDA is unavailable")
    if torch.version.cuda != "13.0":
        raise TrainingSetupError(f"CUDA 13.0 wheel required, got {torch.version.cuda}")
    if not torch.cuda.is_bf16_supported():
        raise TrainingSetupError("BF16 is not supported by the local GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):
        raise TrainingSetupError(f"local train expects Blackwell sm_120, got {capability}")
    total_vram = torch.cuda.get_device_properties(0).total_memory
    if total_vram < VRAM_LIMIT_BYTES:
        raise TrainingSetupError("GPU has less memory than the configured 14 GiB cap")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    query = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    scores = torch.matmul(query, key.transpose(-2, -1)) * (64**-0.5)
    output = torch.matmul(torch.softmax(scores.float(), dim=-1).to(torch.bfloat16), value)
    output.float().square().mean().backward()
    if not bool(torch.isfinite(output).all().item()):
        raise TrainingSetupError("BF16 eager attention probe produced non-finite output")
    report = {
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "bf16_eager_forward_backward": True,
        "capability": list(capability),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "host_available_bytes": psutil.virtual_memory().available,
        "host_ram_limit_bytes": RAM_LIMIT_BYTES,
        "peak_probe_vram_bytes": torch.cuda.max_memory_reserved(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vram_limit_bytes": VRAM_LIMIT_BYTES,
        "vram_total_bytes": total_vram,
    }
    del output, scores, value, key, query
    torch.cuda.empty_cache()
    return report


def _load_prepared(path: Path) -> list[dict[str, Any]]:
    records = list(_iter_jsonl(path))
    if not records:
        raise TrainingSetupError(f"prepared annotations are empty: {path}")
    return records


def _messages(record: dict[str, Any], *, include_answer: bool) -> list[dict[str, Any]]:
    images = record["image"] if isinstance(record["image"], list) else [record["image"]]
    prompt = str(record["conversations"][0]["value"])
    parts = prompt.split("<image>")
    content: list[dict[str, Any]] = []
    if len(parts) - 1 != len(images):
        raise TrainingSetupError(f"image tag count mismatch: {record.get('id')}")
    if parts[0].strip():
        content.append({"type": "text", "text": parts[0].strip()})
    for image, trailing in zip(images, parts[1:], strict=True):
        content.append(
            {
                "type": "image",
                "image": Path(image).as_uri(),
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            }
        )
        if trailing.strip():
            content.append({"type": "text", "text": trailing.strip()})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if include_answer:
        messages.append({"role": "assistant", "content": record["conversations"][1]["value"]})
    return messages


def _encode_record(processor: Any, record: dict[str, Any], torch: Any) -> dict[str, Any]:
    full_messages = _messages(record, include_answer=True)
    prompt_messages = _messages(record, include_answer=False)
    full = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    labels = full["input_ids"].clone()
    prompt_length = prompt["input_ids"].shape[1]
    labels[:, :prompt_length] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    full["labels"] = labels
    return {
        key: value.to("cuda") if torch.is_tensor(value) else value for key, value in full.items()
    }


def _load_model_and_adapter(
    *, model_root: Path, profile: Profile
) -> tuple[Any, Any, Any, Any, Any]:
    torch, transformers, psutil, peft_parts = _require_training_runtime()
    LoraConfig, get_peft_model = peft_parts
    snapshot = model_root / f"models--Qwen--Qwen3-VL-4B-Instruct/snapshots/{MODEL_REVISION}"
    if not snapshot.is_dir():
        raise TrainingSetupError(f"pinned model snapshot is missing: {snapshot}")
    processor = transformers.AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    process = psutil.Process()
    if process.memory_info().rss > RAM_LIMIT_BYTES:
        raise TrainingSetupError("host RAM cap exceeded before model load")
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    lora = LoraConfig(
        r=profile.lora_rank,
        lora_alpha=profile.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    if torch.cuda.memory_reserved() > VRAM_LIMIT_BYTES:
        raise TrainingSetupError("VRAM cap exceeded by the BF16 base model and LoRA")
    if process.memory_info().rss > RAM_LIMIT_BYTES:
        raise TrainingSetupError("host RAM cap exceeded by the BF16 base model and LoRA")
    return torch, processor, model, process, psutil


def _resource_snapshot(torch: Any, process: Any) -> dict[str, int]:
    snapshot = {
        "ram_rss_bytes": int(process.memory_info().rss),
        "vram_allocated_bytes": int(torch.cuda.memory_allocated()),
        "vram_reserved_bytes": int(torch.cuda.memory_reserved()),
        "vram_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    if snapshot["ram_rss_bytes"] > RAM_LIMIT_BYTES:
        raise TrainingSetupError("10 GiB host RAM cap exceeded")
    if snapshot["vram_reserved_bytes"] > VRAM_LIMIT_BYTES:
        raise TrainingSetupError("14 GiB VRAM cap exceeded")
    return snapshot


def _evaluation_loss(
    model: Any,
    processor: Any,
    records: Sequence[dict[str, Any]],
    torch: Any,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for record in records:
            inputs = _encode_record(processor, record, torch)
            output = model(**inputs)
            if output.loss is None or not bool(torch.isfinite(output.loss).item()):
                raise TrainingSetupError("validation loss is missing or non-finite")
            losses.append(float(output.loss.detach().cpu()))
            del inputs, output
    model.train()
    return sum(losses) / len(losses)


def probe(dataset_root: Path, *, profile_name: str) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    output_root = dataset_root.resolve() / OUTPUT_RELATIVE
    annotations = output_root / "annotations" / f"{profile_name}-train.jsonl"
    first = next(_iter_jsonl(annotations))
    torch, processor, model, process, _ = _load_model_and_adapter(
        model_root=dataset_root.resolve() / "models/huggingface-cache",
        profile=profile,
    )
    torch.cuda.reset_peak_memory_stats()
    model.train()
    inputs = _encode_record(processor, first, torch)
    output = model(**inputs)
    if output.loss is None or not bool(torch.isfinite(output.loss).item()):
        raise TrainingSetupError("probe loss is missing or non-finite")
    output.loss.backward()
    resources = _resource_snapshot(torch, process)
    report = {
        "backward_completed": True,
        "loss": float(output.loss.detach().cpu()),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "no_optimizer_step": True,
        "profile": profile_name,
        "resources": resources,
        "training_started": False,
    }
    _write_json(output_root / f"probe-{profile_name}.json", report)
    return report


def _checkpoint_due_steps(total_optimizer_steps: int) -> dict[int, int]:
    return {
        min(total_optimizer_steps, max(1, math.ceil(total_optimizer_steps * fraction))): int(
            fraction * 100
        )
        for fraction in CHECKPOINT_FRACTIONS
    }


def train(
    dataset_root: Path,
    *,
    profile_name: str,
    confirm_training: bool,
    seed: int,
) -> dict[str, Any]:
    if not confirm_training:
        raise TrainingSetupError(
            "real training is locked; pass --confirm-training only after explicit approval"
        )
    profile = PROFILES[profile_name]
    dataset_root = dataset_root.resolve()
    output_root = dataset_root / OUTPUT_RELATIVE
    train_records = _load_prepared(output_root / "annotations" / f"{profile_name}-train.jsonl")
    validation_records = _load_prepared(
        output_root / "annotations" / f"{profile_name}-validation.jsonl"
    )
    if len(validation_records) > profile.validation_samples:
        validation_records = random.Random(seed).sample(  # noqa: S311 - deterministic evaluation
            validation_records, profile.validation_samples
        )
    torch, processor, model, process, _ = _load_model_and_adapter(
        model_root=dataset_root / "models/huggingface-cache",
        profile=profile,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=profile.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    microsteps_total = len(train_records) * profile.epochs
    optimizer_steps_total = math.ceil(microsteps_total / profile.gradient_accumulation)
    due_steps = _checkpoint_due_steps(optimizer_steps_total)
    optimizer_step = 0
    microstep = 0
    started = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(profile.epochs):
        order = list(range(len(train_records)))
        random.Random(seed + epoch).shuffle(order)  # noqa: S311 - reproducible training order
        for index in order:
            inputs = _encode_record(processor, train_records[index], torch)
            output = model(**inputs)
            if output.loss is None or not bool(torch.isfinite(output.loss).item()):
                raise TrainingSetupError(f"non-finite loss at microstep {microstep + 1}")
            (output.loss / profile.gradient_accumulation).backward()
            microstep += 1
            should_step = (
                microstep % profile.gradient_accumulation == 0 or microstep == microsteps_total
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                resources = _resource_snapshot(torch, process)
                if optimizer_step in due_steps:
                    percent = due_steps[optimizer_step]
                    checkpoint = (
                        output_root / "checkpoints" / profile_name / f"checkpoint-{percent:03d}pct"
                    )
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint, safe_serialization=True)
                    validation_loss = _evaluation_loss(model, processor, validation_records, torch)
                    _write_json(
                        checkpoint / "firewarning-state.json",
                        {
                            "elapsed_seconds": time.time() - started,
                            "microstep": microstep,
                            "optimizer_step": optimizer_step,
                            "profile": profile_name,
                            "resources": resources,
                            "total_optimizer_steps": optimizer_steps_total,
                            "validation_loss": validation_loss,
                            "validation_samples": len(validation_records),
                        },
                    )
            del inputs, output
    final_root = output_root / "adapters" / profile_name
    final_root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_root, safe_serialization=True)
    report = {
        "elapsed_seconds": time.time() - started,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "optimizer_steps": optimizer_step,
        "profile": profile_name,
        "resources": _resource_snapshot(torch, process),
        "training_completed": True,
    }
    _write_json(output_root / f"train-report-{profile_name}.json", report)
    return report


def launch_plan(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_root = dataset_root / OUTPUT_RELATIVE
    prepare_report_path = output_root / "prepare-report.json"
    if not prepare_report_path.is_file():
        raise TrainingSetupError("run prepare before creating the launch plan")
    prepare_report = json.loads(prepare_report_path.read_text(encoding="utf-8"))
    hardware = hardware_preflight()
    profiles: dict[str, Any] = {}
    for name, profile in PROFILES.items():
        rows = int(prepare_report["files"][f"{name}-train"]["rows"])
        microsteps = rows * profile.epochs
        optimizer_steps = math.ceil(microsteps / profile.gradient_accumulation)
        profiles[name] = {
            "checkpoint_steps": _checkpoint_due_steps(optimizer_steps),
            "epochs": profile.epochs,
            "gradient_accumulation": profile.gradient_accumulation,
            "learning_rate": profile.learning_rate,
            "lora_alpha": profile.lora_alpha,
            "lora_rank": profile.lora_rank,
            "microsteps": microsteps,
            "optimizer_steps": optimizer_steps,
            "train_rows": rows,
            "validation_samples_per_checkpoint": min(
                profile.validation_samples,
                int(prepare_report["files"][f"{name}-validation"]["rows"]),
            ),
        }
    plan = {
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "command_locked_without_confirm_training": True,
        "hardware": hardware,
        "model": {
            "id": MODEL_ID,
            "license": MODEL_LICENSE,
            "revision": MODEL_REVISION,
        },
        "precision": TRAINING_PRECISION,
        "profiles": profiles,
        "resource_limits": {
            "host_ram_bytes": RAM_LIMIT_BYTES,
            "vram_bytes": VRAM_LIMIT_BYTES,
        },
        "storage_policy": "local_only_no_docker_image_no_docker_volume",
        "training_started": False,
    }
    _write_json(output_root / "launch-plan.json", plan)
    return plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "preflight", "launch-plan", "download-model"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--dataset-root", type=Path, required=True)
        if command == "prepare":
            subparser.add_argument("--verify-hashes", action="store_true")
    for command in ("probe", "train"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--dataset-root", type=Path, required=True)
        subparser.add_argument("--profile", choices=sorted(PROFILES), required=True)
        if command == "train":
            subparser.add_argument("--confirm-training", action="store_true")
            subparser.add_argument("--seed", type=int, default=42017)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "prepare":
        report = prepare(args.dataset_root, verify_hashes=args.verify_hashes)
    elif args.command == "download-model":
        report = download_model(args.dataset_root)
    elif args.command == "preflight":
        report = hardware_preflight()
    elif args.command == "launch-plan":
        report = launch_plan(args.dataset_root)
    elif args.command == "probe":
        report = probe(args.dataset_root, profile_name=args.profile)
    elif args.command == "train":
        report = train(
            args.dataset_root,
            profile_name=args.profile,
            confirm_training=args.confirm_training,
            seed=args.seed,
        )
    else:  # pragma: no cover - argparse rejects unknown commands
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingSetupError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
