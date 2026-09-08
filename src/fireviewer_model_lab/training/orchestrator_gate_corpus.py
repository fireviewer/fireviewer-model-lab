from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from fireviewer_contracts.stage_contracts import (
    StageCapability,
    StageContract,
    StageRole,
    load_stage_contract_registry,
)
from fireviewer_orchestrator.stage_gates import StageGateEngine

BUNDLE_ID = "orchestrator-gates-sft-v1"
LICENSE = "AGPL-3.0-or-later"
ALLOWED_DECISIONS = {
    "pass",
    "repair",
    "adjudicated",
    "not_applicable",
    "abstain",
    "human_review",
    "failed_retryable",
    "failed_terminal",
}


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return (rendered + "\n").encode("utf-8")


def _stable_id(*parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"OG-{digest}"


def _split(group_id: str) -> str:
    bucket = int(sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "validation"
    if bucket == 1:
        return "test"
    return "train"


def _ordered(capabilities: set[StageCapability]) -> list[str]:
    return sorted(capability.value for capability in capabilities)


def _required_capabilities(contract: StageContract) -> set[StageCapability]:
    required = set(contract.required_all)
    if contract.required_any:
        required.add(contract.required_any[0])
    return required


def _gate_record(
    *,
    contract: StageContract,
    scenario: str,
    inputs: dict[str, object],
    expected: object,
    contract_digest: str,
) -> dict[str, object]:
    group_id = f"stage-gate:{contract.role.value}:{scenario}"
    return {
        "record_id": _stable_id(group_id),
        "task": "stage_gate_decision",
        "group_id": group_id,
        "split": _split(group_id),
        "stage_role": contract.role.value,
        "scenario": scenario,
        "contract_digest": contract_digest,
        "input": {
            "contract": contract.model_dump(mode="json"),
            **inputs,
        },
        "expected": expected,
        "provenance": {
            "kind": "deterministic_code_execution",
            "generator": "fireviewer_model_lab.training.orchestrator_gate_corpus",
            "license": LICENSE,
        },
    }


def _stage_gate_records() -> tuple[list[dict[str, object]], str]:
    registry = load_stage_contract_registry()
    engine = StageGateEngine()
    records: list[dict[str, object]] = []

    for role in sorted(registry, key=lambda value: value.value):
        contract = registry[role]
        required = _required_capabilities(contract)
        preflight_cases: list[tuple[str, set[StageCapability], int]] = [
            ("requirements_satisfied", required, 1),
            ("input_limit_exceeded", required, contract.max_input_items + 1),
        ]
        if contract.required_all or contract.required_any:
            preflight_cases.append(("required_capability_missing", set(), 1))

        for scenario, capabilities, input_items in preflight_cases:
            expected = engine.preflight(
                contract,
                frozenset(capabilities),
                input_items=input_items,
            )
            records.append(
                _gate_record(
                    contract=contract,
                    scenario=f"preflight_{scenario}",
                    inputs={
                        "phase": "preflight",
                        "available_capabilities": _ordered(capabilities),
                        "input_items": input_items,
                    },
                    expected=expected.model_dump(mode="json"),
                    contract_digest=registry.digest,
                )
            )

        before = _required_capabilities(contract)
        produced = contract.minimum_output_any[0]
        if produced in before:
            before.remove(produced)
        postflight_cases = (
            (
                "minimum_output_satisfied",
                "succeeded",
                None,
                min(1.0, float(contract.max_wall_time_seconds)),
                min(1, contract.max_output_items_per_input),
                before | {produced},
            ),
            (
                "minimum_output_missing",
                "succeeded",
                None,
                min(1.0, float(contract.max_wall_time_seconds)),
                0,
                before,
            ),
            (
                "model_runtime_error",
                "failed",
                "model_runtime_error",
                min(1.0, float(contract.max_wall_time_seconds)),
                0,
                before,
            ),
            (
                "invalid_model_output",
                "failed",
                "invalid_model_output",
                min(1.0, float(contract.max_wall_time_seconds)),
                0,
                before,
            ),
            (
                "wall_time_exceeded",
                "succeeded",
                None,
                float(contract.max_wall_time_seconds + 1),
                min(1, contract.max_output_items_per_input),
                before | {produced},
            ),
            (
                "output_limit_exceeded",
                "succeeded",
                None,
                min(1.0, float(contract.max_wall_time_seconds)),
                contract.max_output_items_per_input + 1,
                before | {produced},
            ),
            (
                "skipped_not_applicable",
                "skipped",
                None,
                0.0,
                0,
                before,
            ),
            (
                "skipped_explicit_abstention",
                "skipped",
                "explicit_abstention",
                0.0,
                0,
                before,
            ),
        )
        for scenario, status, error_code, elapsed, output_count, after in postflight_cases:
            expected = engine.postflight(
                contract,
                before=frozenset(before),
                after=frozenset(after),
                status=status,  # type: ignore[arg-type]
                error_code=error_code,
                elapsed_seconds=elapsed,
                maximum_output_items=output_count,
            )
            records.append(
                _gate_record(
                    contract=contract,
                    scenario=f"postflight_{scenario}",
                    inputs={
                        "phase": "postflight",
                        "before_capabilities": _ordered(before),
                        "after_capabilities": _ordered(after),
                        "status": status,
                        "error_code": error_code,
                        "elapsed_seconds": elapsed,
                        "maximum_output_items": output_count,
                    },
                    expected=expected.model_dump(mode="json"),
                    contract_digest=registry.digest,
                )
            )
    return records, registry.digest


def _consensus_records() -> list[dict[str, object]]:
    cases: tuple[dict[str, Any], ...] = (
        {
            "scenario": "quorum_agreement",
            "strategy": "quorum",
            "candidates": [
                {"id": "primary", "status": "succeeded", "value": "fumée visible"},
                {"id": "challenger", "status": "succeeded", "value": "Fumée visible."},
            ],
            "agreement_score": 1.0,
            "expected": {
                "decision": "pass",
                "selected_candidate_id": "primary",
                "invoke_final_judge": False,
                "downstream_allowed": True,
            },
        },
        {
            "scenario": "quorum_contradiction_adjudicated",
            "strategy": "quorum",
            "candidates": [
                {"id": "primary", "status": "succeeded", "value": "fumée visible"},
                {"id": "challenger", "status": "succeeded", "value": "aucun signe"},
            ],
            "agreement_score": 0.0,
            "judge": {"selected_candidate_id": "challenger", "confidence": 0.93},
            "expected": {
                "decision": "adjudicated",
                "selected_candidate_id": "challenger",
                "invoke_final_judge": True,
                "downstream_allowed": True,
            },
        },
        {
            "scenario": "quorum_contradiction_low_judge_confidence",
            "strategy": "quorum",
            "candidates": [
                {"id": "primary", "status": "succeeded", "value": "fumée visible"},
                {"id": "challenger", "status": "succeeded", "value": "aucun signe"},
            ],
            "agreement_score": 0.0,
            "judge": {"selected_candidate_id": "challenger", "confidence": 0.2},
            "expected": {
                "decision": "human_review",
                "selected_candidate_id": None,
                "invoke_final_judge": True,
                "downstream_allowed": False,
            },
        },
        {
            "scenario": "visual_contradiction_without_raw_evidence",
            "strategy": "quorum",
            "candidates": [
                {"id": "primary", "status": "succeeded", "value": "point A"},
                {"id": "challenger", "status": "succeeded", "value": "point B"},
            ],
            "agreement_score": 0.0,
            "judge": {"raw_visual_evidence_available": False},
            "expected": {
                "decision": "abstain",
                "selected_candidate_id": None,
                "invoke_final_judge": True,
                "downstream_allowed": False,
            },
        },
        {
            "scenario": "cascade_primary_valid",
            "strategy": "cascade",
            "candidates": [{"id": "primary", "status": "succeeded", "value": "fait sourcé"}],
            "expected": {
                "decision": "pass",
                "selected_candidate_id": "primary",
                "invoke_final_judge": False,
                "downstream_allowed": True,
            },
        },
        {
            "scenario": "cascade_fallback_valid",
            "strategy": "cascade",
            "candidates": [
                {"id": "primary", "status": "insufficient", "value": None},
                {"id": "challenger", "status": "succeeded", "value": "fait sourcé"},
            ],
            "expected": {
                "decision": "pass",
                "selected_candidate_id": "challenger",
                "invoke_final_judge": False,
                "downstream_allowed": True,
            },
        },
        {
            "scenario": "no_successful_candidate",
            "strategy": "quorum",
            "candidates": [
                {"id": "primary", "status": "failed", "value": None},
                {"id": "challenger", "status": "failed", "value": None},
            ],
            "expected": {
                "decision": "human_review",
                "selected_candidate_id": None,
                "invoke_final_judge": False,
                "downstream_allowed": False,
            },
        },
        {
            "scenario": "single_candidate_repaired",
            "strategy": "cascade",
            "candidates": [
                {
                    "id": "primary",
                    "status": "succeeded",
                    "value": "sortie conforme",
                    "repaired": True,
                }
            ],
            "expected": {
                "decision": "repair",
                "selected_candidate_id": "primary",
                "invoke_final_judge": False,
                "downstream_allowed": True,
            },
        },
    )
    records: list[dict[str, object]] = []
    for case in cases:
        group_id = f"consensus:{case['scenario']}"
        records.append(
            {
                "record_id": _stable_id(group_id),
                "task": "consensus_decision",
                "group_id": group_id,
                "split": _split(group_id),
                "scenario": case["scenario"],
                "input": {
                    key: value for key, value in case.items() if key not in {"scenario", "expected"}
                },
                "expected": case["expected"],
                "provenance": {
                    "kind": "code_test_policy",
                    "source": "firewarning_worker.consensus_and_session_runner",
                    "license": LICENSE,
                },
            }
        )
    return records


def _workflow_examples() -> list[dict[str, object]]:
    common = {
        "incident_id": "SYNTH-FR-00-0001",
        "analysis_day": "2030-01-01",
        "as_of": "2030-01-01T23:59:59Z",
        "sources": ["https://source.example.invalid/firewarning/synthetic"],
        "publication_allowed": False,
    }
    return [
        {
            "workflow_id": "daily_nominal_multisource",
            "split": "train",
            "input": common,
            "stages": [{"role": role.value, "expected_gate": "pass"} for role in StageRole],
            "expected": {
                "result_state": "private_review_ready",
                "requires_human_validation": True,
                "automatic_publication": False,
            },
        },
        {
            "workflow_id": "daily_contradiction_then_abstention",
            "split": "validation",
            "input": {**common, "incident_id": "SYNTH-FR-00-0002"},
            "stages": [
                {"role": "source_research", "expected_gate": "pass"},
                {
                    "role": "fire_pointing",
                    "expected_gate": "pass",
                    "candidate_state": "contradiction",
                },
                {
                    "role": "consensus_judge",
                    "expected_action": "invoke_qwen3_14b",
                    "raw_evidence_available": False,
                    "expected_decision": "abstain",
                },
                {
                    "role": "spatial_projection",
                    "expected_gate": "not_applicable",
                },
                {"role": "situation_report", "expected_gate": "pass"},
            ],
            "expected": {
                "result_state": "private_human_review",
                "spatial_claim_released": False,
                "contradiction_preserved": True,
                "automatic_publication": False,
            },
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _validate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("orchestrator corpus contains duplicate record ids")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        if split not in {"train", "validation", "test"}:
            raise RuntimeError(f"invalid split: {split}")
        group_splits[str(row["group_id"])].add(split)
        expected = row.get("expected")
        if not isinstance(expected, dict) or expected.get("decision") not in ALLOWED_DECISIONS:
            raise RuntimeError(f"invalid expected decision for {row['record_id']}")

    leaking = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if leaking:
        raise RuntimeError(f"split leakage: {leaking[:5]}")

    roles = {str(row["stage_role"]) for row in rows if row["task"] == "stage_gate_decision"}
    expected_roles = {role.value for role in StageRole}
    if roles != expected_roles:
        raise RuntimeError(f"missing stage roles: {sorted(expected_roles - roles)}")

    split_counts = Counter(str(row["split"]) for row in rows)
    if set(split_counts) != {"train", "validation", "test"}:
        raise RuntimeError("all three splits must be populated")
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(Counter(str(row["task"]) for row in rows).items())),
        "stage_roles": sorted(roles),
        "split_leakage_groups": 0,
        "unique_record_ids": len(record_ids),
    }


def _validate_workflows(workflows: list[dict[str, object]]) -> dict[str, object]:
    if len(workflows) != 2:
        raise RuntimeError("exactly two end-to-end workflow examples are required")
    serialized = json.dumps(workflows, ensure_ascii=False)
    if "FR-26-" in serialized or "FR-77-" in serialized:
        raise RuntimeError("operational evaluation incidents must not enter the train bundle")
    if any(
        workflow["expected"].get("automatic_publication") is not False for workflow in workflows
    ):
        raise RuntimeError("workflow examples must remain private until human validation")
    return {
        "workflow_examples": len(workflows),
        "contains_operational_incident": False,
        "automatic_publication": False,
    }


def _build(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir / BUNDLE_ID
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    gate_rows, contract_digest = _stage_gate_records()
    rows = gate_rows + _consensus_records()
    workflows = _workflow_examples()
    row_validation = _validate_rows(rows)
    workflow_validation = _validate_workflows(workflows)

    for split in ("train", "validation", "test"):
        _write_jsonl(root / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    _write_jsonl(root / "end-to-end-examples.jsonl", workflows)
    (root / "README.md").write_text(
        "# FireWarning orchestrator gates SFT v1\n\n"
        "Deterministic gate decisions generated by the real FireWarning stage engine, plus "
        "closed consensus examples and two synthetic end-to-end daily workflows. No real "
        "incident fact or evaluation truth is included. Human validation remains mandatory.\n",
        encoding="utf-8",
    )
    (root / "entrypoint.json").write_bytes(
        _json_bytes(
            {
                "working_directory": "<FIREVIEWER_AI_WORKER_REPOSITORY>",
                "preflight": (
                    "python -m fireviewer_model_lab.training.orchestrator_gate_corpus preflight "
                    "--dataset-root <TRAIN_BUNDLE_ROOT>"
                ),
                "trainer": None,
            },
            pretty=True,
        )
    )

    file_inventory = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        payload = path.read_bytes()
        file_inventory.append(
            {
                "path": path.name,
                "size_bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "bundle_id": BUNDLE_ID,
        "license": LICENSE,
        "dataset_ready": True,
        "training_ready": False,
        "blocking_reasons": [
            "orchestrator_sft_trainer_not_implemented",
            "independent_human_validation_missing",
        ],
        "stage_contract_digest": contract_digest,
        "validation": {**row_validation, **workflow_validation},
        "files": file_inventory,
        "evaluation_excluded": ["operational-reference-a", "operational-reference-b"],
    }
    (root / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))

    zip_path = output_dir / f"{BUNDLE_ID}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            archive.write(path, f"{BUNDLE_ID}/{path.name}")

    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC verification failed: {bad}")
        entry_count = len(archive.infolist())
    zip_digest = sha256(zip_path.read_bytes()).hexdigest()
    (output_dir / f"{BUNDLE_ID}.zip.sha256").write_text(
        f"{zip_digest}  {zip_path.name}\n",
        encoding="ascii",
    )
    report = {
        "schema_version": 1,
        "bundle_id": BUNDLE_ID,
        "package_format": "firewarning-train-bundle-zip-v1",
        "dataset_ready": True,
        "training_ready": False,
        "blocking_reasons": manifest["blocking_reasons"],
        "stage_contract_digest": contract_digest,
        "rows": row_validation,
        "workflows": workflow_validation,
        "zip_validation": {
            "path": zip_path.name,
            "size_bytes": zip_path.stat().st_size,
            "sha256": zip_digest,
            "entry_count": entry_count,
            "crc_verified": True,
            "single_train_root": BUNDLE_ID,
        },
    }
    (output_dir / f"{BUNDLE_ID}.validation.json").write_bytes(_json_bytes(report, pretty=True))
    return report


def _preflight(dataset_root: Path) -> dict[str, object]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        path = dataset_root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") != split:
                raise RuntimeError(f"row split mismatch in {path}")
            rows.append(row)
    workflows = [
        json.loads(line)
        for line in (dataset_root / "end-to-end-examples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    row_validation = _validate_rows(rows)
    workflow_validation = _validate_workflows(workflows)
    if manifest.get("stage_contract_digest") != load_stage_contract_registry().digest:
        raise RuntimeError("stage contract digest does not match the current worker")
    return {
        "bundle_id": manifest.get("bundle_id"),
        "dataset_ready": True,
        "training_ready": bool(manifest.get("training_ready")),
        "blocking_reasons": manifest.get("blocking_reasons", []),
        **row_validation,
        **workflow_validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output-dir", type=Path, required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = _build(args.output_dir) if args.command == "build" else _preflight(args.dataset_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
