"""Build the SageMaker request for the fail-closed DINOv3 corpus composition."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"
DEFAULT_CODE_BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "bundles/dinov3-compose-v1"
UTC_COMPAT = timezone.utc  # noqa: UP017 - SageMaker image currently uses Python 3.10
CODE_BUNDLE_PINS = {'training/__init__.py': '5347f39488a1caf80541f05685b35d2cb110c25ab3756396e38999d3b64be71c', 'training/dinov3_corpus_identity.py': '0fe55f7f5f5437bf25580dd3b70b2626a3d85d48c0f4ddad4241a74e3fd47637', 'training/dinov3_multitask_compose.py': 'c6f2b028f8a40f40791519037a9b376de4aa23b0b0688cb5281ebdcfe46de696', 'training/registries/dinov3-independent-benchmark-denylist-v1.json': '7368b1337e4ea363b0fb9e711fff94c5795aaea329a5f87a7afc0b46c1659a40', 'training/registries/dinov3-multitask-composition-v1.json': '27de9d64c3d758f2078148fa51ab95d97df9f86c5f4b6e56d05732477462a8d4', 'training/dinov3_sagemaker_bundle_bootstrap.py': '29105cc7e074e6feed015b14da0b609a96a82da8d5a882649ecabaf5ac790499'}
BOOTSTRAP_RELATIVE_PATH = "training/dinov3_sagemaker_bundle_bootstrap.py"
CODE_BUNDLE_BOOTSTRAP_VERIFIER = (
    "import hashlib,pathlib,runpy,sys;"
    "p=pathlib.Path(sys.argv[1]);"
    "p.is_file()and hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]"
    "or sys.exit('bootstrap integrity failure');"
    "sys.argv=[str(p),*sys.argv[3:]];"
    "runpy.run_path(str(p),run_name='__main__')"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_code_bundle(root: Path) -> dict[str, str]:
    root = root.resolve()
    for relative, expected_sha256 in CODE_BUNDLE_PINS.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"unsafe code bundle path: {relative}")
        if not path.is_file():
            raise ValueError(f"required code bundle file is missing: {relative}")
        if _sha256(path) != expected_sha256:
            raise ValueError(f"code bundle SHA-256 mismatch: {relative}")

    registry_path = root / "training/registries/dinov3-multitask-composition-v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    denylist_relative = "training/registries/dinov3-independent-benchmark-denylist-v1.json"
    if (
        registry.get("schema_version") != 2
        or registry.get("benchmark_boundary", {}).get("denylist_sha256")
        != CODE_BUNDLE_PINS[denylist_relative]
    ):
        raise ValueError("composition registry does not pin the bundled benchmark denylist")
    return dict(CODE_BUNDLE_PINS)


def _input(name: str, uri: str, local_path: str) -> dict:
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": uri,
            "LocalPath": local_path,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
            "S3CompressionType": "None",
        },
    }


def build_request(args: argparse.Namespace, job_name: str) -> dict:
    bundle_hashes = _validate_code_bundle(Path(args.code_bundle_root))
    bundle_contract = json.dumps(bundle_hashes, sort_keys=True, separators=(",", ":"))
    bundle_contract_sha256 = hashlib.sha256(bundle_contract.encode("utf-8")).hexdigest()
    prefixes = (
        args.pyro_control_prefix,
        args.boreal_prefix,
        args.camp_swift_prefix,
        args.kit_prefix,
        args.ijmond_prefix,
        args.code_prefix,
        args.output_prefix,
    )
    forbidden = ("benchdata", "fireviewer_bench", "independent-benchmark")
    if any(marker in value.casefold() for value in prefixes for marker in forbidden):
        raise ValueError("composition input includes an independent benchmark path")
    output_s3 = f"s3://{args.bucket}/{args.output_prefix.rstrip('/')}/{job_name}"
    code_root = "/opt/ml/processing/input/code"
    registry_path = f"{code_root}/training/registries/dinov3-multitask-composition-v1.json"
    denylist_path = f"{code_root}/training/registries/dinov3-independent-benchmark-denylist-v1.json"
    bootstrap_path = f"{code_root}/{BOOTSTRAP_RELATIVE_PATH}"
    arguments = [
        "-c",
        CODE_BUNDLE_BOOTSTRAP_VERIFIER,
        bootstrap_path,
        bundle_hashes[BOOTSTRAP_RELATIVE_PATH],
        code_root,
        "--registry",
        registry_path,
        "--benchmark-denylist",
        denylist_path,
        "--control",
        "pyro-sdis-negative-qa=/opt/ml/processing/input/pyro-control",
        "--overlay",
        "boreal=/opt/ml/processing/input/boreal",
        "--overlay",
        "camp-swift=/opt/ml/processing/input/camp-swift",
        "--overlay",
        "kit=/opt/ml/processing/input/kit",
        "--overlay",
        "ijmond=/opt/ml/processing/input/ijmond",
        "--work-dir",
        "/opt/ml/processing/work",
        "--output-dir",
        "/opt/ml/processing/output",
    ]
    if any(len(argument) > 256 for argument in arguments):
        raise ValueError("SageMaker container argument exceeds the 256-character limit")
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": arguments,
        },
        "ProcessingInputs": [
            _input(
                "code",
                f"s3://{args.bucket}/{args.code_prefix.rstrip('/')}",
                code_root,
            ),
            _input(
                "pyro-control",
                f"s3://{args.bucket}/{args.pyro_control_prefix.rstrip('/')}",
                "/opt/ml/processing/input/pyro-control",
            ),
            _input(
                "boreal",
                f"s3://{args.bucket}/{args.boreal_prefix.rstrip('/')}",
                "/opt/ml/processing/input/boreal",
            ),
            _input(
                "camp-swift",
                f"s3://{args.bucket}/{args.camp_swift_prefix.rstrip('/')}",
                "/opt/ml/processing/input/camp-swift",
            ),
            _input(
                "kit",
                f"s3://{args.bucket}/{args.kit_prefix.rstrip('/')}",
                "/opt/ml/processing/input/kit",
            ),
            _input(
                "ijmond",
                f"s3://{args.bucket}/{args.ijmond_prefix.rstrip('/')}",
                "/opt/ml/processing/input/ijmond",
            ),
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "multitask-composition-candidate",
                    "S3Output": {
                        "S3Uri": output_s3,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": args.volume_size,
            }
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime},
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "FIREVIEWER_CODE_BUNDLE_CONTRACT_SHA256": bundle_contract_sha256,
            "FIREVIEWER_IDENTITY_MODULE_SHA256": bundle_hashes[
                "training/dinov3_corpus_identity.py"
            ],
            "FIREVIEWER_BENCHMARK_DENYLIST_SHA256": bundle_hashes[
                "training/registries/dinov3-independent-benchmark-denylist-v1.json"
            ],
        },
        "Tags": [
            {"Key": "fireviewer:corpus", "Value": "dinov3-multitask-v4"},
            {"Key": "fireviewer:stage", "Value": "strict-composition-candidate"},
            {"Key": "fireviewer:reviews-admitted", "Value": "false"},
            {"Key": "fireviewer:publication-allowed", "Value": "false"},
            {"Key": "fireviewer:training-ready", "Value": "false"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--pyro-control-prefix", required=True)
    parser.add_argument("--boreal-prefix", required=True)
    parser.add_argument("--camp-swift-prefix", required=True)
    parser.add_argument("--kit-prefix", required=True)
    parser.add_argument("--ijmond-prefix", required=True)
    parser.add_argument("--code-prefix", required=True)
    parser.add_argument(
        "--code-bundle-root",
        type=Path,
        default=DEFAULT_CODE_BUNDLE_ROOT,
        help="Local repository root used to verify the exact bundle before request emission.",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--instance-type", default="ml.t3.large")
    parser.add_argument("--volume-size", type=int, default=30)
    parser.add_argument("--max-runtime", type=int, default=3600)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC_COMPAT).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-dinov3-compose-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"job_name": job_name, "request": str(args.emit_request)}, indent=2))


if __name__ == "__main__":
    main()
