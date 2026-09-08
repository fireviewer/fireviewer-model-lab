# Active public models

Snapshot reconciled with the public Hugging Face API on 8 September 2026. The listed revisions are immutable Hub commits at that snapshot.

| Model | Task | Revision | Registry status | Evidence available |
| --- | --- | --- | --- | --- |
| [`fire-smoke-dfine-m-strict-v1`](https://huggingface.co/fireviewer/fire-smoke-dfine-m-strict-v1) | Detection | `0ef733968748c7ac172273db8c381ea5ea5ff046` | Public research candidate | Frozen external mAP50-95 0.3924; F1 0.6475. |
| [`rtdetr-v2-r50-fire-smoke`](https://huggingface.co/fireviewer/rtdetr-v2-r50-fire-smoke) | Detection | `273e0b37b32aad8c43c1ba5c004b9a7b130974ca` | Public conservative reference | Frozen external mAP50-95 0.3502; F1 0.5598; lower negative false-alarm rate. |
| [`fire-smoke-yolo11m-strict-v1`](https://huggingface.co/fireviewer/fire-smoke-yolo11m-strict-v1) | Detection | `c9ed3ade20b79d27560591fcd592c037ffe9caf3` | Public research baseline | Frozen external mAP50-95 0.1491; F1 0.3145. |
| [`fire-smoke-rfdetr-medium-v110-strict-v1`](https://huggingface.co/fireviewer/fire-smoke-rfdetr-medium-v110-strict-v1) | Detection | `b94fa0b90b2077960d07ff406feefd644844f292` | Public validation candidate | Training validation EMA mAP50-95 0.4683 at epoch 18; no frozen independent benchmark yet. |
| [`segformer-b2-fire-smoke-baseline-v1`](https://huggingface.co/fireviewer/segformer-b2-fire-smoke-baseline-v1) | Segmentation | `9e1418ef700ce06eda9e2ab308520adf4fd0329a` | Public segmentation baseline | Held-out IoU 0.7822; Dice 0.8778. Not comparable with detector scores. |

## Comparable detector evidence

Only D-FINE-M, RT-DETR-v2-R50, and YOLO11-M share the same frozen external detection protocol. RF-DETR's training-validation result is useful for monitoring but is not an independent ranking.

See [BENCHMARKS.md](BENCHMARKS.md) for the protocol, metrics, and limits.

## Promotion boundary

These research artifacts are suitable for reproducible evaluation and product prototyping. They are not autonomous-warning systems: deployments require a calibrated operating threshold, human review, and location-specific validation.
