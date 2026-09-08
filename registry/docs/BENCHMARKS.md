# Frozen external detection benchmark

## Protocol

| Field | Value |
| --- | --- |
| Dataset | `Hajorda/flameye-wildfire-detection@test` |
| Dataset revision | `361a3dea8b877482af9f4ed80eff77ffd39926ef` |
| Images / boxes | 512 / 854 |
| Calibration / held-out split | 129 / 383 images |
| Isolation control | Exact image-hash overlap with FireViewer detector train and validation data: zero. |

## Comparable results

| Model | mAP50-95 | mAP50 | Precision | Recall | F1 | Negative false-alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D-FINE-M strict v1 | 0.3924 | 0.6838 | 0.6279 | 0.6682 | 0.6475 | 0.1913 |
| RT-DETR-v2-R50 | 0.3502 | 0.6332 | 0.7385 | 0.4507 | 0.5598 | 0.0435 |
| YOLO11-M strict v1 | 0.1491 | 0.2784 | 0.4274 | 0.2488 | 0.3145 | 0.3391 |

D-FINE-M is the strongest result on this frozen protocol by mAP50-95 and F1. RT-DETR-v2-R50 has the lower negative false-alarm rate. YOLO11-M remains a public baseline.

RF-DETR-Medium is intentionally absent from this table: its epoch-18 EMA validation result (mAP50-95 0.4683) was measured during training and has not yet been reproduced on this frozen independent benchmark. SegFormer is a segmentation model and is not comparable with detector metrics.

These metrics are research evidence, not a safety guarantee or authorization for autonomous incident decisions.
