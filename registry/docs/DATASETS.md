# Public datasets

Snapshot reconciled with the public Hugging Face API on 8 September 2026. Repository visibility does not, by itself, establish training suitability or downstream redistribution rights; consult every dataset card and its source terms.

| Dataset | Revision | Registry role | Notes |
| --- | --- | --- | --- |
| [`fire-smoke-detection-corpus-v1`](https://huggingface.co/datasets/fireviewer/fire-smoke-detection-corpus-v1) | `b45f08eb8cfb8636abe64002e95ad0548c9cfe8f` | Active detection-corpus authority | 102,257 auto-validated rows; clean authoritative detector corpus. |
| [`dataset-from-simulations`](https://huggingface.co/datasets/fireviewer/dataset-from-simulations) | `71664807b9bb20812a5d423f4dc7de2614436fbc` | Public synthetic research resource | Omniverse/OpenUSD wildfire and geospatial simulations. |
| [`dinov3-cross-view-fireviewer-v1-dataset`](https://huggingface.co/datasets/fireviewer/dinov3-cross-view-fireviewer-v1-dataset) | `d4e6a78739a8e0a8efc3d9635f1bc64551911034` | Public cross-view research resource | Image-to-image and segmentation research data. |
| [`firewarning-train-bundles-v1`](https://huggingface.co/datasets/fireviewer/firewarning-train-bundles-v1) | `3207a080f786e3349349759af34385e3d892b4e5` | Public historical archive | Excluded from the active strict detector corpus. |
| [`firewarning-training-corpus`](https://huggingface.co/datasets/fireviewer/firewarning-training-corpus) | `8528ee1f4a62090d73f966f0fa5c2a721aee8b9d` | Public provenance and evaluation reference | Retained for traceability and evaluation context. |
| [`prithvi-burnscars-training-dataset-v1`](https://huggingface.co/datasets/fireviewer/prithvi-burnscars-training-dataset-v1) | `5a1f6006002000aea031c261daf807ea1500c762` | Public burn-scar research dataset | Distinct remote-sensing task. |
| [`simple-measured-scenes-v1`](https://huggingface.co/datasets/fireviewer/simple-measured-scenes-v1) | `ee102e37fdbb752787fe8988c4d200272aa446e8` | Public measured-map resource | Geospatial and 3D scene data. |

## Detection-corpus replacement boundary

`fire-smoke-detection-corpus-v1` replaces prior detector-training candidates as the sole active public authority for the strict detection training corpus. Historical and task-specific datasets above are preserved as documented resources; they must not be silently merged into the strict corpus.

Deleted, private, or non-resolving repositories are deliberately absent from this registry.
