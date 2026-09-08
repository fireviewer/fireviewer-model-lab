# Hugging Face public inventory

This repository tracks the public FireViewer Hugging Face organization snapshot reconciled on 8 September 2026: five public model repositories and seven public dataset repositories.

The canonical machine-readable records are:

- [`registry/models.json`](../registry/models.json)
- [`registry/datasets.json`](../registry/datasets.json)
- [`registry/huggingface-inventory.json`](../registry/huggingface-inventory.json)

## Scope

The inventory is deliberately limited to resources visible through the public Hugging Face API. It does not enumerate private resources, local artifacts, deleted repositories, raw benchmark payloads, or source-rights assertions.

## Refresh method

Query the public Hugging Face model and dataset APIs for author `fireviewer`, reconcile exact repository IDs and immutable revisions, then update the JSON registry and human-readable pages in the same commit. A model or dataset is not represented until it resolves publicly at the recorded revision.
