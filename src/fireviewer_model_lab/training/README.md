# Generic training-data preparation

This directory contains code for preparing and validating training manifests.
No media, dataset payload, annotation corpus, render, model weight, checkpoint,
or evaluation output is versioned here.

## Rules

- Supply an explicit data root outside the repository.
- Preserve provenance, licence, and SHA-256 for every entry.
- Separate related event or site groups before train/validation/test splitting.
- Exclude protected operational incidents and private productions.
- Never copy training data into a public container image.
- Require human validation before promoting any model.

Examples and tests use synthetic identifiers and content. Versioned JSON
registries describe contracts or source metadata, never the samples themselves.

## Checks

```bash
ruff check training tools
ruff format --check training tools
pytest -q
```

GPU execution, external-corpus admission, independent evaluation, and model
quality are separate gates and are not proven by these local checks.
