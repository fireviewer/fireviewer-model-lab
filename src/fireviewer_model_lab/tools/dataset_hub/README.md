# FireViewer training bundles

The dataset-bundle tooling builds one archive per training objective. Source
archives and raw datasets remain outside the Git checkout; the repository keeps
only executable specifications, manifests, and validation code.

## Bundle contract

Each archive has one root directory named after its training objective and
contains:

- `TRAIN_BUNDLE.json`: sources, licences, gate state, and reproducible commands;
- `PAYLOAD_CHECKSUMS.sha256`: the digest of every payload file;
- source payloads mounted at the paths expected by the selected trainer;
- source-provided train, validation, and test manifests;
- no protected operational incident used as a final evaluation reference.

Bundle declarations live in `train-bundles-v1.json`; executable specifications
live under `specs/`.

## Local build

Use explicit external roots. The following PowerShell example assumes the
operator has set `$FireViewerDataRoot` to an approved local data directory:

```powershell
python tools/dataset_hub/finalize_train_bundle.py `
  --spec tools/dataset_hub/specs/media-filter-fire-smoke-v1.json `
  --source-root "$FireViewerDataRoot/remote" `
  --work-dir "$FireViewerDataRoot/work" `
  --output-dir "$FireViewerDataRoot/ready"
```

The builder rejects truncated archives, invalid hashes, unsafe paths, missing
files, duplicates across independent sources, and split leakage. It then reads
the complete output archive again and verifies each entry's CRC and SHA-256.

## Publication safeguards

Remote replacement is allowed only after:

1. complete local archive validation;
2. a durable local copy of the archive and report;
3. one atomic remote change that adds the new bundle and removes only the
   superseded payloads;
4. a verification download with an identical SHA-256;
5. removal of temporary local material only after the remote equality check.

If the atomic update fails, the existing remote archive remains authoritative
and the local validated copy is retained. Shared sources stay declared in every
bundle that uses them. Evaluation-only references are never promoted into a
training archive.

## Rights and provenance

Every normalised source must retain `SOURCE_MANIFEST.json`,
`VALIDATION_REPORT.json`, `manifest.jsonl`, artifact hashes, licence metadata,
and event- or site-isolated split groups. Discovery or download capability does
not prove redistribution or training rights. Sources with incomplete rights
remain excluded or quarantined until a human decision is recorded.
