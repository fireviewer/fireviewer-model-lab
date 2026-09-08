import hashlib
import json

import pytest

from training.pointing_dataset_v7.split_registry import digest
from training.pointing_dataset_v8 import acquire_dfire
from training.pointing_dataset_v8.acquire_dfire import MIRROR_REVISION, ViewerRows, retained_candidates, select


def authority(tmp_path, names, history=None):
    source = tmp_path / "pointing-dataset-v4-dfire-fire-extension-20260824"
    source.mkdir()
    label = "1 0.5 0.5 0.02 0.03"
    rows = [{"filename": name, "row_index": i,
             "label_text": label, "label_sha256": hashlib.sha256(label.encode()).hexdigest()}
            for i, name in enumerate(names)]
    metadata = source / "source_metadata.jsonl"
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "source_authority.json").write_text(json.dumps({"source_metadata_sha256": digest(metadata)}))
    registry = tmp_path / "fireviewer-pointing-v7-audited-20260827"
    registry.mkdir()
    (registry / "historical_split_registry.json").write_text(json.dumps({"source_groups": history or {}}))
    return source


def test_filename_families_are_explicit_and_default_stays_web_only(tmp_path):
    authority(tmp_path, ["WEB01000.jpg", "AoF01000.jpg", "PublicDataset01000.jpg"])
    assert [row["filename"] for row in select(tmp_path)] == ["WEB01000.jpg"]
    selected = select(tmp_path, ["AoF", "PublicDataset"])
    assert {row["filename"] for row in selected} == {"AoF01000.jpg", "PublicDataset01000.jpg"}
    assert all("not_verified_incident_identity" in row["grouping_basis"] for row in selected)
    assert all("v8_corpus_admitted" not in row for row in selected)


def test_smoke_profile_uses_smoke_labels_without_relaxing_existing_filters(tmp_path):
    source = authority(tmp_path, ["WEB01000.jpg", "WEB01001.jpg"])
    metadata = source / "source_metadata.jsonl"
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    rows[1]["label_text"] = "0 0.5 0.5 0.02 0.03"
    rows[1]["label_sha256"] = hashlib.sha256(rows[1]["label_text"].encode()).hexdigest()
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "source_authority.json").write_text(json.dumps({"source_metadata_sha256": digest(metadata)}))
    assert [r["filename"] for r in select(tmp_path)] == ["WEB01000.jpg"]
    assert [r["filename"] for r in select(tmp_path, target="small_smoke")] == ["WEB01001.jpg"]
    with pytest.raises(ValueError, match="selection profile"):
        select(tmp_path, target="all")


def test_other_filename_family_holdout_aliases_are_not_bypassed(tmp_path):
    authority(tmp_path, ["AoF04801.jpg", "PublicDataset01000.jpg"],
              {"dfire:AoF:96": ["test"], "dfire:publicdataset:block-0010": ["validation"]})
    assert select(tmp_path, ["AoF", "PublicDataset"]) == []


def test_previously_reviewed_families_do_not_reenter_candidate_pool(tmp_path):
    authority(tmp_path, ["AoF01000.jpg", "PublicDataset01000.jpg"])
    pool = tmp_path / "pointing-dataset-v4-old"
    pool.mkdir()
    (pool / "review_pool.jsonl").write_text(json.dumps({"source_record_id": "D-Fire/AoF01000.jpg"}) + "\n")
    assert [row["filename"] for row in select(tmp_path, ["AoF", "PublicDataset"])] == ["PublicDataset01000.jpg"]


def test_metadata_must_still_match_original_authority(tmp_path):
    source = authority(tmp_path, ["AoF01000.jpg"])
    (source / "source_metadata.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="metadata changed"):
        select(tmp_path, ["AoF"])


def test_viewer_metadata_is_paged_once_and_bound_to_labels(monkeypatch):
    calls = []
    def page(params, *, revision):
        calls.append((params, revision))
        return {"rows": [{"row_idx": i, "row": {"filename": f"WEB{i:05d}.jpg", "label": "bound",
                        "image": {"src": f"ephemeral/{i}"}}} for i in range(100, 200)]}
    monkeypatch.setattr(acquire_dfire, "api_rows", page)
    viewer = ViewerRows()
    for i in [101, 103, 101]:
        assert viewer.image_url({"row_index": i, "filename": f"WEB{i:05d}.jpg", "label_text": "bound"}) == f"ephemeral/{i}"
    assert len(calls) == 1 and calls[0][0]["length"] == 100 and calls[0][1] == MIRROR_REVISION
    with pytest.raises(ValueError, match="mismatch"):
        viewer.image_url({"row_index": 101, "filename": "WEB00101.jpg", "label_text": "different"})


def test_resume_keeps_receipts_outside_the_current_selection(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image bytes")
    row = {"filename": "AoF01000.jpg", "source_revision": MIRROR_REVISION,
           "source_image": str(image), "sha256": digest(image)}
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "AoF01000.json").write_text(json.dumps(row))
    (tmp_path / "candidate_manifest.jsonl").write_text(json.dumps(row) + "\n")
    assert retained_candidates(tmp_path) == {row["filename"]: row}
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        retained_candidates(tmp_path)


def test_orphaned_manifest_cannot_silently_disappear_on_resume(tmp_path):
    (tmp_path / "candidate_manifest.jsonl").write_text(json.dumps({"filename": "WEB01.jpg", "sha256": "a"}) + "\n")
    with pytest.raises(ValueError, match="matching retained receipt"):
        retained_candidates(tmp_path)


def test_rate_limit_header_or_conservative_fallback():
    class Response:
        headers = {"Retry-After": "900"}
    assert acquire_dfire.rate_limit_delay(Response()) == 900
    Response.headers = {}
    assert acquire_dfire.rate_limit_delay(Response()) == 600


def test_bulk_mapping_preserves_smoke_zero_to_canonical_one():
    from training.pointing_dataset_v8.prepare_bulk_dfire import strict_objects
    objects = strict_objects('0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.1 0.1', 100, 200)
    assert objects['category'] == [1, 0]
    assert objects['bbox'][0] == [40., 60., 20., 80.]


@pytest.mark.parametrize('label', ['2 0.5 0.5 0.2 0.2', '0.5 0.5 0.5 0.2 0.2',
                                   '1 nan 0.5 0.2 0.2', '1 0.9 0.5 0.5 0.2',
                                   '1 0.5 0.5 0 0.2', '1 0.5 0.5 0.001 0.2'])
def test_bulk_refuses_bad_annotations(label):
    from training.pointing_dataset_v8.prepare_bulk_dfire import strict_objects
    with pytest.raises(ValueError):
        strict_objects(label, 100, 100)


def test_bulk_near_index_matches_brute_force_including_mirrors():
    import random
    from training.pointing_dataset_v8.prepare_bulk_dfire import NearIndex
    rng = random.Random(71)
    values = [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(80)]
    index = NearIndex()
    for i, (a, b) in enumerate(values):
        index.add(f'{a:016x}', f'{b:016x}', str(i))
    queries = []
    for a, b in values:
        for distance in (0, 1, 4, 5):
            mask = sum(1 << bit for bit in rng.sample(range(64), distance))
            queries.append((rng.getrandbits(64), b ^ mask))
    queries.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(50))
    for a, b in queries:
        expected = any(min((a^x).bit_count(), (b^x).bit_count(),
                           (a^y).bit_count(), (b^y).bit_count()) <= 4 for x, y in values)
        assert (index.match(f'{a:016x}', f'{b:016x}') is not None) == expected


def test_bulk_holdout_aliases_remain_locked():
    from training.pointing_dataset_v8.prepare_bulk_dfire import aliases, frozen_groups
    row = {'filename': 'AoF04801.jpg', 'row_index': 109}
    locked = frozen_groups({'source_groups': {'dfire:AoF:96': ['test']}, 'groups': {}}, [])
    assert aliases(row) & locked == {'dfire:aof:96'}


def test_bulk_image_inspection_never_forges_manual_admission(tmp_path):
    from PIL import Image
    from io import BytesIO
    from training.pointing_dataset_v8.prepare_bulk_dfire import inspect_original
    stream = BytesIO()
    Image.new('RGB', (128, 128), '#445533').save(stream, format='JPEG')
    (tmp_path/'source-images').mkdir()
    meta = {'filename': 'WEB01000.jpg', 'label_text': '1 0.5 0.5 0.2 0.2',
            'row_index': 100, 'shard_row': 0}
    raw = {'filename': meta['filename'], 'label': meta['label_text'], 'image': {'bytes': stream.getvalue()}}
    row, error = inspect_original((raw, meta, {'lfs_sha256': 'a'*64, 'path': 'train.parquet'}, tmp_path))
    assert error is None
    assert row['objects']['category'] == [0]
    assert row['annotation_review_complete'] is False
    assert row['v8_corpus_admitted'] is False and row['v8_training_admitted'] is False
    assert digest(tmp_path/'source-images'/f'{row["sha256"]}.jpg') == row['sha256']


def test_bulk_export_preserves_base_annotations_and_holdouts(tmp_path):
    from PIL import Image
    from training.pointing_dataset_v8.prepare_bulk_dfire import export_view
    base, output = tmp_path/'base', tmp_path/'output'
    expected = {}
    for split in ('train', 'valid', 'test'):
        (base/split/'images').mkdir(parents=True)
        Image.new('RGB', (128, 128), 'navy').save(base/split/'images/original.jpg')
        value = {'images': [{'id': 41, 'file_name': 'images/original.jpg', 'width': 128, 'height': 128}],
                 'annotations': [{'id': 9, 'image_id': 41, 'category_id': 0, 'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0}],
                 'categories': [{'id': 0, 'name': 'fire'}, {'id': 1, 'name': 'smoke'}]}
        path = base/split/'_annotations.coco.json'
        path.write_text(json.dumps(value))
        expected[split] = path.read_bytes()
    extra = tmp_path/'extra.jpg'
    Image.new('RGB', (128, 128), 'green').save(extra)
    row = {'sha256': digest(extra), 'source_image': str(extra), 'width': 128, 'height': 128,
           'source_group_id': 'dfire:web:block-0100',
           'objects': {'bbox': [[20, 20, 30, 30]], 'category': [1], 'area': [900]}}
    export_view(base, output, [row])
    view = output/'combined-coco-draft'
    for split in expected:
        assert (base/split/'_annotations.coco.json').read_bytes() == expected[split]
    for split in ('valid', 'test'):
        assert (view/split/'_annotations.coco.json').read_bytes() == expected[split]
    train = json.loads((view/'train/_annotations.coco.json').read_text())
    assert train['annotations'][0] == json.loads(expected['train'])['annotations'][0]
    assert train['annotations'][1]['image_id'] == 42
    assert train['annotations'][1]['category_id'] == 1
    assert train['images'][1]['annotation_review_status'] == 'source_provided_not_manually_reviewed'
