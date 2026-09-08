"""Prepare an upstream-annotated bulk complement without forging V8 admission.

The existing manually reviewed corpus and its gates are immutable. This creates
an independently usable COCO draft and records the unresolved visual admission.
It never starts a trainer and never changes the active dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil

import imagehash
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from pycocotools.coco import COCO

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v7.build_dfire_gap_candidates import MIRROR_REVISION, OFFICIAL_REVISION, parse_labels
from training.pointing_dataset_v8.prepare_extension_review import known_fingerprints


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def strict_objects(label, width, height):
    for line in label.splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError('nonfinite_or_invalid_yolo')
        cls, cx, cy, bw, bh = values
        if cls not in (0, 1) or bw <= 0 or bh <= 0:
            raise ValueError('invalid_yolo_class_or_size')
        if min(cx-bw/2, cy-bh/2) < -1e-7 or max(cx+bw/2, cy+bh/2) > 1+1e-7:
            raise ValueError('out_of_frame_yolo')
    parsed = parse_labels(label, width, height)
    boxes = []
    for obj in parsed:
        x, y, w, h = obj['bbox']
        # Only normalize floating point roundoff at exact image boundaries.
        x, y = max(0., x), max(0., y)
        w, h = min(w, width-x), min(h, height-y)
        if min(w, h) < 1:
            raise ValueError('subpixel_annotation')
        boxes.append([x, y, w, h])
    return {'bbox': boxes, 'category': [o['category'] for o in parsed],
            'area': [b[2]*b[3] for b in boxes]}


class NearIndex:
    """Exact radius-4 Hamming lookup using five disjoint bit bands.

    At most four changed bits cannot affect every band. Candidate retrieval
    therefore has no false negatives; full Hamming checks remove false hits.
    Original and mirrored fingerprints are both indexed and queried.
    """
    BANDS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))

    def __init__(self):
        self.values = {}
        self.buckets = defaultdict(set)

    def add(self, phash, flipped, identity):
        for value in (int(phash, 16), int(flipped, 16)):
            self.values.setdefault(value, identity)
            for i, (shift, bits) in enumerate(self.BANDS):
                self.buckets[i, (value >> shift) & ((1 << bits)-1)].add(value)

    def match(self, phash, flipped):
        for value in (int(phash, 16), int(flipped, 16)):
            candidates = set()
            for i, (shift, bits) in enumerate(self.BANDS):
                candidates.update(self.buckets.get((i, (value >> shift) & ((1 << bits)-1)), ()))
            for candidate in sorted(candidates):
                if (value ^ candidate).bit_count() <= 4:
                    return self.values[candidate]
        return None


def aliases(row):
    m = re.fullmatch(r'(WEB|AoF|PublicDataset)(\d+)\.jpg', row['filename'], re.I)
    if not m:
        raise ValueError('unknown_filename_family')
    family, number = m[1].lower(), int(m[2])
    return {f'dfire:{family}:block-{row["row_index"]//100:04d}',
            f'dfire:{family}:block-{number//100:04d}', f'dfire:{family}:{number//50}'}


def inspect_original(job):
    raw, meta, shard, output = job
    name = raw['filename']
    if name != meta['filename'] or raw['label'] != meta['label_text']:
        raise ValueError('Pinned source projection differs from archive')
    data = raw['image']['bytes']
    sha = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.getexif().get(274, 1) != 1:
                raise ValueError('unresolved_exif_geometry')
            width, height = im.size
            if min(width, height) < 64:
                raise ValueError('image_too_small')
            rgb = im.convert('RGB')
            phash, flipped = str(imagehash.phash(rgb)), str(imagehash.phash(ImageOps.mirror(rgb)))
        objects = strict_objects(raw['label'], width, height)
    except (ValueError, OSError, SyntaxError) as exc:
        return None, {'source_record_id': name, 'sha256': sha, 'reason': 'technical_image_or_annotation', 'detail': str(exc)}
    target = output / 'source-images' / (sha + '.jpg')
    if target.exists():
        if digest(target) != sha:
            raise ValueError('Existing extracted original changed')
    else:
        target.write_bytes(data)
    row = dict(meta, sha256=sha, source_image=str(target.resolve()), width=width, height=height,
               phash=phash, phash_flipped=flipped, objects=objects, source_record_id=name,
               source_dataset='D-Fire', source_family='D-Fire', source_split='train', split='train',
               source_revision=MIRROR_REVISION, source_parquet_sha256=shard['lfs_sha256'],
               source_parquet_path=shard['path'], source_parquet_row=meta['shard_row'],
               source_group_id=f'dfire:{re.match(r"[A-Za-z]+", name)[0].lower()}:block-{meta["row_index"]//100:04d}',
               source_group_aliases=sorted(aliases(meta)),
               grouping_basis='conservative_filename_blocks_not_verified_incident_identity',
               annotation_state='upstream_yolo_boxes_technically_validated',
               annotation_review_complete=False, review_status='upstream_annotations_not_manually_reviewed',
               v8_corpus_admitted=False, v8_training_admitted=False, synthetic=False,
               license='CC0-1.0',
               license_evidence=f'https://github.com/gaia-solutions-on-demand/DFireDataset/blob/{OFFICIAL_REVISION}/LICENSE')
    return row, None


def frozen_groups(history, base):
    frozen = {k.lower() for k, splits in history['source_groups'].items() if any(s != 'train' for s in splits)}
    frozen.update(k.lower() for k, split in history['groups'].items() if split != 'train')
    for row in base:
        if row['split'] != 'train':
            frozen.update(str(row.get(k, '')).lower() for k in ('source_group_id', 'split_group_id', 'scene_group_id'))
            frozen.update(str(x).lower() for x in row.get('source_group_aliases', []))
    frozen.discard('')
    return frozen


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError('Existing export differs')
    else:
        os.link(source, target)


def export_view(base_view, output, additions):
    destination = output / 'combined-coco-draft'
    frozen = {}
    for split in ('train', 'valid', 'test'):
        source_json = base_view / split / '_annotations.coco.json'
        coco = json.loads(source_json.read_text())
        if coco['categories'] != [{'id': 0, 'name': 'fire'}, {'id': 1, 'name': 'smoke'}]:
            raise ValueError('Base category mapping differs')
        for im in coco['images']:
            link(base_view / split / im['file_name'], destination / split / im['file_name'])
        dest_json = destination / split / '_annotations.coco.json'
        if split != 'train':
            shutil.copyfile(source_json, dest_json)
            frozen[split] = {'sha256': digest(source_json), 'images': len(coco['images'])}
            if digest(dest_json) != frozen[split]['sha256']:
                raise ValueError('Holdout annotations changed')
        else:
            next_i = max(i['id'] for i in coco['images']) + 1
            next_a = max(a['id'] for a in coco['annotations']) + 1
            for row in additions:
                filename = 'images/' + row['sha256'] + '.jpg'
                link(Path(row['source_image']), destination / split / filename)
                coco['images'].append({'id': next_i, 'file_name': filename,
                    'width': row['width'], 'height': row['height'], 'fireviewer_sha256': row['sha256'],
                    'fireviewer_source_group_id': row['source_group_id'],
                    'annotation_review_status': 'source_provided_not_manually_reviewed'})
                for box, category, area in zip(row['objects']['bbox'], row['objects']['category'], row['objects']['area']):
                    coco['annotations'].append({'id': next_a, 'image_id': next_i, 'category_id': category,
                                               'bbox': box, 'area': area, 'iscrowd': 0})
                    next_a += 1
                next_i += 1
            write_json(dest_json, coco)
        loaded = COCO(str(dest_json))
        if len(loaded.imgs) != len(coco['images']) or len(loaded.anns) != len(coco['annotations']):
            raise ValueError('COCO reload identifier collision')
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/local'))
    parser.add_argument('--available-only', action='store_true', help='Extract existing shards only; do not finalize a combined view')
    args = parser.parse_args()
    root = args.root.resolve()
    owner = root / 'pointing-v8-completion-20260828'
    output = owner / 'v8p3-dfire-bulk'
    for name in ('source-images', 'shard-receipts'):
        (output / name).mkdir(parents=True, exist_ok=True)
    authority_dir = root / 'pointing-dataset-v4-dfire-fire-extension-20260824'
    authority = json.loads((authority_dir / 'source_authority.json').read_text())
    if digest(authority_dir / 'source_metadata.jsonl') != authority['source_metadata_sha256']:
        raise ValueError('Pinned metadata changed')
    metadata = read_rows(authority_dir / 'source_metadata.jsonl')
    base_path = owner / 'v8p2-aussmoke/corpus/selection_manifest.jsonl'
    base_hash = digest(base_path)
    base = read_rows(base_path)
    history_path = root / 'fireviewer-pointing-v7-audited-20260827/historical_split_registry.json'
    history = json.loads(history_path.read_text())
    frozen = frozen_groups(history, base)
    base_names = {r['source_record_id'].lower() for r in base if r.get('source_family') == 'D-Fire'}
    rejected_names, rejected_hashes = set(), set()
    reviews = list(owner.glob('**/curation_manifest.jsonl'))
    reviews += list(root.glob('pointing-v8-review-*/curation_manifest.jsonl'))
    for path in reviews:
        for row in read_rows(path):
            if row.get('review_status') in {'reject', 'needs_annotation'}:
                rejected_hashes.add(row['sha256'])
                if row.get('source_family') == 'D-Fire':
                    rejected_names.add(row['source_record_id'].lower())
    binding = {'base_sha256': base_hash, 'metadata_sha256': authority['source_metadata_sha256'],
               'frozen_groups': sorted(frozen), 'rejected_names': sorted(rejected_names),
               'base_names': sorted(base_names), 'builder_sha256': digest(Path(__file__))}
    binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    all_rows, exclusions, processed = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for shard in authority['parquet_shards']:
            shard_metadata = [r for r in metadata if r['shard'] == shard['shard']]
            if all(r['filename'].lower().startswith('aof') for r in shard_metadata):
                exclusions.extend({'source_record_id': r['filename'], 'reason': 'aof_fixed_camera_family_quarantined_for_holdout_safety'} for r in shard_metadata)
                processed.append(shard['shard'])
                continue
            archive = owner / 'sources/bulk-dfire/original/parquet' / Path(shard['path']).name
            if not archive.exists():
                if args.available_only:
                    continue
                raise ValueError('An expected train archive is not yet available')
            if archive.stat().st_size != shard['bytes'] or digest(archive) != shard['lfs_sha256']:
                raise ValueError('Archive identity mismatch')
            cache = output / 'shard-receipts' / f'{shard["shard"]:02d}-{binding_hash[:12]}.json'
            if cache.exists():
                saved = json.loads(cache.read_text())
                if saved['binding_sha256'] != binding_hash:
                    raise ValueError('Extraction inputs changed; explicit new output required')
                all_rows.extend(saved['rows']); exclusions.extend(saved['exclusions']); processed.append(shard['shard'])
                continue
            selected = {r['shard_row']: r for r in metadata if r['shard'] == shard['shard']}
            rows, excluded, offset = [], [], 0
            for batch in pq.ParquetFile(archive).iter_batches(batch_size=64, columns=['filename', 'label', 'image']):
                jobs = []
                for local, raw in enumerate(batch.to_pylist()):
                    meta = selected[offset+local]
                    # AoF contains the fixed Camera0001/0002 views already
                    # identified in our historical holdouts. A new date or
                    # filename block cannot establish a new camera. Quarantine
                    # this entire upstream family in an unreviewed bulk import.
                    reason = ('aof_fixed_camera_family_quarantined_for_holdout_safety' if meta['filename'].lower().startswith('aof') else
                              'historical_or_current_holdout_group' if aliases(meta) & frozen else
                              'already_in_base' if meta['filename'].lower() in base_names else
                              'previous_visual_rejection_or_annotation_issue' if meta['filename'].lower() in rejected_names else None)
                    if reason:
                        excluded.append({'source_record_id': meta['filename'], 'reason': reason})
                    else:
                        jobs.append((raw, meta, shard, output))
                for row, error in pool.map(inspect_original, jobs):
                    (rows if row else excluded).append(row or error)
                offset += batch.num_rows
            if offset != shard['row_count']:
                raise ValueError('Archive row count differs')
            write_json(cache, {'binding_sha256': binding_hash, 'rows': rows, 'exclusions': excluded})
            all_rows.extend(rows); exclusions.extend(excluded); processed.append(shard['shard'])
            print(json.dumps({'extracted_shard': shard['shard'], 'technical_candidates': len(all_rows), 'excluded': len(exclusions)}), flush=True)
    if args.available_only:
        print(json.dumps({'preextracted_shards': processed, 'technical_candidates': len(all_rows), 'finalized': False}), flush=True)
        return
    if len(processed) != 9:
        raise ValueError('Incomplete upstream train')
    known = known_fingerprints(root / 'pointing-v7-split-audit-groupwise-20260827/fingerprints.jsonl',
                              Path('fireviewer_bench/data/homefire-pointing-independent-v1-r3/samples.jsonl'))
    known_shas = {r['sha256'] for r in known}
    for row in base:
        if row['sha256'] not in known_shas:
            with Image.open(row['source_image']) as im:
                known.append(dict(row, phash=str(imagehash.phash(im)), phash_flipped=str(imagehash.phash(ImageOps.mirror(im)))))
            known_shas.add(row['sha256'])
    index = NearIndex()
    for row in known:
        index.add(row['phash'], row['phash_flipped'], row['sha256'])
    retained = []
    for row in sorted(all_rows, key=lambda r: (not bool(r['objects']['bbox']), r['sha256'])):
        match = row['sha256'] if row['sha256'] in known_shas else index.match(row['phash'], row['phash_flipped'])
        if row['sha256'] in rejected_hashes:
            reason = 'previous_visual_rejection_or_annotation_issue'
        else:
            reason = 'exact_or_mirrored_near_overlap_le4' if match else None
        if reason:
            exclusions.append({'source_record_id': row['filename'], 'sha256': row['sha256'], 'reason': reason, 'matching_sha256': match})
            continue
        retained.append(row)
        known_shas.add(row['sha256'])
        index.add(row['phash'], row['phash_flipped'], row['sha256'])
    positive = [r for r in retained if r['objects']['bbox']]
    negative = [r for r in retained if not r['objects']['bbox']]
    base_train = [r for r in base if r['split'] == 'train']
    base_negatives = sum(not r['objects']['bbox'] for r in base_train)
    negative_budget = max(0, math.floor((.12*(len(base_train)+len(positive))-base_negatives)/.88))
    additions = positive + negative[:negative_budget]
    reservoir = negative[negative_budget:]
    write_rows(output / 'additional_source_annotations.jsonl', additions)
    write_rows(output / 'negative_reservoir.jsonl', reservoir)
    write_rows(output / 'excluded.jsonl', exclusions)
    write_rows(output / 'combined_selection_manifest.jsonl', base + additions)
    holdouts = export_view(owner / 'v8p2-aussmoke/training-view5048', output, additions)
    if digest(base_path) != base_hash:
        raise ValueError('Base corpus mutated during export')
    boxes = Counter(c for r in additions for c in r['objects']['category'])
    report = {'status': 'complete_source_annotated_coco_draft', 'source_train_rows': 17221,
        'all_six_used_archive_sha256_verified': True,
        'metadata_quarantined_source_rows_not_extracted': 5742, 'base_images_preserved': len(base),
        'base_train_images': len(base_train), 'net_added_train_images': len(additions),
        'new_positive_images': len(positive), 'new_negative_images': len(additions)-len(positive),
        'new_fire_boxes': boxes[0], 'new_smoke_boxes': boxes[1],
        'new_small_fire_images': sum(any(c == 0 and b[2]*b[3]/(r['width']*r['height']) <= .005 for b,c in zip(r['objects']['bbox'],r['objects']['category'])) for r in additions),
        'new_small_smoke_images': sum(any(c == 1 and b[2]*b[3]/(r['width']*r['height']) <= .005 for b,c in zip(r['objects']['bbox'],r['objects']['category'])) for r in additions),
        'combined_train_images': len(base_train)+len(additions), 'combined_total_images': len(base)+len(additions),
        'train_negative_fraction': (base_negatives+len(additions)-len(positive))/(len(base_train)+len(additions)),
        'negative_reservoir_images': len(reservoir), 'excluded': dict(Counter(r['reason'] for r in exclusions)),
        'holdouts_unchanged': holdouts, 'source_test_downloaded': False,
        'base_manifest_sha256': base_hash, 'addition_manifest_sha256': digest(output/'additional_source_annotations.jsonl'),
        'combined_manifest_sha256': digest(output/'combined_selection_manifest.jsonl'),
        'technical_coco_reload_passed': True, 'all_added_images_decoded': True,
        'v8_manual_admission_complete': False, 'training_started': False,
        'limitations': ['Source annotations retained; no fabricated manual review.',
                       'AoF fixed-camera family quarantined; no complete visual audit of remaining aerial views, people, scene identity or box completeness.',
                       'Bulk draft does not claim compliance with V8 manual-review and per-scene admission gates.',
                       'Ready for technical consumption; existing V8 training authorization remains unchanged.',
                       'Larger volume does not prove improved validation metrics.']}
    write_json(output / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
