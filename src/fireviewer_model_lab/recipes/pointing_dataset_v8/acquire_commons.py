"""Bounded original-image Commons acquisition with existing license validation.

Search text is only a candidate hint. Original bytes, current file SHA-1 and
description revision are bound; no photo, annotation or scene is auto-admitted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image, ImageOps
import requests

from training.pointing_dataset_v4.build_wikimedia_commons_metadata import CommonsAPI, validate_file_page
from training.pointing_dataset_v7.split_registry import digest
from training.pointing_dataset_v8.acquire_wui import ByteBudget


QUERIES = (
    'wildfire smoke forest', 'wildfire distant', 'forest fire smoke landscape',
    'wildfire night', 'wildfire sunset', 'bushfire smoke distant',
    'feu foret fumee', 'incendio forestal humo', 'wildfire smoke mountains',
    'prescribed fire landscape', 'wildfire small smoke', 'grass fire distant',
)
ACTIVE_FIRE_QUERIES = (
    'forest fire flames landscape', 'wildfire flames night', 'wildfire flames mountain',
    'wildfire smoke column', 'incendio forestal llamas noche', 'incendio bosque humo',
    'feu foret flammes nuit', 'bushfire flames ridge', 'grass fire flames landscape',
    'prescribed burn flames landscape', 'forest fire distant smoke', 'Waldbrand Flammen Nacht',
)
REJECT_TEXT = re.compile(r'\b(aerial|satellite|drone|map|diagram|illustration|painting|rendering|after(?:math)?|helicopter)\b', re.I)
NON_TARGET_TEXT = re.compile(
    r'\b(prevention|respirator|mascot|air quality|shrouds new york|consumes new jersey|'
    r'metropolitan transportation authority|from a plane|aeroplane|airplane|satellitenaufnahme)\b', re.I)


def acquisition_priority(row):
    """Order candidates only; search text never certifies a visible target."""
    hint = row['title'] + ' ' + row.get('original_description_html', '')
    flame = bool(re.search(r'\b(flames?|flammes?|llamas?|flammen)\b', hint, re.I))
    dark = bool(re.search(r'\b(night|nuit|noche|nacht|sunset|dusk|evening)\b', hint, re.I))
    return (not (flame and dark), not flame,
            not bool(re.search(r'\b(distant|column|panache|ridge|hills)\b', hint, re.I)),
            hashlib.sha256(str(row['pageid']).encode()).hexdigest())


def legacy_sha1_adapter(page):
    """Losslessly adapt today's hex image SHA-1 to the older validator's base36.

    The original API response remains unchanged in metadata/search-*.json.
    This changes representation only, never the digest value or license rules.
    """
    adapted = copy.deepcopy(page)
    for info in adapted.get('imageinfo', []):
        raw = info.get('sha1', '')
        if re.fullmatch(r'[0-9a-f]{40}', raw):
            number, digits = int(raw, 16), ''
            while number:
                number, digit = divmod(number, 36)
                digits = '0123456789abcdefghijklmnopqrstuvwxyz'[digit] + digits
            info['sha1'] = digits or '0'
            info['upstream_sha1_hex'] = raw
    return adapted


def catalogue(output, *, metadata_cache=None, max_original_bytes=3_000_000, allow_share_alike=False,
              query_profile='general'):
    cache = output / 'metadata'
    cache.mkdir(parents=True, exist_ok=True)
    client = CommonsAPI(max_calls=40, timeout=30, attempts=2, request_delay=.75)
    rows, excluded, seen = [], [], set()
    if query_profile not in {'general', 'active_fire'}:
        raise ValueError('Unknown Commons query profile')
    queries = ACTIVE_FIRE_QUERIES if query_profile == 'active_fire' else QUERIES
    prefix = 'active-fire-' if query_profile == 'active_fire' else ''
    for i, query in enumerate(queries):
        path = cache / f'{prefix}search-{i:02d}.json'
        cached_path = path if path.exists() else metadata_cache / path.name if metadata_cache else path
        if cached_path.exists():
            data = json.loads(cached_path.read_text(encoding='utf-8'))
            if data['query'] != query:
                raise ValueError('Cached Commons search changed')
            response = data['response']
        else:
            response = None
        if response is None or response.get('errors') or response.get('error'):
            response = client.request({'action': 'query', 'generator': 'search',
                'gsrsearch': query + ' filetype:bitmap -aerial -satellite -drone -NASA -map',
                'gsrnamespace': 6, 'gsrlimit': 50, 'prop': 'info|imageinfo|revisions',
                'inprop': 'url', 'iiprop': 'timestamp|sha1|url|size|mime|mediatype|extmetadata',
                'iiurlwidth': 1280, 'iilimit': 1, 'iiextmetadatalanguage': 'en',
                'rvprop': 'ids|timestamp|sha1'})
            if response.get('errors') or response.get('error'):
                raise ValueError('Commons API rejected metadata parameters: ' + json.dumps(response.get('errors', response.get('error'))))
            path.write_text(json.dumps({'query': query, 'response': response}), encoding='utf-8')
        for page in response.get('query', {}).get('pages', []):
            if page['pageid'] in seen:
                continue
            seen.add(page['pageid'])
            discovery = {'pageid': page['pageid'], 'enumerated_title': page['title'],
                         'member_categories': [], 'seed_categories': [], 'discovery_depths': []}
            adapted = legacy_sha1_adapter(page)
            row, reasons, _ = validate_file_page(
                adapted, discovery, allow_declared_us_government_public_domain=True)
            if row is not None:
                info = row['current_file_version']
                description = page['imageinfo'][0].get('extmetadata', {}).get('ImageDescription', {}).get('value', '')
                row['original_description_html'] = description
                row['upstream_api_image_sha1'] = page['imageinfo'][0]['sha1']
                row['stored_image_sha1_encoding'] = 'base36_lossless_legacy_adapter'
                allowed_licenses = {'CC0', 'Public domain', 'CC BY'} | ({'CC BY-SA'} if allow_share_alike else set())
                if row['license_family'] not in allowed_licenses:
                    reasons.append('outside_existing_pilot_license_allowlist')
                if info['bytes'] > max_original_bytes or info['mime'] not in {'image/jpeg', 'image/png'}:
                    reasons.append('original_exceeds_bounded_image_budget_or_format')
                if min(info['width'], info['height']) < 300:
                    reasons.append('too_small_for_reliable_annotation')
                if REJECT_TEXT.search(row['title'] + ' ' + description):
                    reasons.append('out_of_scope_source_text_hint')
                if query_profile == 'active_fire' and NON_TARGET_TEXT.search(row['title'] + ' ' + description):
                    reasons.append('active_fire_profile_non_target_hint_not_visual_review')
            if reasons:
                excluded.append({'pageid': page['pageid'], 'title': page['title'], 'reasons': reasons})
            else:
                rows.append(row)
        print(json.dumps({'metadata_queries': i+1, 'unique_pages': len(seen), 'eligible_candidates': len(rows)}), flush=True)
    (output/'metadata_exclusions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in excluded), encoding='utf-8')
    (output/'source_metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    return rows


class RateLimited(RuntimeError):
    pass


def acquire(row, output, budget, request_delay=1.5, *, max_original_bytes=3_000_000):
    receipt = output / 'receipts' / f"commons-{row['pageid']}.json"
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        if digest(Path(cached['source_image'])) != cached['sha256']:
            raise ValueError('Cached Commons bytes changed')
        return cached
    version = row['current_file_version']
    if version.get('bytes', 0) > max_original_bytes:
        raise ValueError('Commons original exceeds individual byte budget before request')
    if budget.used + version.get('bytes', 0) > budget.limit:
        raise ValueError('Commons storage budget insufficient before request')
    # Sequential worker; caller may slow down to the host's observed allowance.
    # A 429 still pauses the entire batch immediately; there is no retry loop.
    time.sleep(request_delay)
    # Deliberately no Retry adapter: a 429 pauses the batch, never sleeps ten
    # minutes inside a worker or retries every remaining file on the same host.
    with requests.get(version['original_url'], timeout=30, stream=True,
                       headers={'User-Agent': 'FireViewerCorpusBuilder/1.0'}) as response:
        if response.status_code == 429:
            raise RateLimited('HTTP 429; Retry-After=' + response.headers.get('Retry-After', 'unknown'))
        response.raise_for_status()
        payload = bytearray()
        for chunk in response.iter_content(65536):
            budget.add(len(chunk))
            payload.extend(chunk)
            if len(payload) > max_original_bytes:
                raise ValueError('Commons original individual byte budget reached')
    if len(payload) != version['bytes'] or int(hashlib.sha1(payload).hexdigest(), 16) != int(version['sha1'], 36):
        raise ValueError('Commons original differs from recorded file version')
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height = image.size
        if (width, height) != (version['width'], version['height']):
            raise ValueError('Commons dimensions differ from source metadata')
        phash, flip = str(imagehash.phash(image)), str(imagehash.phash(ImageOps.mirror(image)))
    sha = hashlib.sha256(payload).hexdigest()
    image_path = output / 'images' / (sha + ('.png' if version['mime'] == 'image/png' else '.jpg'))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image_path.write_bytes(payload)
    license_name = row['extmetadata']['LicenseShortName'].replace('CC BY-SA ', 'CC-BY-SA-').replace('CC BY ', 'CC-BY-').replace('CC0 ', 'CC0-')
    candidate = row | {'candidate_id': f"Commons-{row['pageid']}", 'source_image': str(image_path.resolve()),
        'image_bytes': len(payload), 'sha256': sha, 'width': width, 'height': height,
        'phash': phash, 'phash_flipped': flip, 'source_family': 'Wikimedia-Commons',
        'source_dataset': 'Wikimedia-Commons', 'source_record_id': row['title'],
        'source_revision': str(row['page_revision']['revid']), 'source_split': 'unsplit', 'split': 'train',
        'license': license_name, 'license_evidence': row['canonical_url'] + '?oldid=' + str(row['page_revision']['revid']),
        'objects': {'bbox': [], 'category': [], 'area': []}, 'annotation_state': 'unannotated',
        'review_status': 'pending_visual_annotation', 'synthetic': False,
        'v8_corpus_admitted': False, 'v8_training_admitted': False}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(candidate), encoding='utf-8')
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=350)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--request-delay', type=float, default=1.5)
    parser.add_argument('--max-original-mb', type=int, default=3)
    parser.add_argument('--budget-mb', type=int, default=600)
    parser.add_argument('--allow-share-alike', action='store_true', help='Retain explicitly verified CC BY-SA license and attribution; no redistribution is performed')
    parser.add_argument('--metadata-cache', type=Path, help='Read an existing immutable search cache without changing it')
    parser.add_argument('--query-profile', choices=['general', 'active_fire'], default='general')
    parser.add_argument('--exclude-acquired', type=Path, help='Skip page IDs whose originals were already acquired in this source root')
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        raise ValueError('Commons acquisition count cap exceeded')
    if not 1.5 <= args.request_delay <= 30:
        raise ValueError('Commons request delay must be between 1.5 and 30 seconds')
    if not 1 <= args.max_original_mb <= 10 or not 1 <= args.budget_mb <= 2000:
        raise ValueError('Commons completion storage cap exceeded')
    args.output.mkdir(parents=True, exist_ok=True)
    catalogue_options = {}
    if args.metadata_cache:
        catalogue_options['metadata_cache'] = args.metadata_cache
    if args.max_original_mb != 3:
        catalogue_options['max_original_bytes'] = args.max_original_mb * 1_000_000
    if args.allow_share_alike:
        catalogue_options['allow_share_alike'] = True
    if args.query_profile != 'general':
        catalogue_options['query_profile'] = args.query_profile
    rows = catalogue(args.output, **catalogue_options)
    if args.metadata_only:
        return
    previously_acquired = {json.loads(p.read_text())['pageid'] for p in (args.exclude_acquired/'receipts').glob('*.json')} if args.exclude_acquired else set()
    sort_key = acquisition_priority if args.query_profile == 'active_fire' else lambda r: hashlib.sha256(str(r['pageid']).encode()).hexdigest()
    rows = sorted((r for r in rows if r['pageid'] not in previously_acquired), key=sort_key)[:args.limit]
    budget = ByteBudget(args.budget_mb * 1_000_000, sum(p.stat().st_size for p in (args.output/'images').glob('*')))
    # Preserve already acquired receipts even when a later pass is throttled.
    accepted = [json.loads(p.read_text()) for p in sorted((args.output/'receipts').glob('*.json'))]
    for cached in accepted:
        if digest(Path(cached['source_image'])) != cached['sha256']:
            raise ValueError('Preserved Commons candidate bytes changed')
    acquired_ids = {row['pageid'] for row in accepted}
    errors, paused = [], threading.Event()
    def worker(row):
        if paused.is_set():
            return None, {'pageid': row['pageid'], 'type': 'SkippedAfterRateLimit', 'error': 'No request sent after host rate limit'}
        try:
            options = {'max_original_bytes': args.max_original_mb * 1_000_000} if args.max_original_mb != 3 else {}
            return acquire(row, args.output, budget, args.request_delay, **options), None
        except Exception as exc:
            if isinstance(exc, RateLimited):
                paused.set()
            return None, {'pageid': row['pageid'], 'type': type(exc).__name__, 'error': str(exc).split(' for url:')[0][:150]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        for i, (row, error) in enumerate(pool.map(worker, rows), 1):
            if row and row['pageid'] not in acquired_ids:
                accepted.append(row)
                acquired_ids.add(row['pageid'])
            elif error:
                errors.append(error)
            if i % 20 == 0:
                print(json.dumps({'processed': i, 'acquired': len(accepted), 'errors': len(errors), 'bytes': budget.used}), flush=True)
    for name, values in (('candidate_manifest.jsonl', accepted), ('acquisition_errors.jsonl', errors)):
        (args.output/name).write_text(''.join(json.dumps(r)+'\n' for r in values), encoding='utf-8')
    (args.output/'acquisition_status.json').write_text(json.dumps({
        'status': 'paused_rate_limited_not_complete' if paused.is_set() else 'bounded_acquisition_complete',
        'acquired_this_selection': len(accepted), 'errors': len(errors),
        'automatic_retry_scheduled': False, 'admitted': 0, 'training_started': False,
        'storage_budget_bytes': args.budget_mb * 1_000_000, 'image_bytes_acquired': budget.used,
        'original_file_cap_bytes': args.max_original_mb * 1_000_000,
        'allow_verified_share_alike': args.allow_share_alike,
        'query_profile': args.query_profile,
        'already_acquired_page_ids_skipped': len(previously_acquired),
        'public_redistribution_performed': False}), encoding='utf-8')
    print(json.dumps({'acquired': len(accepted), 'errors': dict(Counter(r['type'] for r in errors)), 'admitted': 0}), flush=True)


if __name__ == '__main__':
    main()
