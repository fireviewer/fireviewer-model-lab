#!/usr/bin/env python3
"""Forward existing DEIM logs to LitLogger without importing or changing its trainer.

Run in the separate logging environment. Log files remain the source of truth;
restarts replay them and LitLogger deduplicates already uploaded metric steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

PROGRESS = re.compile(r'Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)/(\d+)\]')
LOSS = re.compile(r'\bloss:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+)\)')
LR = re.compile(r'\blr:\s*([\d.eE+-]+)')
TIME = re.compile(r'\btime:\s*([\d.eE+-]+)')
COCO_NAMES = ('map', 'map50', 'map75', 'map_small', 'map_medium', 'map_large',
              'ar1', 'ar10', 'ar100', 'ar_small', 'ar_medium', 'ar_large', 'ar50', 'ar75')


def progress_metrics(line, micro_batch, accumulation):
    progress, loss, lr = PROGRESS.search(line), LOSS.search(line), LR.search(line)
    if not (progress and loss and lr):
        return None
    epoch, batch, total = map(int, progress.groups())
    step = epoch * math.ceil(total / accumulation) + batch // accumulation
    metrics = {'train/loss': float(loss[1]), 'train/loss_running_mean': float(loss[2]),
               'train/lr': float(lr[1]), 'train/epoch': float(epoch)}
    timing = TIME.search(line)
    if timing and float(timing[1]) > 0:
        metrics['train/images_per_second_estimate'] = micro_batch / float(timing[1])
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError('Non-finite training metrics')
    return step, metrics


def epoch_metrics(row):
    values = {f'epoch/{key}': float(value) for key, value in row.items()
              if key.startswith('train_') and isinstance(value, (float, int))}
    for name, value in zip(COCO_NAMES, row.get('test_coco_eval_bbox', [])):
        if math.isfinite(value) and value >= 0:
            values[f'validation/{name}'] = float(value)
    return int(row['epoch']) + 1, values


def complete_lines(path, offset):
    """Do not consume a line still being written by the trainer."""
    if not path.exists():
        return [], offset
    with path.open('rb') as stream:
        if path.stat().st_size < offset:
            raise RuntimeError(f'Training log was truncated: {path}')
        stream.seek(offset)
        data = stream.read()
    end = data.rfind(b'\n') + 1
    return data[:end].decode('utf-8').splitlines(), offset + end


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    with os.fdopen(handle, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    os.replace(temporary, path)


def forward(experiment, last_steps, step, metrics):
    count = 0
    for key, value in metrics.items():
        if step <= last_steps.get(key, -1):
            continue
        experiment[key].append(value, step=step)
        last_steps[key] = step
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    state_path = Path(config['state_path'])
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    # These are confirmed synchronous artifact uploads, not metric queue cursors.
    uploaded = previous.get('uploaded_checkpoints', {})
    import litlogger
    experiment = litlogger.init(name=config['name'], teamspace=config['teamspace'],
                                root_dir=str(state_path.parent / 'litlogger-cache'),
                                save_logs=False, metadata={'model': 'DEIM D-FINE L',
                                'dataset': config.get('dataset_label', 'V8 5000 real images'), 'epochs': '40',
                                'effective_batch_size': '16', 'resolution': '960',
                                'logging_source': 'actual DEIM console and COCO validation logs'})
    if experiment.teamspace.name != config['teamspace']:
        raise RuntimeError('Refusing to log outside the requested teamspace')
    offsets, last_steps, sent = {}, {}, 0
    state = {'status': 'running', 'pid': os.getpid(), 'url': experiment.url,
             'experiment_id': experiment.id, 'uploaded_checkpoints': uploaded}
    atomic_json(state_path, state)
    print(json.dumps({'experiment_url': experiment.url, 'pid': os.getpid()}), flush=True)
    try:
        while True:
            for source in config['sources']:
                directory = Path(source['run_dir'])
                manifest_path = directory / 'run_manifest.json'
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text())
                if directory.name not in state.get('registered_sources', []):
                    experiment['manifest_' + directory.name] = litlogger.File(str(manifest_path))
                    experiment['config_' + directory.name] = litlogger.File(manifest['config']['path'])
                    state.setdefault('registered_sources', []).append(directory.name)
                for kind, path in [('progress', Path(source['console_log'])), ('epochs', directory / 'log.txt')]:
                    lines, offsets[str(path)] = complete_lines(path, offsets.get(str(path), 0))
                    for line in lines:
                        if kind == 'progress':
                            result = progress_metrics(line, source['micro_batch'], source['accumulation'])
                        else:
                            row = json.loads(line)
                            result = epoch_metrics(row)
                            state['latest_completed_epoch'] = max(int(row['epoch']), state.get('latest_completed_epoch', -1))
                        if result:
                            sent += forward(experiment, last_steps, *result)
                # Publish a stable resumable checkpoint at the first validation,
                # each ten-epoch boundary and completion. Never upload a live file.
                epoch_log = directory / 'log.txt'
                if epoch_log.exists():
                    rows, _ = complete_lines(epoch_log, 0)
                    if rows:
                        epoch = int(json.loads(rows[-1])['epoch'])
                        key = f'{directory.name}:epoch-{epoch:04d}'
                        if (epoch == 0 or (epoch + 1) % 10 == 0) and key not in uploaded:
                            checkpoint = directory / 'last.pth'
                            staging = state_path.parent / 'litlogger-checkpoints'
                            staging.mkdir(exist_ok=True)
                            snapshot = staging / f'{directory.name}-epoch-{epoch:04d}.pth'
                            before = checkpoint.stat()
                            shutil.copyfile(checkpoint, snapshot)
                            if before.st_mtime_ns != checkpoint.stat().st_mtime_ns:
                                snapshot.unlink()
                                continue
                            import torch
                            payload = torch.load(snapshot, map_location='cpu', weights_only=True)
                            if payload['last_epoch'] != epoch:
                                del payload
                                snapshot.unlink()
                                continue
                            assert all(k in payload for k in ('model', 'optimizer', 'ema'))
                            del payload
                            with snapshot.open('rb') as stream:
                                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                            experiment['checkpoint_epoch_' + str(epoch)] = litlogger.File(str(snapshot))
                            uploaded[key] = {'epoch': epoch, 'sha256': digest, 'bytes': snapshot.stat().st_size,
                                             'artifact_key': 'checkpoint_epoch_' + str(epoch)}
                            atomic_json(state_path, dict(state, uploaded_checkpoints=uploaded))
                            snapshot.unlink()
                state['active_run'] = str(directory)
                state['trainer_status'] = manifest['status']
            state.update(updated_at_unix=time.time(), metrics_enqueued=sent, last_steps=last_steps)
            atomic_json(state_path, state)
            terminal = state.get('active_run') == config['sources'][-1]['run_dir'] and state.get('trainer_status') in ('completed', 'failed')
            if args.once or terminal:
                break
            time.sleep(config.get('poll_seconds', 10))
        experiment.finalize()
        state.update(status='flushed', finalized_at_unix=time.time())
        atomic_json(state_path, state)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
