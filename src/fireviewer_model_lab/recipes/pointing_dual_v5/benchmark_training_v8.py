"""Bounded real training throughput sweep; never modifies the production weights."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .preflight_deim_v5 import atomic_json, sha256_file


class LimitedLoader:
    def __init__(self, loader, count):
        self.loader, self.count = loader, count

    def __len__(self):
        return self.count

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.count)


def child(args):
    import torch
    sys.path.insert(0, str(args.deim_root.resolve()))
    from engine.core import YAMLConfig
    from engine.misc.dist_utils import setup_seed
    from engine.solver import TASKS
    from engine.solver.det_engine import train_one_epoch
    from engine.optim.lr_scheduler import FlatCosineLRScheduler

    setup_seed(42)
    batch = args.micro_batch
    assert batch in (2, 4, 8) and 16 % batch == 0
    accumulation = 16 // batch
    root = args.coco_root.resolve()
    cfg = YAMLConfig(str(args.config.resolve()), tuning=str(args.weights.resolve()),
                     device='cuda:0', use_amp=True, grad_accum_steps=accumulation,
                     output_dir=str(args.output / f'runtime-b{batch}'),
                     train_dataloader={'total_batch_size': batch, 'dataset': {
                         'img_folder': str(root / 'train'), 'ann_file': str(root / 'train/_annotations.coco.json')}},
                     val_dataloader={'total_batch_size': 2, 'dataset': {
                         'img_folder': str(root / 'valid'), 'ann_file': str(root / 'valid/_annotations.coco.json')}})
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.train()
    solver.train_dataloader.set_epoch(0)
    steps_per_epoch = math.ceil(len(solver.train_dataloader) / accumulation)
    scheduler = FlatCosineLRScheduler(solver.optimizer, cfg.lr_gamma, steps_per_epoch,
                                     cfg.epoches, cfg.warmup_iter, cfg.flat_epoch, cfg.no_aug_epoch)

    def run(steps):
        return train_one_epoch(True, scheduler, solver.model, solver.criterion,
            LimitedLoader(solver.train_dataloader, steps * accumulation), solver.optimizer,
            solver.device, 0, max_norm=cfg.clip_max_norm, print_freq=100000,
            ema=solver.ema, scaler=solver.scaler, writer=solver.writer,
            grad_accum_steps=accumulation, amp_dtype=cfg.amp_dtype)

    run(args.warmup_steps)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    stats = run(args.measure_steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert all(math.isfinite(value) for value in stats.values())
    report = {'status': 'passed', 'scope': 'real train loader + original training loop + optimizer + EMA + logging',
              'micro_batch': batch, 'accumulation': accumulation, 'effective_batch': 16,
              'warmup_optimizer_steps': args.warmup_steps, 'measured_optimizer_steps': args.measure_steps,
              'measured_images': args.measure_steps * 16, 'seconds': elapsed,
              'images_per_second': args.measure_steps * 16 / elapsed,
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'weights_sha256': sha256_file(args.weights),
              'coco_receipt_sha256': sha256_file(root/'coco_view_receipt.json'),
              'engine_sha256': sha256_file(args.deim_root/'engine/solver/det_engine.py'),
              'production_checkpoint_written': False, 'loss': stats['loss']}
    atomic_json(args.output / f'batch-{batch}.json', report)
    if solver.writer:
        solver.writer.close()
    del solver, cfg
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deim-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--coco-root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pause-pid', type=int)
    parser.add_argument('--micro-batch', type=int)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--measure-steps', type=int, default=8)
    args = parser.parse_args()
    if args.micro_batch:
        child(args)
        return
    assert args.pause_pid and args.warmup_steps >= 1 and args.measure_steps >= 4
    cmdline = Path(f'/proc/{args.pause_pid}/cmdline').read_bytes().replace(b'\0', b' ')
    assert b'pointing-deim-dfine-v8-lightning-l4-clean-r1' in cmdline and b'train.py' in cmdline
    args.output.mkdir(parents=True, exist_ok=False)
    def interrupt(signum, frame):
        raise InterruptedError(f'Benchmark interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    paused = False
    results = []
    try:
        os.kill(args.pause_pid, signal.SIGSTOP)
        paused = True
        time.sleep(1)
        assert '\nState:\tT' in Path(f'/proc/{args.pause_pid}/status').read_text()
        atomic_json(args.output/'pause.json', {'pid': args.pause_pid, 'state': 'temporarily_paused', 'at': time.time()})
        for batch in (2, 4, 8):
            command = [sys.executable, __file__, '--deim-root', str(args.deim_root),
                       '--config', str(args.config), '--coco-root', str(args.coco_root),
                       '--weights', str(args.weights), '--output', str(args.output),
                       '--micro-batch', str(batch), '--warmup-steps', str(args.warmup_steps),
                       '--measure-steps', str(args.measure_steps)]
            with (args.output/f'batch-{batch}.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if completed.returncode:
                results.append({'micro_batch': batch, 'status': 'failed', 'exit_code': completed.returncode})
                break
            results.append(json.loads((args.output/f'batch-{batch}.json').read_text()))
            print(json.dumps(results[-1]), flush=True)
    finally:
        if paused:
            os.kill(args.pause_pid, signal.SIGCONT)
            atomic_json(args.output/'resume.json', {'pid': args.pause_pid, 'state': 'resumed', 'at': time.time()})
    atomic_json(args.output/'comparison.json', {'results': results, 'production_resumed': True,
                 'limits': 'Short throughput probe; does not measure final accuracy. Larger micro-batches are not bitwise identical.'})


if __name__ == '__main__':
    main()
