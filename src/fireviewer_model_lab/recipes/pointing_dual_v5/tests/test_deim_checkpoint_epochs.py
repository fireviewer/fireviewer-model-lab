"""Regression: stage-two reloads must not falsify resumable epoch numbers."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("save_every_epoch,expected", [(True, [0, 1, 2]), (False, [0])])
def test_checkpoint_epoch_after_best_state_reload(monkeypatch, tmp_path, save_every_epoch, expected):
    root = Path(__file__).resolve().parents[2] / "vendor/DEIM"
    monkeypatch.syspath_prepend(str(root))
    module = importlib.import_module("engine.solver.det_solver")
    epochs_seen, saved = [], []
    loader = SimpleNamespace(set_epoch=epochs_seen.append,
                             collate_fn=SimpleNamespace(stop_epoch=1, ema_restart_decay=.99))
    cfg = SimpleNamespace(lrsheduler=None, epoches=3, checkpoint_freq=10,
                          clip_max_norm=.1, print_freq=50, grad_accum_steps=1,
                          amp_dtype="bf16", yaml_cfg={"save_last_every_epoch": save_every_epoch})
    solver = object.__new__(module.DetSolver)
    solver.cfg = cfg
    solver.train = lambda: None
    solver.model = SimpleNamespace(parameters=lambda: [torch.nn.Parameter(torch.ones(1))])
    solver.criterion = solver.postprocessor = solver.optimizer = solver.evaluator = object()
    solver.train_dataloader, solver.val_dataloader = loader, None
    solver.lr_scheduler = SimpleNamespace(step=lambda: None)
    solver.lr_warmup_scheduler = solver.scaler = solver.writer = None
    solver.ema = SimpleNamespace(module=object(), decay=.9999)
    solver.device, solver.output_dir, solver.last_epoch = "cpu", tmp_path, -1
    solver.state_dict = lambda: {"last_epoch": solver.last_epoch}
    # Emulate a best-stage checkpoint with older metadata than the current loop.
    solver.load_resume_state = lambda _: setattr(solver, "last_epoch", -7)
    monkeypatch.setattr(module, "stats", lambda _: (1, "test model"))
    monkeypatch.setattr(module, "train_one_epoch", lambda *a, **kw: {"loss": 1.0})
    monkeypatch.setattr(module, "evaluate", lambda *a, **kw: ({"coco_eval_bbox": [.1 + epochs_seen[-1]*.1]}, None))
    monkeypatch.setattr(module.dist_utils, "is_dist_available_and_initialized", lambda: False)
    monkeypatch.setattr(module.dist_utils, "is_main_process", lambda: True)
    monkeypatch.setattr(module.dist_utils, "save_on_master", lambda state, path: saved.append((path.name, state["last_epoch"])))
    solver.fit()
    assert [epoch for name, epoch in saved if name == "last.pth"] == expected
    assert solver.last_epoch == 2


def test_health_probe_handles_fresh_schedule_without_inventing_validation(monkeypatch, tmp_path):
    import importlib.util
    import json
    root = Path(__file__).resolve().parents[3]
    path = root / 'artifacts/local/pointing-v8-completion-20260828/lightning-control/training_health_probe.py'
    spec = importlib.util.spec_from_file_location('fresh_schedule_health', path)
    health = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(health)
    run = tmp_path / 'new-run'
    run.mkdir()
    (run / 'run_manifest.json').write_text(json.dumps({'status': 'training'}))
    console = tmp_path / 'train.log'
    console.write_text('Epoch: [0] [0/847] eta: 0:12:00 loss: 40.0 (40.0)\n')
    reference = {'source_epoch': 7, 'map': .3037360553}
    (tmp_path / 'active_training_run.json').write_text(json.dumps({
        'new_run': str(run), 'old_run': str(tmp_path / 'previous'),
        'console_log': str(console), 'trainer_pid': 999999999,
        'history_sources': [{'run_dir': str(run)}], 'initial_reference': reference,
        'initialization_checkpoint': 'verified_epoch7.pth'}))
    monkeypatch.setattr(health, 'CONTROL', tmp_path)
    reports = []
    monkeypatch.setattr(health, 'write_report', reports.append)
    health.main()
    assert reports[0]['validation_history'] == []
    assert reports[0]['best_validation'] is None
    assert reports[0]['initial_reference'] == reference
    assert reports[0]['checkpoint'] is None
    assert 'First completed validation' in reports[0]['pending']
