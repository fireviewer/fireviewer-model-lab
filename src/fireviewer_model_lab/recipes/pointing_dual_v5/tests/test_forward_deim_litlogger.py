import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('forward_deim_litlogger', Path(__file__).parents[1] / 'forward_deim_litlogger.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('batch,total,accum,micro,expected', [(50, 2060, 8, 2, 264), (50, 515, 2, 8, 283)])
def test_optimizer_steps_across_microbatch_change(batch, total, accum, micro, expected):
    line = f'Epoch: [1]  [ {batch}/{total}] eta: 1:00 lr: 0.000004 loss: 71.27 (68.25) time: 0.4 data: 0.001'
    step, values = module.progress_metrics(line, micro, accum)
    assert step == expected
    assert values['train/loss'] == 71.27
    assert values['train/images_per_second_estimate'] == micro / 0.4


def test_validation_is_not_training_loss_and_ignores_unavailable_area_metrics():
    step, values = module.epoch_metrics({'epoch': 0, 'train_loss': 74.2, 'test_coco_eval_bbox': [.009, .03, .002, -1]})
    assert step == 1
    assert values == {'epoch/train_loss': 74.2, 'validation/map': .009, 'validation/map50': .03, 'validation/map75': .002}


def test_partial_log_line_is_retried_and_truncation_rejected(tmp_path):
    path = tmp_path / 'train.log'
    path.write_bytes(b'first\nsec')
    lines, offset = module.complete_lines(path, 0)
    assert (lines, offset) == (['first'], 6)
    with path.open('ab') as stream:
        stream.write(b'ond\n')
    assert module.complete_lines(path, offset) == (['second'], 13)
    path.write_bytes(b'')
    with pytest.raises(RuntimeError, match='truncated'):
        module.complete_lines(path, offset)


def test_replayed_progress_does_not_regress_steps():
    points = []
    experiment = {'train/loss': SimpleNamespace(append=lambda value, step: points.append((step, value)))}
    steps = {}
    assert module.forward(experiment, steps, 10, {'train/loss': 2.0}) == 1
    assert module.forward(experiment, steps, 10, {'train/loss': 3.0}) == 0
    assert module.forward(experiment, steps, 9, {'train/loss': 4.0}) == 0
    assert points == [(10, 2.0)]


def test_non_progress_lines_are_ignored():
    assert module.progress_metrics('Test: [1/2] time: 0.1', 8, 2) is None
