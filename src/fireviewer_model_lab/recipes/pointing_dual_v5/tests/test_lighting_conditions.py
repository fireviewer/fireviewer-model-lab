import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch


@pytest.fixture
def lighting(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'vendor/DEIM'))
    return importlib.import_module('engine.data.transforms.lighting').RandomLightingConditions


@pytest.fixture
def sample():
    pixels = np.arange(64 * 80 * 3, dtype=np.uint8).reshape(64, 80, 3)
    return Image.fromarray(pixels, 'RGB')


@pytest.mark.parametrize('profile', ['morning', 'clear', 'overcast', 'dusk', 'grayscale'])
def test_profiles_change_pixels_without_mutating_source_or_geometry(lighting, sample, profile):
    transform = lighting()
    before = np.asarray(sample).copy()
    output = transform.apply_profile(sample, profile)
    assert output.mode == 'RGB' and output.size == sample.size
    assert not np.array_equal(np.asarray(output), before)
    np.testing.assert_array_equal(np.asarray(sample), before)
    if profile == 'grayscale':
        array = np.asarray(output)
        np.testing.assert_array_equal(array[:, :, 0], array[:, :, 1])
        np.testing.assert_array_equal(array[:, :, 1], array[:, :, 2])


def test_torch_seed_reproduces_random_variants(lighting, sample):
    transform = lighting(weights=[0, .2, .2, .2, .2, .2])
    torch.manual_seed(1234)
    first = [np.asarray(transform(sample)) for _ in range(5)]
    torch.manual_seed(1234)
    second = [np.asarray(transform(sample)) for _ in range(5)]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_targets_and_dataset_metadata_are_untouched(lighting, sample):
    from torchvision import tv_tensors
    transform = lighting(weights=[0, 0, 0, 0, 0, 1])
    boxes = tv_tensors.BoundingBoxes([[2, 3, 22, 34]], format='XYXY', canvas_size=(64, 80))
    target = {'boxes': boxes, 'labels': torch.tensor([1]), 'image_id': torch.tensor([7])}
    dataset = SimpleNamespace(epoch=10)
    image, output, same_dataset = transform((sample, target, dataset))
    assert output['boxes'] is boxes and output['labels'] is target['labels']
    assert output['image_id'] is target['image_id'] and same_dataset is dataset
    assert image.mode == 'RGB'


def test_identity_and_profile_frequency(lighting, sample):
    assert lighting(weights=[1, 0, 0, 0, 0, 0])(sample) is sample
    torch.manual_seed(42)
    transform = lighting()
    profiles = [transform.sample_profile() for _ in range(10000)]
    for name, expected in zip(transform.PROFILES, transform.weights):
        assert abs(profiles.count(name) / len(profiles) - expected) < .02


@pytest.mark.parametrize('weights', [[1], [0] * 6, [1, 0, 0, 0, 0, -.1], [float('nan'), 0, 0, 0, 0, 0]])
def test_invalid_probabilities_rejected(lighting, weights):
    with pytest.raises(ValueError):
        lighting(weights=weights)


def test_train_policy_stops_at_epoch_35_and_validation_stays_unchanged(lighting, sample):
    from engine.core.yaml_utils import load_config
    from engine.data.transforms.container import Compose
    config_root = Path(__file__).resolve().parents[2] / 'vendor/DEIM/configs/fireviewer'
    original = load_config(str(config_root / 'deim_dfine_l_pointing_v8.yml'), {})
    augmented = load_config(str(config_root / 'deim_dfine_l_pointing_v8p1_photometric.yml'), {})
    assert augmented['val_dataloader']['dataset']['transforms'] == original['val_dataloader']['dataset']['transforms']
    transform = lighting(weights=[0, 0, 0, 0, 0, 1])
    compose = Compose([transform], policy={'name': 'stop_epoch', 'epoch': 35, 'ops': ['RandomLightingConditions']})
    target, dataset = {}, SimpleNamespace(epoch=34)
    assert compose(sample, target, dataset)[0] is not sample
    dataset.epoch = 35
    assert compose(sample, target, dataset)[0] is sample
