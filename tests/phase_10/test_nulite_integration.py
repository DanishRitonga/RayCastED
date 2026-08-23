"""Phase 10 tests — NuLite model integration.

Validates that the v1 framework (RayCastTrainer) can build and wire the NuLite
architecture through the `architecture: nulite` config branch.

Run with: uv run python -m pytest tests/phase_10/test_nulite_integration.py -x -q
"""

import os
import tempfile

import numpy as np
import yaml

from raycasted.model.head import RayCastDetect
from raycasted.model.loss import RayCastE2ELoss
from raycasted.model.nulite_model import NuLiteRayCastModel
from raycasted.model.train import RayCastTrainer

NULITE_TCFG = {
    'architecture': 'nulite',
    'n_rays': 64,
    'pretrained': False,
    'lambda_seg': 1.0,
    'hierarchical_cls': True,
    'hierarchical_cls_detach': True,
    'head_channel_scale': 0.5,
    'head_channel_min': 64,
    'cls_channel_scale': 1.0,
    'cls_channel_min': 0,
}


def _make_data_yaml(tmp_dir, nc=5, imgsz=256):
    """Create a data YAML + synthetic tile dirs (n_rays=64). Returns YAML path."""
    for split in ('train', 'val'):
        split_dir = os.path.join(tmp_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(2):
            image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
            annotations = np.zeros((3, 3 + 64), dtype=np.float32)
            annotations[:, 0] = np.random.randint(0, nc, 3)
            annotations[:, 1] = np.random.uniform(0.1, 0.9, 3)
            annotations[:, 2] = np.random.uniform(0.1, 0.9, 3)
            annotations[:, 3:] = np.random.uniform(0.01, 0.05, (3, 64))
            np.savez(
                os.path.join(split_dir, f'tile_{i}.npz'),
                image=image,
                annotations=annotations,
                tissue=np.int32(1),
                content_h=np.int32(imgsz),
                content_w=np.int32(imgsz),
            )
    data = {
        'train': os.path.join(tmp_dir, 'train'),
        'val': os.path.join(tmp_dir, 'val'),
        'nc': nc,
        'names': {i: f'class_{i}' for i in range(nc)},
    }
    yaml_path = os.path.join(tmp_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path


def _make_trainer(tmp_dir, tcfg=None):
    """Create a RayCastTrainer with synthetic data + nulite training_config."""
    yaml_path = _make_data_yaml(tmp_dir)
    overrides = {
        'model': 'yolo11n.yaml',
        'data': yaml_path,
        'epochs': 1,
        'batch': 2,
        'imgsz': 256,
        'mosaic': 0.0,
        'mixup': 0.0,
        'cache': False,
        'pretrained': False,
        'device': 'cpu',
        'workers': 0,
        'verbose': False,
    }
    return RayCastTrainer(overrides=overrides, training_config=tcfg or NULITE_TCFG)


def _setup_model(trainer):
    """Mimic partial _setup_train: create model + set attributes."""
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    return trainer.model


def test_get_model_returns_nulite():
    """get_model with architecture=nulite returns NuLiteRayCastModel."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        assert isinstance(model, NuLiteRayCastModel), type(model).__name__
        print('PASS: get_model returns NuLiteRayCastModel')


def test_nulite_head_is_raycast():
    """NuLite model's model[-1] is a RayCastDetect head."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        head = model.model[-1]
        assert isinstance(head, RayCastDetect), type(head).__name__
        assert head.n_rays == 64
        assert head.nc == 5
        print(f'PASS: head is RayCastDetect — nc={head.nc}, n_rays={head.n_rays}')


def test_nulite_stride_and_end2end():
    """NuLite model has stride [4,8,16] and end2end=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        assert model.stride.tolist() == [4.0, 8.0, 16.0], model.stride
        assert model.end2end is True
        print('PASS: stride=[4,8,16], end2end=True')


def test_nulite_criterion_is_raycast_e2e():
    """init_criterion returns RayCastE2ELoss."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        criterion = model.init_criterion()
        assert isinstance(criterion, RayCastE2ELoss), type(criterion).__name__
        print('PASS: init_criterion returns RayCastE2ELoss')


def test_nulite_training_args():
    """set_model_attributes stores training_args with nulite strides."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        ta = model.training_args
        assert ta['n_rays'] == 64
        assert ta['strides'] == [4.0, 8.0, 16.0]
        assert ta['crop_size'] == 256
        print(f'PASS: training_args — {ta}')


def test_nulite_forward_shapes():
    """Forward pass in train/eval mode yields correct shapes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        x = np.random.rand(1, 3, 256, 256).astype(np.float32)
        import torch

        t = torch.from_numpy(x)

        model.train()
        preds = model(t)
        assert isinstance(preds, dict)
        assert 'np' in preds and 'one2many' in preds and 'one2one' in preds
        assert preds['np'].shape == (1, 1, 256, 256)

        model.eval()
        out = model(t)
        assert isinstance(out, tuple)
        assert out[0].shape == (1, 100, 68)  # B, max_det, raycast_dim+2
        print('PASS: train preds dict + eval (1,100,68) tuple')
