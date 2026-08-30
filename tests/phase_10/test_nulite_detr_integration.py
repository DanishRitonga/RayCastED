"""Phase 10 tests — NuLite DETR model integration.

Validates that the v1 framework wires the single-head DETR architecture
(`architecture: nulite_detr`) and that the lightweight backbone variants
build correctly: FastViT s12 / t8 (NuLite decoder) and YOLO11n (FPN top-down
+ stride-4 NP seed head).

Run with: uv run python -m pytest tests/phase_10/test_nulite_detr_integration.py -x -q
"""

import os
import tempfile

import numpy as np
import torch
import yaml

from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder
from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel
from raycasted.model.train import RayCastTrainer

DETR_TCFG = {
    'architecture': 'nulite_detr',
    'n_rays': 64,
    'pretrained': False,
    'lambda_seg': 1.0,
    'seed_map_target': True,
    'detr_ndl': 3,
    'detr_hd': 256,
    'detr_d_ffn': 1024,
    'detr_nq': 300,
    'detr_grid_size': 0.05,
    'detr_query_selection': 'grid',
    'detr_seed_feature': True,
    'detr_no_object': True,
    'detr_seed_in_content': True,
    'detr_no_object_weight': 2.0,
    'detr_mds': False,
    'detr_cost_inside': 10.0,
    'detr_use_decoder': True,
    'detr_backbone': 'nulite',
    'detr_use_seed': True,
    'yolo11_scale': 'n',
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
    """Create a RayCastTrainer with synthetic data + nulite_detr training_config."""
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
    return RayCastTrainer(overrides=overrides, training_config=tcfg or DETR_TCFG)


def _setup_model(trainer):
    """Mimic partial _setup_train: create model + set attributes."""
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    return trainer.model


def _synthetic_batch(bs=2, anns=7, h=256, nc=5, n_rays=64):
    """Build a collate-compatible batch dict (normalized bboxes + seed_map)."""
    torch.manual_seed(0)
    bboxes = torch.rand(anns, 2 + n_rays)
    bboxes[:, 0] = 0.2 + 0.6 * torch.rand(anns)
    bboxes[:, 1] = 0.2 + 0.6 * torch.rand(anns)
    bboxes[:, 2:] = 0.02 + 0.04 * torch.rand(anns, n_rays)
    seed_map = torch.zeros(bs, 1, h, h)
    seed_map[:, :, 40:50, 40:50] = 1.0
    return {
        'img': torch.rand(bs, 3, h, h),
        'batch_idx': torch.zeros(anns, dtype=torch.long),
        'cls': torch.randint(0, nc, (anns,)),
        'bboxes': bboxes,
        'seed_map': seed_map,
    }


def test_get_model_returns_nulite_detr():
    """get_model with architecture=nulite_detr returns NuLiteRayCastDETRModel."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        assert isinstance(model, NuLiteRayCastDETRModel), type(model).__name__
        print('PASS: get_model returns NuLiteRayCastDETRModel')


def test_detr_head_is_decoder():
    """model[-1] is a NuLiteDETRDecoder with nc=5, n_rays=64, stride [4,8,16]."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        head = model.model[-1]
        assert isinstance(head, NuLiteDETRDecoder), type(head).__name__
        assert head.nc == 5
        assert head.n_rays == 64
        assert model.stride.tolist() == [4.0, 8.0, 16.0]
        print('PASS: head is NuLiteDETRDecoder — nc=5, n_rays=64, stride=[4,8,16]')


def test_detr_eval_forward_detect_format():
    """Eval forward yields (B, nq, 68) Detect format in pixel coords."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        x = torch.zeros(1, 3, 256, 256)
        model.eval()
        y_detect, out = model(x)
        assert y_detect.shape == (1, 400, 68), y_detect.shape
        assert y_detect[0, :, 0].min() >= 0.0  # pixel coords, no negatives
        assert (y_detect[0, :, 66] >= 0.0).all() and (y_detect[0, :, 66] <= 1.0).all()  # conf in [0,1]
        assert (y_detect[0, :, 67].long() >= 0).all() and (y_detect[0, :, 67].long() < 5).all()  # cls ids
        print('PASS: eval (1,400,68) Detect format, pixel-scaled')


def test_detr_loss_backward():
    """Train loss backward is finite (s12, full seed path)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _setup_model(_make_trainer(tmp_dir))
        model.train()
        total, items = model.loss(_synthetic_batch())
        total.backward()
        assert torch.isfinite(total).item()
        assert items.shape == (5,)
        n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
        n_tot = sum(1 for p in model.parameters())
        assert n_grad > 0.8 * n_tot, f'grad coverage {n_grad}/{n_tot}'
        print(f'PASS: loss backward finite ({total.item():.1f}), grads {n_grad}/{n_tot}')


def test_fastvit_t8_variant():
    """fastvit_t8 builds with auto-scaled channels (~8.9M params)."""
    model = NuLiteRayCastDETRModel(nc=5, n_rays=64, variant='fastvit_t8', pretrained=False, verbose=False)
    n_params = sum(p.numel() for p in model.parameters())
    assert 8.5e6 < n_params < 9.3e6, n_params
    assert model.decoder0.conv.in_channels == 3
    assert model.np_head.conv1.in_channels == 96  # 2 * dims[0] (48)
    assert [p[0].in_channels for p in model.detr.input_proj] == [49, 97, 193]
    print(f'PASS: fastvit_t8 — {n_params / 1e6:.2f}M params, seed 96ch, input_proj [49,97,193]')


def test_yolo11_backbone_stride4_seed():
    """yolo11n backbone builds with stride-4 NP seed head (~5.4M params)."""
    model = NuLiteRayCastDETRModel(
        nc=5, n_rays=64, backbone='yolo11', pretrained=False, use_seed=True, lambda_seg=1.0, verbose=False
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 5.0e6 < n_params < 5.8e6, n_params
    assert model.decoder0 is None
    assert model.np_head.conv1.in_channels == 32  # P2 width
    assert [p[0].in_channels for p in model.detr.input_proj] == [33, 65, 129]
    model.eval()
    y_detect, _ = model(torch.zeros(1, 3, 256, 256))
    assert y_detect.shape == (1, 400, 68)
    assert model._last_seed.shape == (1, 1, 64, 64)  # stride-4 seed map
    model.train()
    total, _ = model.loss(_synthetic_batch())
    total.backward()
    assert torch.isfinite(total).item()
    n_grad = sum(1 for p in model.np_head.parameters() if p.grad is not None)
    assert n_grad == 5, f'np_head grads {n_grad}/5'
    print(f'PASS: yolo11n — {n_params / 1e6:.2f}M params, seed (1,1,64,64), np_head 5/5 grads')
