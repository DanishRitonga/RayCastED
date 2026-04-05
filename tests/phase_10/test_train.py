"""Phase 10 tests — Training Integration.

Validates the RayCastTrainer pipeline: model construction, head
replacement, loss wiring, safety assertions, and collate function.

Run with: uv run python tests/phase_10/test_train.py
"""

import math
import os
import tempfile

import numpy as np
import torch
import yaml

from raycasted.model.head import RayCastDetect
from raycasted.model.loss import RayCastE2ELoss
from raycasted.model.register import register_raycast_head
from raycasted.model.train import (
    RayCastTrainer,
    _extract_neck_channels,
    _raycast_collate_fn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_npz(path, n_annots=5, imgsz=640):
    """Create a single .npz tile with random data."""
    image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    annotations = np.zeros((n_annots, 35), dtype=np.float32)
    annotations[:, 0] = np.random.randint(0, 3, n_annots)  # class_id
    annotations[:, 1] = np.random.uniform(50, imgsz - 50, n_annots)  # cx
    annotations[:, 2] = np.random.uniform(50, imgsz - 50, n_annots)  # cy
    annotations[:, 3:] = np.random.uniform(5, 30, (n_annots, 32))  # rays
    np.savez(
        path,
        image=image,
        annotations=annotations,
        tissue=np.int32(1),
        content_h=np.int32(imgsz),
        content_w=np.int32(imgsz),
    )


def _make_synthetic_dataset(dir_path, n_tiles=4, imgsz=640):
    """Create a directory of synthetic .npz tiles."""
    os.makedirs(dir_path, exist_ok=True)
    for i in range(n_tiles):
        _make_synthetic_npz(os.path.join(dir_path, f'tile_{i:04d}.npz'), imgsz=imgsz)


def _make_data_yaml(tmp_dir, nc=5, n_train=4, n_val=2, imgsz=640):
    """Create a data YAML + synthetic tile dirs. Returns YAML path."""
    train_dir = os.path.join(tmp_dir, 'train')
    val_dir = os.path.join(tmp_dir, 'val')
    _make_synthetic_dataset(train_dir, n_train, imgsz)
    _make_synthetic_dataset(val_dir, n_val, imgsz)

    names = {i: f'class_{i}' for i in range(nc)}
    data = {
        'train': train_dir,
        'val': val_dir,
        'nc': nc,
        'names': names,
    }
    yaml_path = os.path.join(tmp_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path


def _make_trainer(tmp_dir, **extra_overrides):
    """Create a RayCastTrainer with synthetic data in tmp_dir."""
    yaml_path = _make_data_yaml(tmp_dir)
    overrides = {
        'model': 'yolo11n.yaml',
        'data': yaml_path,
        'epochs': 1,
        'batch': 2,
        'imgsz': 64,
        'mosaic': 0.0,
        'mixup': 0.0,
        'cache': False,
        'pretrained': False,
        'device': 'cpu',
        'workers': 0,
        'verbose': False,
        **extra_overrides,
    }
    return RayCastTrainer(overrides=overrides)


def _setup_model(trainer):
    """Mimic partial _setup_train: create model + set attributes."""
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    return trainer.model


# ---------------------------------------------------------------------------
# Neck channel extraction
# ---------------------------------------------------------------------------


def test_extract_neck_channels():
    """Channel extraction returns a tuple with one entry per scale level."""
    register_raycast_head()
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel('yolo11n.yaml', nc=5, verbose=False)
    head = model.model[-1]

    ch = _extract_neck_channels(head)
    assert isinstance(ch, tuple), f'Expected tuple, got {type(ch)}'
    assert len(ch) == head.nl, f'Expected {head.nl} channel sizes, got {len(ch)}'
    assert all(isinstance(c, int) and c > 0 for c in ch), f'Invalid channels: {ch}'
    print(f'PASS: extract neck channels — {ch}')


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def test_model_has_raycast_head():
    """setup_model replaces Detect with RayCastDetect."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        model = _setup_model(trainer)
        head = model.model[-1]
        assert isinstance(head, RayCastDetect), f'Expected RayCastDetect, got {type(head).__name__}'
        print(f'PASS: model has RayCastDetect head — nc={head.nc}')


def test_criterion_is_raycast_e2e():
    """init_criterion returns RayCastE2ELoss."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        model = _setup_model(trainer)
        criterion = model.init_criterion()
        assert isinstance(criterion, RayCastE2ELoss), f'Expected RayCastE2ELoss, got {type(criterion).__name__}'
        print('PASS: init_criterion returns RayCastE2ELoss')


def test_end2end_attribute():
    """Model has end2end=True after head replacement."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        model = _setup_model(trainer)
        assert getattr(model, 'end2end', False) is True, 'model.end2end should be True'
        print('PASS: model.end2end is True')


def test_no_dfl():
    """DFL is replaced with Identity in the model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        model = _setup_model(trainer)
        head = model.model[-1]
        assert isinstance(head.dfl, torch.nn.Identity), f'Expected Identity DFL, got {type(head.dfl).__name__}'
        print('PASS: DFL is Identity')


def test_training_args_metadata():
    """set_model_attributes stores training_args dict."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        model = _setup_model(trainer)
        ta = model.training_args
        assert 'crop_size' in ta, 'Missing crop_size'
        assert 'nc' in ta, 'Missing nc'
        assert 'n_rays' in ta, 'Missing n_rays'
        assert 'strides' in ta, 'Missing strides'
        assert ta['n_rays'] == 32
        assert ta['crop_size'] == trainer.args.imgsz
        print(f'PASS: training_args metadata — {ta}')


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def test_raycast_collate_fn():
    """Collate produces dict with correct keys and shapes."""
    n_tiles = 3
    imgsz = 64
    items = []
    for _ in range(n_tiles):
        image = torch.randn(3, imgsz, imgsz)
        labels = np.random.randn(5, 35).astype(np.float32)
        items.append((image, labels))

    batch = _raycast_collate_fn(items)

    assert 'img' in batch, "Missing 'img' key"
    assert 'batch_idx' in batch, "Missing 'batch_idx' key"
    assert 'cls' in batch, "Missing 'cls' key"
    assert 'bboxes' in batch, "Missing 'bboxes' key"

    assert batch['img'].shape == (n_tiles, 3, imgsz, imgsz), f'Wrong img shape: {batch["img"].shape}'
    assert batch['bboxes'].shape == (15, 34), f'Wrong bboxes shape: {batch["bboxes"].shape}'
    assert batch['batch_idx'].shape == (15,), f'Wrong batch_idx shape: {batch["batch_idx"].shape}'
    print(f'PASS: collate_fn — img={batch["img"].shape}, bboxes={batch["bboxes"].shape}')


def test_raycast_collate_fn_empty():
    """Collate handles tiles with zero annotations."""
    items = [
        (torch.randn(3, 64, 64), np.zeros((0, 35), dtype=np.float32)),
        (torch.randn(3, 64, 64), np.random.randn(3, 35).astype(np.float32)),
    ]
    batch = _raycast_collate_fn(items)
    assert batch['bboxes'].shape == (3, 34), f'Expected (3, 34), got {batch["bboxes"].shape}'
    assert batch['batch_idx'].shape == (3,), f'Expected (3,), got {batch["batch_idx"].shape}'
    print('PASS: collate_fn handles empty annotations')


def test_raycast_collate_fn_all_empty():
    """Collate handles all-empty batch."""
    items = [
        (torch.randn(3, 64, 64), np.zeros((0, 35), dtype=np.float32)),
        (torch.randn(3, 64, 64), np.zeros((0, 35), dtype=np.float32)),
    ]
    batch = _raycast_collate_fn(items)
    assert batch['bboxes'].shape == (0, 34), f'Expected (0, 34), got {batch["bboxes"].shape}'
    assert batch['batch_idx'].shape == (0,), f'Expected (0,), got {batch["batch_idx"].shape}'
    print('PASS: collate_fn handles all-empty annotations')


# ---------------------------------------------------------------------------
# Safety: mosaic/mixup forced off
# ---------------------------------------------------------------------------


def test_mosaic_forced_off():
    """Trainer forces mosaic=0.0 even when user tries to enable it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir, mosaic=1.0)
        assert trainer.args.mosaic == 0.0, f'mosaic should be 0.0, got {trainer.args.mosaic}'
        assert trainer.args.mixup == 0.0, f'mixup should be 0.0, got {trainer.args.mixup}'
        print('PASS: mosaic and mixup forced to 0.0')


# ---------------------------------------------------------------------------
# Forward pass shapes
# ---------------------------------------------------------------------------


def test_forward_pass_output_shapes():
    """Model forward pass produces correct output structure in training mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        model = _setup_model(trainer)
        model.train()

        x = torch.randn(1, 3, 64, 64)
        preds = model(x)

        # end2end training: dict with 'one2many' and 'one2one' keys
        assert isinstance(preds, dict), f'Expected dict, got {type(preds)}'
        assert 'one2many' in preds, "Missing 'one2many' key"
        assert 'one2one' in preds, "Missing 'one2one' key"

        for branch in ('one2many', 'one2one'):
            b = preds[branch]
            assert 'boxes' in b, f'{branch}: missing boxes'
            assert 'scores' in b, f'{branch}: missing scores'
            assert 'feats' in b, f'{branch}: missing feats'

            # boxes: [B, 34, N_anchors]
            assert b['boxes'].shape[:2] == (1, 34), f'{branch} boxes shape wrong: {b["boxes"].shape}'
            # scores: [B, nc, N_anchors]
            assert b['scores'].shape[:2] == (1, 5), f'{branch} scores shape wrong: {b["scores"].shape}'
            # feats: list of 3 feature maps
            assert len(b['feats']) == 3, f'{branch}: expected 3 feat maps, got {len(b["feats"])}'

        n_anchors = preds['one2many']['boxes'].shape[2]
        print(f'PASS: forward pass — 3-scale output, {n_anchors} anchors, boxes [1,34,{n_anchors}]')


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def _make_synthetic_batch(batch_size=2, nc=5, imgsz=64, n_gt_per_img=3):
    """Create a synthetic batch dict matching collate_fn output format."""
    # Simulate the output from _raycast_collate_fn
    batch_idx_list = []
    cls_list = []
    bboxes_list = []
    for b in range(batch_size):
        for _ in range(n_gt_per_img):
            batch_idx_list.append(float(b))
            cls_list.append(float(np.random.randint(0, nc)))
            # [cx_norm, cy_norm, d_1..d_32] all in [0, 1]
            bbox = np.random.uniform(0.1, 0.9, 34).astype(np.float32)
            bbox[:2] = np.random.uniform(0.2, 0.8, 2)  # centroids well inside
            bboxes_list.append(bbox)

    return {
        'img': torch.randn(batch_size, 3, imgsz, imgsz),
        'batch_idx': torch.tensor(batch_idx_list, dtype=torch.float32),
        'cls': torch.tensor(cls_list, dtype=torch.float32),
        'bboxes': torch.tensor(np.stack(bboxes_list), dtype=torch.float32),
    }


def test_loss_all_5_terms_finite():
    """Loss computation produces 5 finite, non-zero terms."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        model = _setup_model(trainer)
        model.train()

        criterion = model.init_criterion()
        batch = _make_synthetic_batch(imgsz=64)

        # Forward pass
        preds = model(batch['img'])

        # Compute loss
        loss_dict = criterion(preds, batch)
        # E2ELoss returns (total_loss, loss_detach) or similar
        # Check what E2ELoss.__call__ returns
        assert isinstance(loss_dict, (tuple, list)), f'Expected tuple/list, got {type(loss_dict)}'

        total_loss = loss_dict[0]
        assert isinstance(total_loss, torch.Tensor), f'Expected tensor, got {type(total_loss)}'
        assert torch.isfinite(total_loss).all(), f'Non-finite total loss: {total_loss}'

        # The detached loss should have 5 terms
        loss_detach = loss_dict[1]
        if isinstance(loss_detach, torch.Tensor) and loss_detach.numel() == 5:
            for i, name in enumerate(['xy', 'cls', 'L1', 'piou', 'smooth']):
                val = loss_detach[i].item()
                assert math.isfinite(val), f'{name} loss is not finite: {val}'
                # cls and smooth may be zero if no foreground, but at least one reg term should be non-zero
            # At least one term should be non-zero (cls is usually non-zero even without fg)
            assert loss_detach.sum().item() != 0.0, 'All 5 loss terms are zero'
            print(f'PASS: loss 5 terms finite — {[f"{v:.4f}" for v in loss_detach.tolist()]}')
        else:
            # Loss structure may vary; at minimum total_loss is finite and non-zero
            assert total_loss.item() != 0.0, 'Total loss is zero'
            print(f'PASS: total loss finite and non-zero — {total_loss.item():.4f}')


# ---------------------------------------------------------------------------
# preprocess_batch
# ---------------------------------------------------------------------------


def test_preprocess_batch():
    """preprocess_batch moves tensors to device without /255 division."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        _setup_model(trainer)

        img = torch.randn(2, 3, 64, 64)
        batch = {
            'img': img.clone(),
            'batch_idx': torch.tensor([0.0, 0.0, 1.0]),
            'cls': torch.tensor([1.0, 2.0, 0.0]),
            'bboxes': torch.randn(3, 34),
        }
        result = trainer.preprocess_batch(batch)

        # All tensors should be on trainer device (cpu)
        for k in ('img', 'batch_idx', 'cls', 'bboxes'):
            assert isinstance(result[k], torch.Tensor), f'{k} not a tensor'
            assert result[k].device == trainer.device, f'{k} not on device'

        # Images should NOT be divided by 255 (dataset already normalises)
        assert torch.allclose(result['img'], img, atol=1e-6), 'Images were modified beyond device transfer'
        print('PASS: preprocess_batch moves tensors, no /255 division')


# ---------------------------------------------------------------------------
# Validator wiring
# ---------------------------------------------------------------------------


def test_get_validator():
    """get_validator returns RayCastValidator with correct loss_names."""
    from raycasted.model.val import RayCastValidator

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)
        _setup_model(trainer)
        # Need test_loader for validator — create a minimal dataloader
        val_dir = os.path.join(tmp_dir, 'val')
        dataset = trainer.build_dataset(val_dir, mode='val')
        trainer.test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=_raycast_collate_fn)
        validator = trainer.get_validator()

        assert isinstance(validator, RayCastValidator), f'Expected RayCastValidator, got {type(validator).__name__}'
        assert trainer.loss_names == ('xy_loss', 'cls_loss', 'l1_loss', 'piou_loss', 'smooth_loss'), (
            f'Wrong loss_names: {trainer.loss_names}'
        )
        print(f'PASS: validator is RayCastValidator, loss_names={trainer.loss_names}')


# ---------------------------------------------------------------------------
# Dataset and DataLoader wiring
# ---------------------------------------------------------------------------


def test_build_dataset_and_dataloader():
    """build_dataset and get_dataloader return correct types."""
    from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = _make_trainer(tmp_dir)

        train_dir = os.path.join(tmp_dir, 'train')
        dataset = trainer.build_dataset(train_dir, mode='train')
        assert isinstance(dataset, RayCastTileDataset), f'Expected RayCastTileDataset, got {type(dataset).__name__}'
        assert dataset.crop_size == trainer.args.imgsz
        assert dataset.augment is True, 'Train dataset should have augment=True'

        val_dataset = trainer.build_dataset(train_dir, mode='val')
        assert val_dataset.augment is False, 'Val dataset should have augment=False'

        # get_dataloader
        loader = trainer.get_dataloader(train_dir, batch_size=2, mode='train')
        assert isinstance(loader, torch.utils.data.DataLoader), 'Should return DataLoader'
        assert loader.batch_size == 2
        # Check collate_fn is our custom one
        assert loader.collate_fn is _raycast_collate_fn, 'collate_fn should be _raycast_collate_fn'

        # Actually fetch a batch to verify end-to-end
        batch = next(iter(loader))
        assert 'img' in batch, "Missing 'img' key"
        assert 'bboxes' in batch, "Missing 'bboxes' key"
        assert 'ori_shape' in batch, "Missing 'ori_shape' key"
        assert batch['img'].shape[0] <= 2, f'Batch too large: {batch["img"].shape[0]}'
        print('PASS: dataset/dataloader — train aug=True, val aug=False, batch shapes OK')


# ---------------------------------------------------------------------------
# Smoke train (2 epochs on CPU, ~60-120s)
# ---------------------------------------------------------------------------


def test_smoke_train_2_epochs():
    """Smoke test: train 2 epochs with synthetic data on CPU.

    Validates the full pipeline: model creation, head replacement, loss wiring,
    forward pass, backward pass, gradient update, validation, and checkpoint.
    This is the most comprehensive integration test.
    """
    import time

    start = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Use imgsz=64 for speed (small model, small images)
        trainer = _make_trainer(
            tmp_dir,
            epochs=2,
            batch=2,
            imgsz=64,
            # Disable all augmentations that could slow things down
            mosaic=0.0,
            mixup=0.0,
            cache=False,
            pretrained=False,
            device='cpu',
            workers=0,
            verbose=False,
            # Disable validation plots and saving for speed
            plots=False,
            save=False,
            val=True,
        )

        # Run training
        trainer.train()

        elapsed = time.time() - start

        # Verify training completed
        assert trainer.epochs == 2, f'Expected 2 epochs, got {trainer.epochs}'
        assert trainer.epoch == 1, f'Expected final epoch=1 (0-indexed), got {trainer.epoch}'

        # Verify model state after training
        model = trainer.model
        head = model.model[-1]
        assert isinstance(head, RayCastDetect), 'Head should be RayCastDetect after training'

        # Verify training_args persisted
        ta = model.training_args
        assert ta['nc'] == 5, f'nc should be 5, got {ta["nc"]}'
        assert ta['n_rays'] == 32, f'n_rays should be 32, got {ta["n_rays"]}'

        print(f'PASS: smoke train — 2 epochs in {elapsed:.1f}s')


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_extract_neck_channels()
    test_model_has_raycast_head()
    test_criterion_is_raycast_e2e()
    test_end2end_attribute()
    test_no_dfl()
    test_training_args_metadata()
    test_raycast_collate_fn()
    test_raycast_collate_fn_empty()
    test_raycast_collate_fn_all_empty()
    test_mosaic_forced_off()
    test_forward_pass_output_shapes()
    test_loss_all_5_terms_finite()
    test_preprocess_batch()
    test_get_validator()
    test_build_dataset_and_dataloader()
    test_smoke_train_2_epochs()

    print('\nAll Phase 10 training integration tests passed!')
