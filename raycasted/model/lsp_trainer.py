#!/usr/bin/env python3
"""Standalone LSP-DETR trainer — zero Ultralytics training dependency.

Exact LSP-DETR hyperparameters:
  AdamW(lr=1e-4, wd=1e-4)
  Cosine LR: warmup 1e-7→1e-4 (10 epochs), decay 1e-4→1e-6 (120 epochs)
  Gradient clipping 0.1
  Backbone frozen 30 epochs, unfrozen with lr×0.1
  Batch 16, 130 epochs, AMP

CLI:
    uv run python -m raycasted.model.lsp_trainer \
        --train-dir /path/to/train_npz \
        --val-dir /path/to/val_npz \
        --output runs/lsp_detr

API (used by pipeline.py):
    from raycasted.model.lsp_trainer import train_lsp
    train_lsp(train_dir=..., val_dir=..., output_dir=...)
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset, collate_fn
from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch
from raycasted.data.etl.utils.constants import configure_rays
from raycasted.model.lsp_detr_model import LSPDetrDetectionModel
from raycasted.model.metrics import compute_bpq_from_iou

_logger = logging.getLogger('lsp_trainer')


def _get_lr(
    epoch: int,
    warmup: int = 10,
    max_epochs: int = 130,
    lr0: float = 1e-4,
    lr_min: float = 1e-6,
) -> float:
    if epoch < warmup:
        return float(1e-7 + epoch * (lr0 - 1e-7) / warmup)
    t = (epoch - warmup) / max(1, max_epochs - warmup)
    return float((math.cos(math.pi * t) + 1) / 2 * (lr0 - lr_min) + lr_min)


@torch.no_grad()
def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    nc: int,
    n_rays: int,
    crop_size: int = 256,
    conf_threshold: float = 0.25,
) -> dict[str, float]:
    model.eval()
    total_bpq = total_bsq = total_bdq = 0.0
    count = 0

    for images, targets in tqdm(val_loader, desc='Val', unit='step', leave=False):
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=False):
            out = model.predict(images)

        logits = out['pred_logits']
        points = out['pred_points']
        radial = out['pred_radial']

        cls_prob = logits[..., :-1].softmax(dim=-1)
        conf, _cls_pred = cls_prob.max(dim=-1)
        is_object = logits.argmax(dim=-1) != nc
        keep = (conf > conf_threshold) & is_object

        for i in range(images.shape[0]):
            mask = keep[i]
            pred_bboxes = torch.cat([points[i, mask] * crop_size, radial[i, mask].exp()], dim=-1)

            img_mask = targets[:, 0].long() == i
            gt_bboxes = (targets[img_mask, 2:] * crop_size).to(device)

            n_pred = pred_bboxes.shape[0]
            n_gt = gt_bboxes.shape[0]

            if n_gt == 0 and n_pred == 0:
                total_bpq += 1.0
                total_bsq += 1.0
                total_bdq += 1.0
            elif n_gt > 0 and n_pred > 0:
                pred_rays = pred_bboxes[:, 2:].unsqueeze(1).expand(n_pred, n_gt, n_rays)
                gt_rays = gt_bboxes[:, 2:].unsqueeze(0).expand(n_pred, n_gt, n_rays)
                iou = polar_iou_pairwise_flat_torch(pred_rays, gt_rays).cpu().numpy()
                bpq, bsq, bdq = compute_bpq_from_iou(iou)
                total_bpq += bpq
                total_bsq += bsq
                total_bdq += bdq
            count += 1

    model.train()
    if count == 0:
        return {'bPQ': 0.0, 'bSQ': 0.0, 'bDQ': 0.0}
    return {'bPQ': total_bpq / count, 'bSQ': total_bsq / count, 'bDQ': total_bdq / count}


def train_lsp(
    *,
    train_dir: str,
    val_dir: str,
    output_dir: str,
    epochs: int = 130,
    batch_size: int = 16,
    lr: float = 1e-4,
    wd: float = 1e-4,
    warmup: int = 10,
    freeze_epochs: int = 30,
    backbone_lr_ratio: float = 0.1,
    clip_grad: float = 0.1,
    n_rays: int = 64,
    nc: int = 5,
    crop_size: int = 256,
    conf: float = 0.25,
    resume: str | None = None,
    workers: int = 4,
    seed: int = 42,
    device: torch.device | str = 'cuda',
) -> None:
    """Run LSP-DETR training with exact reference hyperparameters.

    Args:
        train_dir: Directory of .npz training tiles.
        val_dir: Directory of .npz validation tiles.
        output_dir: Directory for checkpoints.
        epochs: Total training epochs (default 130 matching LSP-DETR).
        batch_size: Batch size (default 16 matching LSP-DETR).
        lr: Peak learning rate (default 1e-4).
        wd: Weight decay (default 1e-4).
        warmup: Linear warmup epochs (default 10).
        freeze_epochs: Epochs to keep backbone frozen (default 30).
        backbone_lr_ratio: Backbone LR multiplier after unfreeze (default 0.1).
        clip_grad: Gradient clipping norm (default 0.1).
        n_rays: Number of ray distances (default 64).
        nc: Number of classes (default 5 for PanNuke).
        crop_size: Input size (default 256).
        conf: Validation confidence threshold.
        resume: Checkpoint path to resume from.
        workers: DataLoader workers.
        seed: Random seed.
        device: torch.device or 'cuda'/'cpu'/'0'.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    _device = torch.device(device) if isinstance(device, str) else device
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _logger.info('Device: %s', _device)
    configure_rays(n_rays)

    aug_config = {
        'stain_jitter': True,
        'stain_hsv_h': 0.05,
        'stain_hsv_s': 0.3,
        'stain_hsv_v': 0.2,
        'stain_blur_prob': 0.2,
        'stain_blur_sigma': 1.0,
        'scale_augment': True,
        'scale_range': [0.7, 1.3],
        'translate_augment': True,
        'translate_range': 0.1,
    }

    train_ds = RayCastTileDataset(train_dir, crop_size=crop_size, augment=True, augment_config=aug_config)
    val_ds = RayCastTileDataset(val_dir, crop_size=crop_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=True,
    )

    _logger.info('Train: %d tiles, Val: %d tiles', len(train_ds), len(val_ds))
    _logger.info('Steps per epoch: %d', len(train_loader))

    model = LSPDetrDetectionModel(nc=nc, n_rays=n_rays, crop_size=crop_size)
    model = model.to(_device)
    _logger.info('Model: %.1fM params', sum(p.numel() for p in model.parameters()) / 1e6)

    for p in model.backbone.parameters():
        p.requires_grad_(False)

    decoder_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(decoder_params, lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_bpq = 0.0
    backbone_unfrozen = freeze_epochs < 0

    if resume:
        ckpt = torch.load(resume, map_location=_device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_bpq = ckpt.get('best_bpq', 0.0)
        _logger.info('Resumed from epoch %d (best bPQ=%.4f)', start_epoch, best_bpq)

    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, epochs):
        _lr = _get_lr(epoch, warmup, epochs, lr)
        for pg in optimizer.param_groups:
            btn_name = pg.get('name', '')
            pg['lr'] = _lr * backbone_lr_ratio if btn_name == 'backbone' else _lr

        if epoch == freeze_epochs and not backbone_unfrozen:
            already = any(pg.get('name') == 'backbone' for pg in optimizer.param_groups)
            if not already:
                for p in model.backbone.parameters():
                    p.requires_grad_(True)
                n_bbn = sum(p.numel() for p in model.backbone.parameters())
                _logger.info('Unfreezing backbone (%.1fM params) at epoch %d', n_bbn / 1e6, epoch)
                optimizer.add_param_group(
                    {
                        'params': model.backbone.parameters(),
                        'lr': _lr * backbone_lr_ratio,
                        'weight_decay': wd,
                        'name': 'backbone',
                    }
                )
            backbone_unfrozen = True

        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        t0 = time.perf_counter()

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}', unit='step', leave=False)
        for step, (images, targets) in enumerate(train_pbar):
            images = images.to(_device, non_blocking=True)
            batch_dict = {
                'img': images,
                'batch_idx': targets[:, 0].long(),
                'cls': targets[:, 1].long(),
                'bboxes': targets[:, 2:],
            }

            with torch.amp.autocast('cuda'):
                loss, loss_items = model.loss(batch_dict)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_ce += loss_items[0].item()

            if step % 10 == 0:
                train_pbar.set_postfix(loss=f'{loss.item():.3f}', ce=f'{loss_items[0].item():.3f}')

        train_time = time.perf_counter() - t0

        val_m = _validate(model, val_loader, _device, nc, n_rays, crop_size, conf)
        bpq, bsq, bdq = val_m['bPQ'], val_m['bSQ'], val_m['bDQ']

        lr_dec = optimizer.param_groups[0]['lr']
        lr_bbn = next((pg['lr'] for pg in optimizer.param_groups if pg.get('name') == 'backbone'), 0.0)

        _logger.info(
            'Epoch %3d | loss=%.4f ce=%.4f | bPQ=%.4f bSQ=%.4f bDQ=%.4f | LR dec=%.2e bbn=%.2e | %.1fs',
            epoch,
            epoch_loss / steps_per_epoch,
            epoch_ce / steps_per_epoch,
            bpq,
            bsq,
            bdq,
            lr_dec,
            lr_bbn,
            train_time,
        )

        if bpq > best_bpq:
            best_bpq = bpq
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_bpq': best_bpq,
                },
                out_dir / 'best.pt',
            )
            _logger.info('  ⭐ New best bPQ=%.4f', bpq)

        if epoch % 20 == 0:
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_bpq': best_bpq,
                },
                out_dir / f'epoch_{epoch:04d}.pt',
            )

    _logger.info('Done. Best bPQ=%.4f', best_bpq)


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Standalone LSP-DETR trainer')
    parser.add_argument('--train-dir', required=True, help='Directory of .npz training tiles')
    parser.add_argument('--val-dir', required=True, help='Directory of .npz validation tiles')
    parser.add_argument('--output', default='runs/lsp_detr', help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=130)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--freeze', type=int, default=30)
    parser.add_argument('--backbone-lr-ratio', type=float, default=0.1)
    parser.add_argument('--clip-grad', type=float, default=0.1)
    parser.add_argument('--n-rays', type=int, default=64)
    parser.add_argument('--nc', type=int, default=5)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--resume')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_lsp(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        warmup=args.warmup,
        freeze_epochs=args.freeze,
        backbone_lr_ratio=args.backbone_lr_ratio,
        clip_grad=args.clip_grad,
        n_rays=args.n_rays,
        nc=args.nc,
        crop_size=args.crop_size,
        conf=args.conf,
        resume=args.resume,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    main()
