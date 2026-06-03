#!/usr/bin/env python3
"""Standalone LSP-DETR trainer — zero Ultralytics training dependency.

Exact LSP-DETR hyperparameters:
  AdamW(lr=1e-4, wd=1e-4)
  Cosine LR: warmup 1e-7→1e-4 (10 epochs), decay 1e-4→1e-6 (120 epochs)
  Gradient clipping 0.1
  Backbone frozen 30 epochs, unfrozen with lr×0.1
  Batch 16, 130 epochs, AMP
  HuggingFace PanNuke dataset with albumentations augmentations + WeightedClassAndTissueSampler

CLI:
    uv run python -m raycasted.model.lsp_trainer \
        --train-fold 1 \
        --val-fold 2 \
        --output runs/lsp_detr

API (used by pipeline.py):
    from raycasted.model.lsp_trainer import train_lsp
    train_lsp(train_fold=[1], val_fold=2, output_dir=...)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from raycasted.data.etl.loader.collate_fn_lsp import LSPCollateFn, gpu_prepare_batch
from raycasted.data.etl.loader.gpu_augment import GPUAugment
from raycasted.data.etl.loader.lsp_dataset import LSPDataset, load_pannuke_folds
from raycasted.data.etl.loader.weighted_class_and_tissue import WeightedClassAndTissueSampler
from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch
from raycasted.data.etl.utils.constants import configure_rays
from raycasted.model.lsp_detr_model import LSPDetrDetectionModel
from raycasted.model.metrics import compute_bpq_from_iou

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
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
    allow_overlaps: bool = True,
) -> dict[str, float]:
    model.eval()
    total_bpq = total_bsq = total_bdq = 0.0
    count = 0

    for batch_dict in tqdm(val_loader, desc='Val', unit='step', leave=False):
        batch_dict = gpu_prepare_batch(batch_dict, augment=None, n_rays=n_rays, allow_overlaps=allow_overlaps)
        images = batch_dict['img']
        targets = batch_dict['targets']

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

            tgt = targets[i]
            gt_labels = tgt['labels']
            gt_centroids = tgt['centroids']
            gt_radial = tgt['radial_distances']  # (2, n_rays, H, W)

            if len(gt_labels) == 0:
                n_pred = pred_bboxes.shape[0]
                if n_pred == 0:
                    total_bpq += 1.0
                    total_bsq += 1.0
                    total_bdq += 1.0
                count += 1
                continue

            # Grid-sample upper-bound rays at GT centroid positions
            grid = gt_centroids.to(device) * 2 - 1  # (N, 2) → [-1, 1]
            grid = grid.view(1, -1, 1, 2)  # (1, N, 1, 2)
            gr = gt_radial[1:2].to(device).float()  # (1, n_rays, H, W) — upper bound only
            sampled = torch.nn.functional.grid_sample(
                gr, grid, mode='bilinear', align_corners=False, padding_mode='border'
            )  # (1, n_rays, 1, N)
            sampled = sampled.squeeze(0).squeeze(2).t()  # (N, n_rays)
            gt_rays_px = sampled.clamp(min=1.0)

            n_pred = pred_bboxes.shape[0]
            n_gt = len(gt_labels)

            if n_pred > 0 and gt_rays_px.shape[0] == n_gt:
                pred_rays = pred_bboxes[:, 2:].unsqueeze(1).expand(n_pred, n_gt, n_rays)
                gt_rays = gt_rays_px.unsqueeze(0).expand(n_pred, n_gt, n_rays)
                iou = polar_iou_pairwise_flat_torch(pred_rays, gt_rays).cpu().numpy()
                iou = np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
                bpq, bsq, bdq = compute_bpq_from_iou(iou)
                total_bpq += bpq
                total_bsq += bsq
                total_bdq += bdq

            count += 1

    model.train()
    if count == 0:
        return {'bPQ': 0.0, 'bSQ': 0.0, 'bDQ': 0.0}
    return {
        'bPQ': total_bpq / count,
        'bSQ': total_bsq / count,
        'bDQ': total_bdq / count,
    }

    return {
        'bPQ': total_bpq / count,
        'bSQ': total_bsq / count,
        'bDQ': total_bdq / count,
    }


def train_lsp(
    *,
    train_fold: list[int] | int = 1,
    val_fold: int = 2,
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
    workers: int = 8,
    seed: int = 42,
    device: torch.device | str = 'cuda',
    allow_overlaps: bool = True,
    backbone_name: str = 'facebook/convnextv2-nano-1k-224',
    use_gnn: bool = False,
    gnn_k: int = 8,
) -> None:
    """Train LSP-DETR model on PanNuke using exact LSP-DETR recipe."""
    import os

    os.environ.setdefault('RAYON_NUM_THREADS', '1')
    import cv2

    cv2.setNumThreads(0)  # noqa: F811

    torch.manual_seed(seed)
    np.random.seed(seed)
    from glob import glob

    _device = torch.device(device) if isinstance(device, str) else device
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_idx = 1
    while True:
        suffix = '' if run_idx == 1 else str(run_idx)
        out_dir = base / f'train{suffix}'
        if not out_dir.exists():
            break
        run_idx += 1
    out_dir.mkdir(parents=True, exist_ok=True)

    _logger.info('Device: %s', _device)
    configure_rays(n_rays)

    data_paths = sorted(glob('raycasted/data/dataset/PanNuke/data/fold*-*.parquet'))
    if not data_paths:
        raise FileNotFoundError('No PanNuke parquet files found in raycasted/data/dataset/PanNuke/data/')
    _logger.info('Found %d PanNuke parquet files', len(data_paths))

    train_data = load_pannuke_folds(data_paths, folds=train_fold if isinstance(train_fold, list) else [train_fold])
    val_data = load_pannuke_folds(data_paths, folds=[val_fold])

    train_ds = LSPDataset(train_data, n_rays=n_rays)
    val_ds = LSPDataset(val_data, n_rays=n_rays)

    train_sampler = WeightedClassAndTissueSampler(
        tissues=np.array(train_data['tissue']),
        classes=[np.array(c, dtype=np.uint8) for c in train_data['categories']],
        num_classes=len(train_data.features['categories'].feature.names),
        num_samples=len(train_data),
    )

    gpu_augment = GPUAugment()
    train_collate = LSPCollateFn(n_rays=n_rays, allow_overlaps=allow_overlaps)
    val_collate = LSPCollateFn(n_rays=n_rays, allow_overlaps=allow_overlaps)

    mp.set_start_method('spawn', force=True)

    nw = min(2, mp.cpu_count())
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=train_collate,
        num_workers=nw,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate,
        num_workers=nw,
        prefetch_factor=2,
    )

    _logger.info('Train: %d samples, Val: %d samples', len(train_ds), len(val_ds))
    _logger.info('Steps per epoch: %d', len(train_loader))

    model = LSPDetrDetectionModel(
        nc=nc, n_rays=n_rays, crop_size=crop_size, backbone_name=backbone_name, use_gnn=use_gnn, gnn_k=gnn_k
    )
    model = model.to(_device)
    model = torch.compile(model, dynamic=True)
    _logger.info('Model: %.1fM params (compiled)', sum(p.numel() for p in model.parameters()) / 1e6)

    for p in model.backbone.parameters():
        p.requires_grad_(False)

    decay_p, no_decay_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or 'norm' in n.lower():
            no_decay_p.append(p)
        else:
            decay_p.append(p)

    pgs = []
    if decay_p:
        pgs.append({'params': decay_p, 'weight_decay': wd, 'lr': lr, 'name': 'decoder'})
    if no_decay_p:
        pgs.append({'params': no_decay_p, 'weight_decay': 0.0, 'lr': lr, 'name': 'decoder'})

    optimizer = torch.optim.AdamW(pgs)
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

    _gpu_mem = '0G'
    _header_printed = False

    csv_path = out_dir / 'results.csv'
    csv_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow(
                [
                    'epoch',
                    'train/loss',
                    'train/ce',
                    'train/centroid',
                    'train/radial',
                    'val/bPQ',
                    'val/bSQ',
                    'val/bDQ',
                    'lr',
                ]
            )

    for epoch in range(start_epoch, epochs):
        _lr = _get_lr(epoch, warmup, epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = _lr * (backbone_lr_ratio if pg.get('name') == 'backbone' else 1.0)

        if not backbone_unfrozen and epoch >= freeze_epochs:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            bb_decay, bb_nodecay = [], []
            for n, p in model.backbone.named_parameters():
                if p.ndim <= 1 or 'norm' in n.lower():
                    bb_nodecay.append(p)
                else:
                    bb_decay.append(p)
            if bb_decay:
                optimizer.add_param_group(
                    {
                        'params': bb_decay,
                        'weight_decay': wd,
                        'lr': _lr * backbone_lr_ratio,
                        'name': 'backbone',
                    }
                )
            if bb_nodecay:
                optimizer.add_param_group(
                    {
                        'params': bb_nodecay,
                        'weight_decay': 0.0,
                        'lr': _lr * backbone_lr_ratio,
                        'name': 'backbone',
                    }
                )
            backbone_unfrozen = True

        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_centroid = 0.0
        epoch_radial = 0.0
        total_instances = 0
        n_batches = 0

        train_pbar = tqdm(
            train_loader,
            desc=f'{epoch:>6d}/{epochs}',
            unit='batch',
            bar_format='{desc}{percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]',
            leave=False,
        )
        for batch_dict in train_pbar:
            batch_dict = gpu_prepare_batch(
                batch_dict, augment=gpu_augment, n_rays=n_rays, allow_overlaps=allow_overlaps
            )
            targets = batch_dict['targets']
            total_instances += sum(len(t['labels']) for t in targets)

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
            epoch_centroid += loss_items[1].item()
            epoch_radial += loss_items[2].item()
            n_batches += 1

            train_pbar.set_postfix(
                loss=f'{loss.item():.3f}',
                ce=f'{loss_items[0].item():.3f}',
                cent=f'{loss_items[1].item():.3f}',
                rad=f'{loss_items[2].item():.3f}',
            )

        _gpu_mem = f'{torch.cuda.max_memory_reserved(_device) / 1e9:.1f}G' if _device.type == 'cuda' else '0G'

        avg_loss = epoch_loss / n_batches if n_batches else 0
        avg_ce = epoch_ce / n_batches if n_batches else 0
        avg_centroid = epoch_centroid / n_batches if n_batches else 0
        avg_radial = epoch_radial / n_batches if n_batches else 0

        if not _header_printed:
            print()
            print(
                f'{"Epoch":>6s} {"GPU":>5s} {"loss":>8s} {"ce":>8s} {"cent":>8s} {"rad":>8s} '
                f'{"Inst":>6s} {"Size":>5s} {"bPQ":>8s} {"bSQ":>8s} {"bDQ":>8s} {"LR":>10s}'
            )
            _header_printed = True

        val_m = _validate(model, val_loader, _device, nc, n_rays, crop_size, conf, allow_overlaps=allow_overlaps)
        bpq, bsq, bdq = val_m['bPQ'], val_m['bSQ'], val_m['bDQ']

        lr_dec = optimizer.param_groups[0]['lr']
        lr_bbn = next((pg['lr'] for pg in optimizer.param_groups if 'backbone' in pg.get('name', '')), 0.0)
        lr_str = f'dec={lr_dec:.2e}' if lr_bbn == 0 else f'dec={lr_dec:.2e}/bbn={lr_bbn:.2e}'

        star = ' ⭐' if bpq > best_bpq else ''
        print(
            f'{epoch:>6d} {_gpu_mem:>5s} {avg_loss:>8.4f} {avg_ce:>8.4f} {avg_centroid:>8.4f} {avg_radial:>8.4f} '
            f'{total_instances:>6d} {crop_size:>5d} {bpq:>8.4f} {bsq:>8.4f} {bdq:>8.4f} {lr_str:>10s}{star}'
        )

        with open(csv_path, 'a', newline='') as csv_file:
            csv.writer(csv_file).writerow(
                [
                    epoch,
                    avg_loss,
                    avg_ce,
                    avg_centroid,
                    avg_radial,
                    bpq,
                    bsq,
                    bdq,
                    lr_dec,
                ]
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
    parser.add_argument('--train-fold', type=int, nargs='+', default=[1], help='PanNuke folds for training (fold1=1)')
    parser.add_argument('--val-fold', type=int, default=2, help='PanNuke fold for validation (fold2=2)')
    parser.add_argument('--output', default='runs/lsp_detr', help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=130)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--freeze', type=int, default=30)
    parser.add_argument('--backbone-lr-ratio', type=float, default=0.1)
    parser.add_argument('--backbone', type=str, default='facebook/convnextv2-nano-1k-224')
    parser.add_argument('--gnn', action='store_true', default=False)
    parser.add_argument('--gnn-k', type=int, default=8)
    parser.add_argument('--clip-grad', type=float, default=0.1)
    parser.add_argument('--n-rays', type=int, default=64)
    parser.add_argument('--nc', type=int, default=5)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--resume')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_lsp(
        train_fold=args.train_fold,
        val_fold=args.val_fold,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        warmup=args.warmup,
        freeze_epochs=args.freeze,
        backbone_lr_ratio=args.backbone_lr_ratio,
        backbone_name=args.backbone,
        use_gnn=args.gnn,
        gnn_k=args.gnn_k,
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
