"""Train a clean fold1-only NuLite-T segmentation oracle.

Replicates the released NuLite-T architecture EXACTLY (timm fastvit_s12 features_only
encoder + NuLite decoder0/decoder/np_head U-Net) so the saved checkpoint is drop-in
compatible with pretrained/NuLite-T-Weights.pth key format
({'model_state_dict', 'arch', 'config'}).

Purpose: the released NuLite-T weights were trained on the WHOLE PanNuke (incl fold3),
so they are a soft oracle for fold3. This trainer produces a model trained on fold1
ONLY (rotation-1 train split) -> genuinely held-out for the fold3 evals.

The default legacy mode uses two-class CE plus soft Dice. ``--fg-center-target``
instead gives the two output channels independent meanings:

0. foreground occupancy logit, trained with BCE plus soft Dice;
1. instance-centroid potential logit, trained with CenterNet focal loss plus
   explicit centroid-peak and touching-instance valley terms.

Aug: random rot90 k in {0..3} + h/v flips. Normalize (x-0.5)/0.5 -> [-1,1].
Opt: AdamW 3e-4 / wd 1e-4, cosine + 3ep warmup. AMP fp16 on CUDA.
Checkpoint: best-by-val-fg-Dice (fold2) + last, written as {'model_state_dict',...}.

Only the model (no tissue/hv/type heads) is trained.
"""

import argparse
import random
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from raycasted.data.etl.utils import constants as _const
from raycasted.evals.eval_lsp import ParquetGT, roi_id_from_path

_const.configure_rays(64)


def parse_args(argv=None):
    """Parse and validate oracle-training arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-tiles', required=True, help='fold1 tile dir (sorted glob == parquet order)')
    ap.add_argument('--train-parquet', required=True, help='fold1 dense GT parquet')
    ap.add_argument('--val-tiles', required=True, help='fold2 tile dir')
    ap.add_argument('--val-parquet', required=True, help='fold2 dense GT parquet')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--warmup-epochs', type=int, default=3)
    ap.add_argument('--variant', default='fastvit_s12')
    ap.add_argument(
        '--separable-decoder',
        action='store_true',
        help='depthwise-separable decoder convs (MobileNet-style) to compress the oracle',
    )
    ap.add_argument(
        '--edt-target',
        action='store_true',
        help='regress normalized EDT (distance-to-boundary, bg=0) instead of binary CE; '
        'ridge structure supports h-maxima peak extraction for touching nuclei',
    )
    ap.add_argument('--edt-cap', type=float, default=20.0, help='px distance mapped to target 1.0')
    ap.add_argument(
        '--center-potential-target',
        action='store_true',
        help='train a one-channel max-of-per-instance Gaussian centroid potential',
    )
    ap.add_argument(
        '--fg-center-target',
        action='store_true',
        help='train two independent logits: foreground occupancy and centroid potential',
    )
    ap.add_argument('--center-sigma-scale', type=float, default=0.25)
    ap.add_argument('--center-sigma-min', type=float, default=2.5)
    ap.add_argument('--center-sigma-max', type=float, default=4.0)
    ap.add_argument(
        '--center-loss-weight',
        type=float,
        default=0.1,
        help='centroid-objective weight in dual-target mode',
    )
    ap.add_argument('--center-peak-weight', type=float, default=1.0)
    ap.add_argument('--center-valley-weight', type=float, default=1.0)
    ap.add_argument(
        '--val-center-weight',
        type=float,
        default=0.5,
        help='weight of centroid F1 in dual-target checkpoint score; foreground Dice gets 1-weight',
    )
    ap.add_argument('--val-n', type=int, default=500, help='tiles used per val pass')
    ap.add_argument('--limit', type=int, default=0, help='0=all tiles, else cap (smoke tests)')
    ap.add_argument('--device', default='0')
    ap.add_argument('--out', required=True, help='best checkpoint path (dir auto-created)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    target_modes = sum((args.edt_target, args.center_potential_target, args.fg_center_target))
    if target_modes > 1:
        ap.error('--edt-target, --center-potential-target, and --fg-center-target are mutually exclusive')
    if not 0.0 <= args.val_center_weight <= 1.0:
        ap.error('--val-center-weight must be in [0, 1]')
    return args


# ---------------- NuLite-T exact module ----------------
class _Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0, separable=False):
        super().__init__()
        if separable:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Dropout(dropout),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=(kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.block(x)


class _FastViT(nn.Module):
    def __init__(self, name):
        super().__init__()
        import timm

        self.fast_vit = timm.create_model(f'{name}.apple_in1k', features_only=True, pretrained=True)

    def forward(self, x):
        return self.fast_vit(x)


class NuLiteT(nn.Module):
    """Encoder + decoder0 + full U-Net decoder + np_head (binary 2ch)."""

    def __init__(self, name='fastvit_s12', separable_decoder=False, out_ch=2):
        super().__init__()
        self.encoder = _FastViT(name)
        emb = list(self.encoder.fast_vit.feature_info.channels())  # variant-derived (e.g. t8=[48,96,192,384])
        sep = separable_decoder
        self.decoder0 = nn.Sequential(_Conv2DBlock(3, emb[0], 3))
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        'bottleneck_upsampler',
                        nn.Sequential(
                            _Conv2DBlock(emb[-1], emb[-2], separable=sep), nn.ConvTranspose2d(emb[-2], emb[-2], 2, 2)
                        ),
                    ),
                    (
                        'decoder4_upsampler',
                        nn.Sequential(
                            _Conv2DBlock(2 * emb[-2], emb[-2], separable=sep),
                            _Conv2DBlock(emb[-2], emb[-3], separable=sep),
                            nn.ConvTranspose2d(emb[-3], emb[-3], 2, 2),
                        ),
                    ),
                    (
                        'decoder3_upsampler',
                        nn.Sequential(
                            _Conv2DBlock(2 * emb[-3], emb[-3], separable=sep),
                            _Conv2DBlock(emb[-3], emb[-4], separable=sep),
                            nn.ConvTranspose2d(emb[-4], emb[-4], 2, 2),
                        ),
                    ),
                    (
                        'decoder2_upsampler',
                        nn.Sequential(
                            _Conv2DBlock(2 * emb[-4], emb[-4], separable=sep),
                            nn.ConvTranspose2d(emb[-4], emb[-4], 2, 2),
                        ),
                    ),
                    (
                        'decoder1_upsampler',
                        nn.Sequential(
                            _Conv2DBlock(emb[-4], emb[-4], separable=sep), nn.ConvTranspose2d(emb[-4], emb[-4], 2, 2)
                        ),
                    ),
                ]
            )
        )
        self.np_head = nn.Sequential(
            _Conv2DBlock(2 * emb[0], emb[0]),
            nn.Conv2d(emb[0], out_ch, 1),
        )

    def forward(self, x):
        """Full U-Net forward: encoder -> decoder -> np_head logits (out_ch channels)."""
        z1, z2, z3, z4 = self.encoder(x)
        d = self.decoder
        b5 = d.bottleneck_upsampler(z4)
        b4 = d.decoder4_upsampler(torch.cat([z3, b5], dim=1))
        b3 = d.decoder3_upsampler(torch.cat([z2, b4], dim=1))
        b2 = d.decoder2_upsampler(torch.cat([z1, b3], dim=1))
        b1 = d.decoder1_upsampler(b2)
        xt = torch.cat([self.decoder0(x), b1], dim=1)
        return self.np_head(xt)


# ---------------- data: memmap caches, aligned by sorted glob order ----------------
def _centroid_potential(
    instance_masks,
    height=256,
    width=256,
    sigma_scale=0.25,
    sigma_min=2.5,
    sigma_max=4.0,
):
    """Build a max-of-instance-Gaussians target and touching-pair valley mask."""
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    target = np.zeros((height, width), np.float32)
    valley = np.zeros((height, width), np.uint8)
    centers = []
    masks = []
    for raw_mask in instance_masks:
        mask = np.asarray(raw_mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        radius = float(np.sqrt(len(xs) / np.pi))
        sigma = float(np.clip(sigma_scale * radius, sigma_min, sigma_max))
        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
        ix = int(np.clip(round(cx), 0, width - 1))
        iy = int(np.clip(round(cy), 0, height - 1))
        gaussian[iy, ix] = 1.0
        target = np.maximum(target, gaussian.astype(np.float32))
        centers.append((cx, cy))
        masks.append(mask.astype(np.uint8))

    kernel = np.ones((3, 3), np.uint8)
    dilated = [cv2.dilate(mask, kernel, iterations=1).astype(bool) for mask in masks]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if not (np.any(dilated[i] & masks[j]) or np.any(dilated[j] & masks[i])):
                continue
            mx = int(np.clip(round((centers[i][0] + centers[j][0]) * 0.5), 0, width - 1))
            my = int(np.clip(round((centers[i][1] + centers[j][1]) * 0.5), 0, height - 1))
            cv2.circle(valley, (mx, my), 1, 1, thickness=-1)
    return target, valley


def build_cache(
    cache_path,
    tile_dir,
    parquet_path,
    limit=0,
    center_potential=False,
    sigma_scale=0.25,
    sigma_min=2.5,
    sigma_max=4.0,
):
    """Build or reuse image, foreground-mask, and optional centroid caches."""
    cache_path = Path(cache_path)
    # Cache tags contain decimal sigma values. Path.with_suffix() would treat
    # those decimals as suffixes and can make distinct cache tags collide.
    img_path = Path(f'{cache_path}.imgs.npy')
    msk_path = Path(f'{cache_path}.masks.npy')
    ctr_path = Path(f'{cache_path}.centers.npy')
    valley_path = Path(f'{cache_path}.valleys.npy')
    center_cache_ready = center_potential and ctr_path.exists() and valley_path.exists()
    if img_path.exists() and msk_path.exists() and (not center_potential or center_cache_ready):
        imgs = np.load(img_path, mmap_mode='r')
        masks = np.load(msk_path, mmap_mode='r')
        n = len(imgs)
        print(f'[cache] reuse {img_path.name} n={n}')
        centers = np.load(ctr_path, mmap_mode='r') if center_potential else None
        valleys = np.load(valley_path, mmap_mode='r') if center_potential else None
        return imgs, masks, centers, valleys
    tiles = sorted(Path(tile_dir).glob('*.npz'))
    if limit:
        tiles = tiles[:limit]
    gt = ParquetGT(parquet_path)
    n = len(tiles)
    imgs = np.lib.format.open_memmap(str(img_path), mode='w+', dtype=np.uint8, shape=(n, 256, 256, 3))
    masks = np.lib.format.open_memmap(str(msk_path), mode='w+', dtype=np.uint8, shape=(n, 256, 256))
    centers = (
        np.lib.format.open_memmap(str(ctr_path), mode='w+', dtype=np.float16, shape=(n, 256, 256))
        if center_potential
        else None
    )
    valleys = (
        np.lib.format.open_memmap(str(valley_path), mode='w+', dtype=np.uint8, shape=(n, 256, 256))
        if center_potential
        else None
    )
    for i, tp in enumerate(tiles):
        z = np.load(tp)
        im = z['image']
        roi = roi_id_from_path(tp)
        gm, _gl, _tiss = gt.get(roi)
        m = np.zeros((256, 256), np.uint8)
        for g in gm:
            m[g] = 1
        imgs[i] = im.astype(np.uint8)
        masks[i] = m
        if center_potential:
            center, valley = _centroid_potential(
                gm,
                sigma_scale=sigma_scale,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            centers[i] = center.astype(np.float16)
            valleys[i] = valley
    imgs.flush()
    masks.flush()
    if center_potential:
        centers.flush()
        valleys.flush()
    print(f'[cache] built {img_path.name} / {msk_path.name} n={n}')
    return imgs, masks, centers, valleys


class _TileDS(Dataset):
    def __init__(
        self,
        imgs,
        masks,
        train=True,
        edt_target=False,
        edt_cap=20.0,
        centers=None,
        valleys=None,
        dual_target=False,
    ):
        self.imgs, self.masks, self.train = imgs, masks, train
        self.edt_target = edt_target
        self.edt_cap = edt_cap
        self.centers = centers
        self.valleys = valleys
        self.dual_target = dual_target

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = np.asarray(self.imgs[i]).copy()  # uint8 (256,256,3)
        m = np.asarray(self.masks[i]).copy()  # uint8 (256,256)
        center = np.asarray(self.centers[i], dtype=np.float32).copy() if self.centers is not None else None
        valley = np.asarray(self.valleys[i]).copy() if self.valleys is not None else None
        if self.train:
            k = random.randint(0, 3)
            if k:
                im = np.rot90(im, k).copy()
                m = np.rot90(m, k).copy()
                if center is not None:
                    center = np.rot90(center, k).copy()
                    valley = np.rot90(valley, k).copy()
            if random.random() < 0.5:
                im = im[:, ::-1].copy()
                m = m[:, ::-1].copy()
                if center is not None:
                    center = center[:, ::-1].copy()
                    valley = valley[:, ::-1].copy()
            if random.random() < 0.5:
                im = im[::-1].copy()
                m = m[::-1].copy()
                if center is not None:
                    center = center[::-1].copy()
                    valley = valley[::-1].copy()
        t = torch.from_numpy(im).float().permute(2, 0, 1) / 127.5 - 1.0  # [-1,1]
        if center is not None:
            center_tensor = torch.from_numpy(center)
            valley_tensor = torch.from_numpy(valley.astype(np.float32))
            if self.dual_target:
                return t, torch.from_numpy(m.astype(np.float32)), center_tensor, valley_tensor
            return t, center_tensor, valley_tensor
        if self.edt_target:
            # EDT regression target: distance-to-boundary (px) inside the union
            # mask, clipped and normalized to [0, 1]; background pinned to 0.
            # The ridge structure (medial-axis "spine") carries per-instance
            # peaks even where nuclei touch - h-maxima on this map separates
            # merged blobs that a binary fg map cannot.
            edt = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
            edt = np.clip(edt / self.edt_cap, 0.0, 1.0)
            return t, torch.from_numpy(edt).float()
        return t, torch.from_numpy(m).long()


def _dice(pred_mask_bool, gt_bool):
    inter = (pred_mask_bool & gt_bool).sum()
    return 2.0 * inter / max(pred_mask_bool.sum() + gt_bool.sum(), 1)


def _match_center_peaks(prob, target, radius=4.0, threshold=0.25):
    """Return one-to-one centroid TP, FP, and FN counts for one batch."""
    peaks = prob == F.max_pool2d(prob.unsqueeze(1), 9, stride=1, padding=4).squeeze(1)
    peaks &= prob >= threshold
    tp = fp = fn = 0
    for j in range(prob.shape[0]):
        gt_yx = torch.nonzero(target[j] >= 0.999, as_tuple=False).float()
        pred_yx = torch.nonzero(peaks[j], as_tuple=False).float()
        matched = 0
        if len(gt_yx) and len(pred_yx):
            distances = torch.cdist(gt_yx, pred_yx)
            while True:
                flat_index = distances.argmin()
                distance = distances.flatten()[flat_index]
                if not torch.isfinite(distance) or distance > radius:
                    break
                gt_index = flat_index // distances.shape[1]
                pred_index = flat_index % distances.shape[1]
                distances[gt_index, :] = float('inf')
                distances[:, pred_index] = float('inf')
                matched += 1
        tp += matched
        fp += len(pred_yx) - matched
        fn += len(gt_yx) - matched
    return tp, fp, fn


def evaluate(model, dl, device, use_amp, max_batches=None, edt_mode=False, center_mode=False, dual_mode=False):
    """Evaluate foreground Dice and/or one-to-one centroid F1."""
    model.eval()
    dices = []
    center_tp = center_fp = center_fn = 0
    with torch.no_grad():
        for bi, sample in enumerate(dl):
            if max_batches and bi >= max_batches:
                break
            x = sample[0]
            x = x.to(device)
            with torch.autocast('cuda', dtype=torch.float16) if use_amp else _NullCtx():
                logits = model(x)
            if dual_mode:
                fg_target, center_target = sample[1], sample[2]
                fg_pred = logits[:, 0].float().sigmoid().cpu() >= 0.5
                for j in range(fg_pred.shape[0]):
                    dices.append(_dice(fg_pred[j].numpy(), fg_target[j].numpy().astype(bool)))
                counts = _match_center_peaks(logits[:, 1].float().sigmoid().cpu(), center_target)
                center_tp += counts[0]
                center_fp += counts[1]
                center_fn += counts[2]
            elif center_mode:
                counts = _match_center_peaks(logits.squeeze(1).float().sigmoid().cpu(), sample[1])
                center_tp += counts[0]
                center_fp += counts[1]
                center_fn += counts[2]
            elif edt_mode:
                m = sample[1]
                pr = (logits.squeeze(1).float() > 0.05).cpu()
                gt_bool = m.numpy() > 0  # edt target: fg = edt > 0
            else:
                m = sample[1]
                pr = logits.argmax(1).cpu() == 1
                gt_bool = m.numpy().astype(bool)
            if not (center_mode or dual_mode):
                for j in range(pr.shape[0]):
                    dices.append(_dice(pr[j].numpy(), gt_bool[j]))
    model.train()
    center_f1 = 2.0 * center_tp / max(2 * center_tp + center_fp + center_fn, 1)
    return {
        'fg_dice': float(np.mean(dices)) if dices else 0.0,
        'center_f1': center_f1 if center_mode or dual_mode else 0.0,
    }


def _center_focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet heatmap focal loss, normalized by the number of instance peaks."""
    pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    positive = target >= 0.999
    negative = ~positive
    positive_loss = -(1.0 - pred).pow(alpha) * pred.log() * positive
    negative_weight = (1.0 - target).pow(beta)
    negative_loss = -(pred.pow(alpha)) * (1.0 - pred).log() * negative_weight * negative
    count = positive.sum().clamp(min=1)
    return (positive_loss.sum() + negative_loss.sum()) / count


def _soft_dice_loss(logits, target):
    """Differentiable foreground Dice loss for one sigmoid logit."""
    prob = logits.float().sigmoid()
    target = target.float()
    inter = (prob * target).sum()
    return 1.0 - (2.0 * inter + 1.0) / (prob.sum() + target.sum() + 1.0)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main(args=None):
    """Train a fold1-only NuLite-T oracle and save the best validation checkpoint."""
    args = parse_args() if args is None else args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.device in ('0', 'cuda') else 'cpu')
    use_amp = device.type == 'cuda'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cache_root = Path('/mnt/data/projects/RayCastED/output/oracle')
    cache_root.mkdir(parents=True, exist_ok=True)
    tag = f'tr-{Path(args.train_tiles).name}-va-{Path(args.val_tiles).name}-l{args.limit}'
    needs_centers = args.center_potential_target or args.fg_center_target
    if needs_centers:
        tag += f'-center-s{args.center_sigma_scale:g}-{args.center_sigma_min:g}-{args.center_sigma_max:g}'
    cache_kwargs = {
        'center_potential': needs_centers,
        'sigma_scale': args.center_sigma_scale,
        'sigma_min': args.center_sigma_min,
        'sigma_max': args.center_sigma_max,
    }
    imgs_tr, msk_tr, ctr_tr, valley_tr = build_cache(
        cache_root / f'{tag}-tr', args.train_tiles, args.train_parquet, args.limit, **cache_kwargs
    )
    imgs_va, msk_va, ctr_va, valley_va = build_cache(
        cache_root / f'{tag}-va', args.val_tiles, args.val_parquet, args.limit, **cache_kwargs
    )
    n = len(imgs_tr)
    print(f'[data] train tiles={n}  val tiles={len(imgs_va)}')

    one_channel = args.edt_target or args.center_potential_target
    model = NuLiteT(args.variant, separable_decoder=args.separable_decoder, out_ch=1 if one_channel else 2)
    if needs_centers:
        # CenterNet-style negative prior: sparse heatmaps should begin near
        # zero probability instead of paying a huge loss over every pixel.
        center_channel = 1 if args.fg_center_target else 0
        with torch.no_grad():
            model.np_head[-1].bias[center_channel].fill_(-2.19)
    model.to(device)
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    dec_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith('decoder.'))
    print(
        f'[model] NuLite-T fold1-only ({args.variant},'
        f' separable_decoder={args.separable_decoder}, edt_target={args.edt_target},'
        f' center_target={args.center_potential_target}, fg_center_target={args.fg_center_target})'
        f' trainable {nparams / 1e6:.2f}M'
        f' (decoder {dec_params / 1e6:.2f}M)'
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_ep = max(int(np.ceil(n / args.batch)), 1)
    total_steps = steps_per_ep * args.epochs
    warmup = steps_per_ep * args.warmup_epochs

    def lr_lb(step):
        if step < warmup:
            return step / max(warmup, 1)
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lb)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    dl_tr = DataLoader(
        _TileDS(
            imgs_tr,
            msk_tr,
            train=True,
            edt_target=args.edt_target,
            edt_cap=args.edt_cap,
            centers=ctr_tr,
            valleys=valley_tr,
            dual_target=args.fg_center_target,
        ),
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
    )
    dl_va = DataLoader(
        _TileDS(
            imgs_va,
            msk_va,
            train=False,
            edt_target=args.edt_target,
            edt_cap=args.edt_cap,
            centers=ctr_va,
            valleys=valley_va,
            dual_target=args.fg_center_target,
        ),
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
    )
    val_batches = max(int(np.ceil(args.val_n / args.batch)), 1)

    best_score, best_ep = -float('inf'), -1
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        ep_primary, ep_aux, ep_center, nb = 0.0, 0.0, 0.0, 0
        for sample in dl_tr:
            x = sample[0].to(device)
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16) if use_amp else _NullCtx():
                logits = model(x)
                center_loss = logits.sum() * 0.0
                if args.fg_center_target:
                    fg_target = sample[1].to(device)
                    center_target = sample[2].to(device)
                    valley = sample[3].to(device)
                    fg_raw = logits[:, 0].float()
                    center_raw = logits[:, 1].float()
                    primary = F.binary_cross_entropy_with_logits(fg_raw, fg_target)
                    aux = _soft_dice_loss(fg_raw, fg_target)
                    heatmap_loss = _center_focal_loss(center_raw, center_target)
                    peaks = center_target >= 0.999
                    peak_loss = F.softplus(-center_raw[peaks]).mean() if peaks.any() else center_raw.sum() * 0.0
                    valleys = valley > 0
                    valley_loss = F.softplus(center_raw[valleys]).mean() if valleys.any() else center_raw.sum() * 0.0
                    center_loss = (
                        heatmap_loss + args.center_peak_weight * peak_loss + args.center_valley_weight * valley_loss
                    )
                    loss = primary + aux + args.center_loss_weight * center_loss
                elif args.center_potential_target:
                    center_target = sample[1].to(device)
                    valley = sample[2].to(device)
                    center_raw = logits.squeeze(1).float()
                    primary = _center_focal_loss(center_raw, center_target)
                    peaks = center_target >= 0.999
                    peak_loss = F.softplus(-center_raw[peaks]).mean() if peaks.any() else center_raw.sum() * 0.0
                    valleys = valley > 0
                    valley_loss = F.softplus(center_raw[valleys]).mean() if valleys.any() else center_raw.sum() * 0.0
                    aux = peak_loss + valley_loss
                    center_loss = (
                        primary + args.center_peak_weight * peak_loss + args.center_valley_weight * valley_loss
                    )
                    loss = center_loss
                elif args.edt_target:
                    m = sample[1].to(device)
                    # EDT regression: L1 on the distance map + soft-Dice on
                    # binarized prediction (pred>0.05) vs fg to sharpen ridges.
                    primary = F.l1_loss(logits.squeeze(1), m)
                    p = logits.squeeze(1).float()
                    mf = (m > 0).float()
                    pb = (p > 0.05).float()
                    inter = (pb * mf).sum()
                    aux = 1.0 - (2.0 * inter + 1.0) / (pb.sum() + mf.sum() + 1.0)
                    loss = primary + aux
                else:
                    m = sample[1].to(device)
                    primary = F.cross_entropy(logits, m)
                    aux = _soft_dice_loss(logits[:, 1], m)
                    loss = primary + aux
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ep_primary += primary.item()
            ep_aux += aux.item()
            ep_center += center_loss.item()
            nb += 1
        metrics = evaluate(
            model,
            dl_va,
            device,
            use_amp,
            max_batches=val_batches,
            edt_mode=args.edt_target,
            center_mode=args.center_potential_target,
            dual_mode=args.fg_center_target,
        )
        if args.fg_center_target:
            val_score = (1.0 - args.val_center_weight) * metrics['fg_dice'] + args.val_center_weight * metrics[
                'center_f1'
            ]
        elif args.center_potential_target:
            val_score = metrics['center_f1']
        else:
            val_score = metrics['fg_dice']
        ep_s = time.time() - t0
        print(
            f'ep {ep:3d}/{args.epochs}  primary {ep_primary / nb:.4f}  aux {ep_aux / nb:.4f}'
            f'  centerL {ep_center / nb:.4f}  valFG {metrics["fg_dice"]:.4f}'
            f'  valCenterF1 {metrics["center_f1"]:.4f}  valScore {val_score:.4f}'
            f'  lr {opt.param_groups[0]["lr"]:.2e}  {ep_s:7.0f}s',
            flush=True,
        )
        if val_score > best_score:
            best_score, best_ep = val_score, ep
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if args.fg_center_target:
                oracle_kind = 'fg_centroid'
                training_loss = 'foreground-BCE+dice + center-focal+peak+touching-valley'
                channel_semantics = ['foreground_logit', 'centroid_potential_logit']
            elif args.center_potential_target:
                oracle_kind = 'center_potential'
                training_loss = 'center-focal+peak+touching-valley'
                channel_semantics = ['centroid_potential_logit']
            elif args.edt_target:
                oracle_kind = 'edt'
                training_loss = 'EDT-L1+dice'
                channel_semantics = ['edt']
            else:
                oracle_kind = 'binary'
                training_loss = 'CE+dice'
                channel_semantics = ['background_logit', 'foreground_logit']
            ck = {
                'model_state_dict': sd,
                'arch': f'NuLite-T({args.variant}) fold1-only',
                'config': {
                    'data.dataset': 'PanNuke',
                    'folds': {'train': 0, 'val': 1},
                    'training.loss': training_loss,
                    'oracle_kind': oracle_kind,
                    'channel_semantics': channel_semantics,
                    'center_sigma': {
                        'scale': args.center_sigma_scale,
                        'min': args.center_sigma_min,
                        'max': args.center_sigma_max,
                    },
                    'loss_weights': {
                        'center': args.center_loss_weight,
                        'center_peak': args.center_peak_weight,
                        'center_valley': args.center_valley_weight,
                    },
                    'center_initial_bias': -2.19 if needs_centers else None,
                    'validation': {
                        **metrics,
                        'score': val_score,
                        'center_weight': args.val_center_weight if args.fg_center_target else None,
                    },
                    'note': 'oracle, fold1 only',
                    'separable_decoder': bool(args.separable_decoder),
                },
            }
            torch.save(ck, str(out))
            print(f'  -> saved best valScore {best_score:.4f} @ ep {ep}')
    print(f'[done] best valScore {best_score:.4f} @ ep {best_ep} -> {out}')


if __name__ == '__main__':
    main()
