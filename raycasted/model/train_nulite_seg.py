"""Train a clean fold1-only NuLite-T foreground model (oracle seg prior).

Replicates the released NuLite-T architecture EXACTLY (timm fastvit_s12 features_only
encoder + NuLite decoder0/decoder/np_head U-Net) so the saved checkpoint is drop-in
compatible with pretrained/NuLite-T-Weights.pth key format
({'model_state_dict', 'arch', 'config'}).

Purpose: the released NuLite-T weights were trained on the WHOLE PanNuke (incl fold3),
so they are a soft oracle for fold3. This trainer produces a model trained on fold1
ONLY (rotation-1 train split) -> genuinely held-out for the fold3 evals.

Loss: CE(2ch logits) + soft-Dice on sigmoid(channel1) vs binary nucleus mask.
Aug: random rot90 k in {0..3} + h/v flips. Normalize (x-0.5)/0.5 -> [-1,1].
Opt: AdamW 3e-4 / wd 1e-4, cosine + 3ep warmup. AMP fp16 on CUDA.
Checkpoint: best-by-val-fg-Dice (fold2) + last, written as {'model_state_dict',...}.

Only the model (no tissue/hv/type heads) is trained - binary foreground is all we need.
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
ap.add_argument('--val-n', type=int, default=500, help='tiles used per val pass')
ap.add_argument('--limit', type=int, default=0, help='0=all tiles, else cap (smoke tests)')
ap.add_argument('--device', default='0')
ap.add_argument('--out', required=True, help='best checkpoint path (dir auto-created)')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

seed_everything = getattr(torch, 'manual_seed', lambda s: None)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


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


# ---------------- data: memmap caches (images + binary masks), aligned by sorted glob order ===
def build_cache(cache_path, tile_dir, parquet_path, limit=0):
    """Build (or reuse) memmap .npy caches of tile images and binary masks."""
    cache_path = Path(cache_path)
    img_path = cache_path.with_suffix('.imgs.npy')
    msk_path = cache_path.with_suffix('.masks.npy')
    if img_path.exists() and msk_path.exists():
        imgs = np.load(img_path, mmap_mode='r')
        masks = np.load(msk_path, mmap_mode='r')
        n = len(imgs)
        print(f'[cache] reuse {img_path.name} n={n}')
        return imgs, masks
    tiles = sorted(Path(tile_dir).glob('*.npz'))
    if limit:
        tiles = tiles[:limit]
    gt = ParquetGT(parquet_path)
    n = len(tiles)
    imgs = np.lib.format.open_memmap(str(img_path), mode='w+', dtype=np.uint8, shape=(n, 256, 256, 3))
    masks = np.lib.format.open_memmap(str(msk_path), mode='w+', dtype=np.uint8, shape=(n, 256, 256))
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
    imgs.flush()
    masks.flush()
    print(f'[cache] built {img_path.name} / {msk_path.name} n={n}')
    return imgs, masks


class _TileDS(Dataset):
    def __init__(self, imgs, masks, train=True, edt_target=False, edt_cap=20.0):
        self.imgs, self.masks, self.train = imgs, masks, train
        self.edt_target = edt_target
        self.edt_cap = edt_cap

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = np.asarray(self.imgs[i]).copy()  # uint8 (256,256,3)
        m = np.asarray(self.masks[i]).copy()  # uint8 (256,256)
        if self.train:
            k = random.randint(0, 3)
            if k:
                im = np.rot90(im, k).copy()
                m = np.rot90(m, k).copy()
            if random.random() < 0.5:
                im = np[:, ::-1].copy()
                m = m[:, ::-1].copy()
            if random.random() < 0.5:
                im = im[::-1].copy()
                m = m[::-1].copy()
        t = torch.from_numpy(im).float().permute(2, 0, 1) / 127.5 - 1.0  # [-1,1]
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


def evaluate(model, dl, device, use_amp, max_batches=None, edt_mode=False):
    """Foreground Dice vs GT binary masks over the dataloader.

    edt_mode: model outputs 1ch normalized EDT; fg prediction = edt > 0.05.
    """
    model.eval()
    dices = []
    with torch.no_grad():
        for bi, (x, m) in enumerate(dl):
            if max_batches and bi >= max_batches:
                break
            x = x.to(device)
            with torch.autocast('cuda', dtype=torch.float16) if use_amp else _NullCtx():
                logits = model(x)
            if edt_mode:
                pr = (logits.squeeze(1).float() > 0.05).cpu()
                gt_bool = m.numpy() > 0  # edt target: fg = edt > 0
            else:
                pr = logits.argmax(1).cpu() == 1
                gt_bool = m.numpy().astype(bool)
            for j in range(pr.shape[0]):
                dices.append(_dice(pr[j].numpy(), gt_bool[j]))
    model.train()
    return float(np.mean(dices)) if dices else 0.0


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    """Train fold1-only NuLite-T foreground oracle; save best-by-val-dice ckpt."""
    device = torch.device('cuda' if args.device in ('0', 'cuda') else 'cpu')
    use_amp = device.type == 'cuda'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cache_root = Path('/mnt/data/projects/RayCastED/output/oracle')
    cache_root.mkdir(parents=True, exist_ok=True)
    tag = f'tr-{Path(args.train_tiles).name}-va-{Path(args.val_tiles).name}-l{args.limit}'
    imgs_tr, msk_tr = build_cache(cache_root / f'{tag}-tr', args.train_tiles, args.train_parquet, args.limit)
    imgs_va, msk_va = build_cache(cache_root / f'{tag}-va', args.val_tiles, args.val_parquet, args.limit)
    n = len(imgs_tr)
    print(f'[data] train tiles={n}  val tiles={len(imgs_va)}')

    model = NuLiteT(args.variant, separable_decoder=args.separable_decoder, out_ch=1 if args.edt_target else 2)
    model.to(device)
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    dec_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith('decoder.'))
    print(
        f'[model] NuLite-T fold1-only ({args.variant},'
        f' separable_decoder={args.separable_decoder}, edt_target={args.edt_target}) trainable {nparams / 1e6:.2f}M'
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dl_tr = DataLoader(
        _TileDS(imgs_tr, msk_tr, train=True, edt_target=args.edt_target, edt_cap=args.edt_cap),
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
    )
    dl_va = DataLoader(
        _TileDS(imgs_va, msk_va, train=False, edt_target=args.edt_target, edt_cap=args.edt_cap),
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
    )
    val_batches = max(int(np.ceil(args.val_n / args.batch)), 1)

    best_dice, best_ep = 0.0, -1
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        ep_ce, ep_dl, nb = 0.0, 0.0, 0
        for x, m in dl_tr:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16) if use_amp else _NullCtx():
                logits = model(x)
                if args.edt_target:
                    # EDT regression: L1 on the distance map + soft-Dice on
                    # binarized prediction (pred>0.05) vs fg to sharpen ridges.
                    l1 = F.l1_loss(logits.squeeze(1), m)
                    p = logits.squeeze(1).float()
                    mf = (m > 0).float()
                    pb = (p > 0.05).float()
                    inter = (pb * mf).sum()
                    dice_loss = 1.0 - (2.0 * inter + 1.0) / (pb.sum() + mf.sum() + 1.0)
                    loss = l1 + dice_loss
                    ce = l1  # reuse the epoch logger slot
                else:
                    ce = F.cross_entropy(logits, m)
                    p = logits[:, 1].sigmoid().float()
                    mf = m.float()
                    inter = (p * mf).sum()
                    dice_loss = 1.0 - (2.0 * inter + 1.0) / (p.sum() + mf.sum() + 1.0)
                    loss = ce + dice_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ep_ce += ce.item()
            ep_dl += dice_loss.item()
            nb += 1
        vd = evaluate(model, dl_va, device, use_amp, max_batches=val_batches, edt_mode=args.edt_target)
        ep_s = time.time() - t0
        print(
            f'ep {ep:3d}/{args.epochs}  ce {ep_ce / nb:.4f}  diceL {ep_dl / nb:.4f}'
            f'  valDice {vd:.4f}  lr {opt.param_groups[0]["lr"]:.2e}  {ep_s:7.0f}s',
            flush=True,
        )
        if vd > best_dice:
            best_dice, best_ep = vd, ep
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ck = {
                'model_state_dict': sd,
                'arch': f'NuLite-T({args.variant}) fold1-only',
                'config': {
                    'data.dataset': 'PanNuke',
                    'folds': {'train': 0, 'val': 1},
                    'training.loss': ('EDT-L1+dice' if args.edt_target else 'CE+dice'),
                    'note': 'oracle, fold1 only',
                    'separable_decoder': bool(args.separable_decoder),
                },
            }
            torch.save(ck, str(out))
            print(f'  -> saved best valDice {best_dice:.4f} @ ep {ep}')
    print(f'[done] best valDice {best_dice:.4f} @ ep {best_ep} -> {out}')


if __name__ == '__main__':
    main()
