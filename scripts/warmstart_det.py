"""Warm-start a NuLite-DETR detector from a stripped checkpoint (option B).

`--resume` cannot be used for this: `final_eval` calls `strip_optimizer`, which
removes epoch/optimizer/updates from the saved checkpoint, and ultralytics
`check_resume` then restores the *stale* `train_args` wholesale (including
`epochs: 600`), so `--epochs` overrides are silently ignored.

This script instead constructs the trainer directly with an explicit weight
transfer and a fresh cosine schedule of the requested length. The source
checkpoint is left untouched; output goes to a new run directory.

Usage:
    .venv/bin/python scripts/warmstart_det.py \
        --recipe main/pannuke-thin-s12.yaml \
        --weights output/v2/train52_fastvit-s12/weights/last_ep332.pt \
        --epochs 68 --name train52_fastvit-s12-warm68 --device 0 --workers 4
"""

import argparse
import math

import torch

from raycasted.data.etl.utils import constants as _const
from raycasted.data.etl.utils.config import ETLConfig
from raycasted.model.train import RayCastTrainer


def parse_args(argv=None):
    """Parse warm-start arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', required=True, help='ETL recipe yaml (defines the architecture)')
    ap.add_argument('--weights', required=True, help='stripped checkpoint to warm-start from')
    ap.add_argument('--epochs', type=int, required=True, help='length of the new cosine schedule')
    ap.add_argument('--name', required=True, help='new run name (must not exist)')
    ap.add_argument('--output', default='output/v2', help='project dir')
    ap.add_argument('--data', default='output/v2/transformed/data.yaml', help='dataset yaml')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr0', type=float, default=None, help='override lr0 (default: from checkpoint args)')
    return ap.parse_args(argv)


def main(argv=None):
    """Load weights into a freshly built model and train a new cosine schedule."""
    args = parse_args(argv)
    _const.configure_rays(64)

    training_config = ETLConfig(args.recipe).training

    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    src_args = dict(ckpt.get('train_args') or {})
    model_obj = ckpt.get('model') if ckpt.get('model') is not None else ckpt.get('ema')
    if model_obj is None:
        raise SystemExit(f'no weights found in {args.weights}')
    state = model_obj.state_dict()

    lr0 = args.lr0 if args.lr0 is not None else float(src_args.get('lr0', 0.01))
    overrides = {
        'epochs': args.epochs,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'optimizer': src_args.get('optimizer', 'MuSGD'),
        'lr0': lr0,
        'lrf': float(src_args.get('lrf', 0.01)),
        'warmup_epochs': float(src_args.get('warmup_epochs', 3.0)),
        'cos_lr': True,
        'pretrained': False,
        'model': args.weights,  # placeholder; real transfer happens below
        'data': args.data,
        'imgsz': 256,
        'project': args.output,
        'name': args.name,
        'exist_ok': False,
        'resume': False,
        'plots': bool(src_args.get('plots', True)),
        'amp': bool(training_config.get('amp', src_args.get('amp', False))),
    }

    trainer = RayCastTrainer(overrides=overrides, training_config=training_config, end_epoch=args.epochs)

    # Build the model, then transfer the weights before training starts.
    trainer.model = trainer.get_model(cfg=None, weights=model_obj, verbose=True)
    print(f'[warmstart] loaded {len(state)} tensors from {args.weights}', flush=True)
    print('[warmstart] done', flush=True)
    del state

    print(f'[warmstart] schedule: {args.epochs} epochs, lr0={lr0}, lrf={overrides["lrf"]}', flush=True)
    for ep in (0, args.epochs // 2, args.epochs):
        t = (1 - math.cos(ep * math.pi / args.epochs)) / 2
        print(f'[warmstart]   ep {ep:3d} lr {lr0 * (t * (overrides["lrf"] - 1) + 1):.6f}', flush=True)

    trainer.train()


if __name__ == '__main__':
    main()
