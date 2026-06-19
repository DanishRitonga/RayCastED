"""Zero-Shot Generalization Eval — binary nuclei detection on external datasets.

Reuses load_model, run_inference, metrics from eval_pannuke.py.
All class predictions mapped to binary (fg/bg).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.utils import constants as _const
from raycasted.evals.eval_pannuke import load_model, run_inference, _simple_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.49)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)
    model = load_model(args.weights, device)
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 256)
    n_rays = training_args.get('n_rays', 64)

    _const.configure_rays(n_rays)

    data_dir = Path(args.data_dir)
    npz_files = list(data_dir.glob('*.npz'))
    print(f'Test tiles: {len(npz_files)} ({args.dataset})')

    dataset = RayCastTileDataset(data_dir=str(data_dir), crop_size=crop_size, augment=False)
    dl = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=_simple_collate)

    t0 = time.perf_counter()
    results = run_inference(model, dl, device, conf_threshold=args.conf)
    elapsed = time.perf_counter() - t0
    n_pred = sum(len(r['pred_polys']) for r in results)
    n_gt = sum(len(r['gt_polys']) for r in results)
    print(
        f'Inference: {len(results)} imgs in {elapsed:.1f}s ({elapsed * 1000 / len(results):.1f}ms/img), {n_pred} preds, {n_gt} GT'
    )

    # Binary metrics via eval_pannuke streaming for nc=1
    # Patch: set all classes to 0
    for r in results:
        r['pred_cls'] = np.zeros(len(r['pred_cls']), dtype=int)
        r['gt_cls'] = np.zeros(len(r['gt_cls']), dtype=int)

    from raycasted.evals.eval_pannuke import compute_metrics_streaming

    metrics = compute_metrics_streaming(results, num_classes=1)
    f12 = metrics['centroid']

    print(f'\n{"=" * 50}')
    print(f'  {args.dataset} Zero-Shot (binary)')
    print(f'{"=" * 50}')
    print(f'  AJI:      {metrics["aji"]:.4f}')
    print(f'  bPQ:      {metrics["bpq"]:.4f}')
    print(f'  F1:       {f12["f1"]:.4f}')
    print(f'  Precision:{f12["precision"]:.4f}')
    print(f'  Recall:   {f12["recall"]:.4f}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
