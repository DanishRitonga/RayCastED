"""PanNuke Fold3 Evaluation — LSP-DETR (RT-DETR RayCast decoder).

Handles both full-model-object checkpoints and state_dict checkpoints.
The RT-DETR decoder outputs [cx, cy, rays, cls_0..cls_nc-1] per query
(multi-class VFL sigmoid scores), unlike FCN which outputs [cx, cy, rays, conf, class].

Shares metrics computation with eval_pannuke.py.
"""

import argparse
import atexit
import signal
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from main.eval_pannuke import (
    _uncompile_module,
    benchmark_inference,
    compute_metrics_streaming,
)
from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.model.register import register_raycast_head


def load_model(weights_path: str, device: torch.device):
    """Load LSP-DETR model from checkpoint (handles both formats)."""
    register_raycast_head()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        model = ckpt.get('model') or ckpt.get('ema')
        if model is not None:
            model = _uncompile_module(model)
            model = model.float().to(device)
            model.eval()
            return model

        if 'state_dict' in ckpt or 'model_state_dict' in ckpt:
            from raycasted.model.rtdetr_model import RayCastRTDETRDetectionModel

            sd = ckpt.get('model_state_dict') or ckpt.get('state_dict')
            training_args = ckpt.get('training_args', {})
            nc = training_args.get('nc', 5)

            model = RayCastRTDETRDetectionModel(
                cfg='raycasted/cfg/yolo26s-rtdetr-p234.yaml',
                ch=3,
                nc=nc,
                verbose=False,
            )
            model.load_state_dict(sd, strict=False)
            model = model.float().to(device)
            model.eval()
            return model

        raise ValueError('Checkpoint has no model/ema/state_dict key. Keys: ' + ', '.join(ckpt.keys()))

    model = _uncompile_module(ckpt)
    model = model.float().to(device)
    model.eval()
    return model


def _decode_rtdetr_output(decoded: torch.Tensor, raycast_dim: int, nc: int):
    """Convert RT-DETR output [B, nq, raycast_dim+nc] to numpy per-image lists.

    RT-DETR multi-class VFL: cls scores in columns raycast_dim:raycast_dim+nc.
    Take max across class dim for confidence, argmax for class label.
    """
    det = decoded.cpu().numpy()
    n_cols = det.shape[1] if det.ndim == 2 else 0

    if det.ndim != 2 or n_cols < raycast_dim + nc:
        return np.zeros((0, raycast_dim + 2), dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=int)

    cls_scores = det[:, raycast_dim : raycast_dim + nc]
    pred_confs = cls_scores.max(axis=1)
    pred_cls = cls_scores.argmax(axis=1).astype(int)

    n_pred = det.shape[0]
    result = np.zeros((n_pred, raycast_dim + 2), dtype=np.float32)
    result[:, :raycast_dim] = det[:, :raycast_dim]
    result[:, raycast_dim] = pred_confs
    result[:, raycast_dim + 1] = pred_cls.astype(np.float32)
    return result, pred_confs, pred_cls


def run_inference(model, dataloader, device, conf_threshold=0.20, debug=False):
    """Run inference for LSP-DETR models, decoding multi-class VFL output."""
    results = []
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 640)
    n_rays = training_args.get('n_rays', 64)
    nc = training_args.get('nc', 5)
    raycast_dim = 2 + n_rays

    with torch.no_grad():
        for _batch_idx, batch in enumerate(dataloader):
            images = batch['img'].to(device)
            raw_out = model(images)
            decoded = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            batch_size = images.shape[0]

            if debug and _batch_idx == 0:
                print(
                    f'  [debug] decoded.shape={decoded.shape}, n_rays={n_rays}, raycast_dim={raycast_dim}, nc={nc}',
                    flush=True,
                )
                d0 = decoded[0].cpu().numpy()
                if d0.ndim == 2 and d0.shape[1] >= raycast_dim + nc:
                    cls_scores = d0[:, raycast_dim : raycast_dim + nc]
                    confs = cls_scores.max(axis=1)
                    print(
                        f'  [debug] conf range=[{confs.min():.4f}, {confs.max():.4f}], '
                        f'n_above_{conf_threshold:.2f}={(confs > conf_threshold).sum()}',
                        flush=True,
                    )

            for si in range(batch_size):
                mask = batch['batch_idx'] == si
                gt_cls = batch['cls'][mask].numpy().flatten()
                gt_poly = batch['bboxes'][mask].numpy()

                if gt_poly.shape[0] > 0:
                    gt_poly = gt_poly.copy()
                    gt_poly[:, 0] *= crop_size
                    gt_poly[:, 1] *= crop_size
                    gt_poly[:, 2:] *= crop_size

                det, all_confs, all_cls = _decode_rtdetr_output(decoded[si], raycast_dim, nc)

                if det.shape[0] > 0:
                    conf_mask = all_confs > conf_threshold
                    det = det[conf_mask]
                    pred_confs = all_confs[conf_mask]
                    pred_cls = all_cls[conf_mask]
                else:
                    pred_confs = np.array([], dtype=np.float32)
                    pred_cls = np.array([], dtype=int)

                pred_poly = det[:, :raycast_dim] if det.shape[0] > 0 else np.zeros((0, raycast_dim), dtype=np.float32)

                results.append(
                    {
                        'pred_polys': pred_poly,
                        'pred_confs': pred_confs,
                        'gt_polys': gt_poly,
                        'pred_cls': pred_cls,
                        'gt_cls': gt_cls.astype(int),
                        'imgsz': crop_size,
                    }
                )

    return results


def main():
    def _crash_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f'\nFATAL: received {sig_name} — process dying', file=sys.stderr, flush=True)
        sys.exit(128 + signum)

    def _clean_exit():
        print('__CLEAN_EXIT__', flush=True)

    atexit.register(_clean_exit)

    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        signal.signal(sig, _crash_handler)

    parser = argparse.ArgumentParser(description='PanNuke Fold3 Evaluation (LSP-DETR)')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--conf', type=float, default=0.20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--watershed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max-images', type=int, default=0)
    args = parser.parse_args()

    try:
        _main(args)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _main(args):
    data_dir = args.data_dir
    if not data_dir and args.config is not None and args.output is not None:
        from raycasted.data.etl.transform.transform_orchestrator import TransformOrchestrator
        from raycasted.data.etl.utils.config import ETLConfig

        config = ETLConfig(args.config)
        transformed_dir = Path(args.output) / 'transformed'
        test_dir = transformed_dir / 'test'

        if not test_dir.exists() or not any(test_dir.glob('*.npz')):
            print(f'Test tiles not found at {test_dir}. Running transform...')
            t = TransformOrchestrator(
                config_manager=config,
                ingested_dir=str(Path(args.output) / 'ingested'),
                final_output_dir=str(transformed_dir),
            )
            t.run_pipeline()
            from raycasted.pipeline import RayCastPipeline

            RayCastPipeline._organize_by_split(None, t.registry)
        data_dir = str(test_dir)

    if not data_dir:
        raise ValueError('Either --data-dir or both --config and --output must be provided')

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    npz_files = list(data_dir.glob('*.npz'))
    print(f'Test tiles: {len(npz_files)} files in {data_dir}')

    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)

    print(f'Loading model: {args.weights}', flush=True)
    model = load_model(args.weights, device)
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 640)
    nc = training_args.get('nc', 5)
    n_rays = training_args.get('n_rays', 64)
    print(f'  crop_size={crop_size}, nc={nc}, n_rays={n_rays}', flush=True)

    from raycasted.data.etl.utils.constants import configure_rays

    configure_rays(n_rays)

    dataset = RayCastTileDataset(data_dir=str(data_dir), crop_size=crop_size, augment=False)
    from eval_pannuke import _simple_collate

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_simple_collate,
    )

    print('Running inference...', flush=True)
    results = run_inference(model, dataloader, device, conf_threshold=args.conf, debug=args.debug)
    if args.max_images > 0:
        results = results[: args.max_images]
        n_pred_total = sum(len(r['pred_polys']) for r in results)
        n_gt_total = sum(len(r['gt_polys']) for r in results)
        print(f'  Limited to {args.max_images} images: {n_pred_total} predictions, {n_gt_total} GT', flush=True)
    else:
        n_pred_total = sum(len(r['pred_polys']) for r in results)
        n_gt_total = sum(len(r['gt_polys']) for r in results)
        print(f'  Processed {len(results)} images: {n_pred_total} predictions, {n_gt_total} GT', flush=True)

    print('Computing metrics (streaming)...', flush=True)
    metrics = compute_metrics_streaming(results, num_classes=nc, watershed=args.watershed)

    ap_results = metrics['ap']
    ap50 = ap_results.get(0.5, {}).get('AP', 0.0)
    ap70 = ap_results.get(0.7, {}).get('AP', 0.0)
    ap90 = ap_results.get(0.9, {}).get('AP', 0.0)
    ap50_95 = np.mean([ap_results[t]['AP'] for t in sorted(ap_results.keys())])
    f12 = metrics['centroid']

    from ultralytics.utils.torch_utils import model_info

    n_params = sum(p.numel() for p in model.parameters())
    params_m = n_params / 1e6
    try:
        _, _, _, gflops = model_info(model, imgsz=crop_size, verbose=True)
    except Exception:
        gflops = 0.0

    print('Benchmarking inference time...', flush=True)
    inf_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_simple_collate)
    avg_ms = benchmark_inference(model, inf_dl, device)

    print('\n' + '=' * 60, flush=True)
    print('PanNuke Fold3 Evaluation Results (LSP-DETR Protocol)')
    print('=' * 60)
    print(f'{"Metric":<25} {"Value":>12}')
    print('-' * 37)
    print(f'{"AJI":<25} {metrics["aji"]:>12.4f}')
    print(f'{"AP@0.5":<25} {ap50:>12.4f}')
    print(f'{"AP@0.7":<25} {ap70:>12.4f}')
    print(f'{"AP@0.9":<25} {ap90:>12.4f}')
    print(f'{"AP@0.5:0.05:0.95":<25} {ap50_95:>12.4f}')
    print(f'{"bPQ":<25} {metrics["bpq"]:>12.4f}')
    print(f'{"bMPQ":<25} {metrics["bmpq"]:>12.4f}')
    print(f'{"mPQ":<25} {metrics["mpq"]:>12.4f}')
    print(f'{"mMPQ":<25} {metrics["mmpq"]:>12.4f}')
    print(f'{"F1 (r=12px)":<25} {f12["f1"]:>12.4f}')
    print(f'{"Precision":<25} {f12["precision"]:>12.4f}')
    print(f'{"Recall":<25} {f12["recall"]:>12.4f}')
    print(f'{"Params (M)":<25} {params_m:>12.2f}')
    print(f'{"FLOPs (G)":<25} {gflops:>12.1f}')
    print(f'{"Inference (ms)":<25} {avg_ms:>12.2f}')
    print(f'{"Predicted":<25} {n_pred_total:>12,}')
    print(f'{"GT nuclei":<25} {n_gt_total:>12,}')
    print(f'{"Conf threshold":<25} {args.conf:>12.2f}')
    if args.watershed:
        print(f'{"Post-processing":<25} {"watershed":>12}')
    print('=' * 60)
    print('__DONE__', flush=True)


if __name__ == '__main__':
    main()
