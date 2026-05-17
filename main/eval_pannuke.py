"""PanNuke Fold3 Evaluation — LSP-DETR aligned metrics.

Runs inference on PanNuke test tiles and computes metrics matching
LSP-DETR's evaluation protocol:
  AJI, AP@0.5, AP@0.7, AP@0.9, AP@0.5:0.05:0.95,
  bPQ, bMPQ, mPQ, mMPQ,
  F1 (centroid, r=12), Precision, Recall,
  Params, FLOPs, Inference Time.

All instance-matching metrics use Hungarian assignment (not greedy)
to match LSP-DETR exactly.

Usage:
    # With pre-transformed test tiles
    uv run python main/eval_pannuke.py \
        --weights train4/weights/best.pt \
        --data-dir output/pannuke/transformed/test \
        --batch 16 --device 0

    # Auto-transform from config
    uv run python main/eval_pannuke.py \
        --config main/pannuke.yaml \
        --output output/pannuke \
        --weights train4/weights/best.pt \
        --batch 16 --device 0
"""

import argparse
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from ultralytics.utils.torch_utils import model_info

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.utils import constants as _const
from raycasted.model.metrics import (
    compute_aji,
    resolve_mask_overlaps,
)
from raycasted.model.register import register_raycast_head


def _polygons_to_masks_fast(detections: np.ndarray, img_h: int, img_w: int) -> list[np.ndarray]:
    """Rasterize raycast polygons to binary masks using OpenCV (fast)."""
    masks = []
    cos = _const.RAY_COS
    sin = _const.RAY_SIN
    for det in detections:
        cx, cy = det[0], det[1]
        rays = det[2:]
        if not np.isfinite(rays).all() or not np.isfinite(cx) or not np.isfinite(cy):
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
        vx = cx + rays * cos
        vy = cy + rays * sin
        pts = np.stack([vx, vy], axis=-1)
        pts = np.clip(pts, -32768, 32767).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
    return masks


def load_model(weights_path: str, device: torch.device):
    """Load trained RayCastED model from checkpoint."""
    register_raycast_head()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'] if isinstance(ckpt, dict) else ckpt
    model = model.float().to(device)
    model.eval()
    return model


def centroid_nms(pred_polys, pred_confs, pred_cls, min_distance_px=6.0):
    """Remove duplicate predictions whose centroids are closer than min_distance_px.

    Keeps the highest-confidence prediction in each cluster.
    """
    if len(pred_polys) == 0:
        return pred_polys, pred_confs, pred_cls

    order = np.argsort(-pred_confs)
    pred_polys = pred_polys[order]
    pred_confs = pred_confs[order]
    pred_cls = pred_cls[order]

    centroids = pred_polys[:, :2]
    keep = []
    for i in range(len(centroids)):
        should_keep = True
        for j in keep:
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_distance_px:
                should_keep = False
                break
        if should_keep:
            keep.append(i)

    keep = np.array(keep)
    return pred_polys[keep], pred_confs[keep], pred_cls[keep]


def run_inference(model, dataloader, device, conf_threshold=0.20, nms_dist=6.0):
    """Run inference over all tiles, collecting predictions and GT.

    Returns:
        results: list of dicts, one per image, with keys:
            'pred_polys', 'pred_confs', 'gt_polys', 'pred_cls', 'gt_cls', 'imgsz'
    """
    results = []
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 640)
    n_rays = training_args.get('n_rays', 32)
    raycast_dim = 2 + n_rays

    with torch.no_grad():
        for _batch_idx, batch in enumerate(dataloader):
            images = batch['img'].to(device)
            raw_out = model(images)
            decoded = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            batch_size = images.shape[0]

            for si in range(batch_size):
                # --- GT ---
                mask = batch['batch_idx'] == si
                gt_cls = batch['cls'][mask].numpy().flatten()
                gt_poly = batch['bboxes'][mask].numpy()

                if gt_poly.shape[0] > 0:
                    gt_poly = gt_poly.copy()
                    gt_poly[:, 0] *= crop_size
                    gt_poly[:, 1] *= crop_size
                    gt_poly[:, 2:] *= crop_size

                # --- Predictions ---
                det = decoded[si].cpu().numpy()
                n_cols = det.shape[1] if det.ndim == 2 else 0

                if det.ndim == 2 and n_cols >= raycast_dim + 2:
                    pred_confs = det[:, raycast_dim]
                    pred_cls = det[:, raycast_dim + 1].astype(int)
                    conf_mask = pred_confs > conf_threshold
                    det = det[conf_mask]
                    pred_confs = pred_confs[conf_mask]
                    pred_cls = pred_cls[conf_mask]
                else:
                    det = det[:0] if det.ndim == 2 else np.zeros((0, raycast_dim + 2), dtype=np.float32)
                    pred_confs = np.array([], dtype=np.float32)
                    pred_cls = np.array([], dtype=int)

                pred_poly = det[:, :raycast_dim] if det.shape[0] > 0 else np.zeros((0, raycast_dim), dtype=np.float32)

                # Centroid NMS to remove duplicate detections
                if len(pred_poly) > 0 and nms_dist > 0:
                    pred_poly, pred_confs, pred_cls = centroid_nms(pred_poly, pred_confs, pred_cls, nms_dist)

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


# ---------------------------------------------------------------------------
# Mask IoU helpers
# ---------------------------------------------------------------------------


def _mask_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise mask IoU between predictions and GT."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)

    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)

    intersection = pred_stack @ gt_stack.T
    pred_area = pred_stack.sum(axis=1, keepdims=True)
    gt_area = gt_stack.sum(axis=1, keepdims=True)
    union = pred_area + gt_area.T - intersection

    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)


def _compute_pq_masked(pred_masks, gt_masks, iou_threshold=0.5, mask=None):
    """Compute PQ with optional foreground mask (bMPQ / mMPQ style)."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    if mask is not None:
        pred_masks = [m & mask for m in pred_masks]
        gt_masks = [m & mask for m in gt_masks]

    iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold

    tp = valid.sum()
    fp = n_pred - tp
    fn = n_gt - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0
    pq = sq * dq
    return float(pq), float(sq), float(dq)


def _compute_ap(recall, precision):
    """Compute average precision from recall and precision arrays."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


# ---------------------------------------------------------------------------
# Streaming metric computation (per-image, memory-efficient)
# ---------------------------------------------------------------------------


def compute_metrics_streaming(results, num_classes):
    """Compute all metrics in a single pass, one image at a time.

    Rasterizes masks, computes per-image metrics, then frees masks.
    Peak memory = O(max_masks_per_image * H * W) instead of O(total_masks * H * W).
    """
    iou_thresholds = [0.5, 0.75] + [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]

    # Accumulators
    aji_scores = []
    bpq_scores = []
    bmpq_scores = []
    class_pq = {c: [] for c in range(num_classes)}
    class_mpq = {c: [] for c in range(num_classes)}
    centroid_tp = 0
    centroid_fp = 0
    centroid_fn = 0

    # AP accumulators — per class, per threshold
    class_set = set()
    for r in results:
        class_set.update(r['gt_cls'].tolist())
        class_set.update(r['pred_cls'].tolist())

    ap_stats = {}
    for cls_id in sorted(class_set):
        ap_stats[cls_id] = {t: {'tp': [], 'fp': [], 'conf': [], 'n_gt': 0} for t in iou_thresholds}

    for i, r in enumerate(results):
        imgsz = r['imgsz']
        pred_polys = r['pred_polys']
        gt_polys = r['gt_polys']
        pred_cls = r['pred_cls']
        gt_cls = r['gt_cls']
        pred_confs = r['pred_confs']

        if i == 0:
            print(f'  First image: {len(pred_polys)} preds, {len(gt_polys)} GT, imgsz={imgsz}', flush=True)

        # Rasterize masks for this image only
        gt_masks = _polygons_to_masks_fast(gt_polys, imgsz, imgsz) if len(gt_polys) > 0 else []
        pred_masks = _polygons_to_masks_fast(pred_polys, imgsz, imgsz) if len(pred_polys) > 0 else []
        if pred_masks:
            pred_masks = resolve_mask_overlaps(pred_masks)

        # --- AJI ---
        aji_scores.append(compute_aji(pred_masks, gt_masks))

        # --- bPQ / bMPQ ---
        if len(pred_masks) > 0:
            pred_binary = np.stack(pred_masks).max(axis=0).astype(np.uint8)
        else:
            pred_binary = np.zeros((imgsz, imgsz), dtype=np.uint8)

        if len(gt_masks) > 0:
            gt_binary = np.stack(gt_masks).max(axis=0).astype(np.uint8)
        else:
            gt_binary = np.zeros((imgsz, imgsz), dtype=np.uint8)

        bpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary])
        bpq_scores.append(bpq)

        if gt_binary.sum() > 0:
            fg = gt_binary > 0
            bmpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary], mask=fg)
        else:
            bmpq = 0.0
        bmpq_scores.append(bmpq)

        # --- mPQ / mMPQ ---
        for cls_id in range(num_classes):
            pred_idx = [j for j, c in enumerate(pred_cls) if c == cls_id]
            gt_idx = [j for j, c in enumerate(gt_cls) if c == cls_id]

            pred_cls_masks = [pred_masks[j] for j in pred_idx]
            gt_cls_masks = [gt_masks[j] for j in gt_idx]

            pq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks)
            class_pq[cls_id].append(pq)

            if len(gt_masks) > 0:
                gt_any = np.stack(gt_masks).max(axis=0).astype(np.uint8)
                fg = gt_any > 0
                mpq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks, mask=fg)
            else:
                mpq = 0.0
            class_mpq[cls_id].append(mpq)

        # --- AP (per-class Hungarian matching) ---
        for cls_id in sorted(class_set):
            cls_pred_mask = pred_cls == cls_id
            cls_pred_masks = [
                pred_masks[j] for j in range(len(pred_masks)) if j < len(pred_cls) and pred_cls[j] == cls_id
            ]
            cls_gt_masks = [gt_masks[j] for j in range(len(gt_masks)) if j < len(gt_cls) and gt_cls[j] == cls_id]
            cls_confs = pred_confs[cls_pred_mask]

            n_pred_cls = len(cls_pred_masks)
            n_gt_cls = len(cls_gt_masks)

            for t in iou_thresholds:
                ap_stats[cls_id][t]['n_gt'] += n_gt_cls

            if n_pred_cls > 0 and n_gt_cls > 0:
                iou_mat = _mask_iou_matrix(cls_pred_masks, cls_gt_masks)
                row_ind, col_ind = linear_sum_assignment(-iou_mat)
                matched_iou = iou_mat[row_ind, col_ind]
            else:
                row_ind = np.array([], dtype=int)
                matched_iou = np.array([], dtype=float)

            for t in iou_thresholds:
                if n_gt_cls > 0:
                    valid = matched_iou >= t
                    matched_pred = set(row_ind[valid].tolist())
                else:
                    matched_pred = set()

                for pi in range(n_pred_cls):
                    is_tp = pi in matched_pred
                    ap_stats[cls_id][t]['conf'].append(float(cls_confs[pi]) if pi < len(cls_confs) else 0.0)
                    ap_stats[cls_id][t]['tp'].append(is_tp)
                    ap_stats[cls_id][t]['fp'].append(not is_tp)

        # --- Centroid F1 ---
        n_pred = len(pred_polys)
        n_gt = len(gt_polys)
        if n_pred > 0 and n_gt > 0:
            dist_matrix = np.linalg.norm(pred_polys[:, :2][:, None, :] - gt_polys[:, :2][None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            tp = int((dist_matrix[row_ind, col_ind] <= 12).sum())
        elif n_pred > 0:
            tp = 0
        else:
            tp = 0
        centroid_tp += tp
        centroid_fp += n_pred - tp
        centroid_fn += n_gt - tp

        # Free masks for this image
        del gt_masks, pred_masks

        if (i + 1) % 500 == 0:
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f'  Processed {i + 1}/{len(results)} images (RSS={mem_mb:.0f}MB)', flush=True)

    # --- Aggregate ---

    # AJI
    mean_aji = np.mean(aji_scores)

    # bPQ / bMPQ
    mean_bpq = np.mean(bpq_scores)
    mean_bmpq = np.mean(bmpq_scores)

    # mPQ / mMPQ
    mpq_values = []
    mmpq_values = []
    for c in range(num_classes):
        valid_pq = [v for v in class_pq[c] if v > 0]
        valid_mpq = [v for v in class_mpq[c] if v > 0]
        if valid_pq:
            mpq_values.append(np.mean(valid_pq))
        if valid_mpq:
            mmpq_values.append(np.mean(valid_mpq))
    mean_mpq = np.mean(mpq_values) if mpq_values else 0.0
    mean_mmpq = np.mean(mmpq_values) if mmpq_values else 0.0

    # AP
    ap_results = {}
    for t in iou_thresholds:
        aps = []
        for cls_id in sorted(class_set):
            stats = ap_stats[cls_id][t]
            n_gt = stats['n_gt']
            if n_gt == 0:
                continue
            confs = np.array(stats['conf'])
            tps = np.array(stats['tp'])
            fps = np.array(stats['fp'])
            if len(confs) == 0:
                aps.append(0.0)
                continue
            order = np.argsort(-confs)
            tps = tps[order]
            fps = fps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / n_gt
            aps.append(_compute_ap(recall, precision))
        ap_results[t] = {'AP': np.mean(aps) if aps else 0.0}

    # Centroid F1
    prec = centroid_tp / (centroid_tp + centroid_fp) if (centroid_tp + centroid_fp) > 0 else 0.0
    rec = centroid_tp / (centroid_tp + centroid_fn) if (centroid_tp + centroid_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'aji': mean_aji,
        'bpq': mean_bpq,
        'bmpq': mean_bmpq,
        'mpq': mean_mpq,
        'mmpq': mean_mmpq,
        'ap': ap_results,
        'centroid': {'precision': prec, 'recall': rec, 'f1': f1},
    }


def benchmark_inference(model, dataloader, device, n_warmup=10):
    """Benchmark average inference time per image."""
    times = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['img'].to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(images)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

            if i >= n_warmup:
                times.append(elapsed / images.shape[0])

    return np.mean(times) if times else 0.0


def main():
    import atexit
    import signal
    import sys

    def _crash_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f'\nFATAL: received {sig_name} — process dying', file=sys.stderr, flush=True)
        sys.exit(128 + signum)

    def _clean_exit():
        print('__CLEAN_EXIT__', flush=True)

    atexit.register(_clean_exit)

    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        signal.signal(sig, _crash_handler)

    parser = argparse.ArgumentParser(description='PanNuke Fold3 Evaluation')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--conf', type=float, default=0.20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--nms-dist', type=float, default=6.0)
    args = parser.parse_args()

    try:
        _main(args)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _main(args):
    data_dir = args.data_dir
    if data_dir is None and args.config is not None and args.output is not None:
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

    # Load model
    print(f'Loading model: {args.weights}', flush=True)
    model = load_model(args.weights, device)
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 640)
    nc = training_args.get('nc', 1)
    n_rays = training_args.get('n_rays', 32)
    print(f'  crop_size={crop_size}, nc={nc}, n_rays={n_rays}', flush=True)

    # Configure ray geometry
    from raycasted.data.etl.utils.constants import configure_rays

    configure_rays(n_rays)

    # Build dataset
    dataset = RayCastTileDataset(data_dir=str(data_dir), crop_size=crop_size, augment=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_simple_collate,
    )

    # --- Run inference ---
    print('Running inference...', flush=True)
    results = run_inference(model, dataloader, device, conf_threshold=args.conf, nms_dist=args.nms_dist)
    n_pred_total = sum(len(r['pred_polys']) for r in results)
    n_gt_total = sum(len(r['gt_polys']) for r in results)
    print(f'  Processed {len(results)} images: {n_pred_total} predictions, {n_gt_total} GT', flush=True)

    # --- Compute all metrics (streaming, memory-efficient) ---
    print('Computing metrics (streaming)...', flush=True)
    metrics = compute_metrics_streaming(results, num_classes=nc)

    ap_results = metrics['ap']
    ap50 = ap_results.get(0.5, {}).get('AP', 0.0)
    ap70 = ap_results.get(0.7, {}).get('AP', 0.0)
    ap90 = ap_results.get(0.9, {}).get('AP', 0.0)
    ap50_95 = np.mean([ap_results[t]['AP'] for t in sorted(ap_results.keys())])
    f12 = metrics['centroid']

    # --- Model stats ---
    n_params = sum(p.numel() for p in model.parameters())
    params_m = n_params / 1e6

    try:
        _, _, _, gflops = model_info(model, imgsz=crop_size, verbose=True)
    except Exception:
        gflops = 0.0

    # --- Inference time ---
    print('Benchmarking inference time...', flush=True)
    inf_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_simple_collate)
    avg_ms = benchmark_inference(model, inf_dl, device)

    # --- Print results ---
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
    print(f'{"F1 (centroid, r=12)":<25} {f12["f1"]:>12.4f}')
    print(f'{"Precision (centroid)":<25} {f12["precision"]:>12.4f}')
    print(f'{"Recall (centroid)":<25} {f12["recall"]:>12.4f}')
    print(f'{"Params (M)":<25} {params_m:>12.2f}')
    print(f'{"GFLOPs":<25} {gflops:>12.2f}')
    print(f'{"Inference Time (ms/img)":<25} {avg_ms:>12.2f}')
    print('=' * 37)

    print(f'\nImages evaluated: {len(results)}')
    print(f'Confidence threshold: {args.conf}')
    print(f'NMS distance: {args.nms_dist}')
    print(f'Total predictions (after NMS): {n_pred_total}')
    print(f'Total GT instances: {n_gt_total}')


def _simple_collate(batch):
    """Collate function matching _raycast_collate_fn from train.py."""
    import torch as _torch

    images = _torch.stack([item[0] for item in batch])
    labels_list = [item[1] for item in batch]

    target_list = []
    for batch_idx, labels in enumerate(labels_list):
        if labels.shape[0] == 0:
            continue
        batch_col = np.full((labels.shape[0], 1), batch_idx, dtype=np.float32)
        target_list.append(np.concatenate([batch_col, labels], axis=1))

    if target_list:
        targets = _torch.from_numpy(np.concatenate(target_list, axis=0))
    else:
        ann_width = next((lbl.shape[1] for lbl in labels_list if lbl.ndim == 2 and lbl.shape[1] > 0), 35)
        targets = _torch.zeros((0, 2 + ann_width), dtype=_torch.float32)

    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': targets[:, 2:],
    }


if __name__ == '__main__':
    main()
