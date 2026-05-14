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
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from ultralytics.utils.torch_utils import model_info

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.model.metrics import (
    compute_aji,
    polygons_to_masks,
    resolve_mask_overlaps,
)
from raycasted.model.register import register_raycast_head


def load_model(weights_path: str, device: torch.device):
    """Load trained RayCastED model from checkpoint."""
    register_raycast_head()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    # Ultralytics saves a dict with 'model', 'train_args', etc.
    model = ckpt['model'] if isinstance(ckpt, dict) else ckpt
    model = model.float().to(device)
    model.eval()
    return model


def run_inference(model, dataloader, device, conf_threshold=0.20):
    """Run inference over all tiles, collecting predictions and GT.

    Returns:
        results: list of dicts, one per image, with keys:
            'pred_polys': [N_pred, raycast_dim] denormalised polygons
            'gt_polys': [N_gt, raycast_dim] denormalised polygons
            'gt_cls': [N_gt] class labels
            'pred_cls': [N_pred] class labels
            'imgsz': int, tile size
    """
    results = []
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 640)
    n_rays = training_args.get('n_rays', 32)
    raycast_dim = 2 + n_rays

    with torch.no_grad():
        for _batch_idx, batch in enumerate(dataloader):
            images = batch['img'].to(device)

            # Forward pass — model(images) returns (decoded_preds, training_dict)
            raw_out = model(images)
            decoded = raw_out[0] if isinstance(raw_out, tuple) else raw_out

            batch_size = images.shape[0]

            for si in range(batch_size):
                # --- GT ---
                mask = batch['batch_idx'] == si
                gt_cls = batch['cls'][mask].numpy().flatten()
                gt_poly = batch['bboxes'][mask].numpy()  # [N_gt, raycast_dim] normalised

                # Denormalise GT
                if gt_poly.shape[0] > 0:
                    gt_poly = gt_poly.copy()
                    gt_poly[:, 0] *= crop_size
                    gt_poly[:, 1] *= crop_size
                    gt_poly[:, 2:] *= crop_size

                # --- Predictions ---
                det = decoded[si].cpu().numpy()  # [max_det, raycast_dim+2]
                # Filter by confidence
                if det.ndim == 2 and det.shape[1] == raycast_dim + 2:
                    conf_mask = det[:, raycast_dim] > conf_threshold
                    det = det[conf_mask]

                if det.shape[0] > 0:
                    pred_poly = det[:, :raycast_dim]  # [N_pred, raycast_dim]
                    pred_cls = det[:, raycast_dim + 1].astype(int)
                    pred_confs = det[:, raycast_dim]  # confidence score
                else:
                    pred_poly = np.zeros((0, raycast_dim), dtype=np.float32)
                    pred_cls = np.array([], dtype=int)
                    pred_confs = np.array([], dtype=np.float32)

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


def centroid_nms(pred_polys, pred_confs, pred_cls, min_distance_px=6.0):
    """Remove duplicate predictions whose centroids are closer than min_distance_px.

    Keeps the highest-confidence prediction in each cluster. This matches the
    deduplication that the training validator performs implicitly through AP
    matching but which instance-level metrics (F1, PQ) require explicitly.

    Args:
        pred_polys: [N, raycast_dim] polygon predictions (denormalised).
        pred_confs: [N] confidence scores.
        pred_cls: [N] class labels.
        min_distance_px: Minimum allowed centroid distance (pixels).

    Returns:
        Filtered (pred_polys, pred_confs, pred_cls).
    """
    if len(pred_polys) == 0:
        return pred_polys, pred_confs, pred_cls

    # Sort by confidence descending — keep highest conf in each cluster
    order = np.argsort(-pred_confs)
    pred_polys = pred_polys[order]
    pred_confs = pred_confs[order]
    pred_cls = pred_cls[order]

    centroids = pred_polys[:, :2]  # [N, 2]
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


def _mask_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise mask IoU between predictions and GT.

    Args:
        pred_masks: List of [H, W] uint8 binary masks.
        gt_masks: List of [H, W] uint8 binary masks.

    Returns:
        IoU matrix of shape (N_pred, N_gt).
    """
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


def compute_centroid_f1(results, distance_thresholds=None):
    """Compute centroid-based F1 at specified distance thresholds (LSP-DETR style).

    Uses Hungarian matching on centroid Euclidean distances, then filters
    by radius. Matches LSP-DETR's F1Score exactly.

    Args:
        results: List of dicts with 'pred_polys', 'gt_polys', 'pred_cls', 'gt_cls'.
        distance_thresholds: List of distance thresholds in pixels (default: [6, 8, 10, 12]).

    Returns:
        dict: {threshold: {'precision': float, 'recall': float, 'f1': float}}
    """
    if distance_thresholds is None:
        distance_thresholds = [6, 8, 10, 12]

    per_threshold = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in distance_thresholds}

    for r in results:
        pred_polys = r['pred_polys']
        gt_polys = r['gt_polys']
        n_pred = len(pred_polys)
        n_gt = len(gt_polys)

        if n_pred == 0 and n_gt == 0:
            continue
        if n_pred == 0:
            for t in distance_thresholds:
                per_threshold[t]['fn'] += n_gt
            continue
        if n_gt == 0:
            for t in distance_thresholds:
                per_threshold[t]['fp'] += n_pred
            continue

        # Centroid distance matrix
        pred_c = pred_polys[:, :2]  # [N_pred, 2]
        gt_c = gt_polys[:, :2]  # [N_gt, 2]
        dist_matrix = np.linalg.norm(pred_c[:, None, :] - gt_c[None, :, :], axis=2)  # [N_pred, N_gt]

        # Hungarian matching (minimize total distance)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # Evaluate each threshold
        for t in distance_thresholds:
            valid = dist_matrix[row_ind, col_ind] <= t
            tp = int(valid.sum())
            fp = n_pred - tp
            fn = n_gt - tp
            per_threshold[t]['tp'] += tp
            per_threshold[t]['fp'] += fp
            per_threshold[t]['fn'] += fn

    results_dict = {}
    for t in distance_thresholds:
        tp = per_threshold[t]['tp']
        fp = per_threshold[t]['fp']
        fn = per_threshold[t]['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results_dict[t] = {'precision': precision, 'recall': recall, 'f1': f1}

    return results_dict


def compute_ap_2018_dsb(results, iou_thresholds=None):
    """Compute AP at specified IoU thresholds using mask IoU (2018 DSB style).

    Matches LSP-DETR's AveragePrecision2018DSB: Hungarian assignment on
    mask IoU, then per-threshold TP/FP counting. Predictions are sorted
    globally by confidence and the precision-recall curve is integrated.

    Returns:
        dict with AP, precision, recall at each threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75] + [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]

    # Collect all predictions and GT per class
    class_set = set()
    for r in results:
        class_set.update(r['gt_cls'].tolist())
        class_set.update(r['pred_cls'].tolist())

    per_class_stats = {}
    for cls_id in sorted(class_set):
        per_class_stats[cls_id] = {t: {'tp': [], 'fp': [], 'conf': [], 'n_gt': 0} for t in iou_thresholds}

        for r in results:
            gt_mask = r['gt_cls'] == cls_id
            pred_mask = r['pred_cls'] == cls_id
            gt_polys = r['gt_polys'][gt_mask]
            pred_polys = r['pred_polys'][pred_mask]
            pred_confs = r.get('pred_confs', np.ones(len(pred_polys)))

            n_pred = len(pred_polys)
            n_gt = len(gt_polys)

            for t in iou_thresholds:
                per_class_stats[cls_id][t]['n_gt'] += n_gt

            if n_pred == 0:
                # No predictions for this class in this image
                continue

            # Compute pairwise mask IoU
            imgsz = r['imgsz']
            if n_gt > 0:
                gt_masks = polygons_to_masks(gt_polys, imgsz, imgsz)
                pred_masks = polygons_to_masks(pred_polys, imgsz, imgsz)
                iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)

                # Hungarian matching (maximize total IoU)
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_iou = iou_matrix[row_ind, col_ind]
            else:
                # No GT: every prediction is FP at all thresholds
                row_ind = np.array([], dtype=int)
                matched_iou = np.array([], dtype=float)

            # Per-threshold TP/FP (same matching for all thresholds)
            for t in iou_thresholds:
                if n_gt > 0:
                    valid = matched_iou >= t
                    matched_pred = set(row_ind[valid].tolist())
                else:
                    matched_pred = set()

                for pi in range(n_pred):
                    is_tp = pi in matched_pred
                    per_class_stats[cls_id][t]['conf'].append(float(pred_confs[pi]))
                    per_class_stats[cls_id][t]['tp'].append(is_tp)
                    per_class_stats[cls_id][t]['fp'].append(not is_tp)

    # Compute AP per class per threshold
    results_dict = {}
    for t in iou_thresholds:
        aps = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for cls_id in sorted(class_set):
            stats = per_class_stats[cls_id][t]
            n_gt = stats['n_gt']
            if n_gt == 0:
                continue

            confs = np.array(stats['conf'])
            tps = np.array(stats['tp'])
            fps = np.array(stats['fp'])

            if len(confs) == 0:
                aps.append(0.0)
                total_fn += n_gt
                continue

            # Sort by confidence descending
            order = np.argsort(-confs)
            tps = tps[order]
            fps = fps[order]

            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / n_gt

            ap = _compute_ap(recall, precision)
            aps.append(ap)

            total_tp += int(cum_tp[-1]) if len(cum_tp) > 0 else 0
            total_fp += int(cum_fp[-1]) if len(cum_fp) > 0 else 0
            total_fn += n_gt - (int(cum_tp[-1]) if len(cum_tp) > 0 else 0)

        map_val = np.mean(aps) if aps else 0.0
        precision_val = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall_val = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
        results_dict[t] = {
            'AP': map_val,
            'precision': precision_val,
            'recall': recall_val,
            'F1': f1,
        }

    return results_dict


def _compute_ap(recall, precision):
    """Compute average precision from recall and precision arrays."""
    # Prepend/append sentinels
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum Δrecall × precision
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def _compute_pq_masked(pred_masks, gt_masks, iou_threshold=0.5, mask=None):
    """Compute PQ with optional foreground mask (bMPQ / mMPQ style).

    If ``mask`` is provided, only pixels where ``mask == True`` are considered
    when computing intersections, unions, and areas.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    if mask is not None:
        pred_masks = [m & mask for m in pred_masks]
        gt_masks = [m & mask for m in gt_masks]

    # Re-use vectorised IoU matrix from metrics module
    from raycasted.model.metrics import _compute_mask_iou_matrix

    iou_matrix = _compute_mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold

    tp = valid.sum()
    fp = n_pred - tp
    fn = n_gt - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0
    pq = sq * dq
    return float(pq), float(sq), float(dq)


def compute_binary_pq(results):
    """Compute binary PQ and masked binary PQ (bPQ / bMPQ).

    Merges all instances into a single foreground mask per image,
    then computes PQ between the binary masks.
    """
    bpq_scores = []
    bmpq_scores = []

    for r in results:
        imgsz = r['imgsz']
        gt_masks = polygons_to_masks(r['gt_polys'], imgsz, imgsz) if r['gt_polys'].shape[0] > 0 else []
        pred_masks = polygons_to_masks(r['pred_polys'], imgsz, imgsz) if r['pred_polys'].shape[0] > 0 else []
        if pred_masks:
            pred_masks = resolve_mask_overlaps(pred_masks)

        # Merge to binary foreground masks
        if len(pred_masks) > 0:
            pred_binary = np.stack(pred_masks).max(axis=0).astype(np.uint8)
        else:
            pred_binary = np.zeros((imgsz, imgsz), dtype=np.uint8)

        if len(gt_masks) > 0:
            gt_binary = np.stack(gt_masks).max(axis=0).astype(np.uint8)
        else:
            gt_binary = np.zeros((imgsz, imgsz), dtype=np.uint8)

        # bPQ: standard binary PQ on full image
        bpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary])
        bpq_scores.append(bpq)

        # bMPQ: masked to foreground pixels in GT
        if gt_binary.sum() > 0:
            foreground_mask = gt_binary > 0
            bmpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary], mask=foreground_mask)
        else:
            bmpq = 0.0
        bmpq_scores.append(bmpq)

    return np.mean(bpq_scores), np.mean(bmpq_scores)


def compute_multiclass_pq(results, num_classes=6):
    """Compute multiclass PQ and masked multiclass PQ (mPQ / mMPQ).

    Per-class PQ is averaged over classes that have at least one GT instance
    across the dataset.
    """
    # Accumulate per-class stats
    class_pq = {c: [] for c in range(num_classes)}
    class_mpq = {c: [] for c in range(num_classes)}

    for r in results:
        imgsz = r['imgsz']
        pred_polys = r['pred_polys']
        gt_polys = r['gt_polys']
        pred_cls = r['pred_cls']
        gt_cls = r['gt_cls']

        # Rasterize all masks
        all_pred_masks = polygons_to_masks(pred_polys, imgsz, imgsz) if pred_polys.shape[0] > 0 else []
        all_gt_masks = polygons_to_masks(gt_polys, imgsz, imgsz) if gt_polys.shape[0] > 0 else []
        if all_pred_masks:
            all_pred_masks = resolve_mask_overlaps(all_pred_masks)

        # Build per-class binary masks
        for cls_id in range(num_classes):
            pred_idx = [i for i, c in enumerate(pred_cls) if c == cls_id]
            gt_idx = [i for i, c in enumerate(gt_cls) if c == cls_id]

            pred_cls_masks = [all_pred_masks[i] for i in pred_idx]
            gt_cls_masks = [all_gt_masks[i] for i in gt_idx]

            # Unmasked mPQ (per-class instance PQ)
            pq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks)
            class_pq[cls_id].append(pq)

            # Masked mMPQ: ignore pixels with no GT instance of any class
            if len(all_gt_masks) > 0:
                gt_any = np.stack(all_gt_masks).max(axis=0).astype(np.uint8)
                foreground_mask = gt_any > 0
                mpq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks, mask=foreground_mask)
            else:
                mpq = 0.0
            class_mpq[cls_id].append(mpq)

    # Average per-class PQ over images, then over classes with GT
    mpq_values = []
    mmpq_values = []
    for c in range(num_classes):
        # Only include classes that have at least one non-zero PQ image
        valid_pq = [v for v in class_pq[c] if v > 0 or len(class_pq[c]) == 1]
        valid_mpq = [v for v in class_mpq[c] if v > 0 or len(class_mpq[c]) == 1]
        if valid_pq:
            mpq_values.append(np.mean(valid_pq))
        if valid_mpq:
            mmpq_values.append(np.mean(valid_mpq))

    mean_mpq = np.mean(mpq_values) if mpq_values else 0.0
    mean_mmpq = np.mean(mmpq_values) if mmpq_values else 0.0
    return mean_mpq, mean_mmpq


def benchmark_inference(model, dataloader, device, n_warmup=10):
    """Benchmark average inference time per image.

    Returns:
        avg_ms: average milliseconds per image (after warmup).
    """
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
            elapsed = (time.perf_counter() - start) * 1000  # ms

            if i >= n_warmup:
                times.append(elapsed / images.shape[0])  # per image

    return np.mean(times) if times else 0.0


def main():
    """Run PanNuke fold3 evaluation."""
    parser = argparse.ArgumentParser(description='RayCastED PanNuke Evaluation')
    parser.add_argument('--weights', required=True, help='Path to trained .pt checkpoint')
    parser.add_argument('--data-dir', default=None, help='Path to test tiles directory (.npz files)')
    parser.add_argument('--config', default=None, help='ETL config YAML (for auto-transform)')
    parser.add_argument('--output', default=None, help='Output dir (required with --config)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='0', help='Device (cpu, 0, 0,1)')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--debug', action='store_true', help='Print diagnostic info for first 10 images')
    args = parser.parse_args()

    # Resolve data directory
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
            # Reorganize by split
            from raycasted.pipeline import RayCastPipeline

            RayCastPipeline._organize_by_split(None, t.registry)
        data_dir = str(test_dir)

    if data_dir is None:
        parser.error('Either --data-dir or both --config and --output must be provided')

    data_dir = Path(data_dir)
    if not data_dir.exists():
        parser.error(f'Data directory not found: {data_dir}')

    npz_files = list(data_dir.glob('*.npz'))
    print(f'Test tiles: {len(npz_files)} files in {data_dir}')

    # Device
    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)

    # Load model
    print(f'Loading model: {args.weights}')
    model = load_model(args.weights, device)
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 640)
    nc = training_args.get('nc', 1)
    n_rays = training_args.get('n_rays', 32)
    print(f'  crop_size={crop_size}, nc={nc}, n_rays={n_rays}')

    # Configure ray geometry so polygon rasterization uses correct angles
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
    print('Running inference...')
    results = run_inference(model, dataloader, device, conf_threshold=args.conf)
    print(f'  Processed {len(results)} images')

    # --- Pixel-level metrics (AJI, PQ variants) ---
    print('Computing pixel-level metrics (AJI, bPQ, bMPQ, mPQ, mMPQ)...')
    aji_scores = []
    debug_printed = []

    for r in results:
        imgsz = r['imgsz']
        # GT masks
        gt_masks = polygons_to_masks(r['gt_polys'], imgsz, imgsz) if r['gt_polys'].shape[0] > 0 else []
        # Predicted masks
        pred_masks = polygons_to_masks(r['pred_polys'], imgsz, imgsz) if r['pred_polys'].shape[0] > 0 else []
        # Resolve overlaps (largest-first priority, matching LSP-DETR)
        if pred_masks:
            pred_masks = resolve_mask_overlaps(pred_masks)

        if args.debug and len(debug_printed) < 10 and len(gt_masks) > 0 and len(pred_masks) > 0:
            debug_printed.append(True)
            gt_areas = [m.sum() for m in gt_masks[:3]]
            pred_areas = [m.sum() for m in pred_masks[:3]]
            n_gt = r['gt_polys'].shape[0]
            n_pred = r['pred_polys'].shape[0]
            print(f'  Image {len(debug_printed)}: GT={n_gt}, pred={n_pred}')
            gt0 = r['gt_polys'][0]
            gt_ray_min = gt0[2:].min()
            gt_ray_max = gt0[2:].max()
            print(f'    GT[0] cx,cy={gt0[:2]}, rays=[{gt_ray_min:.1f}, {gt_ray_max:.1f}]')
            print(f'    GT mask areas (first 3): {gt_areas}')
            if len(pred_masks) > 0:
                p0 = r['pred_polys'][0]
                p_ray_min = p0[2:].min()
                p_ray_max = p0[2:].max()
                print(f'    Pred[0] cx,cy={p0[:2]}, rays=[{p_ray_min:.1f}, {p_ray_max:.1f}]')
                print(f'    Pred mask areas (first 3): {pred_areas}')

        aji_scores.append(compute_aji(pred_masks, gt_masks))

    mean_aji = np.mean(aji_scores)
    mean_bpq, mean_bmpq = compute_binary_pq(results)
    mean_mpq, mean_mmpq = compute_multiclass_pq(results)

    # --- AP metrics (LSP-DETR style: Hungarian matching) ---
    print('Computing AP metrics (Hungarian matching)...')
    ap_results = compute_ap_2018_dsb(results)

    ap50 = ap_results.get(0.5, {}).get('AP', 0.0)
    ap70 = ap_results.get(0.7, {}).get('AP', 0.0)
    ap90 = ap_results.get(0.9, {}).get('AP', 0.0)
    ap50_95 = np.mean([ap_results[t]['AP'] for t in sorted(ap_results.keys())])

    # --- Centroid F1 (LSP-DETR style: Hungarian matching, r=12) ---
    print('Computing centroid F1 (Hungarian matching)...')
    centroid_results = compute_centroid_f1(results, distance_thresholds=[12])
    f12 = centroid_results[12]

    # --- Model stats ---
    n_params = sum(p.numel() for p in model.parameters())
    params_m = n_params / 1e6

    # GFLOPs — model_info returns (n_layers, n_params, n_grads, gflops)
    try:
        _, _, _, gflops = model_info(model, imgsz=crop_size, verbose=True)
    except Exception:
        gflops = 0.0

    # --- Inference time ---
    print('Benchmarking inference time...')
    inf_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_simple_collate)
    avg_ms = benchmark_inference(model, inf_dl, device)

    # --- Print results (LSP-DETR format) ---
    print('\n' + '=' * 60)
    print('PanNuke Fold3 Evaluation Results (LSP-DETR Protocol)')
    print('=' * 60)
    print(f'{"Metric":<25} {"Value":>12}')
    print('-' * 37)
    print(f'{"AJI":<25} {mean_aji:>12.4f}')
    print(f'{"AP@0.5":<25} {ap50:>12.4f}')
    print(f'{"AP@0.7":<25} {ap70:>12.4f}')
    print(f'{"AP@0.9":<25} {ap90:>12.4f}')
    print(f'{"AP@0.5:0.05:0.95":<25} {ap50_95:>12.4f}')
    print(f'{"bPQ":<25} {mean_bpq:>12.4f}')
    print(f'{"bMPQ":<25} {mean_bmpq:>12.4f}')
    print(f'{"mPQ":<25} {mean_mpq:>12.4f}')
    print(f'{"mMPQ":<25} {mean_mmpq:>12.4f}')
    print(f'{"F1 (centroid, r=12)":<25} {f12["f1"]:>12.4f}')
    print(f'{"Precision (centroid)":<25} {f12["precision"]:>12.4f}')
    print(f'{"Recall (centroid)":<25} {f12["recall"]:>12.4f}')
    print(f'{"Params (M)":<25} {params_m:>12.2f}')
    print(f'{"GFLOPs":<25} {gflops:>12.2f}')
    print(f'{"Inference Time (ms/img)":<25} {avg_ms:>12.2f}')
    print('=' * 37)

    print(f'\nImages evaluated: {len(results)}')
    print(f'Confidence threshold: {args.conf}')


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
        # Derive annotation width from non-empty labels, or fallback to a safe default.
        # When targets are empty the exact width doesn't affect computation.
        ann_width = next((lbl.shape[1] for lbl in labels_list if lbl.ndim == 2 and lbl.shape[1] > 0), 35)
        targets = _torch.zeros((0, 2 + ann_width), dtype=_torch.float32)  # batch_idx + cls + rays

    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': targets[:, 2:],
    }


if __name__ == '__main__':
    main()
