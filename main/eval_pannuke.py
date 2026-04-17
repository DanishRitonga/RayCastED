"""PanNuke Fold3 Evaluation — Comprehensive metrics.

Runs inference on PanNuke test tiles and computes:
  AJI, mAP@0.5, mAP@0.75, mAP@0.5:0.95, PQ, SQ, DQ,
  Precision, Recall, F1, Params, FLOPs, Inference Time.

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
from torch.utils.data import DataLoader
from ultralytics.utils.torch_utils import model_info

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.model.metrics import (
    compute_aji,
    compute_pq,
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


def run_inference(model, dataloader, device, conf_threshold=0.25):
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
                else:
                    pred_poly = np.zeros((0, raycast_dim), dtype=np.float32)
                    pred_cls = np.array([], dtype=int)

                results.append(
                    {
                        'pred_polys': pred_poly,
                        'gt_polys': gt_poly,
                        'pred_cls': pred_cls,
                        'gt_cls': gt_cls.astype(int),
                        'imgsz': crop_size,
                    }
                )

    return results


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

    return np.where(union > 0, intersection / union, 0.0)


def compute_map_metrics(results, iou_thresholds=None):
    """Compute mAP at specified IoU thresholds using mask IoU.

    Uses rasterized polygon masks so both centroid and shape contribute to IoU.

    Returns:
        dict with mAP, precision, recall at each threshold.
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

            per_class_stats[cls_id]['n_gt_per_image'] = per_class_stats[cls_id].get('n_gt_per_image', [])
            per_class_stats[cls_id]['n_gt_per_image'].append(n_gt)

            for t in iou_thresholds:
                per_class_stats[cls_id][t]['n_gt'] += n_gt

            if n_pred == 0 or n_gt == 0:
                for t in iou_thresholds:
                    per_class_stats[cls_id][t]['conf'].extend([0.0] * n_pred)
                    per_class_stats[cls_id][t]['tp'].extend([False] * n_pred)
                    per_class_stats[cls_id][t]['fp'].extend([True] * n_pred)
                continue

            # Compute pairwise mask IoU
            imgsz = r['imgsz']
            gt_masks = polygons_to_masks(gt_polys, imgsz, imgsz)
            pred_masks = polygons_to_masks(pred_polys, imgsz, imgsz)
            iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)

            # Greedy matching per threshold
            for t in iou_thresholds:
                matched_gt = set()
                tp_list = []
                fp_list = []

                for pi in range(n_pred):
                    best_gt = -1
                    best_iou = t
                    for gi in range(n_gt):
                        if gi not in matched_gt and iou_matrix[pi, gi] >= best_iou:
                            best_iou = iou_matrix[pi, gi]
                            best_gt = gi
                    if best_gt >= 0:
                        tp_list.append(True)
                        fp_list.append(False)
                        matched_gt.add(best_gt)
                    else:
                        tp_list.append(False)
                        fp_list.append(True)

                per_class_stats[cls_id][t]['conf'].extend(pred_confs[:n_pred].tolist() if n_pred > 0 else [])
                per_class_stats[cls_id][t]['tp'].extend(tp_list)
                per_class_stats[cls_id][t]['fp'].extend(fp_list)

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

            # AP = area under precision-recall curve (11-point or all-point)
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
            'mAP': map_val,
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
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
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

    # --- Pixel-level metrics (AJI, PQ, SQ, DQ) ---
    print('Computing pixel-level metrics (AJI, PQ, SQ, DQ)...')
    aji_scores = []
    pq_scores = []
    sq_scores = []
    dq_scores = []
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
        pq, sq, dq = compute_pq(pred_masks, gt_masks)
        pq_scores.append(pq)
        sq_scores.append(sq)
        dq_scores.append(dq)

    mean_aji = np.mean(aji_scores)
    mean_pq = np.mean(pq_scores)
    mean_sq = np.mean(sq_scores)
    mean_dq = np.mean(dq_scores)

    # --- mAP metrics ---
    print('Computing mAP metrics...')
    map_results = compute_map_metrics(results)

    map50 = map_results.get(0.5, {}).get('mAP', 0.0)
    map75 = map_results.get(0.75, {}).get('mAP', 0.0)
    map50_95 = np.mean([map_results[t]['mAP'] for t in sorted(map_results.keys())])

    # Use 0.5 threshold for P/R/F1
    prec = map_results.get(0.5, {}).get('precision', 0.0)
    rec = map_results.get(0.5, {}).get('recall', 0.0)
    f1 = map_results.get(0.5, {}).get('F1', 0.0)

    # --- Model stats ---
    n_params = sum(p.numel() for p in model.parameters())
    params_m = n_params / 1e6

    # GFLOPs — model_info returns (n_layers, n_params, n_grads, gflops)
    try:
        _, _, _, gflops = model_info(model, imgsz=crop_size, verbose=False)
    except Exception:
        gflops = 0.0

    # --- Inference time ---
    print('Benchmarking inference time...')
    inf_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_simple_collate)
    avg_ms = benchmark_inference(model, inf_dl, device)

    # --- Print results ---
    print('\n' + '=' * 60)
    print('PanNuke Fold3 Evaluation Results')
    print('=' * 60)
    print(f'{"Metric":<25} {"Value":>12}')
    print('-' * 37)
    print(f'{"AJI":<25} {mean_aji:>12.4f}')
    print(f'{"mAP@0.5":<25} {map50:>12.4f}')
    print(f'{"mAP@0.75":<25} {map75:>12.4f}')
    print(f'{"mAP@0.5:0.95":<25} {map50_95:>12.4f}')
    print(f'{"PQ":<25} {mean_pq:>12.4f}')
    print(f'{"SQ":<25} {mean_sq:>12.4f}')
    print(f'{"DQ":<25} {mean_dq:>12.4f}')
    print(f'{"Precision":<25} {prec:>12.4f}')
    print(f'{"Recall":<25} {rec:>12.4f}')
    print(f'{"F1":<25} {f1:>12.4f}')
    print(f'{"Params (M)":<25} {params_m:>12.2f}')
    print(f'{"GFLOPs":<25} {gflops:>12.2f}')
    print(f'{"Inference Time (ms/img)":<25} {avg_ms:>12.2f}')
    print('=' * 37)

    # Per-class mAP@0.5 breakdown
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
        ann_width = next((l.shape[1] for l in labels_list if l.ndim == 2 and l.shape[1] > 0), 35)
        targets = _torch.zeros((0, 2 + ann_width), dtype=_torch.float32)  # batch_idx + cls + rays

    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': targets[:, 2:],
    }


if __name__ == '__main__':
    main()
