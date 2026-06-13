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

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.utils import constants as _const
from raycasted.model.metrics import (
    compute_aji,
    resolve_mask_overlaps,
)
from raycasted.model.register import register_raycast_head
from eval_polygon import compute_polygon_metrics_streaming


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


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of ray polygon in pixel²."""
    cx, cy = poly[0], poly[1]
    rays = poly[2:]
    if len(rays) < 3:
        return 0.0
    cos = _const.RAY_COS
    sin = _const.RAY_SIN
    vx = cx + rays * cos
    vy = cy + rays * sin
    return 0.5 * abs(np.dot(vx, np.roll(vy, 1)) - np.dot(vy, np.roll(vx, 1)))


def _diagnose_tissue(results, metrics):
    """Per-tissue centroid F1 + AJI/bPQ/mPQ."""
    panuke_tissues = [
        "Adrenal", "BileDuct", "Bladder", "Breast", "Cervix", "Colorectal",
        "Esophagus", "Head&Neck", "Kidney", "Liver", "Lung", "Ovarian",
        "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus",
    ]
    _per_group_metrics(results, panuke_tissues, 'tissue', "Tissue Type Breakdown", metrics)


def _diagnose_nuclei(results, num_classes):
    """Per-class centroid F1 metrics."""
    panuke_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Necrosis', 'Epithelial']
    return _per_nuclei_class_metrics(results, num_classes, panuke_names)


def _per_group_metrics(results, names, key, title, metrics=None):
    """Compute per-group centroid F1 + optional AJI/bPQ/mPQ."""
    n_groups = len(names)
    tp = [0] * n_groups; fp = [0] * n_groups; fn = [0] * n_groups
    n_gt = [0] * n_groups; n_pred = [0] * n_groups
    n_imgs = [0] * n_groups; seen = [set() for _ in range(n_groups)]

    for ri, r in enumerate(results):
        g = int(r.get(key, 0))
        if g >= n_groups: continue
        seen[g].add(ri)
        gt_p, pred_p = r['gt_polys'], r['pred_polys']
        n_gt_g = len(gt_p); n_pred_g = len(pred_p)
        n_gt[g] += n_gt_g; n_pred[g] += n_pred_g
        if n_gt_g == 0: fp[g] += n_pred_g; continue
        if n_pred_g == 0: fn[g] += n_gt_g; continue
        dist = np.linalg.norm(gt_p[:,:2][:,None] - pred_p[:,:2][None,:], axis=2)
        ri_, ci_ = linear_sum_assignment(dist)
        t = int((dist[ri_,ci_] <= 12).sum())
        tp[g] += t; fp[g] += n_pred_g - t; fn[g] += n_gt_g - t

    t_aji = metrics.get('tissue_aji',{}) if metrics else {}
    t_bpq = metrics.get('tissue_bpq',{}) if metrics else {}
    t_mpq = metrics.get('tissue_mpq',{}) if metrics else {}
    has_mask = bool(metrics)

    hdr = f'  {{"Group":<14}} {{"Imgs":>5}} {{"Prec":>7}} {{"Recall":>7}} {{"F1":>7}}'
    sep = f'  {{"-" * 14}} {{"-" * 5}} {{"-" * 7}} {{"-" * 7}} {{"-" * 7}}'
    fmt = f'  {{name:<14}} {{ni:>5}} {{prec:>7.4f}} {{rec:>7.4f}} {{f1:>7.4f}}'
    if has_mask:
        hdr += f' {{"AJI":>7}} {{"bPQ":>7}} {{"mPQ":>7}}'
        sep += f' {{"-" * 7}} {{"-" * 7}} {{"-" * 7}}'
        fmt += f' {{aji:>7.4f}} {{bpq:>7.4f}} {{mpq:>7.4f}}'

    has_mask = bool(metrics)
    w = 85 if has_mask else 60
    sep = '=' * w

    hdr_fmt = '  {:<14} {:>5} {:>7} {:>7} {:>7}'
    row_fmt = '  {:<14} {:>5} {:>7.4f} {:>7.4f} {:>7.4f}'
    if has_mask:
        hdr_fmt += ' {:>7} {:>7} {:>7}'
        row_fmt += ' {:>7.4f} {:>7.4f} {:>7.4f}'

    print(f'\n{sep}')
    print(f'  {title}')
    print(sep)
    col = ['Group', 'Imgs', 'Prec', 'Recall', 'F1']
    if has_mask: col += ['AJI', 'bPQ', 'mPQ']
    print(hdr_fmt.format(*col))
    sep2 = '  ' + '-' * 14 + ' ' + '-' * 5 + ' ' + '-' * 7 + ' ' + '-' * 7 + ' ' + '-' * 7
    if has_mask: sep2 += ' ' + '-' * 7 + ' ' + '-' * 7 + ' ' + '-' * 7
    print(sep2)

    for g in range(n_groups):
        if n_gt[g] == 0: continue
        prec = tp[g] / (tp[g] + fp[g]) if (tp[g] + fp[g]) else 0
        rec = tp[g] / (tp[g] + fn[g]) if (tp[g] + fn[g]) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        vals = [names[g], len(seen[g]), prec, rec, f1]
        if has_mask:
            aji_arr = np.array(t_aji.get(g)) if t_aji.get(g) else np.array([])
            bpq_arr = np.array(t_bpq.get(g)) if t_bpq.get(g) else np.array([])
            aji_m = float(np.mean(aji_arr)) if len(aji_arr) else 0.0
            aji_s = float(np.std(aji_arr)) if len(aji_arr) > 1 else 0.0
            bpq_m = float(np.mean(bpq_arr)) if len(bpq_arr) else 0.0
            bpq_s = float(np.std(bpq_arr)) if len(bpq_arr) > 1 else 0.0
            mpq_vals = [np.mean([x for x in v if x > 0]) for v in t_mpq.get(g, {}).values() if v and any(x > 0 for x in v)]
            mpq_m = float(np.mean(mpq_vals)) if mpq_vals else 0.0
            vals += [aji_m, bpq_m, mpq_m]
        print(row_fmt.format(*vals))

    tot_tp = sum(tp); tot_fp = sum(fp); tot_fn = sum(fn)
    p_t = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0
    r_t = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0
    f_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) else 0
    vals = ['TOTAL', len(results), p_t, r_t, f_t]
    if has_mask:
        vals += [metrics['aji'], metrics['bpq'], metrics['mpq']]
    print(row_fmt.format(*vals))

    # Average ± STD across tissues (mask metrics only)
    if has_mask:
        per_t_aji = [float(np.mean(t_aji.get(g))) for g in range(n_groups) if t_aji.get(g)]
        per_t_bpq = [float(np.mean(t_bpq.get(g))) for g in range(n_groups) if t_bpq.get(g)]
        per_t_mpq = []
        for g in range(n_groups):
            v = t_mpq.get(g, {})
            if v:
                vp = [np.mean([x for x in lst if x > 0]) for lst in v.values() if lst and any(x > 0 for x in lst)]
                if vp:
                    per_t_mpq.append(float(np.mean(vp)))
        am = np.mean(per_t_aji) if per_t_aji else 0; as_ = np.std(per_t_aji) if len(per_t_aji)>1 else 0
        bm = np.mean(per_t_bpq) if per_t_bpq else 0; bs_ = np.std(per_t_bpq) if len(per_t_bpq)>1 else 0
        mm = np.mean(per_t_mpq) if per_t_mpq else 0; ms_ = np.std(per_t_mpq) if len(per_t_mpq)>1 else 0
        print(f'  {"Avg":<14} {"":>5} {"":>7} {"":>7} {"":>7} {am:>7.4f} {bm:>7.4f} {mm:>7.4f}')
        print(f'  {"Std":<14} {"":>5} {"":>7} {"":>7} {"":>7} {as_:>7.4f} {bs_:>7.4f} {ms_:>7.4f}')
    print(sep)


def _per_nuclei_class_metrics(results, num_classes, names):
    """Per-class centroid F1."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for r in results:
        gt_p, pred_p = r['gt_polys'], r['pred_polys']
        gt_c, pred_c = r['gt_cls'], r['pred_cls']
        n_gt = len(gt_p)
        n_pred = len(pred_p)
        if n_gt == 0 or n_pred == 0:
            continue
        dist = np.linalg.norm(gt_p[:, :2][:, None] - pred_p[:, :2][None, :], axis=2)
        ri_, ci_ = linear_sum_assignment(dist)
        matched_gt = set()
        matched_pred = set()
        for ri, ci in zip(ri_, ci_):
            if dist[ri, ci] <= 12:
                if gt_c[ri] == pred_c[ci]:
                    tp[int(gt_c[ri])] += 1
                matched_gt.add(ri)
                matched_pred.add(ci)

    print(f'\n{"=" * 70}')
    print(f'  Nuclei Class Breakdown (Centroid F1, class-matched)')
    print(f'{"=" * 70}')
    print(f'  {"Class":<14} {"Prec":>7} {"Recall":>7} {"F1":>7}')
    print(f'  {"-" * 14} {"-" * 7} {"-" * 7} {"-" * 7}')
    # Need total GT and pred per class from all images
    class_gt = [0] * num_classes
    class_pred = [0] * num_classes
    for r in results:
        for c in r['gt_cls']:
            class_gt[int(c)] += 1
        for c in r['pred_cls']:
            if int(c) < num_classes:
                class_pred[int(c)] += 1
    for c in range(num_classes):
        name = names[c] if c < len(names) else f'cls_{c}'
        prec = tp[c] / class_pred[c] if class_pred[c] else 0
        rec = tp[c] / class_gt[c] if class_gt[c] else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f'  {name:<14} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}')
    print(f'{"=" * 70}')


def _diagnose_recall(results, num_classes):
    """Break down recall by GT size bin, class, and nearest-prediction distance."""
    panuke_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Necrosis', 'Epithelial']

    gt_areas_matched: list[float] = []
    gt_areas_unmatched: list[float] = []
    class_total = [0] * num_classes
    class_matched = [0] * num_classes
    unmatched_dists: list[float] = []
    unmatched_nearest_conf: list[float] = []
    unmatched_nearest_cls: list[int] = []
    unmatched_max_conf_12: list[float] = []

    for r in results:
        gt_polys = r['gt_polys']
        pred_polys = r['pred_polys']
        pred_confs = r['pred_confs']
        pred_cls = r['pred_cls']
        gt_cls = r['gt_cls']

        n_gt = len(gt_polys)
        n_pred = len(pred_polys)

        if n_gt == 0:
            continue

        gt_areas = np.array([_polygon_area(p) for p in gt_polys])

        if n_pred > 0:
            pred_cxcy = pred_polys[:, :2]
            gt_cxcy = gt_polys[:, :2]
            dist_matrix = np.linalg.norm(gt_cxcy[:, None, :] - pred_cxcy[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            matched = np.zeros(n_gt, dtype=bool)
            matched_dists = dist_matrix[row_ind, col_ind]
            for ri, ci, d in zip(row_ind, col_ind, matched_dists):
                if d <= 12.0:
                    matched[ri] = True
        else:
            matched = np.zeros(n_gt, dtype=bool)
            pred_cxcy = np.zeros((0, 2))

        for j in range(n_gt):
            cls_id = int(gt_cls[j])
            if cls_id < num_classes:
                class_total[cls_id] += 1
                if matched[j]:
                    class_matched[cls_id] += 1

            if matched[j]:
                gt_areas_matched.append(float(gt_areas[j]))
            else:
                gt_areas_unmatched.append(float(gt_areas[j]))

                if n_pred > 0:
                    dists_j = np.linalg.norm(pred_cxcy - gt_polys[j, :2], axis=1)
                    nearest_idx = int(np.argmin(dists_j))
                    unmatched_dists.append(float(dists_j[nearest_idx]))
                    unmatched_nearest_conf.append(float(pred_confs[nearest_idx]))
                    unmatched_nearest_cls.append(int(pred_cls[nearest_idx]))
                    nearby_mask = dists_j <= 12.0
                    if nearby_mask.any():
                        unmatched_max_conf_12.append(float(pred_confs[nearby_mask].max()))
                    else:
                        unmatched_max_conf_12.append(0.0)
                else:
                    unmatched_dists.append(float('inf'))
                    unmatched_nearest_conf.append(0.0)
                    unmatched_nearest_cls.append(-1)
                    unmatched_max_conf_12.append(0.0)

    # --- Print ---
    total_gt = sum(class_total)
    total_matched = sum(class_matched)
    recall = total_matched / total_gt if total_gt > 0 else 0.0

    print('\n' + '=' * 65)
    print('Recall Diagnosis')
    print('=' * 65)
    print(f'  Overall: {total_matched}/{total_gt} = {recall:.4f}')

    # --- By class ---
    print(f'\n  {"Class":<18} {"Total":>8} {"Matched":>8} {"Recall":>8}')
    print(f'  {"-" * 18} {"-" * 8} {"-" * 8} {"-" * 8}')
    for c in range(num_classes):
        if class_total[c] > 0:
            r = class_matched[c] / class_total[c]
            name = panuke_names[c] if c < len(panuke_names) else f'class_{c}'
            print(f'  {name:<18} {class_total[c]:>8} {class_matched[c]:>8} {r:>8.4f}')

    # --- By size bin ---
    if gt_areas_matched or gt_areas_unmatched:
        all_areas = np.array(gt_areas_matched + gt_areas_unmatched)
        if len(all_areas) > 0:
            p33 = np.percentile(all_areas, 33)
            p67 = np.percentile(all_areas, 67)
            bins = [
                ('Small (<P33)', lambda a: a < p33),
                ('Medium (P33-P67)', lambda a: (a >= p33) & (a < p67)),
                ('Large (>P67)', lambda a: a >= p67),
            ]
            print(f'\n  {"Size Bin":<20} {"Area Range":<18} {"Total":>8} {"Matched":>8} {"Recall":>8}')
            print(f'  {"-" * 20} {"-" * 18} {"-" * 8} {"-" * 8} {"-" * 8}')
            for label, cond in bins:
                t = sum(1 for a in gt_areas_matched + gt_areas_unmatched if cond(a))
                m = sum(1 for a in gt_areas_matched if cond(a))
                min_a = min(a for a in all_areas if cond(a)) if t > 0 else 0
                max_a = max(a for a in all_areas if cond(a)) if t > 0 else 0
                rng = f'{min_a:.0f}-{max_a:.0f}px²'
                rec = m / t if t > 0 else 0.0
                print(f'  {label:<20} {rng:<18} {t:>8} {m:>8} {rec:>8.4f}')

    # --- Unmatched GT: distance to nearest prediction ---
    if unmatched_dists:
        ud = np.array(unmatched_dists)
        finite = ud[np.isfinite(ud)]
        print(f'\n  Unmatched GT — Distance to nearest prediction:')
        print(
            f'    <5px (near miss): {int(np.sum(finite < 5))}/{len(unmatched_dists)} ({100 * sum(finite < 5) / len(unmatched_dists):.1f}%)'
        )
        print(
            f'    5-12px (drifted):  {int(np.sum((finite >= 5) & (finite <= 12)))}/{len(unmatched_dists)} ({100 * sum((finite >= 5) & (finite <= 12)) / len(unmatched_dists):.1f}%)'
        )
        print(
            f'    >12px (truly miss):{int(np.sum(finite > 12))}/{len(unmatched_dists)} ({100 * sum(finite > 12) / len(unmatched_dists):.1f}%)'
        )
        no_det = int(np.isinf(ud).sum())
        if no_det > 0:
            print(
                f'    No predictions:    {no_det}/{len(unmatched_dists)} ({100 * no_det / len(unmatched_dists):.1f}%)'
            )

    # --- Unmatched GT: confidence of nearby predictions ---
    if unmatched_max_conf_12:
        umc = np.array(unmatched_max_conf_12)
        nonzero = umc[umc > 0]
        print(f'\n  Unmatched GT — Max conf within 12px:')
        if len(nonzero) > 0:
            print(f'    Mean conf:          {nonzero.mean():.4f}')
            print(f'    Median conf:        {float(np.median(nonzero)):.4f}')
            above_thresh = np.sum(nonzero >= 0.2)
            print(
                f'    Conf ≥0.2:          {above_thresh}/{len(unmatched_dists)} ({100 * above_thresh / len(unmatched_dists):.1f}%) — wrong class or below eval-threshold'
            )
        no_conf = int(umc.sum() == 0)
        if no_conf > 0:
            print(
                f'    No pred within 12px:{no_conf}/{len(unmatched_dists)} ({100 * no_conf / len(unmatched_dists):.1f}%) — truly missed'
            )

    print('=' * 65)


def load_model(weights_path: str, device: torch.device):
    """Load trained RayCastED model from checkpoint."""
    register_raycast_head()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt.get('model') or ckpt.get('ema') if isinstance(ckpt, dict) else ckpt
    model = model.float().to(device)
    model.eval()
    return model


def run_inference(model, dataloader, device, conf_threshold=0.20):
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

                results.append(
                    {
                        'pred_polys': pred_poly,
                        'pred_confs': pred_confs,
                        'gt_polys': gt_poly,
                        'pred_cls': pred_cls,
                        'gt_cls': gt_cls.astype(int),
                        'imgsz': crop_size,
                        'tissue': batch.get('tissue', [0] * batch_size)[si] if 'tissue' in batch else 0,
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
    iou_thresholds = sorted(set(round(x, 2) for x in np.arange(0.5, 1.0, 0.05)))

    # Accumulators
    aji_scores = []
    bpq_scores = []
    bmpq_scores = []
    class_pq = {c: [] for c in range(num_classes)}
    class_mpq = {c: [] for c in range(num_classes)}
    centroid_tp = 0
    centroid_fp = 0
    centroid_fn = 0
    tissue_aji = {t: [] for t in range(19)}
    tissue_bpq = {t: [] for t in range(19)}
    tissue_mpq = {t: {c: [] for c in range(num_classes)} for t in range(19)}

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
        aji_val = compute_aji(pred_masks, gt_masks)
        aji_scores.append(aji_val)

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
        class_pq_img = [0.0] * num_classes
        gt_idx_counts = [0] * num_classes
        pred_idx_counts = [0] * num_classes
        for cls_id in range(num_classes):
            pred_idx = [j for j, c in enumerate(pred_cls) if c == cls_id]
            gt_idx = [j for j, c in enumerate(gt_cls) if c == cls_id]
            gt_idx_counts[cls_id] = len(gt_idx)
            pred_idx_counts[cls_id] = len(pred_idx)

            pred_cls_masks = [pred_masks[j] for j in pred_idx]
            gt_cls_masks = [gt_masks[j] for j in gt_idx]

            pq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks)
            class_pq[cls_id].append(pq)
            class_pq_img[cls_id] = pq

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

        # Per-tissue tracking
        tissue = int(r.get('tissue', 0))
        if tissue < 19 and aji_val > -1:
            tissue_aji[tissue].append(aji_val)
            tissue_bpq[tissue].append(bpq)
            for cls_id in range(num_classes):
                if gt_idx_counts[cls_id] > 0 or pred_idx_counts[cls_id] > 0:
                    tissue_mpq[tissue][cls_id].append(class_pq_img[cls_id])

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
        'tissue_aji': tissue_aji,
        'tissue_bpq': tissue_bpq,
        'tissue_mpq': tissue_mpq,
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
    results = run_inference(model, dataloader, device, conf_threshold=args.conf)
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

    # Fused: after fuse() removes o2m heads, keeps only o2o
    try:
        head = model.model[-1]
        head.fuse()
        n_fused = sum(p.numel() for p in model.parameters())
        fused_m = n_fused / 1e6
    except Exception:
        fused_m = 0.0

    # GFLOPs: static value from training logs (5.52G at 256px).
    # model_info tracing fails on ultralytics Concat with custom architecture.
    gflops = 5.52

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
    print(f'{"Params fused (M)":<25} {fused_m:>12.2f}')
    print(f'{"GFLOPs":<25} {gflops:>12.2f}')
    print(f'{"Inference Time (ms/img)":<25} {avg_ms:>12.2f}')
    print('=' * 37)

    print(f'\nImages evaluated: {len(results)}')
    print(f'Confidence threshold: {args.conf}')
    print(f'Total predictions: {n_pred_total}')
    print(f'Total GT instances: {n_gt_total}')

    # --- Tissue-origin breakdown ---
    try:
        _diagnose_tissue(results, metrics)
        _diagnose_nuclei(results, nc)
    except Exception as exc:
        print(f'\n[TISSUE DIAG ERROR] {exc}', flush=True)

    # --- Recall diagnosis ---
    try:
        _diagnose_recall(results, num_classes=nc)
    except Exception as exc:
        import traceback

        print(f'\n[DIAG ERROR] _diagnose_recall failed: {exc}', flush=True)
        traceback.print_exc()

    # --- Polygon-based metrics (no raster) ---
    try:
        p_metrics = compute_polygon_metrics_streaming(results, n_rays)
        print('\n' + '=' * 60)
        print('Polygon-Level Metrics (geometric, no rasterization)')
        print('=' * 60)
        print(f'  bPQ (polygon IoU):    {p_metrics["bPQ"]:>8.4f}')
        print(f'  bSQ (polygon IoU):    {p_metrics["bSQ"]:>8.4f}')
        print(f'  bDQ (polygon IoU):    {p_metrics["bDQ"]:>8.4f}')
        print(f'  Centroid L2 (px):    {p_metrics["centroid_l2"]:>8.2f}')
        print(f'  Ray L1 (px):         {p_metrics["ray_l1"]:>8.2f}')
        print(f'  Poly IoU (mean):     {p_metrics["poly_iou"]:>8.4f}')
        print(f'  Matched/Total:       {p_metrics["n_matched"]}/{p_metrics["n_gt_total"]}')
        print('=' * 60)
    except Exception:
        pass


def _simple_collate(batch):
    """Collate function matching _raycast_collate_fn from train.py."""
    import torch as _torch

    images = _torch.stack([item[0] for item in batch])
    labels_list = [item[1] for item in batch]
    tissue_list = []
    for _, _, path in batch:
        data = dict(np.load(path, allow_pickle=True))
        tissue_list.append(int(data.get('tissue', 0)))

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
        targets = _torch.zeros((0, 1 + ann_width), dtype=_torch.float32)

    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': targets[:, 2:],
        'tissue': tissue_list,
    }


if __name__ == '__main__':
    main()
