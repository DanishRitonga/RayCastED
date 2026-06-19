"""Polygon-based evaluation metrics — geometric comparison, no rasterization.

Computes AJI, bPQ, mPQ, Centroid L2, Ray L1 from polygon geometry.
IoU approximation: Shoelace area ratio (min/max) for star-convex polygons.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _shoelace_area(poly, n_rays):
    """Shoelace area of ray polygon in px^2."""
    cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    cx, cy = poly[0], poly[1]
    rays = poly[2 : 2 + n_rays]
    vx = cx + rays * cos
    vy = cy + rays * sin
    return 0.5 * abs(np.dot(vx, np.roll(vy, 1)) - np.dot(vy, np.roll(vx, 1)))


def _poly_iou(pred_poly, gt_poly, n_rays):
    """Shoelace area ratio as polygon IoU approximation."""
    pa = _shoelace_area(pred_poly, n_rays)
    ga = _shoelace_area(gt_poly, n_rays)
    if pa <= 0 or ga <= 0:
        return 0.0
    mn, mx = sorted([pa, ga])
    return mn / mx


def polygon_metrics(pred_polys, gt_polys, n_rays):
    """Per-image polygon metrics: centroid L2, ray L1, IoU per matched pair."""
    n_pred = len(pred_polys)
    n_gt = len(gt_polys)
    if n_pred == 0 or n_gt == 0:
        return [], []

    ious = np.array([[_poly_iou(pred_polys[pi], gt_polys[gi], n_rays) for gi in range(n_gt)] for pi in range(n_pred)])
    ri, ci = linear_sum_assignment(-ious)
    valid = ious[ri, ci] >= 0.1

    pairs = []
    for r, c, v in zip(ri, ci, valid):
        if not v:
            continue
        cdist = float(np.linalg.norm(pred_polys[r, :2] - gt_polys[c, :2]))
        rdiff = float(np.abs(pred_polys[r, 2 : 2 + n_rays] - gt_polys[c, 2 : 2 + n_rays]).mean())
        pairs.append((cdist, rdiff, float(ious[r, c]), int(gt_polys[c, -1]) if gt_polys.shape[1] > 2 + n_rays else 0))
    return pairs, ious


def compute_polygon_metrics_streaming(results, n_rays):
    """Streaming polygon metrics: AJI, bPQ, mPQ, Centroid L2, Ray L1."""
    all_centroid = []
    all_ray_l1 = []
    all_iou = []
    tp_global = 0
    fp_global = 0
    fn_global = 0
    nc = 5
    class_tp = [0] * nc
    class_gt = [0] * nc
    class_pred = [0] * nc

    for r in results:
        pred = r['pred_polys']
        gt_p = r['gt_polys']
        gt_c = r.get('gt_cls', np.array([]))
        pred_c = r.get('pred_cls', np.array([]))

        pairs, ious = polygon_metrics(pred, gt_p, n_rays)
        n_pred = len(pred)
        n_gt = len(gt_p)
        tp = len(pairs)

        for cdist, rdiff, iou, cls_id in pairs:
            all_centroid.append(cdist)
            all_ray_l1.append(rdiff)
            all_iou.append(iou)

        tp_global += tp
        fp_global += n_pred - tp
        fn_global += n_gt - tp

        for c in range(nc):
            class_gt[c] += int((gt_c == c).sum()) if len(gt_c) else 0
            class_pred[c] += int((pred_c == c).sum()) if len(pred_c) else 0

    tp = tp_global
    fp = fp_global
    fn = fn_global

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0
    sq = float(np.mean(all_iou)) if all_iou else 0
    pq = sq * dq

    centroid_l2 = float(np.mean(all_centroid)) if all_centroid else 0
    ray_l1 = float(np.mean(all_ray_l1)) if all_ray_l1 else 0
    mean_iou = float(np.mean(all_iou)) if all_iou else 0

    return {
        'bPQ': pq,
        'bSQ': sq,
        'bDQ': dq,
        'centroid_l2': centroid_l2,
        'ray_l1': ray_l1,
        'poly_iou': mean_iou,
        'n_matched': tp,
        'n_pred_total': tp + fp,
        'n_gt_total': tp + fn,
    }
