"""Polygon-based evaluation metrics — no rasterization, direct geometric comparison.

Computes metrics from raw polygon vertices: centroid L2, ray L1, and
polygon-to-polygon intersection/union via shoelace area + Cramer's rule.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _decode_vertices(polys, n_rays):
    """Reconstruct polygon vertices from ray representation."""
    cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    cx, cy = polys[:, 0], polys[:, 1]
    rays = polys[:, 2:2 + n_rays]
    vx = cx[:, None] + rays * cos[None, :]
    vy = cy[:, None] + rays * sin[None, :]
    return np.stack([vx, vy], axis=-1)


def _edge_intersection(px1, py1, px2, py2, rx0, ry0, rx_dir, ry_dir):
    """Ray-segment intersection: find t where ray hits edge [p1, p2]."""
    dx = px2 - px1
    dy = py2 - py1
    det = rx_dir * dy - ry_dir * dx
    ok = np.abs(det) > 1e-10
    t = np.full_like(det, np.inf)
    u = np.full_like(det, np.inf)
    np.divide((px1 - rx0) * dy - (py1 - ry0) * dx, det, where=ok, out=t)
    np.divide((px1 - rx0) * ry_dir - (py1 - ry0) * rx_dir, det, where=ok, out=u)
    valid = ok & (t > 0) & (u >= 0) & (u <= 1)
    t[~valid] = np.inf
    return t


def polygon_iou(pred_verts, gt_verts, n_rays):
    """Compute polygon IoU via ray-casting inside the convex hull of vertices."""
    cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    cx = np.mean(pred_verts[:, 0])
    cy = np.mean(pred_verts[:, 1])
    gc = np.mean(gt_verts[:, 0])

    pred_rays = np.full(n_rays, 0.0)
    gt_rays = np.full(n_rays, 0.0)
    for i in range(n_rays):
        rx, ry = cos[i], sin[i]
        tp = _edge_intersection(pred_verts[:-1,0], pred_verts[:-1,1], pred_verts[1:,0], pred_verts[1:,1], cx, cy, rx, ry)
        tg = _edge_intersection(gt_verts[:-1,0], gt_verts[:-1,1], gt_verts[1:,0], gt_verts[1:,1], gc, cy, rx, ry)
        pred_rays[i] = float(tp[tp < np.inf].min()) if (tp < np.inf).any() else 0
        gt_rays[i] = float(tg[tg < np.inf].min()) if (tg < np.inf).any() else 0

    if pred_rays.max() <= 0 or gt_rays.max() <= 0:
        return 0.0

    pred_area = 0.5 * np.sum(pred_rays[:-1] * pred_rays[1:] * np.sin(2*np.pi/n_rays))
    inter_rays = np.minimum(pred_rays, gt_rays)
    inter_area = 0.5 * np.sum(inter_rays[:-1] * inter_rays[1:] * np.sin(2*np.pi/n_rays))
    gt_area = 0.5 * np.sum(gt_rays[:-1] * gt_rays[1:] * np.sin(2*np.pi/n_rays))
    union = pred_area + gt_area - inter_area
    return inter_area / union if union > 0 else 0.0


def polygon_metrics(pred_polys, gt_polys, n_rays):
    """Compute polygon-level metrics: mean centroid L2, mean ray L1, mean poly IoU.

    Args:
        pred_polys: [N_pred, 2+n_rays]
        gt_polys: [N_gt, 2+n_rays]
        n_rays: number of rays

    Returns:
        dict with centroid_l2, ray_l1, poly_iou, matched_pairs
    """
    n_pred = len(pred_polys)
    n_gt = len(gt_polys)
    if n_pred == 0 or n_gt == 0:
        return {'centroid_l2': 0.0, 'ray_l1': 0.0, 'poly_iou': 0.0, 'matched_pairs': 0}

    dist = np.linalg.norm(pred_polys[:, :2][:, None] - gt_polys[:, :2][None, :], axis=2)
    ri, ci = linear_sum_assignment(dist)
    matched = dist[ri, ci] <= 12

    n_match = int(matched.sum())
    if n_match == 0:
        return {'centroid_l2': 0.0, 'ray_l1': 0.0, 'poly_iou': 0.0, 'matched_pairs': 0}

    centroid_l2_vals = []
    ray_l1_vals = []
    poly_iou_vals = []

    for r, c, m in zip(ri, ci, matched):
        if not m:
            continue
        # Centroid L2
        cdiff = np.sqrt(((gt_polys[c, :2] - pred_polys[r, :2]) ** 2).sum())
        centroid_l2_vals.append(float(cdiff))

        # Ray L1
        rdiff = np.abs(pred_polys[r, 2:2+n_rays] - gt_polys[c, 2:2+n_rays]).mean()
        ray_l1_vals.append(float(rdiff))

        # Polygon IoU via area
        pred_area = _polygon_area(pred_polys[r], n_rays)
        gt_area = _polygon_area(gt_polys[c], n_rays)
        min_area = min(pred_area, gt_area)
        max_area = max(pred_area, gt_area)
        area_iou = min_area / max_area if max_area > 0 else 0.0
        poly_iou_vals.append(float(area_iou))

    return {
        'centroid_l2': float(np.mean(centroid_l2_vals)),
        'ray_l1': float(np.mean(ray_l1_vals)),
        'poly_iou': float(np.mean(poly_iou_vals)),
        'matched_pairs': n_match,
    }


def _polygon_area(poly, n_rays):
    """Shoelace area from ray polygon in pixels."""
    cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    cx, cy = poly[0], poly[1]
    rays = poly[2:2+n_rays]
    vx = cx + rays * cos
    vy = cy + rays * sin
    return 0.5 * abs(np.sum(vx[:-1] * vy[1:] - vx[1:] * vy[:-1]))


def compute_polygon_metrics_streaming(results, n_rays):
    """Compute per-image polygon metrics and aggregate.

    Args:
        results: list of dicts with 'pred_polys', 'gt_polys', 'tissue' keys.
        n_rays: number of rays.

    Returns:
        dict with global means and per-tissue lists.
    """
    all_centroid = []
    all_ray_l1 = []
    all_poly_iou = []
    tissue_centroid = {}
    tissue_ray_l1 = {}
    tissue_poly_iou = {}
    n_matched = 0

    for r in results:
        tissue = int(r.get('tissue', 0))
        if tissue not in tissue_centroid:
            tissue_centroid[tissue] = []
            tissue_ray_l1[tissue] = []
            tissue_poly_iou[tissue] = []

        m = polygon_metrics(r['pred_polys'], r['gt_polys'], n_rays)
        if m['matched_pairs'] > 0:
            n_matched += m['matched_pairs']
            all_centroid.append(m['centroid_l2'])
            all_ray_l1.append(m['ray_l1'])
            all_poly_iou.append(m['poly_iou'])
            if tissue < 19:
                tissue_centroid[tissue].append(m['centroid_l2'])
                tissue_ray_l1[tissue].append(m['ray_l1'])
                tissue_poly_iou[tissue].append(m['poly_iou'])

    return {
        'centroid_l2': np.mean(all_centroid) if all_centroid else 0,
        'ray_l1': np.mean(all_ray_l1) if all_ray_l1 else 0,
        'poly_iou': np.mean(all_poly_iou) if all_poly_iou else 0,
        'n_matched': n_matched,
        'tissue_centroid': tissue_centroid,
        'tissue_ray_l1': tissue_ray_l1,
        'tissue_poly_iou': tissue_poly_iou,
    }
