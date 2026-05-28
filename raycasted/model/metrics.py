"""RayCastED — Instance Segmentation Metrics.

Pixel-level metrics for evaluating polygon detection as instance segmentation:
  AJI, PQ, SQ, DQ, bPQ (Shapely).

Uses polygon rasterization via Shapely + PIL, and Hungarian matching via scipy.
bPQ uses Shapely polygon IoU directly — no rasterization.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from raycasted.data.etl.ops.convert import raycast_to_polygon
from raycasted.data.etl.utils import constants as _const


def polygon_to_mask(cx: float, cy: float, rays: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Rasterize a single raycast polygon to a binary mask.

    Args:
        cx: Centroid x-coordinate (pixel space).
        cy: Centroid y-coordinate (pixel space).
        rays: Ray distances, shape (32,).
        img_h: Image height.
        img_w: Image width.

    Returns:
        Binary mask of shape (img_h, img_w), uint8.
    """
    poly = raycast_to_polygon(rays, cx, cy)
    if poly.is_empty or not poly.is_valid:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    coords = np.array(poly.exterior.coords, dtype=np.float64)
    # Clamp to image bounds
    coords[:, 0] = np.clip(coords[:, 0], 0, img_w - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, img_h - 1)

    img = Image.new('L', (img_w, img_h), 0)
    ImageDraw.Draw(img).polygon([tuple(c) for c in coords], fill=1)
    return np.array(img, dtype=np.uint8)


def _compute_mask_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise IoU matrix between two lists of binary masks.

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

    # Vectorize: stack masks and compute intersection/union
    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)  # (N_pred, H*W)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)  # (N_gt, H*W)

    intersection = pred_stack @ gt_stack.T  # (N_pred, N_gt)
    pred_area = pred_stack.sum(axis=1, keepdims=True)  # (N_pred, 1)
    gt_area = gt_stack.sum(axis=1, keepdims=True)  # (N_gt, 1)
    union = pred_area + gt_area.T - intersection

    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)
    return iou


def compute_aji(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Aggregated Jaccard Index for one image.

    AJI = sum_intersections / (sum_unions + unmatched_pred_area + unmatched_gt_area)

    Uses Hungarian matching at iou_threshold to find optimal pred↔gt pairs.

    Args:
        pred_masks: List of [H, W] uint8 binary masks (predictions).
        gt_masks: List of [H, W] uint8 binary masks (ground truth).
        iou_threshold: Minimum IoU for valid match.

    Returns:
        AJI score in [0, 1]. Returns 0.0 if no GT masks.
    """
    if len(gt_masks) == 0:
        return 0.0
    if len(pred_masks) == 0:
        return 0.0

    iou_matrix = _compute_mask_iou_matrix(pred_masks, gt_masks)  # (N_pred, N_gt)

    # Hungarian matching (maximize IoU → minimize -IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Filter to valid matches above threshold
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold
    match_pred = set(row_ind[valid].tolist())
    match_gt = set(col_ind[valid].tolist())

    # Sum intersection and union for matched pairs
    total_intersection = 0.0
    total_union = 0.0
    for r, c in zip(row_ind[valid], col_ind[valid]):
        p = pred_masks[r].astype(np.float64)
        g = gt_masks[c].astype(np.float64)
        total_intersection += (p * g).sum()
        total_union += (p + g - p * g).sum()

    # Add unmatched predictions (area goes into union only)
    for i in range(len(pred_masks)):
        if i not in match_pred:
            total_union += pred_masks[i].astype(np.float64).sum()

    # Add unmatched GT (area goes into union only)
    for j in range(len(gt_masks)):
        if j not in match_gt:
            total_union += gt_masks[j].astype(np.float64).sum()

    if total_union == 0:
        return 0.0

    return total_intersection / total_union


def compute_pq(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Compute Panoptic Quality (PQ), Segmentation Quality (SQ), Detection Quality (DQ).

    PQ = SQ × DQ
    DQ = TP / (TP + 0.5*FP + 0.5*FN)  [equivalent to F1]
    SQ = mean IoU of matched pairs

    Args:
        pred_masks: List of [H, W] uint8 binary masks (predictions).
        gt_masks: List of [H, W] uint8 binary masks (ground truth).
        iou_threshold: Minimum IoU for valid match.

    Returns:
        (PQ, SQ, DQ) tuple. Returns (0.0, 0.0, 0.0) if no GT masks.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0:
        return 0.0, 0.0, 0.0

    if n_pred == 0:
        return 0.0, 0.0, 0.0

    iou_matrix = _compute_mask_iou_matrix(pred_masks, gt_masks)

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold

    tp = valid.sum()
    fp = n_pred - tp
    fn = n_gt - tp

    # DQ (F1)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0

    # SQ (mean IoU of matched pairs)
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0

    pq = sq * dq
    return float(pq), float(sq), float(dq)


def resolve_mask_overlaps(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Resolve overlapping masks using largest-first priority.

    Processes masks from largest area to smallest. Each pixel is assigned to
    the largest mask that covers it, removing it from smaller overlapping masks.
    This matches LSP-DETR's post-processing for fair comparison.

    Args:
        masks: List of [H, W] uint8 binary masks.

    Returns:
        List of [H, W] uint8 binary masks with overlaps resolved.
    """
    if len(masks) == 0:
        return masks

    n = len(masks)
    h, w = masks[0].shape
    stack = np.stack(masks)  # (N, H, W)
    areas = stack.sum(axis=(1, 2))
    sorted_indices = np.argsort(-areas)  # largest first

    occupied = np.zeros((h, w), dtype=bool)
    resolved = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]

    for idx in sorted_indices:
        resolved[idx] = stack[idx] & ~occupied
        occupied |= resolved[idx].astype(bool)

    return resolved


def polygons_to_masks(
    detections: np.ndarray,
    img_h: int,
    img_w: int,
) -> list[np.ndarray]:
    """Convert raycast polygon detections to binary masks.

    Args:
        detections: [N, 2+n_rays] array where each row is [cx, cy, d_1..d_n].
            Or [N, 4+n_rays] with trailing score/cls columns (ignored).
        img_h: Image height.
        img_w: Image width.

    Returns:
        List of [img_h, img_w] uint8 binary masks.
    """
    masks = []
    for det in detections:
        cx, cy = det[0], det[1]
        rays = det[2:]  # all remaining columns are ray distances
        masks.append(polygon_to_mask(cx, cy, rays, img_h, img_w))
    return masks


def raycast_batch_to_shapely(
    bboxes: np.ndarray,
    n_rays: int | None = None,
) -> list[ShapelyPolygon]:
    """Convert batch of raycast bboxes to Shapely Polygons.

    Args:
        bboxes: [N, 2+n_rays] array where each row is [cx, cy, d_1..d_n].
        n_rays: Override ray count (default: use _const.N_RAYS).

    Returns:
        List of Shapely Polygon objects (empty list if no bboxes).
    """
    if bboxes.shape[0] == 0:
        return []
    nr = n_rays or _const.N_RAYS
    polys = []
    for i in range(bboxes.shape[0]):
        cx, cy = float(bboxes[i, 0]), float(bboxes[i, 1])
        rays = bboxes[i, 2 : 2 + nr]
        poly = raycast_to_polygon(rays, cx, cy)
        if not poly.is_valid:
            poly = make_valid(poly)
        if isinstance(poly, ShapelyPolygon) and not poly.is_empty:
            polys.append(poly)
        else:
            polys.append(ShapelyPolygon())
    return polys


def compute_bpq_shapely(
    pred_polys: list[ShapelyPolygon],
    gt_polys: list[ShapelyPolygon],
    iou_threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    """Compute binary Panoptic Quality using Shapely polygon IoU (no rasterization).

    bPQ = DQ * SQ where:
      DQ = TP / (TP + 0.5*FP + 0.5*FN)   (detection quality, equivalent to F1)
      SQ = mean IoU of matched pairs       (segmentation quality)

    Uses Hungarian matching (scipy) on -IoU, then filters by iou_threshold.

    Args:
        pred_polys: List of Shapely Polygon predictions.
        gt_polys: List of Shapely Polygon ground truths.
        iou_threshold: Minimum IoU for valid match (default 0.5).
        eps: Small value to prevent division by zero.

    Returns:
        (bPQ, bSQ, bDQ) tuple. Returns (0.0, 0.0, 0.0) if no GT.
    """
    n_pred = len(pred_polys)
    n_gt = len(gt_polys)

    if n_gt == 0:
        return 0.0, 0.0, 0.0
    if n_pred == 0:
        return 0.0, 0.0, 0.0

    # Filter out empty polygons
    pred_valid = [(i, p) for i, p in enumerate(pred_polys) if not p.is_empty and p.area > eps]
    gt_valid = [(i, p) for i, p in enumerate(gt_polys) if not p.is_empty and p.area > eps]

    if len(gt_valid) == 0 or len(pred_valid) == 0:
        dq = 0.0
        sq = 0.0
        return dq * sq, sq, dq

    # Compute pairwise IoU matrix [n_pred_valid, n_gt_valid]
    n_p = len(pred_valid)
    n_g = len(gt_valid)
    iou_matrix = np.zeros((n_p, n_g), dtype=np.float64)

    for pi, (p_idx, p_poly) in enumerate(pred_valid):
        for gi, (g_idx, g_poly) in enumerate(gt_valid):
            try:
                inter = p_poly.intersection(g_poly)
                if inter.is_empty:
                    continue
                union = p_poly.area + g_poly.area - inter.area
                if union > eps:
                    iou_matrix[pi, gi] = inter.area / union
            except Exception:
                continue

    # Hungarian matching (maximize IoU → minimize -IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]

    # Filter by threshold
    valid = matched_ious > iou_threshold
    tp = int(valid.sum())
    fp = n_p - tp
    fn = n_g - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + eps)
    sq = float(matched_ious[valid].mean()) if tp > 0 else 0.0
    pq = dq * sq

    return float(pq), float(sq), float(dq)
