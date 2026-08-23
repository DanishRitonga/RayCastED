"""RayCastED — Polar IoU Operations

Sector-area Polar-IoU for star-convex polygons.

Each sector is a triangle from the center to vertex[i] to vertex[i+1].
Sector area_i = 0.5 * sin(2π/n) * d_i * d_{i+1}

The min/max IoU logic operates on these triangular sector areas instead
of per-ray d², capturing adjacent-ray shape interactions that d² misses.

Function shape contracts:
    polar_iou                    [N, R], [N, R] → [N]            (NumPy)
    polar_iou_torch              [..., R], [..., R] → [...]       (Tensor)
    polar_iou_pairwise_flat      [N_cand, N_gt, R] × 2 → [N_cand, N_gt]  (NumPy)
    polar_iou_pairwise_flat_torch [N_cand, N_gt, R] × 2 → [N_cand, N_gt] (Tensor)

The _flat suffix means the batch dimension has already been collapsed by the
caller's per-batch loop. The function never sees the B dimension.
"""

import math

import numpy as np

from ..utils.constants import POLAR_IOU_EPS


def _sector_areas_np(d, n_rays):
    """Compute per-sector triangular areas: 0.5 * sin(2π/n) * d_i * d_{i+1}.

    Args:
        d: ray distances, shape (..., n_rays)
        n_rays: number of rays

    Returns:
        sector areas, shape (..., n_rays)
    """
    sin_theta = math.sin(2.0 * math.pi / n_rays)
    return 0.5 * sin_theta * d * np.roll(d, -1, axis=-1)


def _sector_areas_torch(d, n_rays):
    """Compute per-sector triangular areas: 0.5 * sin(2π/n) * d_i * d_{i+1}.

    Args:
        d: ray distances tensor, shape (..., n_rays)
        n_rays: number of rays

    Returns:
        sector areas tensor, shape (..., n_rays)
    """
    import torch

    sin_theta = math.sin(2.0 * math.pi / n_rays)
    return 0.5 * sin_theta * d * torch.roll(d, -1, dims=-1)


# =============================================================================
# NUMPY VARIANTS (for ETL)
# =============================================================================


def polar_iou(
    d_pred: np.ndarray,
    d_gt: np.ndarray,
    eps: float = POLAR_IOU_EPS,
) -> float:
    """Compute element-wise Polar-IoU between two ray sets using sector areas.

    Args:
        d_pred: Predicted ray distances, shape (N, R) or (R,)
        d_gt: Ground truth ray distances, shape (N, R) or (R,)
        eps: Small constant to prevent division by zero

    Returns:
        iou: Polar-IoU value(s), shape (N,) or scalar
    """
    d_pred = np.asarray(d_pred, dtype=np.float64)
    d_gt = np.asarray(d_gt, dtype=np.float64)

    n_rays = d_pred.shape[-1]
    pred_area = _sector_areas_np(d_pred, n_rays)
    gt_area = _sector_areas_np(d_gt, n_rays)

    intersection = np.sum(np.minimum(pred_area, gt_area), axis=-1)
    union = np.sum(np.maximum(pred_area, gt_area), axis=-1)

    return intersection / (union + eps)


def polar_iou_pairwise_flat(
    d_pred: np.ndarray,
    d_gt: np.ndarray,
    eps: float = POLAR_IOU_EPS,
) -> np.ndarray:
    """Compute pairwise Polar-IoU from pre-expanded flat inputs using sector areas.

    The caller is responsible for expanding d_pred and d_gt to the
    pairwise shape before calling this function.

    Args:
        d_pred: Predicted ray distances, shape (N_cand, N_gt, R) — pre-expanded
        d_gt: Ground truth ray distances, shape (N_cand, N_gt, R) — pre-expanded
        eps: Small constant to prevent division by zero

    Returns:
        iou_matrix: IoU matrix, shape (N_cand, N_gt)
    """
    d_pred = np.asarray(d_pred, dtype=np.float64)
    d_gt = np.asarray(d_gt, dtype=np.float64)

    n_rays = d_pred.shape[-1]
    pred_area = _sector_areas_np(d_pred, n_rays)
    gt_area = _sector_areas_np(d_gt, n_rays)

    intersection = np.sum(np.minimum(pred_area, gt_area), axis=2)  # (N_cand, N_gt)
    union = np.sum(np.maximum(pred_area, gt_area), axis=2)  # (N_cand, N_gt)

    return intersection / (union + eps)


# =============================================================================
# PYTORCH VARIANTS (for training)
# =============================================================================


def polar_iou_torch(d_pred, d_gt, eps=POLAR_IOU_EPS):
    """PyTorch element-wise Polar-IoU using sector areas.

    Args:
        d_pred: Tensor of shape (..., R) — predicted rays
        d_gt: Tensor of shape (..., R) — ground truth rays
        eps: Small constant to prevent division by zero

    Returns:
        iou: Tensor of shape (...) — Polar-IoU
    """
    import torch

    # Cast to float32 — POLAR_IOU_EPS (1e-7) underflows to 0 in float16,
    # causing NaN during AMP validation where model runs in half precision.
    d_pred = d_pred.float()
    d_gt = d_gt.float()

    n_rays = d_pred.shape[-1]
    pred_area = _sector_areas_torch(d_pred, n_rays)
    gt_area = _sector_areas_torch(d_gt, n_rays)

    intersection = torch.sum(torch.minimum(pred_area, gt_area), dim=-1)
    union = torch.sum(torch.maximum(pred_area, gt_area), dim=-1)

    return intersection / (union + eps)


def polar_iou_pairwise_flat_torch(d_pred, d_gt, eps=POLAR_IOU_EPS):
    """PyTorch pairwise Polar-IoU from pre-expanded flat inputs using sector areas.

    The caller is responsible for expanding d_pred and d_gt to the
    pairwise shape before calling this function. Used by RayCastAssigner
    inside its per-batch loop (see §11.2).

    Args:
        d_pred: Tensor of shape (N_cand, N_gt, R) — pre-expanded
        d_gt: Tensor of shape (N_cand, N_gt, R) — pre-expanded
        eps: Small constant to prevent division by zero

    Returns:
        iou_matrix: Tensor of shape (N_cand, N_gt)
    """
    import torch

    # Cast to float32 — same reason as polar_iou_torch.
    d_pred = d_pred.float()
    d_gt = d_gt.float()

    n_rays = d_pred.shape[-1]
    pred_area = _sector_areas_torch(d_pred, n_rays)
    gt_area = _sector_areas_torch(d_gt, n_rays)

    intersection = torch.sum(torch.minimum(pred_area, gt_area), dim=2)  # (N_cand, N_gt)
    union = torch.sum(torch.maximum(pred_area, gt_area), dim=2)  # (N_cand, N_gt)

    iou = intersection / (union + eps)
    # Guard against non-finite entries from degenerate (inf/NaN) rays.
    return torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
