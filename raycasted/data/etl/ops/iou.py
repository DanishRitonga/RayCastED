"""Polygon YOLOv26 — Polar IoU Operations

Polar-IoU computation for star-convex polygons.
Includes both NumPy (ETL) and PyTorch (training) variants.

Formula:
    PolarIoU = Σ min(d_pred_i, d_gt_i)² / Σ max(d_pred_i, d_gt_i)²

This is a sector-area approximation from PolarMask.

Function shape contracts:
    polar_iou                    [N, 32], [N, 32] → [N]            (NumPy)
    polar_iou_torch              [..., 32], [..., 32] → [...]       (Tensor)
    polar_iou_pairwise_flat      [N_cand, N_gt, 32] × 2 → [N_cand, N_gt]  (NumPy)
    polar_iou_pairwise_flat_torch [N_cand, N_gt, 32] × 2 → [N_cand, N_gt] (Tensor)

The _flat suffix means the batch dimension has already been collapsed by the
caller's per-batch loop. The function never sees the B dimension.
"""

import numpy as np

from ..utils.constants import POLAR_IOU_EPS

# =============================================================================
# NUMPY VARIANTS (for ETL)
# =============================================================================


def polar_iou(
    d_pred: np.ndarray,
    d_gt: np.ndarray,
    eps: float = POLAR_IOU_EPS,
) -> float:
    """Compute element-wise Polar-IoU between two ray sets.

    Args:
        d_pred: Predicted ray distances, shape (N, 32) or (32,)
        d_gt: Ground truth ray distances, shape (N, 32) or (32,)
        eps: Small constant to prevent division by zero

    Returns:
        iou: Polar-IoU value(s), shape (N,) or scalar
    """
    d_pred = np.asarray(d_pred, dtype=np.float64)
    d_gt = np.asarray(d_gt, dtype=np.float64)

    pred_sq = d_pred ** 2
    gt_sq = d_gt ** 2

    intersection = np.sum(np.minimum(pred_sq, gt_sq), axis=-1)
    union = np.sum(np.maximum(pred_sq, gt_sq), axis=-1)

    return intersection / (union + eps)


def polar_iou_pairwise_flat(
    d_pred: np.ndarray,
    d_gt: np.ndarray,
    eps: float = POLAR_IOU_EPS,
) -> np.ndarray:
    """Compute pairwise Polar-IoU from pre-expanded flat inputs.

    The caller is responsible for expanding d_pred and d_gt to the
    pairwise shape before calling this function.

    Args:
        d_pred: Predicted ray distances, shape (N_cand, N_gt, 32) — pre-expanded
        d_gt: Ground truth ray distances, shape (N_cand, N_gt, 32) — pre-expanded
        eps: Small constant to prevent division by zero

    Returns:
        iou_matrix: IoU matrix, shape (N_cand, N_gt)
    """
    d_pred = np.asarray(d_pred, dtype=np.float64)
    d_gt = np.asarray(d_gt, dtype=np.float64)

    pred_sq = d_pred ** 2  # (N_cand, N_gt, 32)
    gt_sq = d_gt ** 2      # (N_cand, N_gt, 32)

    intersection = np.sum(np.minimum(pred_sq, gt_sq), axis=2)  # (N_cand, N_gt)
    union = np.sum(np.maximum(pred_sq, gt_sq), axis=2)         # (N_cand, N_gt)

    return intersection / (union + eps)


# =============================================================================
# PYTORCH VARIANTS (for training)
# =============================================================================


def polar_iou_torch(d_pred, d_gt, eps=POLAR_IOU_EPS):
    """PyTorch element-wise Polar-IoU for use in loss function.

    Args:
        d_pred: Tensor of shape (..., 32) — predicted rays
        d_gt: Tensor of shape (..., 32) — ground truth rays
        eps: Small constant to prevent division by zero

    Returns:
        iou: Tensor of shape (...) — Polar-IoU
    """
    import torch

    pred_sq = d_pred ** 2
    gt_sq = d_gt ** 2

    intersection = torch.sum(torch.minimum(pred_sq, gt_sq), dim=-1)
    union = torch.sum(torch.maximum(pred_sq, gt_sq), dim=-1)

    return intersection / (union + eps)


def polar_iou_pairwise_flat_torch(d_pred, d_gt, eps=POLAR_IOU_EPS):
    """PyTorch pairwise Polar-IoU from pre-expanded flat inputs.

    The caller is responsible for expanding d_pred and d_gt to the
    pairwise shape before calling this function. Used by PolygonAssigner
    inside its per-batch loop (see §11.2).

    Args:
        d_pred: Tensor of shape (N_cand, N_gt, 32) — pre-expanded
        d_gt: Tensor of shape (N_cand, N_gt, 32) — pre-expanded
        eps: Small constant to prevent division by zero

    Returns:
        iou_matrix: Tensor of shape (N_cand, N_gt)
    """
    import torch

    pred_sq = d_pred ** 2  # (N_cand, N_gt, 32)
    gt_sq = d_gt ** 2      # (N_cand, N_gt, 32)

    intersection = torch.sum(torch.minimum(pred_sq, gt_sq), dim=2)  # (N_cand, N_gt)
    union = torch.sum(torch.maximum(pred_sq, gt_sq), dim=2)         # (N_cand, N_gt)

    return intersection / (union + eps)
