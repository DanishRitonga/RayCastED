"""Polygon YOLOv26 — Loss Operations

Angular smoothness regularization for polygon training.
Includes both NumPy (validation/debugging) and PyTorch (training) variants.

Note: decode_pred_xy is NOT here. It is a method of PolygonDetectionLoss
in ultralytics/utils/loss.py — it requires anchor grid knowledge and
must not be imported in ETL environments.
"""

import numpy as np


def angular_smoothness_loss(rays: np.ndarray) -> float:
    """Compute circular first-difference penalty on ray distances.

    Encourages smooth polygon boundaries by penalizing large differences
    between consecutive rays. The difference between ray 31 and ray 0 is
    included (circular).

    Formula:
        L_smooth = (1/32) × Σ|d_{i+1} - d_i|  for i in [0, 31]
        where d_{32} = d_0

    Args:
        rays: Ray distances, shape (32,) or (N, 32)

    Returns:
        loss: Smoothness penalty, scalar or shape (N,)
    """
    rays = np.asarray(rays, dtype=np.float64)

    if rays.ndim == 1:
        diff = np.diff(rays, append=rays[0])  # circular difference
        return np.mean(np.abs(diff))
    else:
        diff = np.diff(rays, axis=1, append=rays[:, :1])
        return np.mean(np.abs(diff), axis=1)


def angular_smoothness_loss_torch(rays):
    """PyTorch version of angular_smoothness_loss.

    Args:
        rays: Tensor of shape (..., 32)

    Returns:
        loss: Tensor of shape (...) — smoothness penalty
    """
    import torch

    rays_shifted = torch.roll(rays, shifts=-1, dims=-1)
    diff = rays_shifted - rays

    return torch.mean(torch.abs(diff), dim=-1)
