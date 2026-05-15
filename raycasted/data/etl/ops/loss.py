"""RayCastED — Loss Operations

Angular smoothness regularization for polygon training.
Includes both NumPy (validation/debugging) and PyTorch (training) variants.

Note: decode_pred_xy is NOT here. It is a method of RayCastDetectionLoss
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
    """PyTorch version of angular_smoothness_loss (1st-order, kept for compat).

    Args:
        rays: Tensor of shape (..., N_rays)

    Returns:
        loss: Tensor of shape (...) — 1st-order smoothness penalty
    """
    import torch

    rays_shifted = torch.roll(rays, shifts=-1, dims=-1)
    diff = rays_shifted - rays

    return torch.mean(torch.abs(diff), dim=-1)


def curvature_smoothness_loss_torch(rays):
    """Circular 2nd-order difference (curvature) penalty on ray distances.

    Penalises *changes in slope* between consecutive rays, not the slopes
    themselves. This is a discrete curvature regulariser:

        L_curv = mean|d_{i+1} - 2*d_i + d_{i-1}|

    A perfect circle has zero curvature penalty. Smooth but irregular shapes
    (ellipses, organic nuclei) have low penalty. Only sharp kinks / zigzag
    artefacts are penalised — exactly the failure mode that 1st-order smooth
    misses because it also fights against legitimate non-round GT shapes.

    Circular indexing wraps at both ends so every ray contributes.

    Args:
        rays: Tensor of shape (..., N_rays)

    Returns:
        loss: Tensor of shape (...) — curvature penalty per sample
    """
    import torch

    rays_next = torch.roll(rays, shifts=-1, dims=-1)
    rays_prev = torch.roll(rays, shifts=1, dims=-1)
    curvature = rays_next - 2.0 * rays + rays_prev

    return torch.mean(torch.abs(curvature), dim=-1)
