"""RayCastED — Loss Operations

Angular smoothness regularization for polygon training.
PyTorch variant used during training.

Note: decode_pred_xy is NOT here. It is a method of RayCastDetectionLoss
in ultralytics/utils/loss.py — it requires anchor grid knowledge and
must not be imported in ETL environments.
"""


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
