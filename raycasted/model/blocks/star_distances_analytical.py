"""Analytical star distance computation via Cramer's rule ray-polygon intersection.

No image masks, no Triton, no grid_sample — pure PyTorch GPU linear algebra.
Solves ray-edge intersections for all (centroid, polygon, ray) triples via
determinant-based Cramer's rule, then takes per-ray minimum distance.

Usage for Hungarian matcher cost [n_pred, n_gt]:
    cost = analytical_ray_cost(pred_xy_px, gt_vertices, ray_cos, ray_sin)

Usage for matched-pair loss [n_matched, n_rays]:
    gt_rays = analytical_gt_rays(pred_xy_px, gt_vertices, ray_cos, ray_sin)
    loss = (pred_rays_ln - gt_rays.log()).abs().mean()
"""

import torch


def build_ray_directions(n_rays: int, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) direction tensors for n_rays equi-spaced rays.

    Ray 0 = East (+X), angles increase counter-clockwise.

    Returns:
        ray_cos: [n_rays] cos(θ_i)
        ray_sin: [n_rays] sin(θ_i)
    """
    angles = torch.linspace(0, 2 * torch.pi, n_rays + 1, device=device, dtype=dtype)[:n_rays]
    return angles.cos(), angles.sin()


def _normed_rays_to_vertices(centroids_px: torch.Tensor, rays_norm: torch.Tensor, crop_size: float, ray_cos, ray_sin):
    """Convert normalized ray vectors to pixel-space polygon vertices.

    Args:
        centroids_px: [N, 2] pixel-space centroids (cx, cy)
        rays_norm: [N, n_rays] normalized ray lengths [0, 1]
        crop_size: image size in pixels
        ray_cos: [n_rays] precomputed cosine directions
        ray_sin: [n_rays] precomputed sine directions

    Returns:
        vertices: [N, n_rays, 2] pixel-space vertex coordinates
    """
    rays_px = rays_norm * crop_size
    cx, cy = centroids_px[:, 0:1], centroids_px[:, 1:2]
    cos = ray_cos.to(device=rays_px.device, dtype=rays_px.dtype).view(1, -1)
    sin = ray_sin.to(device=rays_px.device, dtype=rays_px.dtype).view(1, -1)
    vx = cx + rays_px * cos
    vy = cy + rays_px * sin
    return torch.stack([vx, vy], dim=-1)


def analytical_ray_distances(
    centroids_px: torch.Tensor,
    vertices: torch.Tensor,
    ray_cos: torch.Tensor,
    ray_sin: torch.Tensor,
):
    """Compute star-distances from each centroid to polygon boundary along each ray direction.

    For each (centroid, polygon, ray) combination, solves the ray-edge
    intersection and takes the minimum positive distance to any edge.

    Args:
        centroids_px: [N_pred, 2] or [N_pred, 1, 2] predicted centroids in pixel space
        vertices: [N_gt, n_verts, 2] GT polygon vertices in pixel space
        ray_cos: [n_rays] cosine of ray directions
        ray_sin: [n_rays] sine of ray directions

    Returns:
        distances: [N_pred, N_gt, n_rays] non-negative distance to polygon boundary
                   along each ray, capped at a reasonable maximum for misses
    """
    device = centroids_px.device
    dtype = centroids_px.dtype
    n_rays = ray_cos.shape[0]

    if centroids_px.dim() == 2:
        centroids_px = centroids_px.unsqueeze(1)  # [N_pred, 1, 2]
    n_pred, _, _ = centroids_px.shape
    n_gt, n_verts, _ = vertices.shape

    ray_cos = ray_cos.to(device=device, dtype=dtype).view(1, 1, 1, n_rays)
    ray_sin = ray_sin.to(device=device, dtype=dtype).view(1, 1, 1, n_rays)

    v1 = vertices.unsqueeze(0)  # [1, N_gt, n_verts, 2]
    v2_shift = vertices.roll(-1, dims=1).unsqueeze(0)  # wrap-around edges
    v1 = v1.expand(n_pred, n_gt, n_verts, 2)
    v2_shift = v2_shift.expand(n_pred, n_gt, n_verts, 2)

    cx = centroids_px[:, :, 0:1, None].expand(n_pred, n_gt, n_verts, n_rays)
    cy = centroids_px[:, :, 1:2, None].expand(n_pred, n_gt, n_verts, n_rays)

    v1x = v1[:, :, :, 0:1].expand(n_pred, n_gt, n_verts, n_rays)
    v1y = v1[:, :, :, 1:2].expand(n_pred, n_gt, n_verts, n_rays)
    v2x = v2_shift[:, :, :, 0:1].expand(n_pred, n_gt, n_verts, n_rays)
    v2y = v2_shift[:, :, :, 1:2].expand(n_pred, n_gt, n_verts, n_rays)

    dx = v2x - v1x
    dy = v2y - v1y

    cos_exp = ray_cos.expand(n_pred, n_gt, n_verts, n_rays)
    sin_exp = ray_sin.expand(n_pred, n_gt, n_verts, n_rays)

    det = -cos_exp * dy + sin_exp * dx
    det_valid = det.abs() > 1e-10

    t_num = (v1x - cx) * (-dy) + (v1y - cy) * dx
    u_num = (v1x - cx) * (-sin_exp) + (v1y - cy) * cos_exp

    safe_det = torch.where(det_valid, det, torch.ones_like(det))
    t = t_num / safe_det
    u = u_num / safe_det

    valid = det_valid & (t > 1e-6) & (u >= 0) & (u <= 1)

    large_val = torch.full_like(t, 1e10)
    dists = torch.where(valid, t, large_val)
    min_dists, _ = dists.min(dim=2)

    diagonal_px = torch.as_tensor((vertices.max() - vertices.min()) * 2.0, device=device, dtype=dtype).clamp(min=256.0)
    min_dists = min_dists.clamp(max=diagonal_px)

    return min_dists


def analytical_ray_cost(
    pred_xy_px: torch.Tensor,
    gt_vertices: torch.Tensor,
    ray_cos: torch.Tensor,
    ray_sin: torch.Tensor,
):
    """Compute Hungarian matcher cost from analytical ray distances.

    Args:
        pred_xy_px: [n_pred, 2] predicted centroids in pixel space
        gt_vertices: [n_gt, n_verts, 2] GT polygon vertices in pixel space
        ray_cos: [n_rays] cosine of ray directions
        ray_sin: [n_rays] sine of ray directions

    Returns:
        cost: [n_pred, n_gt] mean ray distance per (pred, gt) pair
    """
    distances = analytical_ray_distances(pred_xy_px, gt_vertices, ray_cos, ray_sin)
    return distances.mean(dim=-1)


def analytical_gt_rays(
    pred_xy_px: torch.Tensor,
    gt_vertices: torch.Tensor,
    ray_cos: torch.Tensor,
    ray_sin: torch.Tensor,
    crop_size: float,
):
    """Compute GT star-distance rays for matched prediction-GT pairs.

    Args:
        pred_xy_px: [N_matched, 2] predicted centroids in pixel space
        gt_vertices: [N_matched, n_verts, 2] GT polygon vertices in pixel space
        ray_cos: [n_rays] cosine of ray directions
        ray_sin: [n_rays] sine of ray directions
        crop_size: image size for normalisation

    Returns:
        gt_rays: [N_matched, n_rays] normalized GT ray lengths [0, 1]
    """
    distances = analytical_ray_distances(pred_xy_px, gt_vertices, ray_cos, ray_sin)
    diag = torch.arange(distances.shape[0], device=pred_xy_px.device)
    gt_rays_norm = distances[diag, diag] / crop_size
    return gt_rays_norm
