"""Analytical star distance computation via Cramer's rule ray-polygon intersection.

No image masks, no Triton, no grid_sample — pure PyTorch GPU linear algebra.
Solves ray-edge intersections for all (centroid, polygon, ray) triples via
determinant-based Cramer's rule.

Lower bound: minimum distance from centroid to polygon boundary per ray.
Upper bound: distance from centroid to nearest background pixel per ray
(walks sorted intersections across all polygons, parity-tracked).

Usage for Hungarian matcher cost [n_pred, n_gt]:
    cost = analytical_ray_cost(pred_xy_px, gt_vertices, ray_cos, ray_sin)

Usage for matched-pair loss (interval-aware) [n_matched, n_rays]:
    lower, upper = analytical_ray_bounds(pred_xy_px, gt_vertices, ray_cos, ray_sin, crop_size)
    loss = F.relu(lower.log() - pred_rays_ln) + F.relu(pred_rays_ln - upper.log())
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


def _point_in_polygon_vectorized(
    px: torch.Tensor,
    py: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
) -> torch.Tensor:
    """Even-odd rule point-in-polygon test, vectorized over points.

    Args:
        px: [N] x-coordinates of query points
        py: [N] y-coordinates of query points
        vx: [K] x-vertices of polygon (no duplicate close needed)
        vy: [K] y-vertices of polygon

    Returns:
        inside: [N] boolean
    """
    k = vx.shape[0]
    vx_wrap = torch.cat([vx, vx[:1]])
    vy_wrap = torch.cat([vy, vy[:1]])

    n = px.shape[0]
    inside = torch.zeros(n, dtype=torch.bool, device=px.device)

    for i in range(k):
        y1, y2 = vy_wrap[i], vy_wrap[i + 1]
        straddle = (y1 > py) != (y2 > py)
        denom = y2 - y1
        safe_denom = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
        x_int = vx_wrap[i] + (py - y1) * (vx_wrap[i + 1] - vx_wrap[i]) / safe_denom
        inside ^= straddle & (px < x_int) & denom.abs().gt(1e-12)

    return inside


def _compute_d_edge(
    cx: torch.Tensor,
    cy: torch.Tensor,
    cos_r: torch.Tensor,
    sin_r: torch.Tensor,
    crop_size: float,
) -> torch.Tensor:
    """Distance from (cx,cy) to image boundary along ray direction.

    Args:
        cx: [N_pred, n_rays] x-coordinates
        cy: [N_pred, n_rays] y-coordinates
        cos_r: [n_rays]
        sin_r: [n_rays]
        crop_size: image dimension (assumed square H=W=crop_size)

    Returns:
        d_edge: [N_pred, n_rays]
    """
    s = crop_size
    eps = 1e-12

    candidates = [
        ((s - cx) / (cos_r + eps)),
        ((s - cy) / (sin_r + eps)),
        ((0 - cx) / (cos_r + eps)),
        ((0 - cy) / (sin_r + eps)),
    ]

    stacked = torch.stack(candidates, dim=-1)
    stacked[~(stacked > eps)] = float('inf')
    return stacked.min(dim=-1).values


def _cramer_solve_flat(
    centroids_px: torch.Tensor,
    edge_start: torch.Tensor,
    edge_end: torch.Tensor,
    ray_cos: torch.Tensor,
    ray_sin: torch.Tensor,
):
    """Cramer's rule solver for flat edge list (all GT edges concatenated).

    Args:
        centroids_px: [N_pred, 2]
        edge_start: [total_edges, 2]  v1 per edge
        edge_end:   [total_edges, 2]  v2 per edge
        ray_cos: [n_rays]
        ray_sin: [n_rays]

    Returns:
        t:      [N_pred, total_edges, n_rays]  intersection distances
        valid:  [N_pred, total_edges, n_rays]  boolean validity mask
    """
    device = centroids_px.device
    dtype = centroids_px.dtype
    n_pred = centroids_px.shape[0]
    n_edges = edge_start.shape[0]
    n_rays = ray_cos.shape[0]

    ray_cos = ray_cos.to(device=device, dtype=dtype).view(1, 1, 1, n_rays)
    ray_sin = ray_sin.to(device=device, dtype=dtype).view(1, 1, 1, n_rays)

    cx = centroids_px[:, None, 0:1, None].expand(n_pred, n_edges, 1, n_rays)
    cy = centroids_px[:, None, 1:2, None].expand(n_pred, n_edges, 1, n_rays)

    v1x = edge_start[None, :, 0:1, None].expand(n_pred, n_edges, 1, n_rays)
    v1y = edge_start[None, :, 1:2, None].expand(n_pred, n_edges, 1, n_rays)
    v2x = edge_end[None, :, 0:1, None].expand(n_pred, n_edges, 1, n_rays)
    v2y = edge_end[None, :, 1:2, None].expand(n_pred, n_edges, 1, n_rays)

    dx = v2x - v1x
    dy = v2y - v1y

    cos_exp = ray_cos.expand(n_pred, n_edges, 1, n_rays)
    sin_exp = ray_sin.expand(n_pred, n_edges, 1, n_rays)

    det = -cos_exp * dy + sin_exp * dx
    det_valid = det.abs() > 1e-10

    t_num = (v1x - cx) * (-dy) + (v1y - cy) * dx
    u_num = (v1x - cx) * (-sin_exp) + (v1y - cy) * cos_exp

    safe_det = torch.where(det_valid, det, torch.ones_like(det))
    t = t_num / safe_det
    u = u_num / safe_det

    valid = det_valid & (t > 1e-6) & (u >= 0) & (u <= 1)

    return t.squeeze(2), valid.squeeze(2)


def _parity_walk_upper_bounds(
    sorted_t: torch.Tensor,
    sorted_gt_id: torch.Tensor,
    sorted_valid: torch.Tensor,
    initial_inside: torch.Tensor,
    d_edge: torch.Tensor,
    n_gt: int,
) -> torch.Tensor:
    """Parity-walk to find distance to background after exiting each GT.

    Per-centroid-parallel: each centroid gets its own sorted edge list
    and per-ray active-GT tracking.

    Args:
        sorted_t:      [N_pred, total_edges, n_rays]  intersection distances sorted per ray
        sorted_gt_id:  [N_pred, total_edges, n_rays]  GT id of each sorted edge
        sorted_valid:  [N_pred, total_edges, n_rays]  valid intersections mask
        initial_inside: [N_pred, N_gt]  which GTs contain each centroid initially
        d_edge:        [N_pred, n_rays]  distance to image edge
        n_gt:          number of GT polygons

    Returns:
        upper: [N_pred, n_rays]  distance to background along each ray
    """
    device = sorted_t.device
    n_pred, total_edges, n_rays = sorted_t.shape
    upper = d_edge.clone()

    for pred_i in range(n_pred):
        active = initial_inside[pred_i].unsqueeze(1).expand(n_gt, n_rays).clone()
        found = torch.zeros(n_rays, dtype=torch.bool, device=device)
        ray_indices = torch.arange(n_rays, device=device)

        for e in range(total_edges):
            valid_e = sorted_valid[pred_i, e, :] & (~found)
            if not valid_e.any():
                continue

            t_e = sorted_t[pred_i, e, :]
            gt_ids = sorted_gt_id[pred_i, e, :]

            flat_idx = gt_ids.long() * n_rays + ray_indices
            active_flat = active.int().view(-1)
            cur = active_flat[flat_idx]
            active_flat[flat_idx] = 1 - cur
            active = active_flat.view(n_gt, n_rays).bool()

            active_sum = active.sum(dim=0)
            just_became_bg = (active_sum == 0) & valid_e
            upper[pred_i, just_became_bg] = t_e[just_became_bg]
            found = found | just_became_bg

            if found.all():
                break

    return upper


def analytical_ray_bounds(
    centroids_px: torch.Tensor,
    vertices: torch.Tensor,
    ray_cos: torch.Tensor,
    ray_sin: torch.Tensor,
    crop_size: float,
    compute_upper: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute lower and upper bound star distances.

    Lower: minimum distance from each centroid to each GT polygon boundary per ray.
    Upper (optional): distance from each centroid to nearest background pixel per
    ray (parity walk across all polygons).  Expensive — only computed when
    ``compute_upper=True``.

    Args:
        centroids_px: [N_pred, 2] or [N_pred, N_gt, 2] predicted centroids in pixel space
        vertices: [N_gt, n_verts, 2] GT polygon vertices in pixel space
        ray_cos: [n_rays] cosine of ray directions
        ray_sin: [n_rays] sine of ray directions
        crop_size: image dimension (square)
        compute_upper: if False (default), upper is ``None`` and the expensive
            parity walk + point-in-polygon + sort + d_edge are all skipped.

    Returns:
        lower: [N_pred, N_gt, n_rays] distance to each GT's boundary from each centroid
        upper: [N_pred, n_rays] or ``None`` — distance to background from each centroid
    """
    device = centroids_px.device
    dtype = centroids_px.dtype

    if centroids_px.dim() == 2:
        centroids_px = centroids_px.unsqueeze(1)
    n_pred, n_extra, _ = centroids_px.shape
    n_gt, n_verts, _ = vertices.shape
    pts = centroids_px.squeeze(1) if n_extra == 1 else centroids_px[:, 0]

    ray_cos = ray_cos.to(device=device, dtype=dtype)
    ray_sin = ray_sin.to(device=device, dtype=dtype)

    lower = analytical_ray_distances(pts, vertices, ray_cos, ray_sin)

    if not compute_upper:
        return lower, None

    n_rays_val = ray_cos.shape[0]
    total_edges = n_gt * n_verts
    edge_gt_id = torch.arange(n_gt, device=device).repeat_interleave(n_verts)

    v1_all = vertices.reshape(-1, 2)
    v2_all = vertices[:, torch.arange(n_verts).roll(-1), :].reshape(-1, 2)

    t_flat, valid_flat = _cramer_solve_flat(pts, v1_all, v2_all, ray_cos, ray_sin)

    large_val = 1e10
    t_clamped = torch.where(valid_flat, t_flat, torch.full_like(t_flat, large_val))

    sorted_t, sorted_idx = t_clamped.sort(dim=1)
    sorted_valid = sorted_t < large_val

    gt_id_expanded = edge_gt_id.view(1, total_edges, 1).expand(n_pred, total_edges, n_rays_val)
    sorted_gt_id = gt_id_expanded.gather(1, sorted_idx)

    initial_inside = torch.zeros(n_pred, n_gt, dtype=torch.bool, device=device)
    for gt_j in range(n_gt):
        inside = _point_in_polygon_vectorized(pts[:, 0], pts[:, 1], vertices[gt_j, :, 0], vertices[gt_j, :, 1])
        initial_inside[:, gt_j] = inside

    d_edge = _compute_d_edge(
        pts[:, 0:1].expand(n_pred, n_rays_val), pts[:, 1:2].expand(n_pred, n_rays_val), ray_cos, ray_sin, crop_size
    )

    upper = _parity_walk_upper_bounds(sorted_t, sorted_gt_id, sorted_valid, initial_inside, d_edge, n_gt)

    return lower, upper


def analytical_gt_bounds(
    pred_xy_px: torch.Tensor,
    gt_vertices: torch.Tensor,
    ray_cos: torch.Tensor,
    ray_sin: torch.Tensor,
    crop_size: float,
    compute_upper: bool = False,
):
    """Compute lower/upper GT star-distance rays for matched prediction-GT pairs.

    For N_matched pairs (each pred matched to one GT), returns per-pair
    lower (own boundary) and optionally upper (nearest background) bounds.

    Args:
        pred_xy_px: [N_matched, 2] predicted centroids in pixel space
        gt_vertices: [N_gt, n_verts, 2] ALL GT polygon vertices (may include
                     more GTs than N_matched for upper-bound computation)
        ray_cos: [n_rays]
        ray_sin: [n_rays]
        crop_size: image size
        compute_upper: if False (default), upper is ``None`` and expensive computation skipped

    Returns:
        lower: [N_matched, n_rays] normalized lower bounds [0, 1]
        upper: [N_matched, n_rays] or ``None``
    """
    n_matched = pred_xy_px.shape[0]
    lower_all, upper_all = analytical_ray_bounds(
        pred_xy_px, gt_vertices, ray_cos, ray_sin, crop_size, compute_upper=compute_upper
    )
    diag = torch.arange(n_matched, device=pred_xy_px.device)
    lower = lower_all[diag, diag] / crop_size
    upper = upper_all / crop_size if upper_all is not None else None
    return lower, upper
