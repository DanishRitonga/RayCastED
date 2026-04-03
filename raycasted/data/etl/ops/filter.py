"""Polygon YOLOv26 — Filtering Operations

Annotation filtering and clipping for crop regions.
Implements the 4-step filtering pipeline.
"""

import numpy as np

from ..utils.constants import (
    CX_IDX,
    CY_IDX,
    N_RAYS,
    RAY_COS,
    RAY_END_IDX,
    RAY_SIN,
    RAY_START_IDX,
)


def filter_and_clip_annotations(
    annotations: np.ndarray,
    x_start: int,
    y_start: int,
    chunk_w: int,
    chunk_h: int,
    min_rays_after_clip: float = 0.5,
) -> np.ndarray:
    """Filter and clip annotations to a crop region.

    Implements the 4-step pipeline:
        STEP 1: Filter by centroid position
        STEP 2: Translate coordinates to crop-relative space
        STEP 3: Clip rays to crop boundary
        STEP 4: Filter by ray survival rate

    Args:
        annotations: Array of shape (N, 35) in ETL format (pixel space)
        x_start: Left edge of crop region (pixels)
        y_start: Top edge of crop region (pixels)
        chunk_w: Width of crop region (pixels)
        chunk_h: Height of crop region (pixels)
        min_rays_after_clip: Minimum fraction of non-zero rays required
            (0.5 = at least 16 of 32 rays must be non-zero)

    Returns:
        filtered: Array of shape (M, 35) in crop-relative pixel space
            where M <= N (cells outside crop or with too few rays dropped)
    """
    if annotations is None or len(annotations) == 0:
        return np.zeros((0, 35), dtype=np.float32)

    # Ensure correct shape
    if annotations.ndim == 1:
        annotations = annotations.reshape(1, -1)

    n_cells = annotations.shape[0]

    # =========================================================================
    # STEP 1: Filter by centroid position
    # =========================================================================
    # Keep cells where centroid is inside the crop region
    cx = annotations[:, CX_IDX]
    cy = annotations[:, CY_IDX]

    centroid_inside = (cx >= x_start) & (cx < x_start + chunk_w) & (cy >= y_start) & (cy < y_start + chunk_h)

    annotations = annotations[centroid_inside]

    if len(annotations) == 0:
        return np.zeros((0, 35), dtype=np.float32)

    # =========================================================================
    # STEP 2: Translate to crop-relative coordinates
    # =========================================================================
    # Make a copy to avoid modifying original
    annotations = annotations.copy()
    annotations[:, CX_IDX] -= x_start
    annotations[:, CY_IDX] -= y_start
    # Rays remain unchanged (they're distances from centroid)

    # =========================================================================
    # STEP 3: Clip rays to crop boundary
    # =========================================================================
    # For each ray direction, compute the maximum distance before hitting boundary
    cx_rel = annotations[:, CX_IDX]  # Now relative to crop
    cy_rel = annotations[:, CY_IDX]

    # Distance to each boundary from centroid
    # Note: cx_rel, cy_rel are in crop-relative coordinates where (0,0) is top-left
    # +X is right, +Y is down (image coordinates)

    # Get ray directions (adjusting for image coordinate system)
    # RAY_COS[i], RAY_SIN[i] are for standard math coords (CCW from +X)
    # In image coords: +X is right (same), +Y is down (flipped)
    # So we negate the Y component of direction

    for i in range(N_RAYS):
        cos_a = RAY_COS[i]
        sin_a = -RAY_SIN[i]  # Negate for image coordinates (Y down)

        # Distance to each boundary
        # Right boundary: chunk_w - cx_rel, hit if cos_a > 0
        # Left boundary: cx_rel, hit if cos_a < 0
        # Bottom boundary: chunk_h - cy_rel, hit if sin_a > 0
        # Top boundary: cy_rel, hit if sin_a < 0

        # Use small epsilon to avoid division by zero for comparison
        eps = 1e-7

        # Maximum distance in ray direction before hitting boundary
        # Suppress divide-by-zero warnings - we handle these cases with np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            d_right = np.where(cos_a > eps, (chunk_w - cx_rel) / cos_a, np.inf)
            d_left = np.where(cos_a < -eps, -cx_rel / cos_a, np.inf)
            d_bottom = np.where(sin_a > eps, (chunk_h - cy_rel) / sin_a, np.inf)
            d_top = np.where(sin_a < -eps, -cy_rel / sin_a, np.inf)

        # The maximum allowed distance is the minimum of all boundary distances
        # (first boundary the ray hits)
        d_max = np.minimum(np.minimum(d_right, d_left), np.minimum(d_bottom, d_top))

        # Clip the ray
        ray_idx = RAY_START_IDX + i
        annotations[:, ray_idx] = np.minimum(annotations[:, ray_idx], d_max)

    # =========================================================================
    # STEP 4: Filter by ray survival rate
    # =========================================================================
    rays = annotations[:, RAY_START_IDX:RAY_END_IDX]
    n_surviving = np.sum(rays > 0, axis=1)
    survival_rate = n_surviving / N_RAYS

    keep = survival_rate >= min_rays_after_clip
    annotations = annotations[keep]

    return annotations
