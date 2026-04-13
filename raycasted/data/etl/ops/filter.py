"""RayCastED — Filtering Operations.

Annotation filtering and clipping for crop regions.
Implements the 4-step filtering pipeline.

Ray count is derived from the annotation array shape (N, 3+n_rays)
so the same code works with any ray count.
"""

import numpy as np


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
        annotations: Array of shape (N, 3+n_rays) in ETL format (pixel space).
        x_start: Left edge of crop region (pixels).
        y_start: Top edge of crop region (pixels).
        chunk_w: Width of crop region (pixels).
        chunk_h: Height of crop region (pixels).
        min_rays_after_clip: Minimum fraction of non-zero rays required.

    Returns:
        filtered: Array of shape (M, 3+n_rays) in crop-relative pixel space
            where M <= N (cells outside crop or with too few rays dropped).
    """
    if annotations is None or len(annotations) == 0:
        return annotations if annotations is not None else np.zeros((0, 0), dtype=np.float32)

    # Ensure correct shape
    if annotations.ndim == 1:
        annotations = annotations.reshape(1, -1)

    n_rays = annotations.shape[1] - 3  # [class_id, cx, cy, d_1..d_n]

    # =========================================================================
    # STEP 1: Filter by centroid position
    # =========================================================================
    cx = annotations[:, 1]  # CX_IDX
    cy = annotations[:, 2]  # CY_IDX

    centroid_inside = (cx >= x_start) & (cx < x_start + chunk_w) & (cy >= y_start) & (cy < y_start + chunk_h)

    annotations = annotations[centroid_inside]

    if len(annotations) == 0:
        return np.zeros((0, 3 + n_rays), dtype=np.float32)

    # =========================================================================
    # STEP 2: Translate to crop-relative coordinates
    # =========================================================================
    annotations = annotations.copy()
    annotations[:, 1] -= x_start  # CX
    annotations[:, 2] -= y_start  # CY

    # =========================================================================
    # STEP 3: Clip rays to crop boundary
    # =========================================================================
    cx_rel = annotations[:, 1]
    cy_rel = annotations[:, 2]

    # Use precomputed cos/sin matching the actual ray count
    from ..utils import constants as _const

    if n_rays == _const.N_RAYS:
        ray_cos = _const.RAY_COS
        ray_sin = _const.RAY_SIN
    else:
        # Compute on the fly if ray count doesn't match configured constants
        angular_spacing = 2.0 * np.pi / n_rays
        angles = np.array([i * angular_spacing for i in range(n_rays)])
        ray_cos = np.cos(angles)
        ray_sin = np.sin(angles)

    for i in range(n_rays):
        cos_a = ray_cos[i]
        sin_a = ray_sin[i]

        eps = 1e-7

        with np.errstate(divide='ignore', invalid='ignore'):
            d_right = np.where(cos_a > eps, (chunk_w - cx_rel) / cos_a, np.inf)
            d_left = np.where(cos_a < -eps, -cx_rel / cos_a, np.inf)
            d_bottom = np.where(sin_a > eps, (chunk_h - cy_rel) / sin_a, np.inf)
            d_top = np.where(sin_a < -eps, -cy_rel / sin_a, np.inf)

        d_max = np.minimum(np.minimum(d_right, d_left), np.minimum(d_bottom, d_top))

        ray_idx = 3 + i  # RAY_START_IDX + i
        annotations[:, ray_idx] = np.minimum(annotations[:, ray_idx], d_max)

    # =========================================================================
    # STEP 4: Filter by ray survival rate
    # =========================================================================
    rays = annotations[:, 3:]  # all ray columns
    n_surviving = np.sum(rays > 0, axis=1)
    survival_rate = n_surviving / n_rays

    keep = survival_rate >= min_rays_after_clip
    annotations = annotations[keep]

    return annotations
