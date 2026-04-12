"""RayCastED — Filtering Operations.

Annotation filtering and clipping for crop regions.
Implements the 4-step filtering pipeline.
"""

import numpy as np

from ..utils import constants as _const


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
        annotations: Array of shape (N, 3+N_RAYS) in ETL format (pixel space).
        x_start: Left edge of crop region (pixels).
        y_start: Top edge of crop region (pixels).
        chunk_w: Width of crop region (pixels).
        chunk_h: Height of crop region (pixels).
        min_rays_after_clip: Minimum fraction of non-zero rays required.

    Returns:
        filtered: Array of shape (M, 3+N_RAYS) in crop-relative pixel space
            where M <= N (cells outside crop or with too few rays dropped).
    """
    if annotations is None or len(annotations) == 0:
        return np.zeros((0, 3 + _const.N_RAYS), dtype=np.float32)

    # Ensure correct shape
    if annotations.ndim == 1:
        annotations = annotations.reshape(1, -1)

    # =========================================================================
    # STEP 1: Filter by centroid position
    # =========================================================================
    cx = annotations[:, _const.CX_IDX]
    cy = annotations[:, _const.CY_IDX]

    centroid_inside = (cx >= x_start) & (cx < x_start + chunk_w) & (cy >= y_start) & (cy < y_start + chunk_h)

    annotations = annotations[centroid_inside]

    if len(annotations) == 0:
        return np.zeros((0, 3 + _const.N_RAYS), dtype=np.float32)

    # =========================================================================
    # STEP 2: Translate to crop-relative coordinates
    # =========================================================================
    annotations = annotations.copy()
    annotations[:, _const.CX_IDX] -= x_start
    annotations[:, _const.CY_IDX] -= y_start

    # =========================================================================
    # STEP 3: Clip rays to crop boundary
    # =========================================================================
    cx_rel = annotations[:, _const.CX_IDX]
    cy_rel = annotations[:, _const.CY_IDX]

    ray_cos = _const.RAY_COS
    ray_sin = _const.RAY_SIN
    n_rays = _const.N_RAYS

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

        ray_idx = _const.RAY_START_IDX + i
        annotations[:, ray_idx] = np.minimum(annotations[:, ray_idx], d_max)

    # =========================================================================
    # STEP 4: Filter by ray survival rate
    # =========================================================================
    rays = annotations[:, _const.RAY_START_IDX:_const.RAY_END_IDX]
    n_surviving = np.sum(rays > 0, axis=1)
    survival_rate = n_surviving / n_rays

    keep = survival_rate >= min_rays_after_clip
    annotations = annotations[keep]

    return annotations
