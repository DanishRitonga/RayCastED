"""RayCastED — Augmentation Operations.

Geometric augmentations for star-convex polygon annotations.
Uses precomputed permutation indices for efficient ray reordering.
"""

import numpy as np

from ..utils import constants as _const


def flip_horizontal(
    annotations: np.ndarray,
    canvas_w: int,
) -> np.ndarray:
    """Apply horizontal flip to annotations.

    Reflects x-coordinate and permutes ray ordering.

    Args:
        annotations: Array of shape (N, 3+N_RAYS) in ETL format (crop-relative).
        canvas_w: Width of the canvas (crop size).

    Returns:
        flipped: Array of shape (N, 3+N_RAYS) with flipped annotations.
    """
    if annotations is None or len(annotations) == 0:
        return np.zeros((0, 3 + _const.N_RAYS), dtype=np.float32)

    annotations = annotations.copy()

    # Flip x-coordinate: x' = canvas_w - x
    annotations[:, _const.CX_IDX] = canvas_w - annotations[:, _const.CX_IDX]

    # Permute rays
    rays = annotations[:, _const.RAY_START_IDX:_const.RAY_END_IDX]
    rays_flipped = rays[:, _const.FLIP_H_IDX]
    annotations[:, _const.RAY_START_IDX:_const.RAY_END_IDX] = rays_flipped

    return annotations


def flip_vertical(
    annotations: np.ndarray,
    canvas_h: int,
) -> np.ndarray:
    """Apply vertical flip to annotations.

    Reflects y-coordinate and permutes ray ordering.

    Args:
        annotations: Array of shape (N, 3+N_RAYS) in ETL format (crop-relative).
        canvas_h: Height of the canvas (crop size).

    Returns:
        flipped: Array of shape (N, 3+N_RAYS) with flipped annotations.
    """
    if annotations is None or len(annotations) == 0:
        return np.zeros((0, 3 + _const.N_RAYS), dtype=np.float32)

    annotations = annotations.copy()

    # Flip y-coordinate: y' = canvas_h - y
    annotations[:, _const.CY_IDX] = canvas_h - annotations[:, _const.CY_IDX]

    # Permute rays
    rays = annotations[:, _const.RAY_START_IDX:_const.RAY_END_IDX]
    rays_flipped = rays[:, _const.FLIP_V_IDX]
    annotations[:, _const.RAY_START_IDX:_const.RAY_END_IDX] = rays_flipped

    return annotations


def rotate_90(
    annotations: np.ndarray,
    k: int,
    canvas_size: int,
) -> np.ndarray:
    """Apply 90° rotation(s) to annotations.

    Rotation is counter-clockwise. After rotation, coordinates are
    transformed to the new coordinate system.

    Args:
        annotations: Array of shape (N, 3+N_RAYS) in ETL format (crop-relative).
        k: Number of 90° rotations (1, 2, or 3).
        canvas_size: Size of the square canvas (assumes square crop).

    Returns:
        rotated: Array of shape (N, 3+N_RAYS) with rotated annotations.
    """
    if annotations is None or len(annotations) == 0:
        return np.zeros((0, 3 + _const.N_RAYS), dtype=np.float32)

    if k not in [1, 2, 3]:
        raise ValueError(f'k must be 1, 2, or 3, got {k}')

    annotations = annotations.copy()

    cx = annotations[:, _const.CX_IDX].copy()
    cy = annotations[:, _const.CY_IDX].copy()

    # Apply rotation to centroid
    # For a square canvas, rotation formulas:
    # k=1 (90° CCW): (x, y) → (y, canvas_size - x)
    # k=2 (180°): (x, y) → (canvas_size - x, canvas_size - y)
    # k=3 (270° CCW): (x, y) → (canvas_size - y, x)

    if k == 1:
        new_cx = cy
        new_cy = canvas_size - cx
    elif k == 2:
        new_cx = canvas_size - cx
        new_cy = canvas_size - cy
    else:  # k == 3
        new_cx = canvas_size - cy
        new_cy = cx

    annotations[:, _const.CX_IDX] = new_cx
    annotations[:, _const.CY_IDX] = new_cy

    # Permute rays
    perm_idx = _const.ROT_INDICES[k]
    rays = annotations[:, _const.RAY_START_IDX:_const.RAY_END_IDX]
    rays_rotated = rays[:, perm_idx]
    annotations[:, _const.RAY_START_IDX:_const.RAY_END_IDX] = rays_rotated

    return annotations
