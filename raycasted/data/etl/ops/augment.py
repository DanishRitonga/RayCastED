"""RayCastED — Augmentation Operations.

Geometric augmentations for star-convex polygon annotations.
Uses precomputed permutation indices for efficient ray reordering.

Ray count is derived from the annotation array shape (N, 3+n_rays)
so the same code works with any ray count.
"""

import numpy as np


def _get_permutation_indices(n_rays: int) -> dict:
    """Get flip/rotation permutation indices for the given ray count.

    Uses precomputed indices from constants if available, otherwise computes on the fly.
    """
    from ..utils import constants as _const

    if n_rays == _const.N_RAYS:
        return {
            'flip_h': _const.FLIP_H_IDX,
            'flip_v': _const.FLIP_V_IDX,
            'rot': _const.ROT_INDICES,
        }

    # Compute on the fly
    flip_h = np.array([(n_rays // 2 - i) % n_rays for i in range(n_rays)], dtype=np.int64)
    flip_v = np.array([(n_rays - i) % n_rays for i in range(n_rays)], dtype=np.int64)
    rot = {}
    for k in (1, 2, 3):
        shift = (k * n_rays) // 4
        rot[k] = np.array([(i + shift) % n_rays for i in range(n_rays)], dtype=np.int64)

    return {'flip_h': flip_h, 'flip_v': flip_v, 'rot': rot}


def flip_horizontal(
    annotations: np.ndarray,
    canvas_w: int,
) -> np.ndarray:
    """Apply horizontal flip to annotations.

    Reflects x-coordinate and permutes ray ordering.

    Args:
        annotations: Array of shape (N, 3+n_rays) in ETL format (crop-relative).
        canvas_w: Width of the canvas (crop size).

    Returns:
        flipped: Array of shape (N, 3+n_rays) with flipped annotations.
    """
    if annotations is None or len(annotations) == 0:
        return annotations if annotations is not None else np.zeros((0, 0), dtype=np.float32)

    annotations = annotations.copy()

    # Flip x-coordinate: x' = canvas_w - x
    annotations[:, 1] = canvas_w - annotations[:, 1]  # CX

    # Permute rays
    n_rays = annotations.shape[1] - 3
    perm = _get_permutation_indices(n_rays)
    rays = annotations[:, 3:]
    annotations[:, 3:] = rays[:, perm['flip_h']]

    return annotations


def flip_vertical(
    annotations: np.ndarray,
    canvas_h: int,
) -> np.ndarray:
    """Apply vertical flip to annotations.

    Reflects y-coordinate and permutes ray ordering.

    Args:
        annotations: Array of shape (N, 3+n_rays) in ETL format (crop-relative).
        canvas_h: Height of the canvas (crop size).

    Returns:
        flipped: Array of shape (N, 3+n_rays) with flipped annotations.
    """
    if annotations is None or len(annotations) == 0:
        return annotations if annotations is not None else np.zeros((0, 0), dtype=np.float32)

    annotations = annotations.copy()

    # Flip y-coordinate: y' = canvas_h - y
    annotations[:, 2] = canvas_h - annotations[:, 2]  # CY

    # Permute rays
    n_rays = annotations.shape[1] - 3
    perm = _get_permutation_indices(n_rays)
    rays = annotations[:, 3:]
    annotations[:, 3:] = rays[:, perm['flip_v']]

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
        annotations: Array of shape (N, 3+n_rays) in ETL format (crop-relative).
        k: Number of 90° rotations (1, 2, or 3).
        canvas_size: Size of the square canvas (assumes square crop).

    Returns:
        rotated: Array of shape (N, 3+n_rays) with rotated annotations.
    """
    if annotations is None or len(annotations) == 0:
        return annotations if annotations is not None else np.zeros((0, 0), dtype=np.float32)

    if k not in [1, 2, 3]:
        raise ValueError(f'k must be 1, 2, or 3, got {k}')

    annotations = annotations.copy()

    cx = annotations[:, 1].copy()  # CX
    cy = annotations[:, 2].copy()  # CY

    # Apply rotation to centroid
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

    annotations[:, 1] = new_cx
    annotations[:, 2] = new_cy

    # Permute rays
    n_rays = annotations.shape[1] - 3
    perm = _get_permutation_indices(n_rays)
    rays = annotations[:, 3:]
    annotations[:, 3:] = rays[:, perm['rot'][k]]

    return annotations
