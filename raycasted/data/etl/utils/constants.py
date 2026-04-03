"""Polygon YOLOv26 — Constants and Permutation Indices.

Angular Convention (IMMUTABLE):
    θ₁ = 0° → East (+X axis)
    Angles increase counter-clockwise.
    Angular spacing: 11.25° (= 2π / 32)

This file is the single source of truth for all angular constants.
Import from here — NEVER recompute inline.
"""

from typing import Final

import numpy as np

# =============================================================================
# RAY GEOMETRY CONSTANTS
# =============================================================================

N_RAYS: Final[int] = 32
"""Number of radial rays for polygon parameterization."""

ANGULAR_SPACING: Final[float] = 2.0 * np.pi / N_RAYS  # 11.25° in radians
"""Angular spacing between consecutive rays (radians)."""

# Precomputed angles for each ray (in radians)
# θ_i = i * ANGULAR_SPACING for i in [0, N_RAYS)
RAY_ANGLES: Final[np.ndarray] = np.array([i * ANGULAR_SPACING for i in range(N_RAYS)], dtype=np.float64)
"""Array of angles for each ray [θ_0, θ_1, ..., θ_31] in radians."""

# Precomputed cosine and sine for each ray direction
RAY_COS: Final[np.ndarray] = np.cos(RAY_ANGLES)
"""Cosine of each ray angle — x-component of ray direction."""

RAY_SIN: Final[np.ndarray] = np.sin(RAY_ANGLES)
"""Sine of each ray angle — y-component of ray direction."""


# =============================================================================
# ANNOTATION FORMAT INDICES
# =============================================================================

# ETL Internal Format (pixel space): [cx, cy, d_1, ..., d_32, class_id]
# Shape: (N, 35)
CLASS_IDX: Final[int] = 0
CX_IDX: Final[int] = 1
CY_IDX: Final[int] = 2
RAY_START_IDX: Final[int] = 3
RAY_END_IDX: Final[int] = 35  # exclusive


# Collated Batch Format: [batch_idx, class_id, cx, cy, d_1, ..., d_32]
# Shape: (sum_M, 36)
BATCH_IDX: Final[int] = 0
BATCH_CLASS_IDX: Final[int] = 1
BATCH_CX_IDX: Final[int] = 2
BATCH_CY_IDX: Final[int] = 3
BATCH_RAY_START_IDX: Final[int] = 4
BATCH_RAY_END_IDX: Final[int] = 36  # exclusive


# =============================================================================
# PERMUTATION INDICES FOR GEOMETRIC AUGMENTATIONS
# =============================================================================


def _compute_flip_h_indices() -> np.ndarray:
    """Compute permutation indices for horizontal flip.

    Horizontal flip reflects the x-axis:
        θ → π - θ  (mod 2π)
    """
    indices = np.zeros(N_RAYS, dtype=np.int64)
    for i in range(N_RAYS):
        indices[i] = (N_RAYS // 2 - i) % N_RAYS
    return indices


def _compute_flip_v_indices() -> np.ndarray:
    """Compute permutation indices for vertical flip.

    V-flip reflects across horizontal axis (y → -y):
        θ → -θ (mod 2π)
    """
    indices = np.zeros(N_RAYS, dtype=np.int64)
    for i in range(N_RAYS):
        indices[i] = (N_RAYS - i) % N_RAYS
    return indices


def _compute_rotation_indices(k: int) -> np.ndarray:
    """Compute permutation indices for 90° rotation (counter-clockwise).

    k=1: rotate 90° CCW  → shift by 8
    k=2: rotate 180°     → shift by 16
    k=3: rotate 270° CCW → shift by 24
    """
    indices = np.zeros(N_RAYS, dtype=np.int64)
    shift = (k * N_RAYS) // 4
    for i in range(N_RAYS):
        indices[i] = (i + shift) % N_RAYS
    return indices


# Precomputed permutation indices
FLIP_H_IDX: Final[np.ndarray] = _compute_flip_h_indices()
FLIP_V_IDX: Final[np.ndarray] = _compute_flip_v_indices()

ROT_90_IDX: Final[np.ndarray] = _compute_rotation_indices(1)
ROT_180_IDX: Final[np.ndarray] = _compute_rotation_indices(2)
ROT_270_IDX: Final[np.ndarray] = _compute_rotation_indices(3)

ROT_INDICES: Final[dict] = {1: ROT_90_IDX, 2: ROT_180_IDX, 3: ROT_270_IDX}


# =============================================================================
# POLAR IOU CONSTANTS
# =============================================================================

POLAR_IOU_EPS: Final[float] = 1e-7
"""Small epsilon to prevent division by zero in Polar-IoU computation."""


# =============================================================================
# VERIFICATION
# =============================================================================


def verify_permutation_indices() -> None:
    """Verify that permutation indices are correct."""
    for name, idx_arr in [
        ('FLIP_H_IDX', FLIP_H_IDX),
        ('FLIP_V_IDX', FLIP_V_IDX),
        ('ROT_90_IDX', ROT_90_IDX),
        ('ROT_180_IDX', ROT_180_IDX),
        ('ROT_270_IDX', ROT_270_IDX),
    ]:
        assert idx_arr.min() >= 0, f'{name} has negative index'
        assert idx_arr.max() < N_RAYS, f'{name} has index >= N_RAYS'
        assert len(np.unique(idx_arr)) == N_RAYS, f'{name} is not valid permutation'

    # Check involutions (flip twice = identity)
    for name, idx_arr in [('FLIP_H_IDX', FLIP_H_IDX), ('FLIP_V_IDX', FLIP_V_IDX)]:
        double_perm = idx_arr[idx_arr]
        assert np.array_equal(double_perm, np.arange(N_RAYS)), f'Double {name} != identity'

    print('All permutation index verifications passed!')


if __name__ == '__main__':
    verify_permutation_indices()
    print(f'\nN_RAYS = {N_RAYS}')
    print(f'ANGULAR_SPACING = {np.degrees(ANGULAR_SPACING):.2f}°')
