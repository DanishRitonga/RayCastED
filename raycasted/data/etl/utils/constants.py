"""RayCastED — Constants and Permutation Indices.

Angular Convention (IMMUTABLE):
    θ₁ = 0° → East (+X axis)
    Angles increase counter-clockwise.
    Angular spacing: 360° / N_RAYS

This file is the single source of truth for all angular constants.
Import from here — NEVER recompute inline.

Use ``configure_rays(n_rays)`` to change the ray count at runtime
(e.g. 32 → 64) before running ETL ingestion.
"""

import numpy as np

# =============================================================================
# RAY GEOMETRY CONSTANTS  (mutable — use configure_rays() to change)
# =============================================================================

N_RAYS = 32
"""Number of radial rays for polygon parameterization."""

# Precomputed angles for each ray (in radians)
# θ_i = i * ANGULAR_SPACING for i in [0, N_RAYS)
RAY_ANGLES = np.array([i * (2.0 * np.pi / N_RAYS) for i in range(N_RAYS)], dtype=np.float64)
"""Array of angles for each ray [θ_0, θ_1, ..., θ_31] in radians."""

# Precomputed cosine and sine for each ray direction
RAY_COS = np.cos(RAY_ANGLES)
"""Cosine of each ray angle — x-component of ray direction."""

RAY_SIN = np.sin(RAY_ANGLES)
"""Sine of each ray angle — y-component of ray direction."""


# =============================================================================
# ANNOTATION FORMAT INDICES  (depend on N_RAYS)
# =============================================================================

# ETL Internal Format (pixel space): [class_id, cx, cy, d_1, ..., d_N]
# Shape: (N, 3 + N_RAYS)
CLASS_IDX = 0
CX_IDX = 1
CY_IDX = 2
RAY_START_IDX = 3
RAY_END_IDX = 3 + N_RAYS  # exclusive


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

    k=1: rotate 90° CCW  → shift by N_RAYS//4
    k=2: rotate 180°     → shift by N_RAYS//2
    k=3: rotate 270° CCW → shift by 3*N_RAYS//4
    """
    indices = np.zeros(N_RAYS, dtype=np.int64)
    shift = (k * N_RAYS) // 4
    for i in range(N_RAYS):
        indices[i] = (i + shift) % N_RAYS
    return indices


# Precomputed permutation indices
FLIP_H_IDX = _compute_flip_h_indices()
FLIP_V_IDX = _compute_flip_v_indices()

ROT_90_IDX = _compute_rotation_indices(1)
ROT_180_IDX = _compute_rotation_indices(2)
ROT_270_IDX = _compute_rotation_indices(3)

ROT_INDICES = {1: ROT_90_IDX, 2: ROT_180_IDX, 3: ROT_270_IDX}


# =============================================================================
# POLAR IOU CONSTANTS
# =============================================================================

POLAR_IOU_EPS = 1e-7
"""Small epsilon to prevent division by zero in Polar-IoU computation."""


# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================


def configure_rays(n_rays: int) -> None:
    """Reconfigure the ray count at runtime.

    Updates all module-level constants that depend on the number of rays.
    Must be called **before** ETL ingestion and model construction.

    Args:
        n_rays: Number of radial rays (e.g. 32, 64).
    """
    import raycasted.data.etl.utils.constants as _self

    if n_rays < 4 or n_rays % 4 != 0:
        raise ValueError(f'n_rays must be a multiple of 4 and >= 4, got {n_rays}')

    _self.N_RAYS = n_rays
    angular_spacing = 2.0 * np.pi / n_rays
    _self.RAY_ANGLES = np.array([i * angular_spacing for i in range(n_rays)], dtype=np.float64)
    _self.RAY_COS = np.cos(_self.RAY_ANGLES)
    _self.RAY_SIN = np.sin(_self.RAY_ANGLES)
    _self.RAY_END_IDX = 3 + n_rays

    # Recompute permutation indices
    _self.FLIP_H_IDX = _self._compute_flip_h_indices()
    _self.FLIP_V_IDX = _self._compute_flip_v_indices()
    _self.ROT_90_IDX = _self._compute_rotation_indices(1)
    _self.ROT_180_IDX = _self._compute_rotation_indices(2)
    _self.ROT_270_IDX = _self._compute_rotation_indices(3)
    _self.ROT_INDICES = {1: _self.ROT_90_IDX, 2: _self.ROT_180_IDX, 3: _self.ROT_270_IDX}


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
    import raycasted.data.etl.utils.constants as _c

    _c.verify_permutation_indices()
    print(f'\nN_RAYS = {_c.N_RAYS}')

    # Test with 64 rays
    print('\n--- Testing configure_rays(64) ---')
    configure_rays(64)
    _c.verify_permutation_indices()
    print(f'N_RAYS = {_c.N_RAYS}')
    print(f'RAY_END_IDX = {_c.RAY_END_IDX}')
