"""RayCastED — Utility Operations

Helper functions for annotation validation and quality monitoring.
"""

import numpy as np

from ..utils import constants as _const


def count_zero_rays(annotations: np.ndarray) -> int:
    """Count cells with more than 5 zero rays.

    Used for monitoring geometry quality during ingestion.
    High rates indicate data quality issues or centroid placement problems.

    Args:
        annotations: Array of shape (N, 35) in ETL format

    Returns:
        count: Number of cells with > 5 zero rays
    """
    if annotations is None or len(annotations) == 0:
        return 0

    rays = annotations[:, _const.RAY_START_IDX : _const.RAY_END_IDX]
    n_zero = np.sum(rays == 0, axis=1)

    return int(np.sum(n_zero > 5))


def validate_annotation_format(annotations: np.ndarray, name: str = 'annotations') -> None:
    """Validate annotation array format.

    Raises AssertionError if format is invalid.

    Args:
        annotations: Array to validate
        name: Name for error messages
    """
    assert annotations is not None, f'{name} is None'
    assert isinstance(annotations, np.ndarray), f'{name} must be numpy array'
    assert annotations.ndim == 2, f'{name} must be 2D, got {annotations.ndim}D'
    assert annotations.shape[1] == 3 + _const.N_RAYS, (
        f'{name} must have {3 + _const.N_RAYS} columns, got {annotations.shape[1]}'
    )
    assert annotations.dtype == np.float32, f'{name} must be float32, got {annotations.dtype}'
