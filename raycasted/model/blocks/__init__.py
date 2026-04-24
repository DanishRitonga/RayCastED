"""Custom building blocks for RayCastED models.

This module contains custom neural network blocks used in RayCastED:
- head: RayCastDetect polygon detection head
- resoconv: ResoConv wavelet transform block (TODO)
- lk_block: C3k2_LK large kernel block (TODO)
"""

from raycasted.model.blocks.head import (
    RAYCAST_DIM,
    LargeKernelRefinementBlock,
    RayCastDetect,
    RayRefinementBlock,
)

__all__ = [
    'LargeKernelRefinementBlock',
    'RayCastDetect',
    'RayRefinementBlock',
    'RAYCAST_DIM',
]
