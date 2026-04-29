"""Custom building blocks for RayCastED models.

This module contains custom neural network blocks used in RayCastED:
- head: RayCastDetect polygon detection head
- resoconv: ResoConv, ResoConvDS, HFResidual wavelet blocks
- lk_block: C3k2_LK large kernel block with dilated reparameterization
"""

from raycasted.model.blocks.head import (
    RAYCAST_DIM,
    LargeKernelRefinementBlock,
    RayCastDetect,
    RayRefinementBlock,
)
from raycasted.model.blocks.lk_block import C3k2_LK
from raycasted.model.blocks.resoconv import HFResidual, ResoConv, ResoConvDS, ResoConvDS_Hybrid, ResoConvHybrid

__all__ = [
    'C3k2_LK',
    'HFResidual',
    'LargeKernelRefinementBlock',
    'RayCastDetect',
    'RayRefinementBlock',
    'ResoConv',
    'ResoConvDS',
    'ResoConvDS_Hybrid',
    'ResoConvHybrid',
    'RAYCAST_DIM',
]
