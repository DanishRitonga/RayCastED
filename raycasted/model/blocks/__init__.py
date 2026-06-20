"""Custom building blocks for RayCastED models.

This module contains custom neural network blocks used in RayCastED:
- head: RayCastDetect polygon detection head
- resoconv: ResoConv (wavelet-based downsampling with SE attention)
- lk_block: C3k2_LK large kernel block with dilated reparameterization
"""

from raycasted.model.blocks.head import (
    RAYCAST_DIM,
    LargeKernelRefinementBlock,
    RayCastDetect,
    RayRefinementBlock,
)
from raycasted.model.blocks.lk_block import C3k2_LK
from raycasted.model.blocks.dcn_blocks import C3k2_DCN
from raycasted.model.blocks.resoconv import ResoConv, ResoConvHybrid

__all__ = [
    'C3k2_DCN',
    'C3k2_LK',
    'LargeKernelRefinementBlock',
    'RayCastDetect',
    'RayRefinementBlock',
    'ResoConv',
    # Backward-compat alias (old checkpoints)
    'ResoConvHybrid',
    'RAYCAST_DIM',
]
