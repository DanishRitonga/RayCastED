"""RayCastED — AIFI-Lite Block.

AIFI (Attention-based Intra-scale Feature Interaction) from RT-DETR (CVPR 2024).
Applies multi-head self-attention on a single feature map to enable global
context interaction between spatial positions.

AIFIBlock encapsulates: project-in (optional) → AIFI → project-out (optional)
as a single module with identity channel dimensions (c1 in = c1 out).
Projection layers are added only when c1 != aifi_dim.

Design choices:
- P4 (16x16 = 256 tokens): full quadratic O(N^2) attention is trivial
- Channel projection when needed (e.g., variant m where P4=512ch)
- Sincos positional encoding (fixed, not learned) — resolution-agnostic
- No residual around the block (matches RT-DETR where AIFI replaces features)
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.transformer import AIFI


class AIFIBlock(nn.Module):
    """AIFI-Lite: channel projection (optional) → AIFI self-attention → channel projection (optional).

    Encapsulates AIFI with optional projection layers as a single module that
    preserves input channel dimensions. Designed for insertion between backbone
    and neck (e.g., on P4 features before FPN upsample).

    When c1 == aifi_dim, projections are replaced with Identity (no extra params).
    When c1 != aifi_dim, 1x1 convs project channels for AIFI at the target dimension.

    Args:
        c1: Input/output channels (identity dimension).
        aifi_dim: Internal AIFI dimension. 256 matches RT-DETR's standard.
        cm: FFN hidden dimension inside AIFI's TransformerEncoderLayer.
        num_heads: Number of attention heads. Must divide aifi_dim evenly.
        dropout: Dropout rate for attention and FFN.
    """

    def __init__(
        self,
        c1: int,
        aifi_dim: int = 256,
        cm: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj_in = nn.Conv2d(c1, aifi_dim, 1) if c1 != aifi_dim else nn.Identity()
        self.aifi = AIFI(aifi_dim, cm, num_heads, dropout)
        self.proj_out = nn.Conv2d(aifi_dim, c1, 1) if c1 != aifi_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply AIFI-Lite: project → self-attention → project back."""
        x = self.proj_in(x)
        x = self.aifi(x)
        x = self.proj_out(x)
        return x
