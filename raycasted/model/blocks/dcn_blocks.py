"""C3k2_DCN: DCNv2 variant of C3k2.

Drop-in replacement for standard C3k2 that uses DCNv2 in the first 3×3
bottleneck conv. Zero-init offsets+mask → starts as standard conv at init.

Architecture:
  C3k2_DCN (replicates C2f split/merge):
    cv1: 1x1 Conv(c_in -> 2*c)
    m:   n x DCNBottleneck(c, c)
    cv2: 1x1 Conv((2+n)*c -> c_out)

  DCNBottleneck(c, c):
    DCNConv(c, c, 3)               # DCNv2 replacing first 3×3 Conv
    Conv(c, c, 3)                  # standard second 3×3 Conv + BN + SiLU
    + residual shortcut
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

from raycasted.model.blocks.dcn import DCNConv


class DCNBottleneck(nn.Module):
    """Bottleneck with DCNv2 in first 3×3 Conv.

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Whether to add residual connection.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = DCNConv(c1, c2, 3, 1)
        self.cv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass: DCNv2 → Conv + optional residual."""
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3k2_DCN(nn.Module):  # noqa: N801
    """C3k2 with DCNv2 bottlenecks.

    Replicates C2f's split/merge structure but uses DCNBottleneck instead
    of standard Bottleneck for adaptive receptive fields.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of DCNBottleneck blocks.
        c3k: Unused (C3k2 compat).
        e: Expansion ratio for hidden channels.
        attn: Unused (C3k2 compat).
        g: Unused (C3k2 compat).
        shortcut: Whether DCNBottleneck blocks use shortcut.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DCNBottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        """Forward pass: C2f-style split/process/merge with DCN bottlenecks."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
