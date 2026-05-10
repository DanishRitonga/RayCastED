"""C3k2_LK: Large-kernel depthwise variant of C3k2.

Drop-in replacement for standard C3k2 that uses large-kernel depthwise
convolutions with dilated reparameterization (UniRepLKNet-style). Inherits
C2f's split/merge structure but replaces inner Bottleneck with LKBottleneck.

Architecture:
  C3k2_LK(C2f):
    cv1: 1x1 Conv(c_in -> 2*c)         # split into [c, c]
    m:   n x LKBottleneck(c, c, K)     # large-kernel sub-blocks
    cv2: 1x1 Conv((2+n)*c -> c_out)    # merge all branches

  LKBottleneck(c, c, K):
    1x1 PW expand (c -> 2c)
    DilatedReparamDW(K)                 # parallel dilated DW branches
    1x1 PW project (2c -> c)
    + residual shortcut

At inference, DilatedReparamDW fuses all branches into a single KxK DW conv
with zero overhead.

References:
  - UniRepLKNet: Ding et al., "Universal Perception Large-Kernel ConvNet"
  - LKCell: Cui et al., arXiv:2407.18054
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


def _dilated_decomposition(kernel_size: int) -> list[tuple[int, int]]:
    """Compute dilated branch decomposition for a target kernel size.

    Returns list of (kernel, dilation) pairs. First entry is always the
    main branch (full kernel, dilation=1). Subsequent entries use smaller
    kernels with increasing dilation to cover the same spatial extent.

    Constraint from UniRepLKNet: (k-1)*d + 1 <= K

    Args:
        kernel_size: Target large kernel size (e.g. 7, 9, 13).

    Returns:
        List of (kernel_size, dilation) tuples.
    """
    branches = [(kernel_size, 1)]
    dilation = 2
    while True:
        max_k = kernel_size - 2 * (dilation - 1)
        if max_k < 3:
            break
        k = min(max_k, 5)
        if (k - 1) * dilation + 1 > kernel_size:
            k = (kernel_size - 1) // dilation + 1
            if k < 3:
                break
        branches.append((k, dilation))
        dilation += 1
    return branches


class DilatedReparamDW(nn.Module):
    """Training: parallel dilated DW branches. Inference: single fused DW conv.

    Multiple parallel depthwise conv branches with different dilations provide
    different spatial granularities. At inference, all branches fuse into a
    single KxK depthwise conv via reparameterization.

    Args:
        dim: Number of channels (input == output for depthwise).
        kernel_size: Target large kernel size.
    """

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.branches = nn.ModuleList()

        for k, d in _dilated_decomposition(kernel_size):
            padding = (k - 1) * d // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, k, padding=padding, groups=dim, bias=False, dilation=d),
                    nn.BatchNorm2d(dim),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(b(x) for b in self.branches)


class LKBottleneck(nn.Module):
    """Large-kernel bottleneck block with dilated reparameterization.

    Architecture:
        1x1 Conv expand (c -> 2c)
        DilatedReparamDW(2c, K)
        1x1 Conv project (2c -> c)
        + residual shortcut

    Args:
        c1: Input channels.
        c2: Output channels.
        kernel_size: Large kernel size for DilatedReparamDW.
        shortcut: Whether to add residual connection.
    """

    def __init__(self, c1: int, c2: int, kernel_size: int = 7, shortcut: bool = True):
        super().__init__()
        self.cv1 = Conv(c1, c1 * 2, 1, 1)
        self.dw = DilatedReparamDW(c1 * 2, kernel_size)
        self.cv2 = Conv(c1 * 2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.dw(self.cv1(x)))
        return x + out if self.add else out


class C3k2_LK(nn.Module):
    """C3k2 with large-kernel depthwise bottlenecks.

    Replicates C2f's split/merge structure but uses LKBottleneck instead
    of standard Bottleneck for larger receptive field.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of LKBottleneck blocks.
        c3k: Unused (kept for C3k2 arg compatibility).
        e: Expansion ratio for hidden channels.
        attn: Unused (kept for C3k2 arg compatibility).
        g: Groups (unused, kept for compatibility).
        shortcut: Whether bottleneck blocks use shortcut.
        kernel_size: Large kernel size for DilatedReparamDW.
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
        kernel_size: int = 7,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(LKBottleneck(self.c, self.c, kernel_size, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
