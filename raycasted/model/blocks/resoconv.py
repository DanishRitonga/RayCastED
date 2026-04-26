"""ResoConv: Wavelet-based downsampling convolution via bior2.2 DWT.

Replaces strided 3x3 Conv in backbone/neck with DWT-based downsampling:
- bior2.2 (biorthogonal-2.2) DWT splits input into 4 sub-bands (LL, LH, HL, HH)
- Concatenate sub-bands with identity shortcut → 1x1 Conv projection
- Output is at half spatial resolution (H/2 x W/2) — same as strided Conv
- Explicitly preserves high-frequency information in LH, HL, HH sub-bands

Key differences from standard strided Conv:
- Anti-aliasing by design (bior2.2 has 2 vanishing moments)
- Explicit frequency decomposition (4 sub-bands vs single mixed output)
- Fewer parameters (1x1 conv for channel mixing, not spatial)

Wavelet choice rationale (bior2.2 over db2):
- Symmetric filters → linear phase → zero spatial shift in sub-band responses
- Reflection padding corrects most phase issues, but symmetric filters give
  cleaner sub-band symmetry around features (verified on Gaussian/blob tests)
- 6-tap (vs db2's 4-tap): slightly wider support, same vanishing moments
- Not orthogonal (dec ≠ rec filters), but irrelevant for single-pass DWT

Wavelet filter generation:
- Uses pywt.Wavelet('bior2.2').dec_lo / dec_hi if pywavelets is installed (default)
- Falls back to hardcoded bior2.2 values if pywavelets is not installed
- Both paths produce identical filter values (verified)

References:
- WaveCNet: Williams & Li, "Wavelet-based Pooling for Deep CNNs", CVPR 2020
- DWT-UNet: DWT downsampling in U-Net for medical segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

_BIOR_PAD = 2  # bior2.2: 6-tap filters → pad=2 for exact H/2 output with stride=2

# bior2.2 (biorthogonal-2.2) wavelet decomposition filters
# 6-tap symmetric filters → linear phase → zero spatial shift in sub-bands
try:
    import pywt

    _BIOR_LO = torch.tensor(pywt.Wavelet('bior2.2').dec_lo, dtype=torch.float32)
    _BIOR_HI = torch.tensor(pywt.Wavelet('bior2.2').dec_hi, dtype=torch.float32)
except ImportError:
    _BIOR_LO = torch.tensor(
        [
            0.0,
            -0.1767766952966369,
            0.3535533905932738,
            1.0606601717798212,
            0.3535533905932738,
            -0.1767766952966369,
        ],
        dtype=torch.float32,
    )

    _BIOR_HI = torch.tensor(
        [
            0.0,
            0.3535533905932738,
            -0.7071067811865476,
            0.3535533905932738,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )


def _create_bior_filters(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create 2D bior2.2 wavelet filters for conv2d weight initialization.

    Constructs 4 filters (LL, LH, HL, HH) by outer product of 1D bior2.2 filters.
    Output shape: [4, 1, 6, 6] for conv2d with in_channels=1, out_channels=4.

    Args:
        device: Target device for filters.
        dtype: Target dtype for filters.

    Returns:
        Filter tensor of shape [4, 1, 6, 6].
    """
    lo = _BIOR_LO.to(device=device, dtype=dtype)
    hi = _BIOR_HI.to(device=device, dtype=dtype)

    ll = lo[:, None] * lo[None, :]
    lh = hi[:, None] * lo[None, :]
    hl = lo[:, None] * hi[None, :]
    hh = hi[:, None] * hi[None, :]

    filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # [4, 1, 6, 6]
    return filters


class DWT2D(nn.Module):
    """2D Discrete Wavelet Transform using fixed bior2.2 filters.

    Splits input into 4 sub-bands via depthwise conv2d:
    - LL: Low-low (approximation)
    - LH: Low-high (horizontal edges)
    - HL: High-low (vertical edges)
    - HH: High-high (diagonal edges)

    Filters are initialized from bior2.2 constants and frozen (no gradient).
    Output channels = 3*C (drop_hh=True) or 4*C × input_channels.

    Args:
        in_channels: Number of input channels.
        drop_hh: If True, discard HH sub-band and return 3 sub-bands only.
    """

    def __init__(self, in_channels: int, drop_hh: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.drop_hh = drop_hh

        filters = _create_bior_filters(device='cpu', dtype=torch.float32)
        filters_tiled = filters.repeat(in_channels, 1, 1, 1)

        self.register_buffer('dwt_weight', filters_tiled, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D DWT via depthwise conv2d with reflection padding.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Wavelet sub-bands [B, 3*C, H/2, W/2] (drop_hh) or [B, 4*C, H/2, W/2].
        """
        x = F.pad(x, (_BIOR_PAD, _BIOR_PAD, _BIOR_PAD, _BIOR_PAD), mode='reflect')
        out = F.conv2d(
            x,
            self.dwt_weight,
            bias=None,
            stride=2,
            padding=0,
            groups=self.in_channels,
        )
        if self.drop_hh:
            n_c = self.in_channels
            out = torch.cat([out[:, :n_c], out[:, n_c : 3 * n_c]], dim=1)
        return out


class ResoConv(nn.Module):
    """Wavelet-based downsampling convolution via bior2.2 DWT.

    Replaces strided 3x3 Conv with DWT-based 2x downsampling:
        Input [B, C_in, H, W]
          ↓
        DWT2D (bior2.2 filters, stride=2) → [B, n_sub*C_in, H/2, W/2]
          ↓
        Concat with downsampled identity shortcut (optional) → [B, in_proj, H/2, W/2]
          ↓
        1×1 Conv projection → [B, C_out, H/2, W/2]

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Whether to add downsampled identity shortcut before projection.
        drop_hh: If True, discard HH sub-band before projection.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, drop_hh: bool = False):
        super().__init__()
        self.shortcut = shortcut

        self.dwt = DWT2D(c1, drop_hh=drop_hh)

        n_sub = 3 if drop_hh else 4
        in_proj = c1 * (1 + n_sub) if shortcut else c1 * n_sub
        self.proj = Conv(in_proj, c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ResoConv: DWT downsample → concat shortcut → 1x1 project.

        Args:
            x: Input tensor [B, c1, H, W].

        Returns:
            Output tensor [B, c2, H/2, W/2] — half spatial resolution.
        """
        x_dwt = self.dwt(x)  # [B, 4*c1, H/2, W/2]

        if self.shortcut:
            x_down = F.avg_pool2d(x, kernel_size=2, stride=2)
            x_cat = torch.cat([x_down, x_dwt], dim=1)
        else:
            x_cat = x_dwt

        return self.proj(x_cat)
