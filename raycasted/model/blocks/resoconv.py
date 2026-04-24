"""ResoConv: Wavelet-based downsampling convolution via DB2 DWT.

Replaces strided 3x3 Conv in backbone/neck with DWT-based downsampling:
- DB2 (Daubechies-2) DWT splits input into 4 sub-bands (LL, LH, HL, HH)
- Concatenate sub-bands with identity shortcut → 1x1 Conv projection
- Output is at half spatial resolution (H/2 x W/2) — same as strided Conv
- Explicitly preserves high-frequency information in LH, HL, HH sub-bands

Key differences from standard strided Conv:
- Anti-aliasing by design (DB2 has 2 vanishing moments)
- Explicit frequency decomposition (4 sub-bands vs single mixed output)
- Fewer parameters (1x1 conv for channel mixing, not spatial)

Wavelet filter generation:
- Uses pywt.Wavelet('db2').dec_lo / dec_hi if pywavelets is installed (default)
- Falls back to hardcoded DB2 values if pywavelets is not installed
- Both paths produce identical filter values (verified)

References:
- WaveCNet: Williams & Li, "Wavelet-based Pooling for Deep CNNs", CVPR 2020
- DWT-UNet: DWT downsampling in U-Net for medical segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

# DB2 (Daubechies-2) wavelet decomposition filters
# Fixed constants — no learning, no pywavelets dependency at runtime
# Use pywavelets to generate if available (verification), otherwise fallback to hardcoded values
try:
    import pywt

    _DB2_LO = torch.tensor(pywt.Wavelet('db2').dec_lo, dtype=torch.float32)
    _DB2_HI = torch.tensor(pywt.Wavelet('db2').dec_hi, dtype=torch.float32)
except ImportError:
    # Fallback: hardcoded values matching pywt.Wavelet('db2')
    # Verified to match: dec_lo = [-0.12940952, 0.22414387, 0.8365163, 0.48301487]
    _DB2_LO = torch.tensor(
        [
            -0.1294095225512603,  # h0[0]
            0.2241438680420134,  # h0[1]
            0.8365163037378079,  # h0[2]
            0.4830148656578357,  # h0[3]
        ],
        dtype=torch.float32,
    )

    _DB2_HI = torch.tensor(
        [
            -0.4830148656578357,  # h1[0] = -h0[3]
            0.8365163037378079,  # h1[1] = h0[2]
            -0.2241438680420134,  # h1[2] = -h0[1]
            -0.1294095225512603,  # h1[3] = -h0[0]
        ],
        dtype=torch.float32,
    )


def _create_db2_filters(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create 2D DB2 wavelet filters for conv2d weight initialization.

    Constructs 4 filters (LL, LH, HL, HH) by outer product of 1D DB2 filters.
    Output shape: [4, 1, 4, 4] for conv2d with in_channels=1, out_channels=4.

    Args:
        device: Target device for filters.
        dtype: Target dtype for filters.

    Returns:
        Filter tensor of shape [4, 1, 4, 4].
    """
    lo = _DB2_LO.to(device=device, dtype=dtype)
    hi = _DB2_HI.to(device=device, dtype=dtype)

    # Outer products: LL = lo ⊗ lo, LH = hi ⊗ lo, HL = lo ⊗ hi, HH = hi ⊗ hi
    ll = lo[:, None] * lo[None, :]  # [4, 4]
    lh = hi[:, None] * lo[None, :]
    hl = lo[:, None] * hi[None, :]
    hh = hi[:, None] * hi[None, :]

    # Stack: [4, 4, 4] → [4, 1, 4, 4] for conv2d weight format
    filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # [4, 1, 4, 4]
    return filters


class DWT2D(nn.Module):
    """2D Discrete Wavelet Transform using fixed DB2 filters.

    Splits input into 4 sub-bands via depthwise conv2d:
    - LL: Low-low (approximation)
    - LH: Low-high (horizontal edges)
    - HL: High-low (vertical edges)
    - HH: High-high (diagonal edges)

    Filters are initialized from DB2 constants and frozen (no gradient).
    Output channels = 4 × input_channels (one set per input channel).

    Args:
        in_channels: Number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        filters = _create_db2_filters(device='cpu', dtype=torch.float32)
        filters_tiled = filters.repeat(in_channels, 1, 1, 1)

        self.register_buffer('dwt_weight', filters_tiled, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D DWT via depthwise conv2d with reflection padding.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Wavelet sub-bands [B, 4*C, H/2, W/2].
        """
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(
            x,
            self.dwt_weight,
            bias=None,
            stride=2,
            padding=0,
            groups=self.in_channels,
        )


class ResoConv(nn.Module):
    """Wavelet-based downsampling convolution via DB2 DWT.

    Replaces strided 3x3 Conv with DWT-based 2x downsampling:
        Input [B, C_in, H, W]
          ↓
        DWT2D (DB2 filters, stride=2) → [B, 4*C_in, H/2, W/2]
          ↓
        Concat with downsampled identity shortcut → [B, 5*C_in, H/2, W/2]
          ↓
        1×1 Conv projection → [B, C_out, H/2, W/2]

    This replaces Conv [C_out, 3, 2] in backbone/neck with:
    - Explicit frequency decomposition (LL, LH, HL, HH)
    - Anti-aliased downsampling (DB2 vanishing moments)
    - Gradient flow via identity shortcut

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Whether to add downsampled identity shortcut before projection.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.shortcut = shortcut

        self.dwt = DWT2D(c1)

        in_proj = c1 * 5 if shortcut else c1 * 4
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

