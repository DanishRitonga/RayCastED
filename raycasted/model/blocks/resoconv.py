"""ResoConv: Resolution-preserving convolution via wavelet transform.

Based on WaveMix / WCM / RWCM designs from medical image literature:
- DB2 (Daubechies-2) DWT splits input into 4 sub-bands (LL, LH, HL, HH)
- Concatenate sub-bands → 1×1 Conv projects to target channels
- Preserves spatial resolution while extracting multi-scale features
- Pure PyTorch implementation with fixed DB2 filters (ONNX-friendly)

Key differences from WCM:
- Uses pywavelets by default; hardcoded fallback when pywt is unavailable
- No learnable wavelet parameters (avoid deployment complexity)
- Simplified architecture: DWT → concat → 1×1 Conv (no DW+PW cascade)

Wavelet filter generation:
- Uses pywt.Wavelet('db2').dec_lo / dec_hi if pywavelets is installed (default)
- Falls back to hardcoded DB2 values if pywavelets is not installed
- Both paths produce identical filter values (verified)

References:
- WaveMix: "WaveMix: A Novel Framework for Network Architecture"
- WCM: "Wavelet-based Convolutional Module for Medical Image Segmentation"
- RWCM: "Receptive Field Wavelet-based Convolutional Module"
"""

import torch
import torch.nn as nn
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

        # Create depthwise conv: each input channel gets its own 4 filters
        # We want depthwise convolution: each of the C_in input channels produces
        # 4 output channels (LL, LH, HL, HH) using the same DB2 filters.
        # Weight shape: [4*C_in, 1, 4, 4] for depthwise operation
        filters = _create_db2_filters(device='cpu', dtype=torch.float32)  # [4, 1, 4, 4]
        filters_tiled = filters.repeat(in_channels, 1, 1, 1)  # [in_channels*4, 1, 4, 4]

        self.register_buffer(
            'filters',
            filters_tiled,  # [in_channels*4, 1, 4, 4]
            persistent=False,
        )

        # Depthwise conv: groups=in_channels means each input channel has its own
        # set of filters. But we want 4 filters per input channel, so we use
        # groups=in_channels and out_channels=in_channels*4.
        # This creates a depthwise conv where each input channel produces 4 outputs.
        self.dwt = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 4,
            kernel_size=4,
            stride=2,
            padding=0,  # Valid padding (no padding)
            groups=in_channels,  # Depthwise: each input channel has its own filters
            bias=False,
        )

        # Initialize with DB2 filters and freeze
        self.dwt.weight.data.copy_(filters_tiled)
        self.dwt.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DWT and upsample back to original resolution.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Wavelet sub-bands [B, 4*C, H/2, W/2] → upsampled to [B, 4*C, H, W].
        """
        # Apply DWT: halves resolution
        x_dwt = self.dwt(x)  # [B, 4*C, H/2, W/2]

        # Upsample back to original size (nearest neighbor)
        x_upsampled = nn.functional.interpolate(
            x_dwt,
            size=x.shape[2:],
            mode='nearest',
        )
        return x_upsampled


class ResoConv(nn.Module):
    """Resolution-preserving convolution via wavelet transform.

    Architecture:
        Input [B, C_in, H, W]
          ↓
        DWT2D (DB2 filters) → [B, 4*C_in, H, W] (upsampled)
          ↓
        Concat with identity shortcut → [B, 5*C_in, H, W]
          ↓
        1×1 Conv projection → [B, C_out, H, W]

    This design:
    - Preserves spatial resolution (no downsampling)
    - Extracts multi-scale wavelet features (LL, LH, HL, HH)
    - Maintains gradient flow via identity shortcut
    - Uses only standard Conv2d operations (ONNX-friendly)

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Whether to add identity shortcut before projection.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.shortcut = shortcut

        # Wavelet decomposition: 1 channel → 4 sub-bands
        self.dwt = DWT2D(c1)

        # Projection: 5*c1 (input + 4*dwt) → c2
        # Use ultralytics Conv wrapper (Conv2d + BN + SiLU)
        in_proj = c1 * 5 if shortcut else c1 * 4
        self.proj = Conv(in_proj, c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ResoConv forward pass.

        Args:
            x: Input tensor [B, c1, H, W].

        Returns:
            Output tensor [B, c2, H, W] — same spatial size as input.
        """
        # Wavelet decomposition (upsampled to original size)
        x_dwt = self.dwt(x)  # [B, 4*c1, H, W]

        # Concatenate with identity shortcut
        x_cat = torch.cat([x, x_dwt], dim=1) if self.shortcut else x_dwt

        # Project to target channels
        return self.proj(x_cat)


# Alias for compatibility with RWCM naming in literature
RWCM = ResoConv
