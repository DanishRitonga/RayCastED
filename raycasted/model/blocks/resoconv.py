"""ResoConv: Wavelet-based downsampling convolution.

Replaces strided 3x3 Conv in backbone/neck with DWT-based downsampling:
- Configurable wavelet (db2, bior2.2, etc.) splits input into 4 sub-bands (LL, LH, HL, HH)
- Concatenate sub-bands with identity shortcut → 1x1 Conv projection
- Output is at half spatial resolution (H/2 x W/2) — same as strided Conv
- Explicitly preserves high-frequency information in LH, HL, HH sub-bands

Supported wavelets:
- db2: 4-tap asymmetric, orthogonal, sharper edge response (best for H&E cells)
- bior2.2: 6-tap symmetric, linear phase, smoother sub-bands

Dual-stream architecture (ResoConvDS):
- LL stream: DWConv 3×3 + 1x1 projection → continues through backbone
- HF stream: SE attention → 1x1 projection → stored for neck injection
- HFResidual: At neck, adds boundary_conv(hf) to semantic features as residual
- Boundary detail bypasses C3k2_LK smoothing, injected at matching stride

References:
- WaveCNet: Williams & Li, "Wavelet-based Pooling for Deep CNNs", CVPR 2020
- DWT-UNet: DWT-based U-Net for Medical Image Segmentation, IEEE 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

# Hardcoded filter fallbacks (used when pywt is unavailable)
_DB2_FILTERS = {
    'lo': [-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416],
    'hi': [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037],
}

_BIOR22_FILTERS = {
    'lo': [0.0, -0.1767766952966369, 0.3535533905932738, 1.0606601717798212, 0.3535533905932738, -0.1767766952966369],
    'hi': [-0.0, 0.3535533905932738, -0.7071067811865476, 0.3535533905932738, -0.0, 0.0],
}


def _get_wavelet_1d_filters(wavelet_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 1D decomposition filters (lo, hi) for the given wavelet.

    Args:
        wavelet_type: Wavelet name ('db2', 'bior2.2').

    Returns:
        (lo, hi) tensors of shape [filter_len].
    """
    try:
        import pywt

        w = pywt.Wavelet(wavelet_type)
        lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        hi = torch.tensor(w.dec_hi, dtype=torch.float32)
        return lo, hi
    except ImportError:
        if wavelet_type == 'db2':
            lo = torch.tensor(_DB2_FILTERS['lo'], dtype=torch.float32)
            hi = torch.tensor(_DB2_FILTERS['hi'], dtype=torch.float32)
        elif wavelet_type == 'bior2.2':
            lo = torch.tensor(_BIOR22_FILTERS['lo'], dtype=torch.float32)
            hi = torch.tensor(_BIOR22_FILTERS['hi'], dtype=torch.float32)
        else:
            raise ValueError(f'Unknown wavelet_type={wavelet_type} and pywt unavailable for lookup')
        return lo, hi


def _create_wavelet_filters(
    wavelet_type: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    """Create 2D wavelet filters for conv2d weight initialization.

    Constructs 4 filters (LL, LH, HL, HH) by outer product of 1D filters.
    Output shape: [4, 1, K, K] where K is the filter length.

    Args:
        wavelet_type: Wavelet name ('db2', 'bior2.2').
        device: Target device for filters.
        dtype: Target dtype for filters.

    Returns:
        (filters, pad) where filters is [4, 1, K, K] and pad is the
        reflection padding needed for exact H/2 output with stride=2.
    """
    lo, hi = _get_wavelet_1d_filters(wavelet_type)
    lo = lo.to(device=device, dtype=dtype)
    hi = hi.to(device=device, dtype=dtype)

    ll = lo[:, None] * lo[None, :]
    lh = hi[:, None] * lo[None, :]
    hl = lo[:, None] * hi[None, :]
    hh = hi[:, None] * hi[None, :]

    filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # [4, 1, K, K]
    pad = (lo.shape[0] - 2) // 2
    return filters, pad


class SE(nn.Module):
    """Squeeze-and-Excitation block using AdaptiveAvgPool2d.

    Image-size agnostic: global avg pool reduces any (H,W) to (1,1) before FC layers.

    Args:
        channels: Number of input/output channels.
        reduction: Channel reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention: x * sigmoid(fc(avg_pool(x)))."""  # noqa: D401
        return x * self.fc(self.pool(x))


class DWT2D(nn.Module):
    """2D Discrete Wavelet Transform using configurable wavelet filters.

    Splits input into 4 sub-bands via depthwise conv2d:
    - LL: Low-low (approximation)
    - LH: Low-high (horizontal edges)
    - HL: High-low (vertical edges)
    - HH: High-high (diagonal edges)

    Output layout is grouped: [LL_0..LL_C, LH_0..LH_C, HL_0..HL_C, HH_0..HH_C].
    This allows clean slicing: ll = out[:, :C], hf = out[:, C:].

    Args:
        in_channels: Number of input channels.
        drop_hh: If True, discard HH sub-band and return 3 sub-bands only.
        wavelet_type: Wavelet name ('db2', 'bior2.2').
    """

    def __init__(
        self,
        in_channels: int,
        drop_hh: bool = False,
        wavelet_type: str = 'bior2.2',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.drop_hh = drop_hh
        self.wavelet_type = wavelet_type

        filters, pad = _create_wavelet_filters(wavelet_type, device='cpu', dtype=torch.float32)
        filters_tiled = filters.repeat(in_channels, 1, 1, 1)

        self.register_buffer('dwt_weight', filters_tiled, persistent=True)
        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D DWT via depthwise conv2d with reflection padding.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Wavelet sub-bands [B, 3*C, H/2, W/2] (drop_hh) or [B, 4*C, H/2, W/2].
            Grouped layout: all LL channels first, then LH, HL, (HH).
        """
        p = self._pad
        x = F.pad(x, (p, p, p, p), mode='reflect')
        out = F.conv2d(
            x,
            self.dwt_weight,
            bias=None,
            stride=2,
            padding=0,
            groups=self.in_channels,
        )
        batch, _, h2, w2 = out.shape
        c = self.in_channels
        out = out.view(batch, c, 4, h2, w2).permute(0, 2, 1, 3, 4).reshape(batch, -1, h2, w2)
        if self.drop_hh:
            out = out[:, : 3 * c]
        return out


class DWT_LL(nn.Module):
    """Extract LL (low-low approximation) sub-band from DWT.

    Runs DWT and returns only the LL sub-band (low-pass approximation).
    Output channels = input channels, spatial = H/2 x W/2.

    Args:
        c1: Input channels.
        wavelet_type: Wavelet family ('db2', 'bior2.2').
    """

    def __init__(self, c1: int, wavelet_type: str = 'bior2.2'):
        super().__init__()
        self.dwt = DWT2D(c1, drop_hh=False, wavelet_type=wavelet_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DWT and return LL sub-band only."""
        x_dwt = self.dwt(x)
        return x_dwt[:, : self.dwt.in_channels]


class DWT_HF(nn.Module):
    """Extract HF (LH, HL, HH) sub-bands from DWT.

    Runs DWT and returns only the high-frequency sub-bands.
    Output channels = 3*C_in (or 2*C_in if drop_hh), spatial = H/2 x W/2.

    Args:
        c1: Input channels.
        drop_hh: If True, discard HH and return only LH+HL.
        wavelet_type: Wavelet family ('db2', 'bior2.2').
    """

    def __init__(self, c1: int, drop_hh: bool = False, wavelet_type: str = 'bior2.2'):
        super().__init__()
        self.dwt = DWT2D(c1, drop_hh=drop_hh, wavelet_type=wavelet_type)
        self.n_hf = 2 if drop_hh else 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DWT and return HF sub-bands only."""
        x_dwt = self.dwt(x)
        return x_dwt[:, self.dwt.in_channels :]


class ResoConv(nn.Module):
    """Wavelet-based downsampling convolution via bior2.2 DWT (single-stream).

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
        wavelet_type: Wavelet family ('db2', 'bior2.2').
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, drop_hh: bool = False, wavelet_type: str = 'bior2.2'):
        super().__init__()
        self.shortcut = shortcut

        self.dwt = DWT2D(c1, drop_hh=drop_hh, wavelet_type=wavelet_type)

        n_sub = 3 if drop_hh else 4
        in_proj = c1 * (1 + n_sub) if shortcut else c1 * n_sub
        self.proj = Conv(in_proj, c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ResoConv: DWT → concat shortcut → 1×1 project."""
        x_dwt = self.dwt(x)

        if self.shortcut:
            x_down = F.avg_pool2d(x, kernel_size=2, stride=2)
            x_cat = torch.cat([x_down, x_dwt], dim=1)
        else:
            x_cat = x_dwt

        return self.proj(x_cat)


class ResoConvDS(nn.Module):
    """Dual-stream wavelet downsampling: LL → backbone, HF → neck skip.

    Splits DWT output into two streams:
    - LL stream: DWConv 3×3 + BN + SiLU + 1×1 Conv → returned (backbone continues)
    - HF stream: SE attention + 1×1 Conv → stored as self._hf for HFResidual

    The HF stream preserves boundary detail that would otherwise be destroyed
    by C3k2_LK's large kernels. HFResidual injects it at the neck as a residual.

    Args:
        c1: Input channels.
        c2: Output channels (for both LL and HF streams).
        se_r: SE reduction ratio for HF attention.
        drop_hh: If True, discard HH sub-band (only LH+HL in HF stream).
        wavelet_type: Wavelet family ('db2', 'bior2.2').
    """

    def __init__(self, c1: int, c2: int, se_r: int = 4, drop_hh: bool = False, wavelet_type: str = 'bior2.2'):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.drop_hh = drop_hh
        n_hf = 2 if drop_hh else 3

        self.dwt = DWT2D(c1, drop_hh=drop_hh, wavelet_type=wavelet_type)

        self.hf_se = SE(c1 * n_hf, reduction=se_r)
        self.hf_proj = Conv(c1 * n_hf, c2, k=1)

        self.ll_conv = nn.Conv2d(c1, c1, 3, padding=1, bias=False, groups=c1)
        self.ll_bn = nn.BatchNorm2d(c1)
        self.ll_proj = Conv(c1, c2, k=1)

        self._hf = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual-stream DWT: return ll, store _hf."""
        x_dwt = self.dwt(x)
        ll = x_dwt[:, : self.c1]
        hf = x_dwt[:, self.c1 :]

        self._hf = self.hf_proj(self.hf_se(hf))
        ll_out = self.ll_proj(F.silu(self.ll_bn(self.ll_conv(ll))))
        return ll_out


class HFResidual(nn.Module):
    """Inject HF boundary detail as residual to semantic features.

    Reads self._hf from a linked ResoConvDS source, processes through
    a lightweight boundary path (DWConv 3×3 + 1×1), and adds to input:

        output = input + boundary_conv(source._hf)

    Builder links self._source to the correct ResoConvDS after construction.

    Args:
        c2: Semantic feature channel count (output of this module).
        source_c2: HF stream channel count from ResoConvDS._hf (input to boundary_conv).
    """

    def __init__(self, c2: int, source_c2: int):
        super().__init__()
        self._source = None
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(source_c2, source_c2, 3, padding=1, bias=False, groups=source_c2),
            nn.BatchNorm2d(source_c2),
            nn.SiLU(inplace=True),
            Conv(source_c2, c2, k=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add boundary-processed HF as residual: x + boundary_conv(source._hf)."""
        return x + self.boundary_conv(self._source._hf)
