"""ResoConv: Wavelet-based downsampling convolution with SE channel attention.

Replaces strided 3x3 Conv in backbone/neck with DWT-based downsampling:
- Haar wavelet (default) splits input into 4 sub-bands (LL, LH, HL, HH)
- SE (Squeeze-and-Excitation) channel attention re-weights sub-band mix
- 1×1 Conv projection
- Output is at half spatial resolution (H/2 x W/2) — same as strided Conv
- Explicitly preserves high-frequency information in LH, HL, HH sub-bands

Supported wavelets (configurable per layer via yaml):
- haar: 2-tap, simple, fast (default, used by v1.1)
- db2: 4-tap asymmetric, orthogonal, sharper edge response
- bior2.2: 6-tap symmetric, linear phase, smoother sub-bands

Backward compatibility:
    `ResoConvHybrid = ResoConv` alias is exported so existing checkpoints
    serialised with the old class name (docs/runs/v1.1/best.pt, etc.)
    continue to load without migration.

References:
- WaveCNet: Williams & Li, "Wavelet-based Pooling for Deep CNNs", CVPR 2020
- SE: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
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
    'lo': [0.0, 0.1767767, 0.35355338, 1.06066, 1.06066, 0.35355338, 0.1767767, 0.0],
    'hi': [0.0, 0.35355338, -0.70710677, 0.35355338, 0.35355338, -0.70710677, 0.35355338, 0.0],
}

_HAAR_FILTERS = {
    'lo': [1.0 / 1.414213562, 1.0 / 1.414213562],
    'hi': [1.0 / 1.414213562, -1.0 / 1.414213562],
}


def _get_wavelet_1d_filters(wavelet_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 1D decomposition filters (lo, hi) for the given wavelet.

    Args:
        wavelet_type: Wavelet name ('haar', 'db2', 'bior2.2').

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
        elif wavelet_type == 'haar':
            lo = torch.tensor(_HAAR_FILTERS['lo'], dtype=torch.float32)
            hi = torch.tensor(_HAAR_FILTERS['hi'], dtype=torch.float32)
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
        wavelet_type: Wavelet name ('haar', 'db2', 'bior2.2').
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
        wavelet_type: Wavelet name ('haar', 'db2', 'bior2.2').
    """

    def __init__(
        self,
        in_channels: int,
        drop_hh: bool = False,
        wavelet_type: str = 'haar',
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


class DWT2D_Hybrid(nn.Module):
    """DWT split into LL (approximation) and HF (detail) sub-bands.

    Runs a single DWT pass internally and returns the concatenation
    [LL, HF] where LL is the low-low sub-band and HF is the LH+HL+HH
    sub-bands concatenated. Used by ResoConv.

    Args:
        in_channels: Number of input channels.
        drop_hh: If True, discard HH and HF has only LH+HL (2 sub-bands).
        wavelet_type: Wavelet family ('haar', 'db2', 'bior2.2').
    """

    def __init__(self, in_channels: int, drop_hh: bool = False, wavelet_type: str = 'haar'):
        super().__init__()
        self.in_channels = in_channels
        self.drop_hh = drop_hh
        self.dwt_ll = DWT2D(in_channels, drop_hh=False, wavelet_type=wavelet_type)
        self.dwt_hf = DWT2D(in_channels, drop_hh=drop_hh, wavelet_type=wavelet_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll = self.dwt_ll(x)
        x_hf = self.dwt_hf(x)
        ll = x_ll[:, : self.in_channels]
        hf = x_hf[:, self.in_channels :]
        return torch.cat([ll, hf], dim=1)


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    GAP → FC(reduce) → ReLU → FC(restore) → Sigmoid → gate.
    2D FC weights — compatible with Muon optimizer.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        att = self.gap(x).view(b, c)
        return x * self.fc(att).view(b, c, 1, 1)


class ResoConv(nn.Module):
    """Wavelet downsampling with DWT + SE channel attention.

    Haar wavelet (default) splits each channel into 4 sub-bands:
      LL = (x₀+x₁)/2  (approximation, low-pass)
      LH = (x₀−x₁)/2  (horizontal detail, high-pass)
      HL, HH           (vertical/diagonal detail)
    SE re-weights channels dynamically based on global context.
    1×1 Conv projects to the target channel count.

    Replaces strided 3×3 Conv with frequency-aware downsampling that
    explicitly preserves high-frequency boundary information.

    Args:
        c1: Input channels.
        c2: Output channels.
        wavelet_type: Wavelet family ('haar', 'db2', 'bior2.2'). Default 'haar'.
        shortcut: Reserved for backward-compat with old checkpoints. No-op
            (kept only to absorb the kwarg silently if old code passes it).
        drop_hh: If True, discard HH sub-band before SE/projection.
    """

    def __init__(self, c1: int, c2: int, wavelet_type: str = 'haar', shortcut: bool = False, drop_hh: bool = False):
        super().__init__()
        self.shortcut = shortcut
        self.dwt = DWT2D_Hybrid(c1, drop_hh=drop_hh, wavelet_type=wavelet_type)

        n_sub = 3 if drop_hh else 4
        in_proj = c1 * n_sub
        self.se = SE(in_proj)
        self.proj = Conv(in_proj, c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ResoConv: DWT → SE → 1×1 project."""
        out = self.dwt(x)
        if hasattr(self, 'se'):
            out = self.se(out)
        return self.proj(out)


# Backward-compatibility alias — old checkpoints (docs/runs/v1.1/best.pt and
# any runs/detect/train*/weights/*.pt) were serialised with the class name
# `ResoConvHybrid`. The alias lets torch.load() resolve them to the renamed
# `ResoConv` class without migration. Safe to remove once all checkpoints are
# re-exported or no longer needed.
ResoConvHybrid = ResoConv
