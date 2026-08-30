"""NuLite encoder-decoder + auxiliary segmentation modules.

FastViT (timm) encoder with a NuLite-style ConvTranspose2d decoder that
exposes multi-scale detection features (strides 4/8/16) plus a full-resolution
segmentation feature. The NP (nuclei presence) head produces a binary mask used
as a SAM-style self-prompt conditioning detection features.

NuLite reference: Tommasino et al., Biomedical Signal Processing and Control
(2026). The decoder uses the NuLite convention (Conv2d->BN->ReLU), not the YOLO
SiLU convention, for pretrained-weight compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv2d -> BN -> ReLU (NuLite decoder convention)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv-BN-ReLU."""
        return self.relu(self.bn(self.conv(x)))


class NuLiteEncoderDecoder(nn.Module):
    """FastViT encoder + NuLite 5-block upsample decoder.

    Exposes:
        b3 (stride 4, 64ch)  — P2 detection feature source
        b4 (stride 8, 128ch) — P3 detection feature source
        b5 (stride 16, 256ch) — P4 detection feature source
        b1 (stride 1, 64ch)  — full-resolution segmentation feature
    """

    def __init__(self, variant: str = 'fastvit_s12', pretrained: bool = True, use_decoder: bool = True):
        import timm

        super().__init__()
        self.use_decoder = use_decoder
        self.encoder = timm.create_model(f'{variant}.apple_in1k', features_only=True, pretrained=pretrained)
        dims = list(self.encoder.feature_info.channels())  # [64, 128, 256, 512] for s12
        self.embed_dims = dims
        if not use_decoder:
            return
        # dims order: stride4 -> stride32; decoder consumes reversed (coarse -> fine)
        e3, e2, e1, e0 = dims  # 64, 128, 256, 512

        self.bottleneck_upsampler = nn.Sequential(
            _ConvBlock(e0, e1),
            _ConvBlock(e1, e1),
            nn.ConvTranspose2d(e1, e1, kernel_size=2, stride=2),
        )
        self.decoder4_upsampler = nn.Sequential(
            _ConvBlock(2 * e1, e2),
            _ConvBlock(e2, e2),
            nn.ConvTranspose2d(e2, e2, kernel_size=2, stride=2),
        )
        self.decoder3_upsampler = nn.Sequential(
            _ConvBlock(2 * e2, e3),
            _ConvBlock(e3, e3),
            nn.ConvTranspose2d(e3, e3, kernel_size=2, stride=2),
        )
        self.decoder2_upsampler = nn.Sequential(
            _ConvBlock(2 * e3, e3),
            _ConvBlock(e3, e3),
            nn.ConvTranspose2d(e3, e3, kernel_size=2, stride=2),
        )
        self.decoder1_upsampler = nn.Sequential(
            _ConvBlock(e3, e3),
            nn.ConvTranspose2d(e3, e3, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Encode and (optionally) upsample, returning the multi-scale features.

        With ``use_decoder=False`` the raw FastViT stage features (strides
        4/8/16, channels 64/128/256 for s12) are returned directly as b3/b4/b5
        and ``b1`` is None — the upsample decoder is bypassed entirely.
        """
        feats = list(self.encoder(x))  # [st4, st8, st16, st32]
        if not getattr(self, 'use_decoder', True):
            return {'b3': feats[0], 'b4': feats[1], 'b5': feats[2], 'b1': None}
        z4, z3, z2, z1 = feats[::-1]
        b5 = self.bottleneck_upsampler(z4)
        b4 = self.decoder4_upsampler(torch.cat([b5, z3], dim=1))
        b3 = self.decoder3_upsampler(torch.cat([b4, z2], dim=1))
        b2 = self.decoder2_upsampler(torch.cat([b3, z1], dim=1))
        b1 = self.decoder1_upsampler(b2)
        return {'b3': b3, 'b4': b4, 'b5': b5, 'b1': b1}

    def freeze_encoder(self) -> None:
        """Freeze the FastViT encoder parameters."""
        self.encoder.requires_grad_(False)

    def freeze_decoder(self) -> None:
        """Freeze the upsample decoder (bottleneck/decoder*) parameters."""
        for name, p in self.named_parameters():
            if name.startswith(('bottleneck_', 'decoder')):
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        self.requires_grad_(True)

    def load_nulite_ckpt(self, ckpt_path: str) -> None:
        """Load NuLite-pretrained weights into the encoder + decoder."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        # NuLite stores FastViT under 'encoder.fast_vit.'; we store it under 'encoder.'
        own = self.state_dict()
        remapped = {}
        for k, v in own.items():
            ckpt_key = k
            if k.startswith('encoder.'):
                ckpt_key = 'encoder.fast_vit.' + k[len('encoder.') :]
            if ckpt_key in state and state[ckpt_key].shape == v.shape:
                remapped[k] = state[ckpt_key]
        self.load_state_dict(remapped, strict=False)


class NPHead(nn.Module):
    """Binary nuclei-presence head producing raw logits (B, 1, H, W)."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce raw nuclei-presence logits (no sigmoid)."""
        x = self.relu(self.bn(self.conv1(x)))
        return self.conv2(x)


def _coord_channels(h: int, w: int, device, dtype) -> torch.Tensor:
    """Return (1, 2, h, w) normalized xy coordinate grid breaking translation invariance."""
    gx = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    gy = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(gy, gx, indexing='ij')
    return torch.stack([gx, gy], dim=0).unsqueeze(0)


class SAMConditioner(nn.Module):
    """SAM-style self-prompt conditioning: gate detection features by the NP mask.

    Concats [np_prob (1), coords (2), det_feat (C)] -> 1x1 conv -> residual add.
    Zero-init makes the conditioner an identity at the start of training.

    ``gate_scale`` bounds the gate's contribution (``cond = det + gate_scale * fuse``).
    Lowering it forces the detection head to learn discrimination on its own
    instead of leaning on the NP amplification (which inflates same-scale
    duplicate detections — see cond_ablate.py diagnostic).
    """

    def __init__(self, det_channels: int, gate_scale: float = 1.0):
        super().__init__()
        self.gate_scale = gate_scale
        self.fuse = nn.Conv2d(1 + 2 + det_channels, det_channels, kernel_size=1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, np_logits: torch.Tensor, det_feat: torch.Tensor) -> torch.Tensor:
        """Gate det_feat by the (interpolated, sigmoid) NP mask."""
        gate_scale = getattr(self, 'gate_scale', 1.0)  # old checkpoints lack the attr
        _, _, h, w = det_feat.shape
        np_prob = F.interpolate(np_logits, size=(h, w), mode='bilinear', align_corners=False).sigmoid()
        coords = _coord_channels(h, w, det_feat.device, det_feat.dtype).expand(np_prob.shape[0], -1, -1, -1)
        x = torch.cat([np_prob, coords, det_feat], dim=1)
        return det_feat + gate_scale * self.fuse(x)


class PANBottomUp(nn.Module):
    """Bottom-up PAN path over NuLite decoder features (P2->P3->P4).

    Uses YOLO Conv (stride=2 downsampling) + C3k2 fusion, then 1x1 expansion
    to the same channel widths the v1 detection head expects: [128, 256, 512].
    """

    def __init__(self):
        from ultralytics.nn.modules.block import C3k2
        from ultralytics.nn.modules.conv import Conv

        super().__init__()
        self.n3_down = Conv(64, 128, 3, s=2)
        self.n3_fuse = C3k2(256, 128, n=1)
        self.n4_down = Conv(128, 256, 3, s=2)
        self.n4_fuse = C3k2(512, 256, n=1)
        self.expand2 = Conv(64, 128, 1)
        self.expand3 = Conv(128, 256, 1)
        self.expand4 = Conv(256, 512, 1)

    def forward(self, b3: torch.Tensor, b4: torch.Tensor, b5: torch.Tensor) -> tuple:
        """Fuse decoder features bottom-up and expand to detection channels."""
        n2 = b3
        n3 = self.n3_fuse(torch.cat([self.n3_down(n2), b4], dim=1))
        n4 = self.n4_fuse(torch.cat([self.n4_down(n3), b5], dim=1))
        return self.expand2(n2), self.expand3(n3), self.expand4(n4)
