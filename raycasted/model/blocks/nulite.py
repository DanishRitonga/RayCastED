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
    """Conv2d (bias) -> BN -> ReLU -> Dropout, mirroring NuLite's Conv2DBlock.

    NuLite reference (models/utils.py:4): Conv2d(bias=True, k=3, stride=1,
    padding=1) -> BatchNorm2d -> ReLU(inplace) -> Dropout(drop_rate). The
    state-dict layout (``self.block = Sequential(conv, bn, relu, dropout)``)
    must be kept identical so NuLite-pretrained decoder weights load.

    When ``separable=True`` the conv is replaced by a depthwise-separable
    factorization (MobileNet-style): 3x3 depthwise (groups=in) -> BN -> ReLU
    -> 1x1 pointwise -> BN -> ReLU -> Dropout. This preserves per-channel 3x3
    spatial mixing while cutting params from ``in*out*9`` to ``9*in + in*out``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        separable: bool = False,
    ):
        super().__init__()
        self.separable = separable
        if separable:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv-BN-ReLU-dropout."""
        return self.block(x)


class NuLiteEncoderDecoder(nn.Module):
    """FastViT encoder + NuLite 5-block upsample decoder.

    Exposes:
        b3 (stride 4, 64ch)  — P2 detection feature source
        b4 (stride 8, 128ch) — P3 detection feature source
        b5 (stride 16, 256ch) — P4 detection feature source
        b1 (stride 1, 64ch)  — full-resolution segmentation feature
    """

    def __init__(
        self,
        variant: str = 'fastvit_s12',
        pretrained: bool = True,
        use_decoder: bool = True,
        separable_decoder: bool = False,
    ):
        import timm

        super().__init__()
        self.use_decoder = use_decoder
        self.separable_decoder = separable_decoder
        self.encoder = timm.create_model(f'{variant}.apple_in1k', features_only=True, pretrained=pretrained)
        dims = list(self.encoder.feature_info.channels())  # [64, 128, 256, 512] for s12
        self.embed_dims = dims
        if not use_decoder:
            return
        # dims order: stride4 -> stride32; decoder consumes reversed (coarse -> fine)
        e3, e2, e1, e0 = dims  # 64, 128, 256, 512

        # NuLite create_upsampling_branch (models/nulite.py:188-264), exact channels.
        self.bottleneck_upsampler = nn.Sequential(
            _ConvBlock(e0, e1, separable=separable_decoder),
            nn.ConvTranspose2d(e1, e1, kernel_size=2, stride=2),
        )
        self.decoder4_upsampler = nn.Sequential(
            _ConvBlock(2 * e1, e1, separable=separable_decoder),
            _ConvBlock(e1, e2, separable=separable_decoder),
            nn.ConvTranspose2d(e2, e2, kernel_size=2, stride=2),
        )
        self.decoder3_upsampler = nn.Sequential(
            _ConvBlock(2 * e2, e2, separable=separable_decoder),
            _ConvBlock(e2, e3, separable=separable_decoder),
            nn.ConvTranspose2d(e3, e3, kernel_size=2, stride=2),
        )
        self.decoder2_upsampler = nn.Sequential(
            _ConvBlock(2 * e3, e3, separable=separable_decoder),
            nn.ConvTranspose2d(e3, e3, kernel_size=2, stride=2),
        )
        self.decoder1_upsampler = nn.Sequential(
            _ConvBlock(e3, e3, separable=separable_decoder),
            nn.ConvTranspose2d(e3, e3, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor, collect_fusion: bool = False) -> dict:
        """Encode and (optionally) upsample, returning the multi-scale features.

        With ``use_decoder=False`` the raw FastViT stage features (strides
        4/8/16, channels 64/128/256 for s12) are returned directly as b3/b4/b5
        and ``b1`` is None — the upsample decoder is bypassed entirely.

        With ``collect_fusion=True`` the dict additionally carries the raw
        fusion tensors tapped for detection adapters (pre-ConvBlock, so both
        the backbone skip and the upsampled context stay linearly separable):
        ``fuse5`` = z4 (bottleneck input, no skip exists at this level),
        ``fuse4`` = cat([z3, b5]) (512ch at stride 8) and
        ``fuse3`` = cat([z2, b4]) (384ch at stride 4).
        """
        feats = list(self.encoder(x))  # [st4, st8, st16, st32]
        if not getattr(self, 'use_decoder', True):
            return {'b3': feats[0], 'b4': feats[1], 'b5': feats[2], 'b1': None, 'top': feats[3]}
        z4, z3, z2, z1 = feats[::-1]
        fuse5 = z4
        b5 = self.bottleneck_upsampler(z4)
        fuse4 = torch.cat([z3, b5], dim=1)
        b4 = self.decoder4_upsampler(fuse4)
        fuse3 = torch.cat([z2, b4], dim=1)
        b3 = self.decoder3_upsampler(fuse3)
        b2 = self.decoder2_upsampler(torch.cat([z1, b3], dim=1))
        b1 = self.decoder1_upsampler(b2)
        out = {'b3': b3, 'b4': b4, 'b5': b5, 'b1': b1, 'top': feats[3]}
        if collect_fusion:
            out.update({'fuse3': fuse3, 'fuse4': fuse4, 'fuse5': fuse5})
        return out

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
        # NuLite stores FastViT under 'encoder.fast_vit.' (we store 'encoder.')
        # and the decoder under 'decoder.<stage>.' (we store '<stage>.' directly).
        own = self.state_dict()
        remapped = {}
        for k, v in own.items():
            ckpt_key = k
            if k.startswith('encoder.'):
                ckpt_key = 'encoder.fast_vit.' + k[len('encoder.') :]
            elif k.startswith(
                (
                    'bottleneck_upsampler.',
                    'decoder4_upsampler.',
                    'decoder3_upsampler.',
                    'decoder2_upsampler.',
                    'decoder1_upsampler.',
                )
            ):
                ckpt_key = 'decoder.' + k
            if ckpt_key in state and state[ckpt_key].shape == v.shape:
                remapped[k] = state[ckpt_key]
        missing = [k for k in own if k not in remapped]
        print(
            f'NuLite ckpt load: {len(remapped)}/{len(own)} keys matched'
            + (f' (missing {len(missing)}, e.g. {missing[:4]})' if missing else '')
        )
        self.load_state_dict(remapped, strict=False)


class NuLiteSegOracle(nn.Module):
    """Frozen NuLite-T binary, centroid, or dual-output oracle.

    Replicates the trained NuLite-T U-Net EXACTLY (encoder + decoder0 + full
    upsample decoder + configurable np_head) so a fold1-only oracle checkpoint
    saved by ``train_nulite_seg.py`` loads wholesale (state_dict key-for-key,
    622 keys for the separable-decoder variant). Binary checkpoints produce
    two CE logits; centroid-potential checkpoints produce one direct logit.
    """

    def __init__(
        self,
        variant: str = 'fastvit_s12',
        separable_decoder: bool = True,
        output_channels: int = 2,
        oracle_kind: str = 'binary',
    ):
        super().__init__()
        if output_channels not in (1, 2):
            raise ValueError('NuLiteSegOracle output_channels must be 1 or 2')
        if oracle_kind not in {'binary', 'center_potential', 'fg_centroid'}:
            raise ValueError("oracle_kind must be 'binary', 'center_potential', or 'fg_centroid'")
        expected_channels = 1 if oracle_kind == 'center_potential' else 2
        if output_channels != expected_channels:
            raise ValueError(f'oracle kind {oracle_kind!r} requires {expected_channels} output channels')
        self.output_channels = int(output_channels)
        self.oracle_kind = oracle_kind
        self.encoder_decoder = NuLiteEncoderDecoder(
            variant=variant, pretrained=False, use_decoder=True, separable_decoder=separable_decoder
        )
        emb0 = self.encoder_decoder.embed_dims[0]  # 64
        self.decoder0 = _ConvBlock(3, emb0, 3, separable=False)
        self.np_head = nn.Sequential(
            _ConvBlock(2 * emb0, emb0),
            nn.Conv2d(emb0, self.output_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full U-Net and return binary or centroid-potential logits."""
        d = self.encoder_decoder(x)
        x0 = self.decoder0(x)
        xt = torch.cat([x0, d['b1']], dim=1)
        return self.np_head(xt)

    def load_ckpt(self, ckpt_path: str) -> None:
        """Load a NuLite-T oracle checkpoint into all four submodules.

        Remaps our key layout onto the NuLite-T state-dict: ``encoder_decoder.
        encoder.*`` -> ``encoder.fast_vit.*``, ``encoder_decoder.<branch>.*`` ->
        ``decoder.<branch>.*``, ``decoder0.block.*`` -> ``decoder0.0.block.*``
        (oracle decoder0 is a Sequential of one block). ``np_head.*`` matches
        exactly and needs no remap.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        config = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
        checkpoint_kind = config.get('oracle_kind')
        expected_kind = self.oracle_kind
        if checkpoint_kind is not None and checkpoint_kind != expected_kind:
            raise ValueError(f'oracle checkpoint kind is {checkpoint_kind!r}, but model expects {expected_kind!r}')
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        own = self.state_dict()
        remapped = {}
        for k, v in own.items():
            ck = k
            if k.startswith('encoder_decoder.encoder.'):
                ck = 'encoder.fast_vit.' + k[len('encoder_decoder.encoder.') :]
            elif k.startswith('encoder_decoder.'):
                ck = 'decoder.' + k[len('encoder_decoder.') :]
            elif k.startswith('decoder0.'):
                ck = 'decoder0.0.' + k[len('decoder0.') :]
            if ck in state and state[ck].shape == v.shape:
                remapped[k] = state[ck]
        missing = [k for k in own if k not in remapped]
        print(
            f'NuLite seg-oracle load: {len(remapped)}/{len(own)} keys matched'
            + (f' (missing {len(missing)}, e.g. {missing[:4]})' if missing else '')
        )
        if missing:
            raise ValueError(
                f'oracle checkpoint is incompatible: {len(missing)} parameters were not loaded; '
                f'first keys: {missing[:4]}'
            )
        self.load_state_dict(remapped, strict=False)


class _DeformConv2d(nn.Module):
    """Deformable conv with internally-computed offsets (mmcv DCN-style).

    torchvision's DeformConv2d requires an explicit ``offset`` argument, so we
    wrap it and compute the offset (2*deform_groups*k*k channels) via a
    zero-initialized 3x3 conv, matching SPAR-Det's mmcv DCN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        deform_groups: int = 1,
    ):
        from torchvision.ops import DeformConv2d

        super().__init__()
        self.deform = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        offset_channels = 2 * deform_groups * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(in_channels, offset_channels, kernel_size, stride, padding)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute offsets and apply the deformable convolution."""
        offset = self.conv_offset(x)
        return self.deform(x, offset)


class Branch3Gate(nn.Module):
    """SPAR-Det CAHF fusion gate (foreground/background disentanglement).

    Multiplicatively gates the detection feature ``x`` with a dual
    (deformable-spatial x SE-channel) attention map derived from the FUSED
    [det, seg] features. Faithful port of SPAR-Det's Branch3 (cahf.py:207-282):

        px, py = conv1_1(x), conv1_1(y)
        processed = cat[px, py]              # in//2 + in//2 == in channels
        out = spatial_attention(processed) * channel_attention(processed) * x

    ``x`` is the trainable detection-stream feature, ``y`` the frozen
    segmentation-stream feature at the same scale (equal channels). The frozen
    seg stream supplies the FG/BG identity; the gate selectively amplifies
    foreground and suppresses background in the det stream (replacing ``x``,
    not adding to it).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.spatial_attention = nn.Sequential(
            _DeformConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_channels, 2 * in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * in_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gate det feature ``x`` by a dual attention map fused from [det, seg]."""
        px = self.conv1_1(x)
        py = self.conv1_1(y)
        processed = torch.cat([px, py], dim=1)
        spatial = self.spatial_attention(processed)
        channel = self.channel_attention(processed).unsqueeze(-1).unsqueeze(-1)
        return spatial * channel * x


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
