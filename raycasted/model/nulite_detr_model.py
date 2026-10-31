"""NuLite-RayCast DETR detection model.

FastViT s12 encoder + NuLite decoder (seg backbone) feeding a single query-based
DETR head. Queries are seeded from local maxima of the NP seed map (InstanSeg
style self-prompting) instead of FCN multi-scale anchors. Uses the
NuLiteDETRLoss (Hungarian 1:1 matching + VFL + ray-L1 + polar-IoU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel

from raycasted.model.blocks.nulite import NPHead, NuLiteEncoderDecoder, NuLiteSegOracle, _ConvBlock
from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder
from raycasted.model.nulite_detr_loss import NuLiteDETRLoss


class DenseAuxHead(nn.Module):
    """Training-only semantic supervision for the trainable detection pyramid."""

    def __init__(self, channels: tuple[int, int, int], nc: int, hidden: int = 32):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, hidden, 1) for c in channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.objectness = nn.Conv2d(hidden, 1, 1)
        self.types = nn.Conv2d(hidden, nc, 1)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Fuse the three pyramid levels and return objectness and type logits."""
        size = features[0].shape[-2:]
        projected = [self.projections[0](features[0])]
        projected.extend(
            F.interpolate(proj(feat), size=size, mode='bilinear', align_corners=False)
            for proj, feat in zip(self.projections[1:], features[1:])
        )
        fused = self.fuse(torch.cat(projected, dim=1))
        return self.objectness(fused), self.types(fused)


class QueryAppearanceHead(nn.Module):
    """Build a supervised stride-2 appearance map from b1 and b3."""

    def __init__(self, b1_channels: int, b3_channels: int, channels: int = 32):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(b1_channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Conv2d(b3_channels, channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.mask = nn.Conv2d(channels, 1, 1)

    def forward(self, b1, b3):
        """Return the fused appearance representation and union-mask logits."""
        high = self.b1(b1)
        low = F.interpolate(self.b3(b3), size=high.shape[-2:], mode='bilinear', align_corners=False)
        appearance = self.fuse(torch.cat((high, low), 1))
        return appearance, self.mask(appearance)


class FusionAdapter(nn.Module):
    """DW 3x3 -> BN -> GELU -> PW 1x1 -> ConvT 2x2 upsampling adapter.

    Sits between the frozen oracle U-Net's raw fusion concats and the DETR
    input projections. The tapped concats sit one level below the level they
    feed (pre-ConvTranspose), so each adapter also restores the native stride
    of its target level. Module-level (not a local class) so ultralytics can
    pickle the whole model object in checkpoints.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.up = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a fusion concat to the target level's native channels/stride."""
        return self.up(self.pw(F.gelu(self.bn(self.dw(x)))))


class NuLiteRayCastDETRModel(DetectionModel):
    """Single-head DETR detector on the NuLite seg backbone.

    Submodules (plain-list self.model ending in the DETR decoder so that
    model.model[-1] is the head, matching the v1 framework contract):
      encoder_decoder -> NuLiteEncoderDecoder (b3/b4/b5/b1)
      decoder0        -> Conv(3, dims[0], 3)    [use_decoder only]
      np_head         -> NPHead(2*dims[0]) (seed logits at full res) [use_decoder only]
      detr            -> NuLiteDETRDecoder (nc, ch=(dims[0],dims[1],dims[2]), nq=300, ndl=3)

    Channel dimensions derive from the encoder variant (``dims``), so any
    FastViT size works (s12: 15.46M params, t8: ~8.89M).

    With ``use_decoder=False`` the NuLite upsample decoder + NP seed head are
    dropped entirely: the DETR head reads the raw FastViT stage features
    (strides 4/8/16). ~12.2M params vs 15.46M with the decoder.
    """

    def __init__(
        self,
        nc: int = 5,
        n_rays: int = 64,
        variant: str = 'fastvit_s12',
        det_variant: str = 'fastvit_t12',
        pretrained: bool = True,
        lambda_seg: float = 1.0,
        nq: int = 300,
        ndl: int = 3,
        hd: int = 256,
        d_ffn: int = 1024,
        seed_threshold: float = 0.5,
        peak_distance: int = 3,
        grid_size: float = 0.05,
        query_selection: str = 'grid',
        seed_feature: bool = True,
        no_object: bool = True,
        seed_in_content: bool = True,
        no_object_weight: float = 1.0,
        cls_weight: float = 2.0,
        piou_weight: float = 3.0,
        xy_weight: float = 500.0,
        mds: bool = True,
        cost_inside: float = 10.0,
        per_layer_match: bool = False,
        duplicate_train_topk: int = 1,
        detr_lsp: bool = False,
        detr_matcher_lsp: bool = False,
        detr_layer_schedule: str = '',
        hybrid_peaks: int = 100,
        seed_level: bool = False,
        seed_level_stride: int = 4,
        query_block_size: float = 0.0,
        content_top: bool = False,
        seed_resize: str = 'bilinear',
        seed_resize_frac: float = 0.25,
        seed_map_target: bool = True,
        oracle_kind: str = 'binary',
        use_decoder: bool = True,
        shared_layers: bool = False,
        shared_layer_groups: int = 0,
        separable_decoder: bool = False,
        backbone: str = 'nulite',
        use_seed: bool = True,
        yolo11_scale: str = 'n',
        yolo11_ckpt: str | None = None,
        recast: bool = False,
        recast_after_layer: int = 3,
        recast_radius_px: float = 4.0,
        respawn: bool = False,
        respawn_peaks: int = 64,
        respawn_min_seed: float = 0.5,
        aux_dense_weight: float = 0.0,
        aux_dense_type_weight: float = 1.0,
        seg_fg_margin_weight: float = 0.0,
        seg_fg_margin: float = 1.0,
        seg_fg_radius_px: float = 4.0,
        seg_fg_margin_layer_weights: list[float] | tuple[float, ...] | None = None,
        viable_owner_fg_weight: float = 0.0,
        viable_owner_iou: float = 0.5,
        viable_owner_margin: float = 1.0,
        proposal_bal_weight: float = 0.0,
        proposal_bal_cap: float = 5.0,
        nucleus_cdn: int = 0,
        nucleus_cdn_weight: float = 0.0,
        nucleus_cdn_class_embed: bool = False,
        nucleus_cdn_real_to_dn: bool = False,
        frozen_adapter: bool = False,
        seed_in_content_pooled: bool = False,
        seed_in_content_pool_sizes: tuple = (3, 5, 9),
        map_modulation: bool = False,
        score_size_bias: bool = False,
        score_size_bias_hidden: int = 32,
        self_attention_rope: bool = False,
        self_attention_sta: bool = False,
        sta_kernel: int = 3,
        rope_theta: float = 100.0,
        rope_position_scale: float | None = None,
        polygon_masked_attention: bool = False,
        polygon_oracle_mask: bool = False,
        polygon_mask_floor: float = 0.1,
        polygon_mask_temperature: float = 0.1,
        appearance_weight: float = 0.0,
        presence_weight: float = 0.0,
        appearance_imitation_weight: float = 0.0,
        appearance_imitation_temperature: float = 0.1,
        appearance_imitation_small_area: float = 200.0,
        instance_alignment_weight: float = 0.0,
        instance_alignment_temperature: float = 0.1,
        instance_alignment_radius_px: float = 30.0,
        instance_alignment_max_negatives: int = 8,
        appearance_channels: int = 32,
        presence_negative_margin_px: float = 4.0,
        verbose: bool = True,
    ):
        super(DetectionModel, self).__init__()
        self.nc = nc
        self.n_rays = n_rays
        self.lambda_seg = lambda_seg
        self.no_object = no_object
        self.no_object_weight = no_object_weight
        self.cls_weight = cls_weight
        self.piou_weight = piou_weight
        self.xy_weight = xy_weight
        self.cost_inside = cost_inside
        self.per_layer_match = per_layer_match
        self.duplicate_train_topk = int(duplicate_train_topk)
        if self.duplicate_train_topk < 1:
            raise ValueError('duplicate_train_topk must be at least 1')
        self.detr_lsp = detr_lsp
        self.detr_matcher_lsp = detr_matcher_lsp
        self.detr_layer_schedule = detr_layer_schedule
        self.seed_level = seed_level
        self.seed_level_stride = seed_level_stride
        self.query_block_size = query_block_size
        self.content_top = content_top and backbone == 'nulite'
        self.respawn = respawn
        self.respawn_peaks = respawn_peaks
        self.respawn_min_seed = respawn_min_seed
        self.seed_resize = seed_resize
        self.seed_resize_frac = seed_resize_frac
        self.seed_map_target = seed_map_target
        self.oracle_kind = str(oracle_kind)
        if self.oracle_kind not in {'binary', 'center_potential', 'fg_centroid'}:
            raise ValueError("oracle_kind must be 'binary', 'center_potential', or 'fg_centroid'")
        self.aux_dense_weight = aux_dense_weight
        self.aux_dense_type_weight = aux_dense_type_weight
        self.seg_fg_margin_weight = seg_fg_margin_weight
        self.seg_fg_margin = seg_fg_margin
        self.seg_fg_radius_px = seg_fg_radius_px
        self.seg_fg_margin_layer_weights = tuple(float(w) for w in (seg_fg_margin_layer_weights or ()))
        self.viable_owner_fg_weight = float(viable_owner_fg_weight)
        self.viable_owner_iou = float(viable_owner_iou)
        self.viable_owner_margin = float(viable_owner_margin)
        if self.viable_owner_fg_weight < 0 or not 0 < self.viable_owner_iou < 1 or self.viable_owner_margin < 0:
            raise ValueError('viable owner loss requires nonnegative weight/margin and IoU in (0, 1)')
        self.proposal_bal_weight = float(proposal_bal_weight)
        self.proposal_bal_cap = float(proposal_bal_cap)
        if self.proposal_bal_weight < 0 or self.proposal_bal_cap < 1:
            raise ValueError('proposal balanced loss requires nonnegative weight and cap >= 1')
        self.nucleus_cdn = int(nucleus_cdn)
        self.nucleus_cdn_weight = float(nucleus_cdn_weight)
        self.nucleus_cdn_class_embed = bool(nucleus_cdn_class_embed)
        self.nucleus_cdn_real_to_dn = bool(nucleus_cdn_real_to_dn)
        self.seed_in_content_pooled = bool(seed_in_content_pooled)
        self.seed_in_content_pool_sizes = tuple(int(p) for p in seed_in_content_pool_sizes)
        self.map_modulation = bool(map_modulation)
        self.score_size_bias = bool(score_size_bias)
        self.score_size_bias_hidden = int(score_size_bias_hidden)
        if self.score_size_bias and not no_object:
            raise ValueError('score_size_bias requires no_object (empty-column targets)')
        if self.nucleus_cdn < 0 or self.nucleus_cdn_weight < 0:
            raise ValueError('nucleus_cdn and nucleus_cdn_weight must be nonnegative')
        if self.nucleus_cdn > 0 and not no_object:
            raise ValueError('nucleus_cdn requires no_object (empty-column targets)')
        self.appearance_weight = float(appearance_weight)
        self.presence_weight = float(presence_weight)
        self.appearance_imitation_weight = float(appearance_imitation_weight)
        self.appearance_imitation_temperature = float(appearance_imitation_temperature)
        self.appearance_imitation_small_area = float(appearance_imitation_small_area)
        self.instance_alignment_weight = float(instance_alignment_weight)
        self.instance_alignment_temperature = float(instance_alignment_temperature)
        self.instance_alignment_radius_px = float(instance_alignment_radius_px)
        self.instance_alignment_max_negatives = int(instance_alignment_max_negatives)
        self.appearance_channels = int(appearance_channels)
        self.presence_negative_margin_px = float(presence_negative_margin_px)
        if any(w < 0 for w in self.seg_fg_margin_layer_weights):
            raise ValueError('seg_fg_margin_layer_weights must be non-negative')
        self.use_decoder = use_decoder
        self.backbone = backbone
        self.separable_decoder = separable_decoder
        self.encoder_input_norm = '01'
        self.names = {i: str(i) for i in range(nc)}
        self.yaml = {'nc': nc, 'n_rays': n_rays, 'architecture': 'nulite_detr', 'variant': variant}

        if backbone == 'yolo11':
            from raycasted.model.blocks.yolo11_fpn import Yolo11FPNBackbone

            self.encoder_decoder = Yolo11FPNBackbone(scale=yolo11_scale, nc=nc, ch=3, verbose=False)
            if yolo11_ckpt:
                self.encoder_decoder.load_pretrained(yolo11_ckpt)
            elif pretrained:
                self.encoder_decoder.load_pretrained(f'yolo11{yolo11_scale}.pt')
            dims = self.encoder_decoder.out_channels
            self.use_decoder = False
            self.decoder0 = None
            self.np_head = NPHead(in_channels=dims[0]) if use_seed else None
        elif backbone == 'dual':
            # Dual-encoder, SPAR-Det CAHF-style FG/BG disentanglement:
            #   frozen NuLite-T FastViT seg stream (no decoder) gates the
            #   trainable yolo11 detection stream at every scale via a
            #   multiplicative dual (deformable-spatial x SE-channel) attention
            #   gate (Branch3Gate). No seed map / score-head cue (faithful CAHF).
            from raycasted.model.blocks.nulite import Branch3Gate
            from raycasted.model.blocks.yolo11_fpn import Yolo11FPNBackbone

            self.seg_encoder = NuLiteEncoderDecoder(variant=variant, pretrained=False, use_decoder=False)
            self.np_head = None
            self.encoder_decoder = Yolo11FPNBackbone(scale=yolo11_scale, nc=nc, ch=3, verbose=False)
            if yolo11_ckpt:
                self.encoder_decoder.load_pretrained(yolo11_ckpt)
            elif pretrained:
                self.encoder_decoder.load_pretrained(f'yolo11{yolo11_scale}.pt')
            dims = self.encoder_decoder.out_channels
            self.use_decoder = False
            self.decoder0 = None
            seg_dims = self.seg_encoder.embed_dims  # [64,128,256,512] s12
            assert list(dims[:3]) == seg_dims[:3], (
                f'dual-encoder channel mismatch: det {list(dims[:3])} vs seg {seg_dims[:3]}'
            )
            # per-scale CAHF gates fusing the frozen seg stream into the det stream
            self.cahf_gates = nn.ModuleList([Branch3Gate(c) for c in (dims[0], dims[1], dims[2])])
            # the seg stream is frozen (pretrained FG/BG prior); NuLite-T weights
            # are loaded into it later via load_nulite_ckpt().
            self.seg_encoder.freeze_encoder()
        elif backbone == 'two_stream':
            # Two-stream baseline: a frozen NuLite-T oracle seg stream (s12 +
            # separable decoder, full U-Net -> foreground map) + a separate
            # trainable fastvit_t12 detection stream (encoder + NuLite decoder)
            # + DETR head. The seg foreground map is fed to the DETR head as the
            # per-query score cue (Q) and feature channel (F), decoupling 'where
            # nuclei are' (frozen prior) from 'precise regression' (trainable det
            # features) — the SPAR-Det FG/BG disentanglement mapped to our DETR
            # readout (vs CAHF's dense feature gating, which missed the sparse
            # per-query classifier).
            self.seg_encoder = NuLiteSegOracle(
                variant=variant,
                separable_decoder=True,
                output_channels=1 if self.oracle_kind == 'center_potential' else 2,
                oracle_kind=self.oracle_kind,
            )
            self.encoder_decoder = NuLiteEncoderDecoder(
                variant=det_variant, pretrained=pretrained, use_decoder=True, separable_decoder=False
            )
            dims = self.encoder_decoder.embed_dims  # [64,128,256,512] t12 st4/8/16/32
            self.use_decoder = True
            self.decoder0 = None
            self.np_head = None
            self.lambda_seg = 0.0  # seg loss is inert (frozen prior stream)
            self.seg_encoder.requires_grad_(False)  # frozen FG/BG prior
            # ``requires_grad_(False)`` does not freeze BatchNorm running
            # statistics. Keep the oracle in eval mode from construction; the
            # overridden ``train`` method below also restores this after the
            # parent model changes modes.
            self.seg_encoder.eval()
        elif backbone == 'shared':
            # Single-stream multitask: ONE FastViT encoder (frozen) + ONE NuLite
            # upsample decoder (trainable) serving both tasks.
            #   segmentation: b1 -> np_head (frozen NuLite-T oracle head) -> FG map
            #   detection   : b3/b4/b5 -> DETR decoder -> rays/classes
            # This removes the duplicated detection encoder of ``two_stream``
            # (25.9% of the parameters and one whole encode pass) while letting
            # the shared decoder adapt to both objectives.
            self.encoder_decoder = NuLiteEncoderDecoder(
                variant=det_variant, pretrained=pretrained, use_decoder=True, separable_decoder=separable_decoder
            )
            dims = self.encoder_decoder.embed_dims  # [64,128,256,512] t12 st4/8/16/32
            self.use_decoder = True
            self.decoder0 = None
            self.np_head = None
            # Unlike ``two_stream``, the seed map is produced by the trainable
            # shared decoder + np_head, so it must stay anchored to the oracle
            # target (L1 against ``batch['seed_map']``) instead of drifting.
            self.shared_stream = True
            # The oracle's own encoder+decoder are not instantiated: only its
            # decoder0/np_head are needed, and those are created on the shared
            # U-Net by ``load_nulite_ckpt`` below.
            self.seg_encoder = None
        elif backbone == 'frozen':
            # Frozen-oracle thinning: the FULL NuLite-T U-Net (encoder + decoder
            # + decoder0 + np_head) is kept intact and entirely frozen, exactly
            # as in ``two_stream``, but the detection head reads the encoder's
            # b3/b4/b5 directly instead of running a second detection encoder
            # (as ``shared`` must do from the shared decoder). Nothing on the
            # segmentation side can drift; only the DETR head is trained.
            #   [backbone] -[b3/b4/b5]-> [DETR head]
            #   [backbone] -> [seg decoder] -> [FG map] -> [DETR head]
            self.seg_encoder = NuLiteSegOracle(
                variant=variant,
                separable_decoder=True,
                output_channels=1 if self.oracle_kind == 'center_potential' else 2,
                oracle_kind=self.oracle_kind,
            )
            self.encoder_decoder = None
            dims = self.seg_encoder.encoder_decoder.embed_dims  # [64,128,256,512] t12 st4/8/16/32
            self.use_decoder = True
            self.decoder0 = None
            self.np_head = None
            self.lambda_seg = 0.0  # seg stream is a frozen prior; no seg loss
            self.seg_encoder.requires_grad_(False)
            self.seg_encoder.eval()
            self.frozen_oracle = True
            # Optional detection-side adapters on the raw U-Net fusion tensors
            # (pre-ConvBlock concats; b5 has no skip so it taps z4). Pure
            # depthwise->pointwise projections to the native level channels,
            # so the DETR input_proj shapes stay identical to two_stream.
            self.frozen_adapter = bool(frozen_adapter)
            if self.frozen_adapter:
                e3, e2, e1, e0 = dims  # e3=64(st4) e2=128(st8) e1=256(st16) e0=512(st32), cf. nulite.py
                # fuse3 = cat[z2, b4] (128+128=256ch @ stride 8)  -> b3's 64ch  @ stride 4
                # fuse4 = cat[z3, b5] (256+256=512ch @ stride 16) -> b4's 128ch @ stride 8
                # fuse5 = z4          (512ch @ stride 32)         -> b5's 256ch @ stride 16
                self.fusion_adapters = nn.ModuleList(
                    [
                        FusionAdapter(2 * e2, e3),
                        FusionAdapter(2 * e1, e2),
                        FusionAdapter(e0, e1),
                    ]
                )
            else:
                self.fusion_adapters = None
        else:
            self.encoder_decoder = NuLiteEncoderDecoder(
                variant=variant,
                pretrained=pretrained,
                use_decoder=use_decoder,
                separable_decoder=separable_decoder,
            )
            dims = self.encoder_decoder.embed_dims  # [64,128,256,512] s12 / [48,96,192,384] t8
            if use_decoder:
                self.decoder0 = _ConvBlock(
                    3, dims[0], 3, separable=separable_decoder
                )  # NuLite decoder0: Conv2DBlock(3 -> 64) ReLU (not SiLU Conv)
                self.np_head = NPHead(in_channels=2 * dims[0])  # xt = cat([decoder0(x), b1]), both dims[0]-wide
            else:
                self.decoder0 = None
                self.np_head = None
        if backbone == 'dual':
            # CAHF gating replaces the seed-map F/Q fusion entirely (no seed map
            # is produced in dual mode), so the DETR head's seed conditioning is
            # disabled and input_proj stays c-channel.
            detr_seed_feature = False
            detr_seed_in_content = False
        else:
            detr_seed_feature = seed_feature
            detr_seed_in_content = seed_in_content
        if backbone == 'shared':
            # Frozen segmentation readout on the shared features. ``np_head`` is
            # the NuLite-T head (cat[decoder0(x), b1] -> FG logits); decoder0 is
            # the raw-input Conv(3 -> 64) the oracle was trained with.
            emb0 = dims[0]
            self.decoder0 = _ConvBlock(3, emb0, 3, separable=False)
            self.np_head = nn.Sequential(
                _ConvBlock(2 * emb0, emb0),
                nn.Conv2d(emb0, 1 if self.oracle_kind == 'center_potential' else 2, 1),
            )
            # The shared encoder is the oracle's encoder: freeze it. The shared
            # upsample decoder stays trainable so it can serve detection too.
            self.encoder_decoder.freeze_encoder()
        self.detr = NuLiteDETRDecoder(
            nc=nc,
            ch=(dims[0], dims[1], dims[2]),
            hd=hd,
            nq=nq,
            ndl=ndl,
            d_ffn=d_ffn,
            nd=0,
            n_rays=n_rays,
            seed_threshold=seed_threshold,
            peak_distance=peak_distance,
            grid_size=grid_size,
            query_selection=query_selection,
            seed_feature=detr_seed_feature,
            no_object=no_object,
            seed_in_content=detr_seed_in_content,
            mds=mds,
            shared_layers=shared_layers,
            shared_layer_groups=shared_layer_groups,
            query_block_size=query_block_size,
            content_top=self.content_top,
            content_top_channels=dims[3] if (self.content_top and len(dims) > 3) else 0,
            seed_resize=seed_resize,
            seed_resize_frac=seed_resize_frac,
            layer_schedule=detr_layer_schedule,
            hybrid_peaks=hybrid_peaks,
            seed_level=seed_level and backbone in ('nulite', 'two_stream', 'shared', 'frozen'),
            seed_level_stride=seed_level_stride,
            recast=recast,
            recast_after_layer=recast_after_layer,
            recast_radius_px=recast_radius_px,
            respawn=respawn,
            respawn_peaks=respawn_peaks,
            respawn_min_seed=respawn_min_seed,
            self_attention_rope=self_attention_rope,
            self_attention_sta=self_attention_sta,
            sta_kernel=sta_kernel,
            rope_theta=rope_theta,
            rope_position_scale=rope_position_scale,
            polygon_masked_attention=polygon_masked_attention,
            polygon_oracle_mask=polygon_oracle_mask,
            polygon_mask_floor=polygon_mask_floor,
            polygon_mask_temperature=polygon_mask_temperature,
            appearance_channels=(
                self.appearance_channels
                if (
                    self.appearance_weight > 0
                    or self.presence_weight > 0
                    or self.appearance_imitation_weight > 0
                    or self.instance_alignment_weight > 0
                )
                else 0
            ),
            appearance_after_layer=recast_after_layer,
            presence_head=self.presence_weight > 0,
            nucleus_cdn=self.nucleus_cdn,
            nucleus_cdn_class_embed=self.nucleus_cdn_class_embed,
            nucleus_cdn_real_to_dn=self.nucleus_cdn_real_to_dn,
            seed_in_content_pooled=self.seed_in_content_pooled,
            seed_in_content_pool_sizes=self.seed_in_content_pool_sizes,
            map_modulation=self.map_modulation,
            score_size_bias=self.score_size_bias,
            score_size_bias_hidden=self.score_size_bias_hidden,
        )
        self.aux_dense_head = DenseAuxHead((dims[0], dims[1], dims[2]), nc) if self.aux_dense_weight > 0 else None
        self.appearance_head = (
            QueryAppearanceHead(dims[0], dims[0], self.appearance_channels)
            if (
                self.appearance_weight > 0
                or self.presence_weight > 0
                or self.appearance_imitation_weight > 0
                or self.instance_alignment_weight > 0
            )
            else None
        )
        self.appearance_imitation_projection = (
            nn.Sequential(
                nn.Linear(self.appearance_channels * 3, self.appearance_channels, bias=False),
                nn.LayerNorm(self.appearance_channels),
            )
            if self.appearance_imitation_weight > 0
            else None
        )
        self.instance_alignment_projection = (
            nn.Sequential(
                nn.Linear(hd, self.appearance_channels, bias=False),
                nn.LayerNorm(self.appearance_channels),
            )
            if self.instance_alignment_weight > 0
            else None
        )

        # The decoder already stores the right stride vector (3 or 4 entries when
        # the seed map is attended as an extra level); fall back to the default.
        stride = getattr(self.detr, 'stride', None)
        if stride is None:
            stride = torch.tensor([4.0, 8.0, 16.0])
        self.detr.stride = stride
        self.stride = stride

        if backbone == 'dual':
            self.model = (
                [self.seg_encoder]
                + ([self.np_head] if self.np_head is not None else [])
                + [self.encoder_decoder, self.detr]
            )
        elif backbone == 'yolo11':
            self.model = [self.encoder_decoder] + ([self.np_head] if self.np_head is not None else []) + [self.detr]
        elif backbone == 'two_stream':
            self.model = [self.seg_encoder, self.encoder_decoder, self.detr]
        elif backbone == 'frozen':
            self.model = [self.seg_encoder, self.detr]
        elif use_decoder:
            self.model = [self.encoder_decoder, self.decoder0, self.np_head, self.detr]
        else:
            self.model = [self.encoder_decoder, self.detr]
        self.save = []
        # NOTE: end2end stays False. The DetectionModel.end2end getter reads
        # model[-1].end2end (absent on the DETR decoder), and the setter warns +
        # skips when the head lacks the attr. Keeping it False also avoids the
        # ultralytics resume path calling criterion.update() (NuLiteDETRLoss has
        # no anneal method).

        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            print(f'NuLiteRayCastDETRModel: {n_params / 1e6:.2f}M params')

    # -- inference path -----------------------------------------------------

    def train(self, mode: bool = True):
        """Set train/eval mode while keeping the two-stream oracle immutable."""
        super().train(mode)
        if getattr(self, 'backbone', None) == 'two_stream':
            # Ultralytics' setup loop can re-enable every floating parameter
            # before it builds the optimizer.  Keeping this here as well makes
            # every later train-mode transition restore the oracle contract.
            self.seg_encoder.requires_grad_(False)
            self.seg_encoder.eval()
        elif getattr(self, 'backbone', None) == 'shared':
            # The shared encoder is the frozen oracle encoder; its BN statistics
            # must not drift, or the frozen seg head silently degrades. The
            # shared decoder stays in the parent's train/eval mode.
            self.encoder_decoder.encoder.requires_grad_(False)
            self.encoder_decoder.encoder.eval()
        elif getattr(self, 'backbone', None) == 'frozen':
            # The whole NuLite-T oracle (encoder, decoder, decoder0, np_head) is
            # frozen; keep it in eval mode so BN statistics cannot drift. An
            # explicit ``thaw_oracle_encoder()`` (detr_frozen_oracle_unfreeze_encoder
            # or detr_frozen_oracle_unfreeze_decoder) opts part of the U-Net out
            # of this contract.
            if not getattr(self, 'oracle_encoder_thawed', False):
                self.seg_encoder.eval()
                self.seg_encoder.requires_grad_(False)
            else:
                self.seg_encoder.np_head.eval()
                self.seg_encoder.np_head.requires_grad_(False)
                enc = self.seg_encoder.encoder_decoder.encoder
                enc.train(mode)
                enc.requires_grad_(True)
                if getattr(self, 'oracle_full_thaw', False):
                    rest = self.seg_encoder.encoder_decoder
                    rest.requires_grad_(True)
                    for n, p in rest.named_parameters():
                        if not n.startswith('encoder.'):
                            p.requires_grad = True
                    rest.train(mode)
                    self.seg_encoder.decoder0.requires_grad_(True)
                    self.seg_encoder.decoder0.train(mode)
        return self

    @staticmethod
    def _compact_no_object_scores(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert sigmoid DETR scores to compact confidence and real-class id."""
        p_no_obj = scores[..., -1:]
        is_bg = scores.argmax(-1, keepdim=True) == scores.shape[-1] - 1
        conf = (1.0 - p_no_obj).clamp(min=0.0).masked_fill(is_bg, 0.0)
        cls = scores[..., :-1].argmax(-1, keepdim=True)
        return conf, cls

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None, *args, **kwargs):
        """Run encoder -> decoder -> seed head -> DETR decoder.

        Training: returns the decoder's 5-tuple (dec_polygons, dec_scores,
        enc_polygons, enc_scores, dn_meta).
        Eval: returns (y, out) where y = (B, nq, raycast_dim + 2) in the Detect
        format [cx, cy, d1..dn, conf, cls] (eval_pannuke / val.py compatible).
        """
        if getattr(self, 'backbone', 'nulite') == 'nulite' and getattr(self, 'encoder_input_norm', '01') == '11':
            # Oracle-loaded NuLite encoder was trained on [-1, 1] input
            # (0.5/0.5 normalization). Its frozen BN stats are calibrated to
            # that distribution, so feed it the same domain; decoder0 is
            # concatenated with the oracle decoder output and must match too.
            x = (x - 0.5) / 0.5
        if getattr(self, 'backbone', 'nulite') == 'frozen':
            # ONE frozen oracle pass supplies both the FG/centroid map and the
            # b3/b4/b5 features the DETR decoder attends to. With adapters on,
            # the head consumes the raw fusion concats (pre-ConvBlock) instead
            # of the post-fusion decoder outputs.
            seg_x = (x - 0.5) / 0.5
            d = self.seg_encoder.encoder_decoder(seg_x, collect_fusion=self.frozen_adapter)
            seed_logits = self.seg_encoder(seg_x)
            if self.frozen_adapter:
                d = {
                    'b3': self.fusion_adapters[0](d['fuse3']),
                    'b4': self.fusion_adapters[1](d['fuse4']),
                    'b5': self.fusion_adapters[2](d['fuse5']),
                    'b1': d['b1'],
                    'top': d['top'],
                }
        else:
            d = self.encoder_decoder(x)
            seed_logits = None
        if getattr(self, 'backbone', 'nulite') == 'dual':
            # NuLite seg stream was trained with 0.5/0.5 normalization, i.e. it
            # expects input in [-1, 1]; the det stream (yolo11) expects [0, 1].
            # Gate the det stream features with the frozen seg features at each
            # scale (SPAR-Det CAHF) — no seed map / score-head cue.
            seg_x = (x - 0.5) / 0.5
            d_seg = self.seg_encoder(seg_x)
            seed = None
            d = {
                'b3': self.cahf_gates[0](d['b3'], d_seg['b3']),
                'b4': self.cahf_gates[1](d['b4'], d_seg['b4']),
                'b5': self.cahf_gates[2](d['b5'], d_seg['b5']),
            }
        elif getattr(self, 'backbone', 'nulite') == 'yolo11':
            seed = self.np_head(d['b3']) if getattr(self, 'np_head', None) is not None else None
        elif getattr(self, 'backbone', 'nulite') == 'shared':
            # One frozen-encoder + trainable-decoder pass feeds both tasks: the
            # full-resolution b1 goes to the frozen NuLite-T seg head, the
            # stride 4/8/16 features go to the DETR decoder below.
            seg_x = (x - 0.5) / 0.5
            x0 = self.decoder0(seg_x)
            seg_logits = self.np_head(torch.cat([x0, d['b1']], dim=1))
            if self.oracle_kind == 'center_potential':
                seed = seg_logits
                centroid_seed = seg_logits
            elif self.oracle_kind == 'fg_centroid':
                seed = seg_logits[:, 0:1]
                centroid_seed = seg_logits[:, 1:2]
            else:
                seed = seg_logits[:, 1:2] - seg_logits[:, 0:1]
                centroid_seed = seed
        elif getattr(self, 'backbone', 'nulite') == 'frozen':
            # The oracle map is the same as in ``two_stream``; the only change is
            # that its encoder's b3/b4/b5 feed the DETR head directly.
            if self.oracle_kind == 'center_potential':
                seed = seed_logits
                centroid_seed = seed_logits
            elif self.oracle_kind == 'fg_centroid':
                seed = seed_logits[:, 0:1]
                centroid_seed = seed_logits[:, 1:2]
            else:
                seed = seed_logits[:, 1:2] - seed_logits[:, 0:1]
                centroid_seed = seed
        elif getattr(self, 'backbone', 'nulite') == 'two_stream':
            # Frozen NuLite-T oracle seg stream was trained on [-1, 1] input
            # (0.5/0.5 normalization); the det stream expects raw [0, 1]. The
            # Binary CE logits collapse to a foreground logit. A centroid-
            # potential oracle already emits the single seed logit directly.
            seg_x = (x - 0.5) / 0.5
            seg_logits = self.seg_encoder(seg_x)
            if self.oracle_kind == 'center_potential':
                seed = seg_logits
                centroid_seed = seg_logits
            elif self.oracle_kind == 'fg_centroid':
                # Preserve the Train36 foreground contract for score content,
                # recasting, and the matched-FG margin. Keep the independently
                # trained centroid potential as the attended spatial level.
                seed = seg_logits[:, 0:1]
                centroid_seed = seg_logits[:, 1:2]
            else:
                seed = seg_logits[:, 1:2] - seg_logits[:, 0:1]
                centroid_seed = seed
        elif getattr(self, 'use_decoder', True):
            x0 = self.decoder0(x)
            xt = torch.cat([x0, d['b1']], dim=1)
            seed = self.np_head(xt)
        else:
            seed = None
            centroid_seed = None
        if getattr(self, 'backbone', 'nulite') not in ('two_stream', 'shared', 'frozen'):
            centroid_seed = seed
        self._last_seed = seed
        self._last_centroid_seed = centroid_seed
        self._last_aux_dense = None
        self._last_appearance_logits = None
        self._last_appearance_map = None
        appearance_map = None
        if self.appearance_head is not None:
            if 'b1' not in d:
                raise ValueError('query appearance supervision requires the NuLite decoder b1 feature')
            appearance_map, self._last_appearance_logits = self.appearance_head(d['b1'], d['b3'])
            self._last_appearance_map = appearance_map
        if self.training and self.aux_dense_head is not None:
            self._last_aux_dense = self.aux_dense_head((d['b3'], d['b4'], d['b5']))

        top_feat = d.get('top') if getattr(self.detr, 'content_top', False) else None
        out = self.detr(
            [d['b3'], d['b4'], d['b5']],
            batch,
            seed_map=seed,
            seed_level_map=centroid_seed,
            top_feat=top_feat,
            appearance_map=appearance_map,
        )
        if self.training:
            return out

        y, out = out
        poly = y[..., : self.detr.raycast_dim]
        scores = y[..., self.detr.raycast_dim :]  # already sigmoid prob (nc or nc+1)
        if self.no_object:
            # LSP-DETR: last column is the explicit "no object" prob.
            # A compact [polygon, conf, class] output cannot carry an explicit
            # background label, so zero its confidence whenever no-object wins
            # the all-class argmax. This matches eval_lsp and prevents the
            # generic validator from treating a background query as foreground.
            conf, cls = self._compact_no_object_scores(scores)
        else:
            conf = scores.max(-1, keepdim=True).values
            cls = scores.argmax(-1, keepdim=True)
        # decode_polygon outputs NORMALIZED coords (sigmoid xy, exp rays in
        # [0,1]). val.py / eval_pannuke expect PIXEL coords (GT is denormalized
        # by crop_size there), so scale to pixels using the input size.
        imgsz = x.shape[-1]
        poly = poly.clone()
        poly[..., :2] = poly[..., :2] * imgsz
        poly[..., 2:] = poly[..., 2:] * imgsz
        y_detect = torch.cat([poly, conf, cls], -1)
        return y_detect, out

    # -- training path ------------------------------------------------------

    def init_criterion(self):
        """DETR 1:1 Hungarian criterion (no o2m/o2o dual assignment).

        Focal loss with an explicit "no object" class (LSP-DETR style) when
        ``no_object`` is enabled; ``no_object_weight`` scales the ∅ column
        (>1 = suppress redundant per-nucleus queries harder).

        With ``detr_lsp`` the criterion switches to LSP-DETR-exact terms: class
        focal with EQUAL ∅/class columns (no per-column gain), centroid L1 +
        log-space radial (analytical from the predicted centroid), no polar-IoU
        or curvature smoothing, and a near-hard (9999) centroid-inside-GT
        matching gate.
        """
        if getattr(self, 'detr_lsp', False):
            return NuLiteDETRLoss(
                nc=self.nc,
                loss_gain={
                    'class': self.cls_weight,
                    'xy': self.xy_weight,
                    'l1': 14.0,
                    'piou': 0.0,
                    'smooth': 0.0,
                    'no_object': 1.0,
                },
                use_vfl=False,
                no_object=True,
                n_rays=self.n_rays,
                cost_inside=9999.0,
                per_layer_match=self.per_layer_match,
                duplicate_train_topk=self.duplicate_train_topk,
                lsp_mode=True,
            )
        return NuLiteDETRLoss(
            nc=self.nc,
            loss_gain={
                'class': self.cls_weight,
                'xy': self.xy_weight,
                'l1': 14.0,
                'piou': self.piou_weight,
                'smooth': 0.0,
                'no_object': self.no_object_weight,
            },
            use_vfl=not self.no_object,
            no_object=self.no_object,
            n_rays=self.n_rays,
            cost_inside=self.cost_inside,
            per_layer_match=self.per_layer_match,
            duplicate_train_topk=self.duplicate_train_topk,
            matcher_lsp=getattr(self, 'detr_matcher_lsp', False),
        )

    def _segmentation_foreground_margin(self, dec_polygons, dec_scores, targets, gt_groups, seed):
        """Apply segmentation-guided true-class margins at selected decoder layers.

        ``dec_polygons`` and ``dec_scores`` include the encoder proposal stage
        at index zero. Configured weights correspond only to decoder layers
        1..N. An empty weight list preserves the train35/36 final-layer-only
        behavior. Every selected layer receives its own Hungarian assignment.
        """
        zero = dec_scores.sum() * 0.0
        n_decoder_layers = len(dec_scores) - 1
        configured = getattr(self, 'seg_fg_margin_layer_weights', ())
        if configured:
            if len(configured) == n_decoder_layers:
                selected = [(layer_idx, weight) for layer_idx, weight in enumerate(configured, start=1) if weight > 0]
            elif n_decoder_layers == 1:
                # Evaluation returns only decoder.eval_idx (the final layer),
                # so retain its configured multiplier for loss telemetry.
                selected = [(1, configured[-1])] if configured[-1] > 0 else []
            else:
                raise ValueError(
                    'seg_fg_margin_layer_weights must contain one value per decoder layer '
                    f'({len(configured)} configured, {n_decoder_layers} decoder layers)'
                )
        else:
            selected = [(len(dec_scores) - 1, 1.0)]

        seed_prob = seed.detach().sigmoid()
        seed_h, seed_w = seed_prob.shape[-2:]
        offsets = dec_polygons.new_tensor([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        scale = dec_polygons.new_tensor([seed_w, seed_h])
        weighted_margin = zero

        for layer_idx, layer_weight in selected:
            layer_polygons = dec_polygons[layer_idx]
            layer_scores = dec_scores[layer_idx]
            match_indices = self.criterion.matcher(
                layer_polygons, layer_scores, targets['bboxes'], targets['cls'], gt_groups
            )
            margin_terms = []
            for batch_i, (query_idx, target_idx) in enumerate(match_indices):
                if len(query_idx) == 0:
                    continue
                query_idx = query_idx.to(layer_polygons.device)
                target_idx = target_idx.to(layer_polygons.device)
                positions = layer_polygons[batch_i, query_idx, :2].detach()
                grid = positions[:, None, :] + offsets[None] * self.seg_fg_radius_px / scale
                evidence = (
                    F.grid_sample(
                        seed_prob[batch_i : batch_i + 1],
                        (grid * 2 - 1).unsqueeze(0),
                        mode='bilinear',
                        align_corners=False,
                        padding_mode='border',
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .mean(-1)
                )
                true_cls = targets['cls'][target_idx]
                logits = layer_scores[batch_i, query_idx]
                hinge = F.relu(self.seg_fg_margin + logits[:, -1] - logits.gather(1, true_cls[:, None]).squeeze(1))
                margin_terms.append(evidence * hinge)
            if margin_terms:
                weighted_margin = weighted_margin + layer_weight * torch.cat(margin_terms).mean()

        return weighted_margin

    @staticmethod
    def _paired_star_iou_sampled(pred, target, samples=20):
        """Detached, local-grid polygon IoU for matched star polygons only."""
        if len(pred) == 0:
            return pred.new_zeros((0,))
        polygons = torch.stack((pred, target), dim=1).detach().float()
        centers, rays = polygons[..., :2], polygons[..., 2:]
        extent = rays.amax(-1)
        lo = (centers - extent[..., None]).amin(1).clamp(0, 1)
        hi = (centers + extent[..., None]).amax(1).clamp(0, 1)
        unit = (torch.arange(samples, device=pred.device, dtype=polygons.dtype) + 0.5) / samples
        xx = lo[:, None, None, 0] + (hi[:, None, None, 0] - lo[:, None, None, 0]) * unit[None, None, :]
        yy = lo[:, None, None, 1] + (hi[:, None, None, 1] - lo[:, None, None, 1]) * unit[None, :, None]
        points = torch.stack(torch.broadcast_tensors(xx, yy), dim=-1)
        delta = points[:, None] - centers[:, :, None, None]
        distance = delta.square().sum(-1).sqrt()
        ray_count = rays.shape[-1]
        position = torch.remainder(torch.atan2(delta[..., 1], delta[..., 0]), 2 * torch.pi)
        position = position * (ray_count / (2 * torch.pi))
        lower = position.floor().long() % ray_count
        fraction = position - position.floor()
        expanded_rays = rays[:, :, None, None].expand(-1, -1, samples, samples, -1)
        lower_ray = expanded_rays.gather(-1, lower[..., None]).squeeze(-1)
        upper_ray = expanded_rays.gather(-1, ((lower + 1) % ray_count)[..., None]).squeeze(-1)
        inside = distance <= lower_ray * (1 - fraction) + upper_ray * fraction
        intersection = (inside[:, 0] & inside[:, 1]).sum((1, 2))
        union = (inside[:, 0] | inside[:, 1]).sum((1, 2))
        return intersection.float() / union.clamp_min(1)

    def _viable_owner_foreground_loss(self, polygons, scores, targets, gt_groups):
        """Reinforce only geometrically viable one-to-one owners against empty."""
        zero = scores.sum() * 0.0
        with torch.no_grad():
            owners = self.criterion.matcher(polygons, scores, targets['bboxes'], targets['cls'], gt_groups)
        terms = []
        for batch_i, (query_idx, target_idx) in enumerate(owners):
            if len(query_idx) == 0:
                continue
            query_idx = query_idx.to(polygons.device)
            target_idx = target_idx.to(polygons.device)
            with torch.no_grad():
                quality = self._paired_star_iou_sampled(
                    polygons[batch_i, query_idx], targets['bboxes'][target_idx]
                )
                viable = quality >= self.viable_owner_iou
            if not viable.any():
                continue
            logits = scores[batch_i, query_idx[viable]]
            true_class = targets['cls'][target_idx[viable]]
            gap = logits[:, -1] - logits.gather(1, true_class[:, None]).squeeze(1)
            terms.append(F.softplus(self.viable_owner_margin + gap))
        return torch.cat(terms).mean() if terms else zero

    def _nucleus_cdn_class_loss(self, layer_scores, dn_meta):
        """Supervise canonical GT-polygon DN queries to fire on their true class.

        Every DN query sits exactly on a GT polygon (geometry frozen), so the
        only thing to learn is the classification decision it exists to fix:
        a good polygon must beat the empty column. Applied at the encoder
        proposal layer AND every decoder layer (``layer_scores`` is
        (L+1, B, n_dn, nc+1)).

        Padded DN rows (label == nc, no GT) are also trained to stay empty:
        they get the same empty-column target as an unmatched real query, so
        the model cannot satisfy the loss by always firing. Mean is over all
        DN slots (valid + padded) so both terms share one denominator.
        """
        labels = dn_meta['labels']  # (B, n_dn)
        bs, n_dn = labels.shape
        flat_labels = labels.reshape(-1)
        one_hot = torch.zeros(
            (bs * n_dn, self.nc + 1), dtype=torch.int64, device=layer_scores.device
        )
        one_hot.scatter_(1, flat_labels.unsqueeze(-1), 1)
        n_layers = layer_scores.shape[0]
        logits = layer_scores.reshape(n_layers, bs * n_dn, self.nc + 1)
        bce = F.binary_cross_entropy_with_logits(
            logits, one_hot.float().unsqueeze(0).expand_as(logits), reduction='none'
        )
        pred_prob = logits.sigmoid()
        p_t = one_hot.float().unsqueeze(0) * pred_prob + (1 - one_hot.float().unsqueeze(0)) * (1 - pred_prob)
        bce = bce * (1.0 - p_t) ** 2.0
        return bce.mean()

    def _balanced_proposal_loss(self, polygons, scores, targets, gt_groups):
        """Class-balanced focal classification loss at the encoder proposal layer.

        The one-to-one Hungarian criterion is dominated by the majority
        "empty" decision, so the proposal head learns an operating point that
        is over-eager to say background (measured: recall 0.65 at tnr 0.93 on
        features whose balanced linear probe reaches recall 0.93). This term
        re-weights the *positive* (Hungarian-matched) queries by the current
        background/foreground ratio, capped at ``proposal_bal_cap``, which
        shifts that operating point without touching geometry or the
        one-to-one assignment.

        Uses the same matcher as the main criterion so the positive set is
        exactly the one the detector is already trying to own.
        """
        zero = scores.sum() * 0.0
        with torch.no_grad():
            owners = self.criterion.matcher(
                polygons, scores, targets['bboxes'], targets['cls'], gt_groups
            )
        terms = []
        for batch_i, (query_idx, target_idx) in enumerate(owners):
            n_q = scores.shape[1]
            weight = torch.ones(n_q, device=scores.device)
            if len(query_idx) > 0:
                ratio = min(float((n_q - len(query_idx)) / max(len(query_idx), 1)), self.proposal_bal_cap)
                weight[query_idx.to(scores.device)] = ratio
            logits = scores[batch_i]
            target_cls = torch.full((n_q,), self.nc, dtype=torch.long, device=scores.device)
            if len(query_idx) > 0:
                target_cls[query_idx.to(scores.device)] = targets['cls'][target_idx.to(scores.device)]
            one_hot = F.one_hot(target_cls, self.nc + 1).float()
            bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction='none')
            p_t = one_hot * logits.sigmoid() + (1 - one_hot) * (1 - logits.sigmoid())
            focal = bce * (1.0 - p_t) ** 2.0
            # weight applies to the whole query row (fg and its empty column
            # both get the foreground gain), so the positive decision is what
            # is being re-balanced rather than one column of it.
            terms.append((focal.mean(-1) * weight).sum() / weight.sum())
        return torch.stack(terms).mean() if terms else zero

    @staticmethod
    def _point_radial_margin(points, polygons):
        """Signed radial margin for every point/polygon pair; positive is inside."""
        delta = points[:, None, :] - polygons[None, :, :2]
        distance = delta.square().sum(-1).sqrt()
        rays = polygons[:, 2:]
        angle = torch.remainder(torch.atan2(delta[..., 1], delta[..., 0]), 2 * torch.pi)
        position = angle * (rays.shape[-1] / (2 * torch.pi))
        lower = position.floor().long() % rays.shape[-1]
        upper = (lower + 1) % rays.shape[-1]
        fraction = position - position.floor()
        boundary = rays.T.gather(0, lower) * (1 - fraction) + rays.T.gather(0, upper) * fraction
        return boundary - distance

    def _query_presence_loss(self, logits, polygons, targets, gt_groups):
        """Balanced presence BCE with strict positive/negative geometric labels."""
        positive_logits, negative_logits = [], []
        start = 0
        margin = self.presence_negative_margin_px / float(targets['imgsz'])
        for batch_i, count in enumerate(gt_groups):
            gt = targets['bboxes'][start : start + count]
            start += count
            pred = polygons[batch_i]
            if count == 0:
                negative_logits.append(logits[batch_i])
                continue
            pred_center_in_gt = self._point_radial_margin(pred[:, :2], gt) >= 0
            gt_center_in_pred = self._point_radial_margin(gt[:, :2], pred).T >= 0
            positive = (pred_center_in_gt & gt_center_in_pred).any(1)
            safely_outside = (self._point_radial_margin(pred[:, :2], gt) <= -margin).all(1)
            contains_no_centroid = ~gt_center_in_pred.any(1)
            negative = safely_outside & contains_no_centroid & ~positive
            positive_logits.append(logits[batch_i][positive])
            negative_logits.append(logits[batch_i][negative])
        terms = []
        positives = (
            torch.cat([x for x in positive_logits if x.numel()])
            if any(x.numel() for x in positive_logits)
            else None
        )
        negatives = (
            torch.cat([x for x in negative_logits if x.numel()])
            if any(x.numel() for x in negative_logits)
            else None
        )
        if positives is not None:
            terms.append(F.binary_cross_entropy_with_logits(positives, torch.ones_like(positives)))
        if negatives is not None:
            terms.append(F.binary_cross_entropy_with_logits(negatives, torch.zeros_like(negatives)))
        return torch.stack(terms).mean() if terms else logits.sum() * 0.0

    def _appearance_imitation_loss(self, embeddings, polygons, scores, targets, gt_groups):
        """Pull small Hungarian owners toward reliable same-type exemplars.

        Teachers and unmatched nucleus-like competitors are detached.  The loss
        therefore shapes the small owner's appearance representation without
        redefining one-to-one ownership or changing inference.
        """
        projected = F.normalize(self.appearance_imitation_projection(embeddings), dim=-1)
        with torch.no_grad():
            owners = self.criterion.matcher(
                polygons.detach(), scores.detach(), targets['bboxes'], targets['cls'], gt_groups
            )

        students, student_types = [], []
        teacher_by_type = [[] for _ in range(self.nc)]
        duplicate_negatives = []
        start = 0
        angle_sine = torch.sin(polygons.new_tensor(2.0 * torch.pi / self.n_rays))
        for batch_i, ((query_idx, target_idx), count) in enumerate(zip(owners, gt_groups)):
            gt_offset = start
            gt = targets['bboxes'][start : start + count]
            gt_cls = targets['cls'][start : start + count]
            start += count
            if count == 0:
                continue
            # Star-convex polygon area from consecutive radial triangles.
            rays = gt[:, 2:]
            area = 0.5 * angle_sine * (rays * rays.roll(-1, dims=-1)).sum(-1)
            area = area * float(targets['imgsz']) ** 2
            query_idx = query_idx.to(polygons.device)
            owner_gt = target_idx.to(polygons.device) - gt_offset
            owner_types = gt_cls[owner_gt]
            owner_area = area[owner_gt]
            fired_type = scores[batch_i, query_idx].argmax(-1)
            small = owner_area <= self.appearance_imitation_small_area
            reliable = (~small) & (fired_type == owner_types)
            if small.any():
                students.append(projected[batch_i, query_idx[small]])
                student_types.append(owner_types[small])
            for class_id in range(self.nc):
                chosen = reliable & (owner_types == class_id)
                if chosen.any():
                    teacher_by_type[class_id].append(projected[batch_i, query_idx[chosen]].detach())

            matched = torch.zeros(polygons.shape[1], dtype=torch.bool, device=polygons.device)
            matched[query_idx] = True
            pred = polygons[batch_i]
            pred_center_in_gt = self._point_radial_margin(pred[:, :2], gt) >= 0
            gt_center_in_pred = self._point_radial_margin(gt[:, :2], pred).T >= 0
            duplicate = (~matched) & (pred_center_in_gt & gt_center_in_pred).any(1)
            if duplicate.any():
                duplicate_negatives.append(projected[batch_i, duplicate].detach())

        if not students:
            return projected.sum() * 0.0
        prototypes, prototype_types = [], []
        for class_id, chunks in enumerate(teacher_by_type):
            if chunks:
                prototypes.append(F.normalize(torch.cat(chunks).mean(0), dim=0))
                prototype_types.append(class_id)
        if not prototypes:
            return projected.sum() * 0.0
        prototypes = torch.stack(prototypes)
        if duplicate_negatives:
            duplicate_proto = F.normalize(torch.cat(duplicate_negatives).mean(0), dim=0)
            prototypes = torch.cat((prototypes, duplicate_proto[None]), 0)
        student = torch.cat(students)
        types = torch.cat(student_types)
        valid = torch.zeros_like(types, dtype=torch.bool)
        labels = torch.zeros_like(types)
        for index, class_id in enumerate(prototype_types):
            select = types == class_id
            valid |= select
            labels[select] = index
        if not valid.any():
            return projected.sum() * 0.0
        logits = student[valid] @ prototypes.T / self.appearance_imitation_temperature
        return F.cross_entropy(logits, labels[valid])

    def _instance_alignment_loss(self, embeddings, appearance_map, instance_mask, polygons, scores, targets, gt_groups):
        """Align final Hungarian owners with their own dense instance prototype.

        Nearby GT instances are the only negatives, concentrating the auxiliary
        objective on crowded cases. This leaves one-to-one ownership and the
        production classification/inference path unchanged.
        """
        query_features = F.normalize(self.instance_alignment_projection(embeddings), dim=-1)
        masks = (
            F.interpolate(
                instance_mask[:, None].to(appearance_map.device, torch.float32),
                size=appearance_map.shape[-2:],
                mode='nearest',
            )
            .squeeze(1)
            .long()
        )
        with torch.no_grad():
            owners = self.criterion.matcher(
                polygons.detach(), scores.detach(), targets['bboxes'], targets['cls'], gt_groups
            )

        terms = []
        start = 0
        radius = self.instance_alignment_radius_px / float(targets['imgsz'])
        for batch_i, ((query_idx, target_idx), count) in enumerate(zip(owners, gt_groups)):
            gt_offset = start
            gt = targets['bboxes'][start : start + count]
            start += count
            if count < 2 or len(query_idx) == 0:
                continue

            feature = appearance_map[batch_i]
            flat_feature = feature.flatten(1)
            flat_mask = masks[batch_i].flatten()
            # Accumulate all instance means in one pass, including a background
            # bin. No per-instance nonzero/boolean checks synchronizing CUDA.
            sums = flat_feature.new_zeros((count + 1, feature.shape[0])).index_add(
                0, flat_mask, flat_feature.T
            )
            sizes = flat_feature.new_zeros(count + 1).index_add(
                0, flat_mask, flat_feature.new_ones(flat_mask.shape)
            )
            prototype = F.normalize(sums[1:] / sizes[1:, None].clamp_min(1), dim=-1)
            valid_gt = sizes[1:] > 0

            query_idx = query_idx.to(polygons.device)
            local_targets = target_idx.to(polygons.device) - gt_offset
            distances = torch.cdist(gt[:, :2], gt[:, :2])
            owner_distances = distances[local_targets]
            neighbors = (owner_distances > 0) & (owner_distances <= radius) & valid_gt[None]
            ranked = owner_distances.masked_fill(~neighbors, float('inf'))
            # Stable sorting preserves the original GT-index tie break.
            nearest = ranked.argsort(dim=-1, stable=True)[:, : self.instance_alignment_max_negatives]
            negative_valid = neighbors.gather(1, nearest)
            candidates = torch.cat((local_targets[:, None], nearest), dim=1)
            logits = (
                query_features[batch_i, query_idx, None] * prototype[candidates]
            ).sum(-1) / self.instance_alignment_temperature
            candidate_valid = torch.cat((torch.ones_like(local_targets[:, None], dtype=torch.bool), negative_valid), 1)
            logits = logits.masked_fill(~candidate_valid, float('-inf'))
            eligible = valid_gt[local_targets] & negative_valid.any(-1)
            losses = F.cross_entropy(logits, torch.zeros_like(local_targets), reduction='none')
            terms.append((losses.masked_fill(~eligible, 0).sum(), eligible.sum()))

        if not terms:
            return query_features.sum() * 0.0
        return torch.stack([loss for loss, _ in terms]).sum() / torch.stack([n for _, n in terms]).sum().clamp_min(1)

    def loss(self, batch, preds=None):
        """DETR loss + seed-map seg loss.

        Args:
            batch: dict with img (B,3,H,W), batch_idx (M,), cls (M,),
                bboxes (M, 2+n_rays), seed_map (B,1,H,W).
            preds: optional pre-computed decoder output (validator path).
        """
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()

        img = batch['img']
        device = img.device
        bs = img.shape[0]
        batch_idx = batch['batch_idx'].to(device, torch.long).view(-1)
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': batch['cls'].to(device, torch.long).view(-1),
            'bboxes': batch['bboxes'].to(device),
            'batch_idx': batch_idx,
            'gt_groups': gt_groups,
            'imgsz': img.shape[-1],
        }

        computed_here = preds is None
        if computed_here:
            preds = self.predict(img, batch=targets)
        else:
            preds = preds[1] if isinstance(preds, tuple) and len(preds) == 2 else preds

        dec_polygons, dec_scores, enc_polygons, enc_scores, dn_meta = preds
        dn_polygons = dn_scores = dn_enc_polygons = dn_enc_scores = None
        if dn_meta is not None:
            if isinstance(dn_meta, dict):
                # nucleus-CDN: the DN block is the leading ``n_dn`` queries of
                # every layer; split it from the real query block.
                n_dn = int(dn_meta['n_dn'])
                dn_polygons, dec_polygons = dec_polygons.split(
                    [n_dn, dec_polygons.shape[2] - n_dn], dim=2
                )
                dn_scores, dec_scores = dec_scores.split([n_dn, dec_scores.shape[2] - n_dn], dim=2)
                dn_enc_polygons, enc_polygons = enc_polygons.split(
                    [n_dn, enc_polygons.shape[1] - n_dn], dim=1
                )
                dn_enc_scores, enc_scores = enc_scores.split([n_dn, enc_scores.shape[1] - n_dn], dim=1)
            else:
                dn_bs, dn_num_quad, _ = dn_meta
                dn_num_group = dn_num_quad * 2
                split = [dn_bs * dn_num_group, bs]
                dec_polygons = dec_polygons.split(split, dim=1)[1]
                dec_scores = dec_scores.split(split, dim=1)[1]
                enc_polygons = enc_polygons.split(split, dim=1)[1]
                enc_scores = enc_scores.split(split, dim=1)[1]

        dec_polygons = torch.cat([enc_polygons.unsqueeze(0), dec_polygons], dim=0)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores], dim=0)

        # Respawn pass: the decoder records which stack index is the stage-1
        # final layer; here the encoder layer is prepended (+1 offset) so the
        # criterion can residual-match the appended respawn layer.
        rs_idx = getattr(self.detr, '_respawn_layer_idx', None)
        self.criterion.respawn_layer_idx = (rs_idx + 1) if rs_idx is not None else None

        loss = self.criterion((dec_polygons, dec_scores), targets)
        total = sum(loss.values())

        proposal_bal = torch.zeros((), device=device)
        if getattr(self, 'proposal_bal_weight', 0.0) > 0 and self.no_object:
            # Re-balance the encoder proposal layer only: this is where the
            # foreground/empty decision is committed.
            proposal_bal = self._balanced_proposal_loss(
                dec_polygons[0], dec_scores[0], targets, gt_groups
            )
            total = total + self.proposal_bal_weight * proposal_bal

        nucleus_cdn_loss = torch.zeros((), device=device)
        if dn_scores is not None and getattr(self, 'nucleus_cdn_weight', 0.0) > 0:
            # Supervise the canonical GT-polygon queries (proposal + every
            # decoder layer) to put their TRUE GT class above the empty
            # column. Class-only: geometry is canonical/frozen. Padded DN rows
            # (label == nc / no-object) are trained to stay empty, so the
            # denominator is the valid-slot mask.
            dn_layer_scores = torch.cat([dn_enc_scores.unsqueeze(0), dn_scores], dim=0)
            nucleus_cdn_loss = self._nucleus_cdn_class_loss(dn_layer_scores, dn_meta)
            total = total + self.nucleus_cdn_weight * nucleus_cdn_loss

        seed = getattr(self, '_last_seed', None)
        if seed is not None and self.lambda_seg > 0:
            if self.seed_map_target and 'seed_map' in batch:
                seg_gt = batch['seed_map'].to(device)
                if seg_gt.shape[-2:] != seed.shape[-2:]:
                    seg_gt = F.interpolate(seg_gt, size=seed.shape[-2:], mode='bilinear', align_corners=False)
                seg = F.l1_loss(seed, seg_gt)
            elif 'np_mask' in batch:
                seg_gt = batch['np_mask'].to(device)
                if seg_gt.shape[-2:] != seed.shape[-2:]:
                    seg_gt = F.interpolate(seg_gt, size=seed.shape[-2:], mode='nearest')
                seg = F.binary_cross_entropy_with_logits(seed, seg_gt)
            else:
                seg = None
            if seg is not None:
                total = total + self.lambda_seg * seg

        aux_obj = torch.zeros((), device=device)
        aux_type = torch.zeros((), device=device)
        aux_dense = getattr(self, '_last_aux_dense', None) if computed_here else None
        if aux_dense is not None and self.aux_dense_weight > 0:
            obj_logits, type_logits = aux_dense
            obj_gt = batch['np_mask'].to(device)
            semantic_gt = batch['semantic_mask'].to(device, torch.long)
            obj_gt = F.interpolate(obj_gt, size=obj_logits.shape[-2:], mode='nearest')
            semantic_gt = (
                F.interpolate(semantic_gt.unsqueeze(1).float(), size=type_logits.shape[-2:], mode='nearest')
                .squeeze(1)
                .long()
            )
            aux_obj = F.binary_cross_entropy_with_logits(obj_logits, obj_gt)
            fg = semantic_gt > 0
            if fg.any():
                aux_type = F.cross_entropy(type_logits.permute(0, 2, 3, 1)[fg], semantic_gt[fg] - 1)
            else:
                aux_type = type_logits.sum() * 0.0
            total = total + self.aux_dense_weight * (aux_obj + self.aux_dense_type_weight * aux_type)

        seg_fg_margin = torch.zeros((), device=device)
        seed = getattr(self, '_last_seed', None)
        if seed is not None and self.seg_fg_margin_weight > 0 and self.no_object:
            seg_fg_margin = self._segmentation_foreground_margin(dec_polygons, dec_scores, targets, gt_groups, seed)
            total = total + self.seg_fg_margin_weight * seg_fg_margin

        viable_owner_fg = torch.zeros((), device=device)
        if getattr(self, 'viable_owner_fg_weight', 0.0) > 0 and self.no_object:
            viable_owner_fg = self._viable_owner_foreground_loss(
                dec_polygons[-1], dec_scores[-1], targets, gt_groups
            )
            total = total + self.viable_owner_fg_weight * viable_owner_fg

        appearance_loss = torch.zeros((), device=device)
        # The validator first calls predict() and then passes those predictions
        # back into loss(); the caches therefore belong to this same batch and
        # should contribute validation telemetry as well as training loss.
        appearance_logits = getattr(self, '_last_appearance_logits', None)
        if appearance_logits is not None and self.appearance_weight > 0:
            appearance_gt = F.interpolate(
                batch['np_mask'].to(device), size=appearance_logits.shape[-2:], mode='nearest'
            )
            probability = appearance_logits.sigmoid()
            bce = F.binary_cross_entropy_with_logits(appearance_logits, appearance_gt)
            overlap = (probability * appearance_gt).flatten(1).sum(1)
            dice = 1.0 - ((2.0 * overlap + 1.0) / (
                probability.flatten(1).sum(1) + appearance_gt.flatten(1).sum(1) + 1.0
            )).mean()
            appearance_loss = bce + dice
            total = total + self.appearance_weight * appearance_loss

        presence_loss = torch.zeros((), device=device)
        presence_logits = getattr(self.detr, '_last_presence_logits', None)
        presence_polygons = getattr(self.detr, '_last_presence_polygons', None)
        if presence_logits is not None and presence_polygons is not None and self.presence_weight > 0:
            presence_loss = self._query_presence_loss(presence_logits, presence_polygons, targets, gt_groups)
            total = total + self.presence_weight * presence_loss

        imitation_loss = torch.zeros((), device=device)
        appearance_embeddings = getattr(self.detr, '_last_appearance_embeddings', None)
        if appearance_embeddings is not None and self.appearance_imitation_weight > 0:
            imitation_loss = self._appearance_imitation_loss(
                appearance_embeddings, dec_polygons[-1], dec_scores[-1], targets, gt_groups
            )
            total = total + self.appearance_imitation_weight * imitation_loss

        instance_alignment_loss = torch.zeros((), device=device)
        appearance_map = getattr(self, '_last_appearance_map', None)
        appearance_queries = getattr(self.detr, '_last_appearance_queries', None)
        if (
            appearance_queries is not None
            and appearance_map is not None
            and self.instance_alignment_weight > 0
            and 'instance_mask' in batch
        ):
            instance_alignment_loss = self._instance_alignment_loss(
                appearance_queries,
                appearance_map,
                batch['instance_mask'],
                dec_polygons[-1],
                dec_scores[-1],
                targets,
                gt_groups,
            )
            total = total + self.instance_alignment_weight * instance_alignment_loss

        loss_items = torch.as_tensor(
            [
                (loss['loss_class'] + loss.get('loss_class_aux', 0.0)).detach(),
                (loss['loss_xy'] + loss.get('loss_xy_aux', 0.0)).detach(),
                (loss['loss_l1'] + loss.get('loss_l1_aux', 0.0)).detach(),
                (loss['loss_piou'] + loss.get('loss_piou_aux', 0.0)).detach(),
                (loss['loss_smooth'] + loss.get('loss_smooth_aux', 0.0)).detach(),
            ],
            device=device,
        )
        if self.aux_dense_weight > 0:
            loss_items = torch.cat([loss_items, torch.stack([aux_obj.detach(), aux_type.detach()])])
        if self.seg_fg_margin_weight > 0:
            loss_items = torch.cat([loss_items, seg_fg_margin.detach().reshape(1)])
        if getattr(self, 'viable_owner_fg_weight', 0.0) > 0:
            loss_items = torch.cat([loss_items, viable_owner_fg.detach().reshape(1)])
        if getattr(self, 'nucleus_cdn_weight', 0.0) > 0:
            loss_items = torch.cat([loss_items, nucleus_cdn_loss.detach().reshape(1)])
        if getattr(self, 'proposal_bal_weight', 0.0) > 0:
            loss_items = torch.cat([loss_items, proposal_bal.detach().reshape(1)])
        if self.appearance_weight > 0:
            loss_items = torch.cat([loss_items, appearance_loss.detach().reshape(1)])
        if self.presence_weight > 0:
            loss_items = torch.cat([loss_items, presence_loss.detach().reshape(1)])
        if self.appearance_imitation_weight > 0:
            loss_items = torch.cat([loss_items, imitation_loss.detach().reshape(1)])
        if self.instance_alignment_weight > 0:
            loss_items = torch.cat([loss_items, instance_alignment_loss.detach().reshape(1)])
        return total, loss_items

    # -- helpers ------------------------------------------------------------

    def fuse(self, verbose=True):
        """No-op (DETR decoder has no Conv-BN pairs to fold)."""
        return self

    def load_nulite_ckpt(self, ckpt_path):
        """Load NuLite weights into the seg stream (dual) or encoder+decoder (nulite).

        NuLite-T was trained with 0.5/0.5 normalization (input in [-1, 1]).
        When the checkpoint lands in the single-stream ``nulite`` encoder (the
        curriculum / oracle-init path), flag the model so ``predict`` feeds it
        [-1, 1] input; a frozen oracle encoder otherwise receives [0, 1] images
        and its BN stats see a shifted distribution. The ``dual`` stream applies
        its own (x-0.5)/0.5 internally, so it is left alone. The ``two_stream``
        seg oracle (full U-Net) loads via its own ``load_ckpt`` remap.
        """
        if getattr(self, 'backbone', 'nulite') == 'two_stream':
            self.seg_encoder.load_ckpt(ckpt_path)
            return
        if getattr(self, 'backbone', 'nulite') == 'frozen':
            self.seg_encoder.load_ckpt(ckpt_path)
            return
        if getattr(self, 'backbone', 'nulite') == 'shared':
            self._load_shared_nulite_ckpt(ckpt_path)
            return
        target = self.seg_encoder if getattr(self, 'backbone', 'nulite') == 'dual' else self.encoder_decoder
        target.load_nulite_ckpt(ckpt_path)
        if getattr(self, 'backbone', 'nulite') == 'nulite':
            self.encoder_input_norm = '11'

    def _load_shared_nulite_ckpt(self, ckpt_path):
        """Load a NuLite-T oracle checkpoint into the shared encoder+decoder.

        The oracle file stores ``encoder.fast_vit.*``, ``decoder.*``,
        ``decoder0.0.*`` and ``np_head.*``; the shared model holds the same
        tensors as ``encoder_decoder.encoder.*``, ``encoder_decoder.*``,
        ``decoder0.*`` and ``np_head.*``. Every own key must find a shape-match
        match, so a partial load cannot silently leave the oracle uninitialised.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        config = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
        checkpoint_kind = config.get('oracle_kind')
        if checkpoint_kind is not None and checkpoint_kind != self.oracle_kind:
            raise ValueError(f'oracle checkpoint kind is {checkpoint_kind!r}, but model expects {self.oracle_kind!r}')
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
        missing = [k for k in own if k not in remapped and k.startswith(('encoder_decoder.', 'decoder0.', 'np_head.'))]
        print(
            f'NuLite shared-stream load: {len(remapped)}/{len(own)} keys matched'
            + (f' (missing {len(missing)}, e.g. {missing[:4]})' if missing else '')
        )
        if missing:
            raise ValueError(
                f'oracle checkpoint is incompatible with the shared stream: {len(missing)} parameters were not loaded; '
                f'first keys: {missing[:4]}'
            )
        self.load_state_dict(remapped, strict=False)

    def freeze_encoder_decoder_np(self):
        """Freeze encoder + decoder + decoder0 + NP seed head."""
        if getattr(self, 'backbone', 'nulite') == 'shared':
            self.encoder_decoder.encoder.requires_grad_(False)
            return
        if getattr(self, 'backbone', 'nulite') == 'two_stream':
            self.seg_encoder.requires_grad_(False)
            self.encoder_decoder.freeze_encoder()
            if getattr(self.encoder_decoder, 'use_decoder', True):
                self.encoder_decoder.freeze_decoder()
            return
        if getattr(self, 'backbone', 'nulite') == 'frozen':
            # Only the DETR head trains; the whole oracle stays frozen unless an
            # experiment explicitly opted into thawing the encoder / U-Net.
            self.seg_encoder.requires_grad_(False)
            self.oracle_encoder_thawed = False
            self.oracle_full_thaw = False
            self.seg_encoder.eval()
            return
        if getattr(self, 'backbone', 'nulite') == 'dual':
            self.seg_encoder.freeze_encoder()
            self.encoder_decoder.freeze_backbone()
            if getattr(self, 'np_head', None) is not None:
                self.np_head.requires_grad_(False)
            return
        if getattr(self, 'backbone', 'nulite') == 'yolo11':
            self.encoder_decoder.freeze_backbone()
            if getattr(self, 'np_head', None) is not None:
                self.np_head.requires_grad_(False)
            return
        self.encoder_decoder.freeze_encoder()
        if getattr(self, 'use_decoder', True):
            self.encoder_decoder.freeze_decoder()
            self.decoder0.requires_grad_(False)
            self.np_head.requires_grad_(False)

    def unfreeze_all(self):
        """Unfreeze trainable detector parameters while preserving the frozen oracle."""
        self.requires_grad_(True)
        if getattr(self, 'backbone', None) in ('two_stream', 'frozen'):
            self.seg_encoder.requires_grad_(False)
            self.seg_encoder.eval()
            if getattr(self, 'oracle_encoder_thawed', False):
                self.thaw_oracle_encoder(full_unet=getattr(self, 'oracle_full_thaw', False))

    def thaw_oracle_encoder(self, full_unet: bool = False):
        """Expose the frozen oracle U-Net to the optimiser.

        Used by the ``frozen`` layout when an experiment decides the frozen
        features are the bottleneck. With ``full_unet=False`` only the FastViT
        encoder adapts (``detr_frozen_oracle_unfreeze_encoder``). With
        ``full_unet=True`` (``detr_frozen_oracle_unfreeze_decoder``) the whole
        U-Net thaws — encoder, upsample decoder and ``decoder0`` — recreating
        the dedicated-detection-encoder specialization t34 learned, while only
        ``np_head`` stays frozen so the foreground map head itself cannot be
        rewritten. BN statistics across the U-Net follow training.
        """
        if getattr(self, 'backbone', None) != 'frozen':
            raise ValueError('thaw_oracle_encoder() requires backbone="frozen"')
        enc = self.seg_encoder.encoder_decoder.encoder
        enc.requires_grad_(True)
        enc.train(self.training)
        if full_unet:
            self.oracle_full_thaw = True
            rest = self.seg_encoder.encoder_decoder
            for n, p in rest.named_parameters():
                if not n.startswith('encoder.'):
                    p.requires_grad = True
            rest.train(self.training)
            self.seg_encoder.decoder0.requires_grad_(True)
            self.seg_encoder.decoder0.train(self.training)
            self.seg_encoder.np_head.requires_grad_(False)
            self.seg_encoder.np_head.eval()
        self.oracle_encoder_thawed = True
        return self
