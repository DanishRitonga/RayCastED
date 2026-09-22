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
        use_decoder: bool = True,
        shared_layers: bool = False,
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
        self.aux_dense_weight = aux_dense_weight
        self.aux_dense_type_weight = aux_dense_type_weight
        self.seg_fg_margin_weight = seg_fg_margin_weight
        self.seg_fg_margin = seg_fg_margin
        self.seg_fg_radius_px = seg_fg_radius_px
        self.seg_fg_margin_layer_weights = tuple(float(w) for w in (seg_fg_margin_layer_weights or ()))
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
            self.seg_encoder = NuLiteSegOracle(variant=variant, separable_decoder=True)
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
            query_block_size=query_block_size,
            content_top=self.content_top,
            content_top_channels=dims[3] if (self.content_top and len(dims) > 3) else 0,
            seed_resize=seed_resize,
            seed_resize_frac=seed_resize_frac,
            layer_schedule=detr_layer_schedule,
            hybrid_peaks=hybrid_peaks,
            seed_level=seed_level and backbone in ('nulite', 'two_stream'),
            seed_level_stride=seed_level_stride,
            recast=recast,
            recast_after_layer=recast_after_layer,
            recast_radius_px=recast_radius_px,
            respawn=respawn,
            respawn_peaks=respawn_peaks,
            respawn_min_seed=respawn_min_seed,
        )
        self.aux_dense_head = DenseAuxHead((dims[0], dims[1], dims[2]), nc) if self.aux_dense_weight > 0 else None

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
        d = self.encoder_decoder(x)
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
        elif getattr(self, 'backbone', 'nulite') == 'two_stream':
            # Frozen NuLite-T oracle seg stream was trained on [-1, 1] input
            # (0.5/0.5 normalization); the det stream expects raw [0, 1]. The
            # 2-channel CE logits collapse to a foreground logit (sigmoid ==
            # P(nucleus)) for the DETR seed cue.
            seg_x = (x - 0.5) / 0.5
            seg_logits = self.seg_encoder(seg_x)
            seed = seg_logits[:, 1:2] - seg_logits[:, 0:1]
        elif getattr(self, 'use_decoder', True):
            x0 = self.decoder0(x)
            xt = torch.cat([x0, d['b1']], dim=1)
            seed = self.np_head(xt)
        else:
            seed = None
        self._last_seed = seed
        self._last_aux_dense = None
        if self.training and self.aux_dense_head is not None:
            self._last_aux_dense = self.aux_dense_head((d['b3'], d['b4'], d['b5']))

        top_feat = d.get('top') if getattr(self.detr, 'content_top', False) else None
        out = self.detr([d['b3'], d['b4'], d['b5']], batch, seed_map=seed, top_feat=top_feat)
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
        if dn_meta is not None:
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
        target = self.seg_encoder if getattr(self, 'backbone', 'nulite') == 'dual' else self.encoder_decoder
        target.load_nulite_ckpt(ckpt_path)
        if getattr(self, 'backbone', 'nulite') == 'nulite':
            self.encoder_input_norm = '11'

    def freeze_encoder_decoder_np(self):
        """Freeze encoder + decoder + decoder0 + NP seed head."""
        if getattr(self, 'backbone', 'nulite') == 'two_stream':
            self.seg_encoder.requires_grad_(False)
            self.encoder_decoder.freeze_encoder()
            if getattr(self.encoder_decoder, 'use_decoder', True):
                self.encoder_decoder.freeze_decoder()
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
        if getattr(self, 'backbone', None) == 'two_stream':
            self.seg_encoder.requires_grad_(False)
            self.seg_encoder.eval()
