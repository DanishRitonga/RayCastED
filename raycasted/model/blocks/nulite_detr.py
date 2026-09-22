"""NuLite DETR Decoder — single query-based head for the NuLite seg backbone.

Self-contained replacement for the old (removed) RT-DETR RayCast decoder. It
subclasses ultralytics' stock ``RTDETRDecoder`` only for the deformable
transformer decoder machinery (self-attention, deformable cross-attention over
multi-scale features, iterative refinement), and inlines the ray-polygon
specific parts that the RT-DETR version used to own:

- ray regression heads (enc_bbox_head / dec_bbox_head output ``2 + n_rays``
  instead of 4), with xy in sigmoid space and rays in exp space
- 2-dim centroid reference points (query_pos_head takes 2 inputs, not 4)
- **query selection from the NP seed map**: local maxima (InstanSeg-style
  ``torch_peak_local_max``) seed one query per nucleus peak, content is
  bilinearly sampled from the projected P2 features, anchors are
  base-radius circles — no grid anchors, no encoder score head

Denoising is disabled (``nd=0``): it assumes 4-dim boxes and adds no value here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import MLP, MSDeformAttn, multi_scale_deformable_attn_pytorch

from raycasted.data.etl.utils import constants as _const


class SegmentationQueryRecast(nn.Module):
    """Local content correction; nine samples per query, no dense query/pixel map."""

    def __init__(self, channels: int, radius_px: float = 4.0):
        super().__init__()
        if radius_px < 0:
            raise ValueError('recast radius must be nonnegative')
        self.radius_px = radius_px
        self.gate = nn.Sequential(nn.LayerNorm(channels + 2), nn.Linear(channels + 2, 1))
        self.projection = nn.Linear(channels, channels, bias=False)
        # Exact identity initially, with a nonzero gradient into the projection.
        nn.init.zeros_(self.projection.weight)
        offsets = torch.tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=torch.float32)
        self.register_buffer('offsets', offsets, persistent=False)

    def forward(self, queries, positions, scores, features, seed_map):
        """Correct content using detached spatial/objectness routing evidence."""
        h, w = seed_map.shape[-2:]
        scale = positions.new_tensor([w, h])
        grid = positions.detach().unsqueeze(2) + self.offsets.to(positions) * self.radius_px / scale
        grid = grid * 2 - 1
        local = F.grid_sample(features, grid, align_corners=False, padding_mode='border')
        # Routing evidence cannot improve its own loss by manipulating logits.
        fg = F.grid_sample(seed_map.detach().sigmoid(), grid, align_corners=False, padding_mode='border')
        descriptor = (local * fg).sum(-1) / fg.sum(-1).clamp_min(1e-6)
        descriptor = descriptor.transpose(1, 2)
        evidence = fg.mean(-1).transpose(1, 2)
        # The score heads are trained with independent sigmoid/focal targets,
        # including the explicit no-object column.  Keep routing semantics
        # aligned with matching and inference; a softmax here would make the
        # gate depend on the number and relative scale of real-class logits.
        bg = scores.detach()[..., -1:].sigmoid()
        gate = torch.sigmoid(self.gate(torch.cat((queries, bg, evidence), -1)))
        gate = gate * bg * evidence
        return queries + gate * self.projection(descriptor)


class _ScheduledMSDeformAttn(MSDeformAttn):
    """MSDeformAttn with an optional per-layer soft attention-level mask.

    When ``level_mask`` is a length-``n_levels`` boolean tensor, the attention
    logits of disabled levels are set to ``-inf`` before the softmax, so a
    decoder layer attends a single feature scale (LSP-DETR-style per-layer
    round-robin over b5/b4/b3). With ``level_mask=None`` it is numerically
    identical to the parent module, so the locked recipe is unaffected.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level_mask = None

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """Forward with an optional soft mask over attention levels."""
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        if self.level_mask is not None:
            lm = self.level_mask.to(device=attention_weights.device, dtype=torch.bool)
            lm = lm.view(1, 1, 1, self.n_levels, 1).expand(bs, len_q, self.n_heads, self.n_levels, self.n_points)
            attention_weights = attention_weights.view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
            attention_weights = attention_weights.masked_fill(~lm, float('-inf'))
            attention_weights = attention_weights.reshape(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


def encode_polygon(polygon: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encode polygon to logit/log space for iterative refinement.

    xy: inverse_sigmoid (logit space)
    rays: log (log space, inverse of exp)
    """
    xy = polygon[..., :2].clamp(eps, 1 - eps)
    rays = polygon[..., 2:].clamp(min=eps)
    return torch.cat([torch.log(xy / (1 - xy)), torch.log(rays)], dim=-1)


def decode_polygon(logits: torch.Tensor, max_ray: float = 1.0) -> torch.Tensor:
    """Decode polygon from logit/log space.

    xy: sigmoid (bounded [0,1]); rays: exp clamped to ``max_ray`` (normalized
    to crop_size, so physical rays are <= 1.0).

    Logits are clamped to a finite range BEFORE exp/sigmoid: border anchors are
    masked with ``inf`` in logit space (``_generate_ray_anchors``), and
    ``exp(inf).clamp()`` yields a NaN gradient (0 * inf) that poisons weights.
    Clamping the logit first bounds the gradient (``exp(clamp(inf, 20))`` is a
    large finite value with a 0 clamp-gradient, so no inf*0).
    """
    logits = logits.clamp(min=-20.0, max=20.0)
    rays = logits[..., 2:].exp().clamp(max=max_ray)
    return torch.cat([logits[..., :2].sigmoid(), rays], dim=-1)


class NuLiteDETRDecoder(RTDETRDecoder):
    """DETR decoder with seed-map local-maxima query selection.

    Uses the stock ultralytics deformable transformer decoder (self-attn +
    deformable cross-attn over the NuLite b3/b4/b5 features, iterative
    refinement) but seeds its ``nq`` queries from the local maxima of a
    full-resolution NP seed map instead of from a learned encoder score head.
    Each peak becomes one query: centroid = reference point, content =
    bilinear-sampled projected P2 features, anchor = base-radius circle.
    """

    def __init__(
        self,
        nc: int = 5,
        ch: tuple = (64, 128, 256),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 3,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 0,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        n_rays: int | None = None,
        seed_threshold: float = 0.5,
        peak_distance: int = 3,
        grid_size: float = 0.05,
        query_selection: str = 'grid',
        hybrid_peaks: int = 100,
        seed_feature: bool = True,
        no_object: bool = True,
        seed_in_content: bool = True,
        mds: bool = True,
        shared_layers: bool = False,
        query_block_size: float = 0.0,
        content_top: bool = False,
        content_top_channels: int = 0,
        seed_resize: str = 'bilinear',
        seed_resize_frac: float = 0.25,
        layer_schedule: str = '',
        seed_level: bool = False,
        seed_level_stride: int = 4,
        recast: bool = False,
        recast_after_layer: int = 3,
        recast_radius_px: float = 4.0,
        respawn: bool = False,
        respawn_peaks: int = 64,
        respawn_min_seed: float = 0.5,
    ):
        self.n_rays = n_rays or _const.N_RAYS
        self.raycast_dim = 2 + self.n_rays
        # Seed-as-extra-level: give the deformable cross-attention a 4th level
        # holding the (projected) seed map, so objectness evidence is read via a
        # LEARNED attention weight at full/controlled resolution instead of being
        # force-blended into every feature channel (the ``seed_feature`` F path).
        self.seed_level = seed_level
        self.seed_level_stride = int(seed_level_stride)
        # Respawn (recall-repair second decoder pass): after the stage-1 grid
        # decode, seed peaks NOT covered by any fired query polygon are used to
        # re-anchor the most-confidently-EMPTY queries; the shared decoder
        # layers run a second pass starting from those anchors. Stage-2 outputs
        # are appended as extra aux layers; the loss matches them against GT
        # UNMATCHED in stage 1 (residual supervision - duplicates are trained
        # away, not NMS'd).
        self.respawn = respawn
        self.respawn_peaks = int(respawn_peaks)
        self.respawn_min_seed = float(respawn_min_seed)
        base_ch = tuple(ch)
        super_ch = base_ch + (1,) if seed_level else base_ch
        super().__init__(
            nc=nc,
            ch=super_ch,
            hd=hd,
            nq=nq,
            ndp=ndp,
            nh=nh,
            ndl=ndl,
            d_ffn=d_ffn,
            dropout=dropout,
            act=act,
            eval_idx=eval_idx,
            nd=nd,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
        )
        if recast and (respawn or not no_object or not 1 <= recast_after_layer < ndl):
            raise ValueError('recast requires no_object, no respawn, and an insertion layer before the final layer')
        if recast and recast_after_layer > self.decoder.eval_idx:
            raise ValueError('recast must precede the evaluation output layer')
        self.recast_after_layer = recast_after_layer
        self.recast_module = SegmentationQueryRecast(hd, recast_radius_px) if recast else None
        self.seed_threshold = seed_threshold
        self.peak_distance = peak_distance
        self.grid_size = grid_size
        self.query_selection = query_selection
        self.hybrid_peaks = hybrid_peaks
        self.seed_feature = seed_feature
        self.no_object = no_object
        self.seed_in_content = seed_in_content
        self.mds = mds
        # LSP-DETR query-init knobs: query_block_size = grid cell diameter in
        # PIXELS (0 = legacy normalized ``grid_size`` grid); content_top samples
        # query content from the top-level (stride-32) feature (LSP FeatureSampling
        # on their stage-4 neck B) instead of the projected P2 map.
        self.query_block_size = query_block_size
        self.content_top = content_top
        # How the seed prob is downsampled to each feature scale for the F channel:
        # 'bilinear' (LOCKED recipe) averages a cell, washing out small nuclei at
        # coarse scales; 'max' (adaptive_max_pool2d) is peak-preserving but keeps
        # isolated background spikes / neighbor peaks; 'topk' averages the top
        # `seed_resize_frac` of pixels per cell (robust peak-mean: preserves a small
        # nucleus's center while rejecting lone spikes); 'avg' is the pure-average
        # control.
        self.seed_resize = seed_resize
        self.seed_resize_frac = seed_resize_frac
        if content_top and content_top_channels > 0:
            self.top_content_conv = nn.Conv2d(content_top_channels, hd, 1, bias=False)
            self.top_content_norm = nn.LayerNorm(hd)
        elif content_top:
            raise ValueError('content_top=True requires content_top_channels > 0')
        if shared_layers:
            # weight-shared transformer layers: reuse ONE DeformableTransformerDecoderLayer
            # instance across all ndl steps (params-only saving, FLOPs unchanged). Per-layer
            # dec_bbox_head/dec_score_head stay distinct so iterative refinement still varies.
            self.decoder.layers = nn.ModuleList([self.decoder.layers[0]] * self.decoder.num_layers)
        self.shared_layers = shared_layers

        # Per-layer cross-attention scale schedule (LSP-DETR-style round-robin).
        # ``layer_schedule`` is a comma-separated list of feature-level indices
        # (0 = b3/st4 finest, 1 = b4/st8, 2 = b5/st16 coarsest); entry i is the
        # scale decoder layer i attends. Cycled if shorter than ``ndl``; empty
        # string disables it (all scales in every layer = locked recipe).
        self.layer_schedule = None
        self._level_masks = None
        sched = [int(x) for x in str(layer_schedule).replace(' ', '').split(',') if x != '']
        if sched:
            n_levels = self.decoder.layers[0].cross_attn.n_levels
            for v in sched:
                if not 0 <= v < n_levels:
                    raise ValueError(f'layer_schedule entry {v} out of range [0, {n_levels})')
            # Swap in the maskable cross-attn module (weights preserved) for every
            # distinct layer; shared layers alias ONE module across all steps.
            swapped = {}
            for layer in self.decoder.layers:
                ca = layer.cross_attn
                if id(ca) not in swapped:
                    new_ca = _ScheduledMSDeformAttn(ca.d_model, ca.n_levels, ca.n_heads, ca.n_points)
                    new_ca.load_state_dict(ca.state_dict())
                    swapped[id(ca)] = new_ca
                layer.cross_attn = swapped[id(ca)]
            self.layer_schedule = sched
            self._level_masks = []
            for i in range(len(self.decoder.layers)):
                m = torch.zeros(n_levels, dtype=torch.bool)
                m[sched[i % len(sched)]] = True
                self._level_masks.append(m)

        # When the seed map is concatenated as an extra feature channel, the
        # input projection must accept one more channel per scale.
        if self.seed_feature:
            self.input_proj = nn.ModuleList(
                nn.Sequential(nn.Conv2d(c + 1, hd, 1, bias=False), nn.BatchNorm2d(hd)) for c in ch
            )
        # Seed-as-4th-level projection: a STRIDED conv (not bilinear/avg-pool —
        # averaging washes tiny nuclei out; see seed_resize diagnostic) that maps
        # the full-res seed map to stride ``seed_level_stride`` and hd channels.
        if self.seed_level:
            seed_proj = nn.Sequential(
                nn.Conv2d(1, hd, 3, stride=self.seed_level_stride, padding=1, bias=False),
                nn.BatchNorm2d(hd),
            )
            if self.seed_feature:
                self.input_proj.append(seed_proj)  # [b3+1, b4+1, b5+1, seed]
            else:
                self.input_proj[3] = seed_proj  # super built 4 levels (last = 1-ch 1x1)

        # LSP-DETR-style explicit "no object" class: score heads output nc+1
        # logits, unmatched queries train against the ∅ column via focal loss.
        # When seed_in_content, the raw seed logit is concatenated to the head
        # input (undiluted objectness read, differentiable into the NP head),
        # so the Linear input width is hd+1.
        score_in = hd + 1 if seed_in_content else hd
        if self.no_object or seed_in_content:
            self.enc_score_head = nn.Linear(score_in, nc + 1 if self.no_object else nc)
            self.dec_score_head = nn.ModuleList(
                [nn.Linear(score_in, nc + 1 if self.no_object else nc) for _ in range(ndl)]
            )
            b = -math.log(99.0) / 80.0 * nc  # bias_init_with_prob(0.01), ultralytics scaling
            nn.init.constant_(self.enc_score_head.bias, b)
            for h in self.dec_score_head:
                nn.init.constant_(h.bias, b)

        # Ray-polygon heads (2+n_rays outputs) and 2-dim centroid query pos.
        self.enc_bbox_head = MLP(hd, hd, self.raycast_dim, num_layers=3)
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, self.raycast_dim, num_layers=3) for _ in range(ndl)])
        self.query_pos_head = MLP(2, 2 * hd, hd, num_layers=2)
        self._reset_raycast_params()

        if self.seed_level:
            self.stride = torch.tensor([4.0, 8.0, 16.0, float(self.seed_level_stride)])
        else:
            self.stride = torch.tensor([4.0, 8.0, 16.0])

    def _reset_raycast_params(self):
        """Zero-init bbox head outputs; xavier query_pos (scratch-stable)."""
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for reg_ in self.dec_bbox_head:
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _get_encoder_input(self, x, seed_map=None):
        """Project multi-scale features to hd and flatten for cross-attention.

        Returns (feats, shapes, proj2d): flattened (B, N, hd) tokens, the
        per-scale (h, w) shapes, and the projected 2D maps (proj2d[0] is the
        P2/stride-4 map used for grid-sampling query content). When
        ``seed_level`` is on, the seed map is projected as a 4th attended level.
        """
        if self.seed_level:
            if seed_map is None:
                b = x[0].shape[0]
                seed_map = x[0].new_zeros((b, 1, x[0].shape[-2] * 4, x[0].shape[-1] * 4))
            x = list(x) + [seed_map]
        proj = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        shapes = [f.shape[-2:] for f in proj]
        feats = torch.cat([f.flatten(2).transpose(1, 2) for f in proj], 1)
        return feats, shapes, proj

    def forward(self, x, batch=None, seed_map=None, top_feat=None):
        """Run encoder + query selection + decoder forward.

        Args:
            x: list of multi-scale features [b3, b4, b5] at strides 4/8/16.
            batch: optional targets dict (denoising is disabled, nd=0).
            seed_map: (B, 1, H, W) raw NP seed logits at full resolution.
            top_feat: optional top-level (stride-32) backbone feature for
                LSP FeatureSampling query content (``content_top``).
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Inject the seed map as an extra feature channel (SPAR-Det-style seg
        # conditioning): the deformable cross-attention then attends over
        # nucleus-aware features. Downsample the full-res seed prob to each
        # scale and concat along the channel dim. Operator chosen by
        # ``self.seed_resize``: 'bilinear' (LOCKED recipe; average-like and
        # dilutes small nuclei at st8/st16), 'max' (peak-preserving) or 'avg'.
        if self.seed_feature and seed_map is not None:
            seed_prob = seed_map.sigmoid()
            seed_feats = []
            for feat in x:
                size = feat.shape[-2:]
                if self.seed_resize == 'max':
                    resized = F.adaptive_max_pool2d(seed_prob, size)
                elif self.seed_resize == 'avg':
                    resized = F.adaptive_avg_pool2d(seed_prob, size)
                elif self.seed_resize == 'topk':
                    th, tw = size
                    sh, sw = seed_prob.shape[-2], seed_prob.shape[-1]
                    kh, kw = max(1, sh // th), max(1, sw // tw)
                    if sh % th == 0 and sw % tw == 0 and (kh > 1 or kw > 1):
                        patches = F.unfold(seed_prob, (kh, kw), stride=(kh, kw))  # (B, kh*kw, n_cells)
                        k = max(1, int(round(patches.shape[1] * self.seed_resize_frac)))
                        vals = patches.topk(k, dim=1).values.mean(dim=1)  # (B, n_cells)
                        resized = vals.reshape(seed_prob.shape[0], 1, th, tw)
                    else:
                        resized = F.interpolate(seed_prob, size=size, mode='bilinear', align_corners=False)
                else:
                    resized = F.interpolate(seed_prob, size=size, mode='bilinear', align_corners=False)
                seed_feats.append(torch.cat([feat, resized], dim=1))
            x = seed_feats

        feats, shapes, proj2d = self._get_encoder_input(x, seed_map=seed_map)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        # seed_q: raw seed logit bilinearly sampled at each query position,
        # differentiable into the NP head. Concatenated directly to the score
        # head inputs (Path B) so the classifier sees the seed's 3.8-logit
        # nucleus/bg separation undiluted (the 1x1 input_proj mixing washes it
        # out — verified corr(seed, enc_noobj) ~ -0.13 on train8).
        seed_q = None
        p0, confine_s = None, None
        if self.query_selection == 'grid':
            embed, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores, p0 = self._grid_decoder_input(
                feats, proj2d, seed_map=seed_map, dn_embed=dn_embed, dn_bbox=dn_bbox, top_feat=top_feat
            )
            if self.query_block_size > 0.0:
                # LSP grid: cell = query_block_size px; confinement half-width
                # is half the cell (a query never leaves its origin cell).
                crop_px = proj2d[0].shape[-2] * 4
                confine_s = (self.query_block_size / crop_px) / 2.0
            else:
                confine_s = (1.0 / max(1, round(1.0 / self.grid_size))) / 2.0
            if self.seed_in_content and seed_map is not None:
                seed_q = self._sample_seed_at(seed_map, p0)
        elif self.query_selection == 'hybrid':
            # Grid queries UNION seed-peak queries (DQ-DETR-style dynamic
            # placement): the regular grid is kept for coverage + background
            # training, and extra queries are placed exactly on seed-map local
            # maxima so tiny nuclei (which a 12.8px cell may miss or share) get
            # a query anchored on their peak. Peak queries are confined around
            # their peak (per-query p0 origin), grid queries around their cell.
            embed, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores, p0 = self._hybrid_decoder_input(
                feats, proj2d, seed_map, self.hybrid_peaks, dn_embed=dn_embed, dn_bbox=dn_bbox, top_feat=top_feat
            )
            if self.query_block_size > 0.0:
                crop_px = proj2d[0].shape[-2] * 4
                confine_s = (self.query_block_size / crop_px) / 2.0
            else:
                confine_s = (1.0 / max(1, round(1.0 / self.grid_size))) / 2.0
            if self.seed_in_content and seed_map is not None:
                seed_q = self._sample_seed_at(seed_map, p0)
        elif self.query_selection == 'seed':
            # Seed queries sit on nucleus peaks. NO hard centroid confinement:
            # train27 showed tanh-clamping to a half-grid-cell disc saturates
            # (all queries pinned at the disc edge, ~0 gradient through the
            # centroid) and freezes xy/l1/piou; the residual logit chain then
            # cannot recover. Peak slots already start on the nucleus, so
            # confinement is unnecessary (train26 trained fine without it).
            embed, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores, seed_pts = self._seed_decoder_input(
                feats, proj2d, seed_map, dn_embed, dn_bbox
            )
            p0 = None
            confine_s = None
            if self.seed_in_content and seed_map is not None:
                seed_q = self._sample_seed_at(seed_map, seed_pts)
        else:
            # Learned selection across all tokens; no grid-cell confinement.
            embed, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores, seed_pts = (
                self._learned_decoder_input(feats, shapes, seed_map=seed_map, dn_embed=dn_embed, dn_bbox=dn_bbox)
            )
            if self.seed_in_content and seed_map is not None:
                seed_q = self._sample_seed_at(seed_map, seed_pts)

        dec_polygons, dec_scores = self._raycast_decoder(
            embed,
            refer_centroids,
            refer_polygon_logits,
            feats,
            shapes,
            attn_mask,
            p0=p0,
            confine_s=confine_s,
            seed_q=seed_q,
            enc_scores=enc_scores,
            recast_features=proj2d[0],
            recast_seed=seed_map,
        )

        if self.respawn and seed_map is not None:
            dec_polygons, dec_scores = self._respawn_pass(
                dec_polygons, dec_scores, feats, shapes, proj2d, seed_map, embed, attn_mask
            )

        x = dec_polygons, dec_scores, enc_polygons, enc_scores, dn_meta
        if self.training:
            return x
        # [-1] = respawn-merged output when respawn is on (stage-1 finals for
        # stable queries + pass-2 refinement for respawned victims); identical
        # to squeeze(0) for the single-layer non-respawn eval stack.
        y = torch.cat((dec_polygons[-1], dec_scores[-1].sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_ray_anchors(self, shapes, device=None):
        """Generate grid anchors + valid mask over multi-scale feature shapes.

        Returns (anchors_encoded, valid_mask) cached on self: anchors_encoded
        is (1, N, 2+n_rays) in logit space (encode_polygon), masked to inf at
        the image border (valid_mask False). base_radius grows with scale depth
        (grid_size * 2**i), matching RT-DETR's multi-scale anchor scheme.
        """
        device = device if device is not None else self.stride.device
        cache_key = (device, tuple((h, w) for h, w in shapes))
        cached = getattr(self, '_ray_anchor_cache', None)
        if cached is not None and cached[0] == cache_key:
            anchors_encoded, valid_mask = cached[1], cached[2]
            if anchors_encoded.device != device:
                anchors_encoded = anchors_encoded.to(device)
                valid_mask = valid_mask.to(device)
            return anchors_encoded, valid_mask

        dtype = torch.float32
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing='ij',
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)
            grid_xy = (grid_xy + 0.5) / torch.tensor([w, h], device=device, dtype=dtype)
            base_radius = self.grid_size * 2**i
            rays = torch.full((h, w, self.n_rays), base_radius, device=device, dtype=dtype)
            anchor = torch.cat([grid_xy, rays], -1)  # (h, w, 2+n_rays)
            anchors.append(anchor.view(1, -1, self.raycast_dim))
        anchors = torch.cat(anchors, 1)  # (1, N, 2+n_rays)

        eps = 1e-2
        valid_mask = (anchors[..., :2] > eps) & (anchors[..., :2] < 1 - eps)
        valid_mask = valid_mask.all(-1, keepdim=True)  # (1, N, 1)
        anchors_encoded = encode_polygon(anchors).masked_fill(~valid_mask, float('inf'))

        self._ray_anchor_cache = (cache_key, anchors_encoded, valid_mask)
        return anchors_encoded, valid_mask

    @staticmethod
    def _sample_seed_at(seed_map, positions):
        """Bilinearly sample raw seed logits at normalized query positions.

        Args:
            seed_map: (B, 1, H, W) raw NP seed logits (full resolution).
            positions: (B, nq, 2) normalized (x, y) in [0, 1].

        Returns:
            (B, nq, 1) seed logits, differentiable into the seed map.
        """
        grid = positions * 2 - 1
        s = F.grid_sample(seed_map, grid.unsqueeze(1), mode='bilinear', align_corners=False, padding_mode='border')
        return s.squeeze(2).transpose(1, 2)  # (B, nq, 1)

    def _grid_decoder_input(self, feats, proj2d, seed_map=None, dn_embed=None, dn_bbox=None, top_feat=None):
        """LSP-DETR-style grid-query initialization (no learned selection).

        Args:
            feats: flattened (B, N, hd) multi-scale tokens for cross-attention.
            proj2d: list of projected 2D feature maps (proj2d[0] = P2, stride 4).
            seed_map: optional (B, 1, H, W) raw seed logits; when
                seed_in_content is set, the seed logit is sampled at p0 and
                concatenated to the enc score head input.
            dn_embed: optional denoising query embeddings (nd=0, unused).
            dn_bbox: optional denoising reference polygons (nd=0, unused).
            top_feat: optional top-level (stride-32) feature used as the query
                content source when ``content_top`` is set (LSP FeatureSampling).

        Initial nuclei descriptors are placed on a FIXED REGULAR GRID covering
        the whole image: one query per grid cell. With ``query_block_size > 0``
        the cell is that many PIXELS (LSP: 14px = ~3.5um at 256px/0.25mpp;
        ceil(256/14)=19 -> 361 queries, last center may lie past the edge,
        matching LSP); otherwise cell = ``grid_size`` (normalized). Position
        p0 = cell center, content = bilinear-sampled projected feature at p0
        (P2 through enc_output by default; the top-level stage feature through
        Conv1x1 + LayerNorm when ``content_top``), rays = half-cell circles.
        Every nucleus is guaranteed a nearby query (N = grid cells >= M); the
        query-to-cell confinement is applied in ``_raycast_decoder``. Returns a
        6-tuple with p0 (origin grid positions) for that confinement.
        """
        bs = feats.shape[0]
        device, dtype = feats.device, feats.dtype
        crop_px = proj2d[0].shape[-2] * 4
        if self.query_block_size > 0.0:
            n_side = max(1, math.ceil(crop_px / self.query_block_size))
            cell_px = self.query_block_size
        else:
            n_side = max(1, round(1.0 / self.grid_size))
            cell_px = crop_px / n_side
        cell = cell_px / crop_px  # normalized cell width
        self.num_queries = n_side * n_side

        coords_px = (torch.arange(n_side, device=device, dtype=dtype) + 0.5) * cell_px
        gy, gx = torch.meshgrid(coords_px, coords_px, indexing='ij')
        p0 = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).unsqueeze(0).repeat(bs, 1, 1)  # (B, nq, 2) px
        p0 = p0 / crop_px  # normalized; last cell may exceed 1.0 (LSP convention)

        # content: bilinear sample at the grid centers. LSP FeatureSampling reads
        # the TOP-LEVEL (stride-32) feature: Conv1x1 -> grid_sample -> LayerNorm
        # (no extra MLP). Legacy reads the projected P2 map through enc_output.
        grid = p0 * 2 - 1
        if self.content_top and top_feat is not None:
            top_proj = self.top_content_conv(top_feat)  # (B, hd, H32, W32)
            content = F.grid_sample(
                top_proj,
                grid.unsqueeze(1),
                mode='bilinear',
                align_corners=False,
                padding_mode='border',
            )  # (B, hd, 1, nq)
            content = content.squeeze(2).transpose(1, 2)  # (B, nq, hd)
            content = self.top_content_norm(content)
        else:
            content = F.grid_sample(
                proj2d[0],
                grid.unsqueeze(1),
                mode='bilinear',
                align_corners=False,
                padding_mode='border',
            )  # (B, hd, 1, nq)
            content = content.squeeze(2).transpose(1, 2)  # (B, nq, hd)
            content = self.enc_output(content)

        # anchors: circle of radius half-cell at each grid center
        radius = cell / 2.0
        rays = torch.full((bs, self.num_queries, self.n_rays), radius, device=device, dtype=dtype)
        anchors = encode_polygon(torch.cat([p0, rays], -1))

        refer_polygon_logits = self.enc_bbox_head(content) + anchors
        enc_polygons = decode_polygon(refer_polygon_logits)
        refer_centroids = enc_polygons[..., :2]
        enc_scores = self._score(
            self.enc_score_head,
            content,
            self._sample_seed_at(seed_map, p0) if (self.seed_in_content and seed_map is not None) else None,
        )
        embeddings = content

        if self.training:
            refer_polygon_logits = refer_polygon_logits.detach()
            refer_centroids = refer_centroids.detach()
            embeddings = embeddings.detach()

        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
            refer_polygon_logits = torch.cat([dn_bbox, refer_polygon_logits], 1)
            refer_centroids = torch.cat([dn_bbox[..., :2], refer_centroids], 1)
            enc_polygons = torch.cat([decode_polygon(dn_bbox), enc_polygons], 1)

        return embeddings, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores, p0

    def _hybrid_decoder_input(self, feats, proj2d, seed_map, n_peak, dn_embed=None, dn_bbox=None, top_feat=None):
        """Grid queries UNION seed-peak queries (dynamic query placement).

        Keeps the regular grid (coverage + background/∅ training) and appends
        ``n_peak`` extra queries placed on the strongest seed-map local maxima,
        so tiny nuclei get a query anchored on their peak. Samples with fewer
        valid peaks fall back to duplicating the first grid centers (benign
        duplicates learn ∅). Peak queries use the same half-cell init radius
        and are confined around their peak origin (per-query p0).

        Returns the same 6-tuple as ``_grid_decoder_input``; p0 holds the
        per-query confinement origins (grid cell centers, then peaks).
        """
        bs = feats.shape[0]
        device, dtype = feats.device, feats.dtype
        crop_px = proj2d[0].shape[-2] * 4
        if self.query_block_size > 0.0:
            n_side = max(1, math.ceil(crop_px / self.query_block_size))
            cell_px = self.query_block_size
        else:
            n_side = max(1, round(1.0 / self.grid_size))
            cell_px = crop_px / n_side
        cell = cell_px / crop_px
        n_grid = n_side * n_side
        n_peak = min(n_peak, n_grid) if n_peak > 0 else 0
        self.num_queries = n_grid + n_peak

        # --- grid part (identical to _grid_decoder_input) ---
        coords_px = (torch.arange(n_side, device=device, dtype=dtype) + 0.5) * cell_px
        gy, gx = torch.meshgrid(coords_px, coords_px, indexing='ij')
        p0_grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).unsqueeze(0).repeat(bs, 1, 1) / crop_px

        grid_g = p0_grid * 2 - 1
        if self.content_top and top_feat is not None:
            top_proj = self.top_content_conv(top_feat)
            content_g = F.grid_sample(
                top_proj, grid_g.unsqueeze(1), mode='bilinear', align_corners=False, padding_mode='border'
            )
            content_g = self.top_content_norm(content_g.squeeze(2).transpose(1, 2))
        else:
            content_g = F.grid_sample(
                proj2d[0], grid_g.unsqueeze(1), mode='bilinear', align_corners=False, padding_mode='border'
            )
            content_g = self.enc_output(content_g.squeeze(2).transpose(1, 2))

        radius = cell / 2.0
        rays_g = torch.full((bs, n_grid, self.n_rays), radius, device=device, dtype=dtype)
        anchors_g = encode_polygon(torch.cat([p0_grid, rays_g], -1))

        # --- seed-peak part: NOVELTY-selected peaks + background parking ---
        # A peak is used ONLY where it adds coverage the grid lacks: it must
        # sit farther than half a cell from every grid center (off-grid
        # nuclei, e.g. tiny ones sharing a cell with a neighbour). Unused
        # slots are parked on the LOWEST-seed local minima (confident
        # background) — NOT on cloned grid centers: clones are Hungarian
        # cost-ties whose winner flips across epochs and churns the fg/∅
        # supervision (train24 regression).
        if n_peak > 0 and seed_map is not None:
            s_h, s_w = seed_map.shape[-2:]
            seed_prob = seed_map.sigmoid()
            kernel = 2 * self.peak_distance + 1
            pooled, max_inds = F.max_pool2d(
                seed_prob, kernel_size=kernel, stride=1, padding=self.peak_distance, return_indices=True
            )
            inds = torch.arange(0, seed_prob.numel(), device=device).reshape(seed_prob.shape)
            is_peak = (max_inds == inds) & (seed_prob > self.seed_threshold)

            # pixel-center normalized coordinates
            ys_c = ((torch.arange(s_h, device=device, dtype=dtype) + 0.5) / s_h).view(s_h, 1).expand(s_h, s_w)
            xs_c = ((torch.arange(s_w, device=device, dtype=dtype) + 0.5) / s_w).view(1, s_w).expand(s_h, s_w)
            pos_x, pos_y = xs_c, ys_c

            # novelty: farther than half a cell from the NEAREST grid center
            gxi = torch.round(pos_x / cell - 0.5) * cell + cell / 2
            gyi = torch.round(pos_y / cell - 0.5) * cell + cell / 2
            d2 = (pos_x - gxi) ** 2 + (pos_y - gyi) ** 2
            novel = is_peak & (d2 > (cell / 2) ** 2)  # (s_h, s_w)

            flat = seed_prob.flatten(2).squeeze(1)  # (B, s_h*s_w)

            def idx_to_pos(idx):
                ys_ = (torch.div(idx, s_w, rounding_mode='floor')).to(dtype)
                xs_ = (idx % s_w).to(dtype)
                return torch.stack([(xs_ + 0.5) / s_w, (ys_ + 0.5) / s_h], -1)  # (B, k, 2)

            # top novel peaks by seed value
            peak_vals = flat.masked_fill(~novel.flatten(2).squeeze(1), 0.0)  # (B, N)
            k_sel = min(n_peak, s_h * s_w)
            top_vals, top_idx = torch.topk(peak_vals, k_sel, dim=1)
            peak_ok = top_vals > 0  # (B, k_sel) real novel-peak slots

            # background parking: lowest-seed LOCAL MINIMA (spread out)
            npool, ninds = F.max_pool2d(
                -seed_prob, kernel_size=kernel, stride=1, padding=self.peak_distance, return_indices=True
            )
            is_min = ninds == inds  # (B,1,s_h,s_w)
            bg_vals = flat.masked_fill(~is_min.flatten(2).squeeze(1), float('inf'))
            _, bg_idx = torch.topk(bg_vals, k_sel, dim=1, largest=False)
            bg_pos = idx_to_pos(bg_idx)

            peaks = torch.where(peak_ok.unsqueeze(-1), idx_to_pos(top_idx), bg_pos).to(dtype)
        else:
            peaks = p0_grid[:, :n_peak, :] if n_peak > 0 else p0_grid[:, :0, :]

        if n_peak > 0:
            grid_p = peaks * 2 - 1
            content_p = F.grid_sample(
                proj2d[0], grid_p.unsqueeze(1), mode='bilinear', align_corners=False, padding_mode='border'
            )
            content_p = self.enc_output(content_p.squeeze(2).transpose(1, 2))
            rays_p = torch.full((bs, n_peak, self.n_rays), radius, device=device, dtype=dtype)
            anchors_p = encode_polygon(torch.cat([peaks, rays_p], -1))

            content = torch.cat([content_g, content_p], 1)
            anchors = torch.cat([anchors_g, anchors_p], 1)
            p0 = torch.cat([p0_grid, peaks], 1)
        else:
            content = content_g
            anchors = anchors_g
            p0 = p0_grid

        refer_polygon_logits = self.enc_bbox_head(content) + anchors
        enc_polygons = decode_polygon(refer_polygon_logits)
        refer_centroids = enc_polygons[..., :2]
        enc_scores = self._score(
            self.enc_score_head,
            content,
            self._sample_seed_at(seed_map, p0) if (self.seed_in_content and seed_map is not None) else None,
        )
        embeddings = content

        if self.training:
            refer_polygon_logits = refer_polygon_logits.detach()
            refer_centroids = refer_centroids.detach()
            embeddings = embeddings.detach()

        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
            refer_polygon_logits = torch.cat([dn_bbox, refer_polygon_logits], 1)
            refer_centroids = torch.cat([dn_bbox[..., :2], refer_centroids], 1)
            enc_polygons = torch.cat([decode_polygon(dn_bbox), enc_polygons], 1)

        return embeddings, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores, p0

    def _learned_decoder_input(self, feats, shapes, seed_map=None, dn_embed=None, dn_bbox=None):
        """Select queries via the learned encoder score head (RT-DETR style).

        Scores every multi-scale token with enc_score_head, takes the top-nq by
        max REAL class score, and uses their features as query content + their
        grid anchors as reference polygons. Trained end-to-end by the detection
        loss (unlike seed-map local maxima). When ``seed_in_content`` is set the
        raw seed logit is sampled at every token position and concatenated to the
        score-head input (the token positions come from the encoded anchors).
        Returns a 6-tuple; the trailing element is the (B, nq, 2) token positions.
        """
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_ray_anchors(shapes, device=feats.device)
        features = self.enc_output(valid_mask * feats)
        all_positions = decode_polygon(anchors)[..., :2].expand(bs, -1, -1)  # (B, N, 2) normalized xy
        seed_all = (
            self._sample_seed_at(seed_map, all_positions) if self.seed_in_content and seed_map is not None else None
        )
        enc_outputs_scores = self._score(self.enc_score_head, features, seed_all)  # (B, N, nc+1)
        # Rank by the best REAL class score: with no_object=True the head emits
        # nc+1 logits and ranking on the raw max would let the ∅ column win,
        # selecting the most confidently-background tokens as queries.
        rank_scores = enc_outputs_scores[..., :-1] if self.no_object else enc_outputs_scores
        topk_ind = torch.topk(rank_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(bs, device=feats.device).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)
        seed_pts = all_positions[0, topk_ind].view(bs, self.num_queries, 2)

        refer_polygon_logits = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_polygons = decode_polygon(refer_polygon_logits)
        refer_centroids = enc_polygons[..., :2]
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features

        if self.training:
            refer_polygon_logits = refer_polygon_logits.detach()
            refer_centroids = refer_centroids.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()

        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
            refer_polygon_logits = torch.cat([dn_bbox, refer_polygon_logits], 1)
            refer_centroids = torch.cat([dn_bbox[..., :2], refer_centroids], 1)
            enc_polygons = torch.cat([decode_polygon(dn_bbox), enc_polygons], 1)

        return embeddings, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores, seed_pts

    def _seed_decoder_input(self, feats, proj2d, seed_map, dn_embed=None, dn_bbox=None):
        """Select queries from seed-map local maxima (seg-governed queries).

        Every query is born on a nucleus candidate: the top-``nq`` seed-map
        local maxima by seed value. Samples with fewer valid peaks park the
        unused slots on the LOWEST-seed local minima (confident background)
        — NOT on arbitrary/cloned positions: clones are Hungarian cost-ties
        whose winner flips across epochs and churns the fg/∅ supervision
        (train24 regression). Parked slots give the ∅ head clean negatives.

        Args:
            feats: flattened (B, N, hd) multi-scale tokens for cross-attention.
            proj2d: list of projected 2D feature maps (proj2d[0] = P2, stride 4).
            seed_map: (B, 1, H, W) raw seed logits.
            dn_embed: optional denoising query embeddings (nd=0, unused).
            dn_bbox: optional denoising reference polygons (nd=0, unused).

        Returns a 6-tuple; the trailing element is the (B, nq, 2) normalized
        peak/parked positions, used by the caller to sample the per-query seed cue.
        """
        bs = feats.shape[0]
        nq = self.num_queries
        h, w = proj2d[0].shape[-2:]
        s_h, s_w = seed_map.shape[-2:]
        device, dtype = feats.device, feats.dtype

        # --- local maxima on the full-res seed map ---
        seed_prob = seed_map.sigmoid()  # (B, 1, H, W)
        kernel = 2 * self.peak_distance + 1
        pooled, max_inds = F.max_pool2d(
            seed_prob,
            kernel_size=kernel,
            stride=1,
            padding=self.peak_distance,
            return_indices=True,
        )
        inds = torch.arange(0, seed_prob.numel(), device=device).reshape(seed_prob.shape)
        is_peak = (max_inds == inds) & (seed_prob > self.seed_threshold)  # (B, 1, H, W)

        flat = seed_prob.flatten(2).squeeze(1)  # (B, s_h*s_w)

        def idx_to_pos(idx):
            ys_ = torch.div(idx, s_w, rounding_mode='floor').to(dtype)
            xs_ = (idx % s_w).to(dtype)
            return torch.stack([(xs_ + 0.5) / s_w, (ys_ + 0.5) / s_h], -1)  # (B, k, 2)

        # --- top-k peaks by seed value (0.0-filled non-peaks sort last) ---
        peak_vals = flat.masked_fill(~is_peak.flatten(2).squeeze(1), 0.0)  # (B, N)
        top_vals, top_idx = torch.topk(peak_vals, nq, dim=1)
        peak_ok = top_vals > 0  # (B, nq) real peak slots

        # --- background parking: lowest-seed LOCAL MINIMA (spread out) ---
        npool, ninds = F.max_pool2d(
            -seed_prob, kernel_size=kernel, stride=1, padding=self.peak_distance, return_indices=True
        )
        is_min = ninds == inds  # (B, 1, H, W)
        bg_vals = flat.masked_fill(~is_min.flatten(2).squeeze(1), float('inf'))
        _, bg_idx = torch.topk(bg_vals, nq, dim=1, largest=False)
        bg_pos = idx_to_pos(bg_idx)

        centroids = torch.where(peak_ok.unsqueeze(-1), idx_to_pos(top_idx), bg_pos).to(dtype)

        # --- query content: bilinear sample projected P2 at the centroids ---
        grid = centroids * 2 - 1  # [-1, 1] for grid_sample
        content = F.grid_sample(
            proj2d[0],
            grid.unsqueeze(1),
            mode='bilinear',
            align_corners=False,
            padding_mode='border',
        )  # (B, hd, 1, nq)
        content = content.squeeze(2).transpose(1, 2)  # (B, nq, hd)
        content = self.enc_output(content)

        # --- anchors: half-grid-cell radius circle at each centroid ---
        radius = self.grid_size / 2.0
        rays = torch.full((bs, nq, self.n_rays), radius, device=device, dtype=dtype)
        anchors = encode_polygon(torch.cat([centroids, rays], -1))

        refer_polygon_logits = self.enc_bbox_head(content) + anchors
        enc_polygons = decode_polygon(refer_polygon_logits)
        refer_centroids = enc_polygons[..., :2]
        # Sample the raw seed logit at each peak (the query is already placed on
        # a seed maximum, so this is a strong foreground cue) and concat to the
        # score-head input exactly like the grid branch. Without this the head is
        # fed width hd while seed_in_content built it for hd+1 -> crash.
        seed_q = self._sample_seed_at(seed_map, centroids) if self.seed_in_content and seed_map is not None else None
        enc_scores = self._score(self.enc_score_head, content, seed_q)
        embeddings = content

        if self.training:
            refer_polygon_logits = refer_polygon_logits.detach()
            refer_centroids = refer_centroids.detach()
            embeddings = embeddings.detach()

        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
            refer_polygon_logits = torch.cat([dn_bbox, refer_polygon_logits], 1)
            refer_centroids = torch.cat([dn_bbox[..., :2], refer_centroids], 1)
            enc_polygons = torch.cat([decode_polygon(dn_bbox), enc_polygons], 1)

        return embeddings, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores, centroids

    @staticmethod
    def _score(head, feats, seed_q):
        """Apply a score head, optionally concat the seed logit column."""
        if seed_q is not None:
            feats = torch.cat([feats, seed_q], -1)
        return head(feats)

    def _rank_causal_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Build the MDS-DETR rank-causal self-attention mask (True = blocked).

        Queries are ranked by descending objectness ``1 - P(no-object)``; each
        query attends ONLY to higher-confidence queries and self-attention is
        blocked. This is a "parallelized learned NMS": duplicate queries of the
        same nucleus attend to the winner and get suppressed toward background.
        The mask is arg-sort-derived (non-differentiable), so it is detached.

        Args:
            scores: (B, nq, nc+1) previous layer's classification logits (or the
                encoder scores for layer 0); col -1 is the no-object logit.

        Returns:
            (B*nhead, nq, nq) boolean mask for ``nn.MultiheadAttention``.
        """
        conf = 1.0 - scores[..., -1].sigmoid().detach()  # (B, nq) objectness
        nq = conf.shape[-1]
        order = conf.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(nq, device=conf.device).unsqueeze(0).expand_as(order))
        mask = rank.unsqueeze(-2) >= rank.unsqueeze(-1)  # (B, nq, nq): block keys of equal-or-lower conf
        # The top-confidence query has no higher peer to attend to; a fully
        # masked row yields NaN attention (softmax over an empty set). Let it
        # keep self-attention instead — under the residual + LayerNorm this is
        # equivalent to "preserved" (LayerNorm is scale-invariant), and avoids
        # the NaN. (MDS-DETR uses learnable TP sink tokens for this; deferred.)
        eye = torch.eye(nq, device=conf.device, dtype=torch.bool)
        top = rank == 0  # (B, nq) one-hot of the highest-confidence query
        mask = mask & ~(top.unsqueeze(-1) & eye.unsqueeze(0))
        return mask.repeat_interleave(self.nhead, dim=0)  # (B*nhead, nq, nq)

    def _respawn_pass(self, dec_polygons, dec_scores, feats, shapes, proj2d, seed_map, embed, attn_mask):
        """Recall-repair second decoder pass over re-anchored queries.

        Stage 1 = the standard grid decode (dec_polygons/dec_scores, layer dim L).
        This pass: (1) extract seed local maxima (EDT/fg peaks); (2) find peaks
        NOT covered by any FIRED stage-1 polygon (analytic point-in-star, no
        rasterization); (3) re-anchor the K most-confidently-EMPTY queries onto
        the top uncovered peaks (position=peak, content=grid_sample, radius=EDT
        scale when the seed map is an EDT map); (4) run the SAME shared decoder
        layers from those anchors; (5) append pass-2 outputs as extra layers on
        the aux stack so the criterion supervises them (with residual matching
        at the loss level: stage-2 targets = GT unmatched in stage 1).

        Returns stacked (dec_polygons, dec_scores) with L+L2 layers. Non-fg
        queries (no anchor assigned) keep their stage-1 final output as the
        pass-2 prediction so their stage-2 ∅ supervision stays consistent.
        """
        from raycasted.model.nulite_detr_loss import _points_inside_star

        nl = dec_polygons.shape[0]
        nb, nq = dec_scores.shape[1], dec_scores.shape[2]
        device = dec_polygons.device
        final_polys = dec_polygons[-1]  # (B, nq, 2+R) normalized
        final_logits = dec_scores[-1]  # (B, nq, nc+1) raw logits
        fired = (
            final_logits.argmax(-1) != self.nc
            if self.no_object
            else torch.ones(nb, nq, dtype=torch.bool, device=device)
        )
        empty_conf = final_logits[..., -1] if self.no_object else -final_logits.max(-1).values

        seed_prob = seed_map.sigmoid()  # (B, 1, H, W); for EDT maps this is normalized distance
        k = 2 * self.peak_distance + 1
        pooled_idx = F.max_pool2d(seed_prob, k, 1, k // 2, return_indices=True)[1]
        own_idx = (
            torch.arange(pooled_idx.flatten(2).shape[-1], device=device).view(1, 1, -1).expand_as(pooled_idx.flatten(2))
        )
        is_peak = (pooled_idx.flatten(2) == own_idx) & (seed_prob.flatten(2) > self.respawn_min_seed)
        is_peak = is_peak.view(nb, seed_prob.shape[-2], seed_prob.shape[-1])  # (B, H, W)

        # Pass-2 query set assembled as ONE batched tensor so the second decoder
        # pass runs a single batched forward instead of a per-image batch-1 loop
        # (the old `_raycast_decoder(feats[b:b+1])` serialized 16 forwards and
        # caused the ~2.3x epoch slowdown). Non-victims keep stage-1 content and
        # anchors; victims are re-anchored onto uncovered peaks.
        p2_embed = embed.clone()  # (B, nq, hd)
        p2_refer_logits = encode_polygon(final_polys.clamp(1e-4, 1 - 1e-4))  # (B, nq, raycast_dim)
        p2_pos = final_polys[:, :, :2].clone()  # seed_q2 sampling positions
        victim_mask = torch.zeros(nb, nq, dtype=torch.bool, device=device)

        for b in range(nb):
            # 1. candidate peaks (normalized xy + strength)
            ys, xs = torch.nonzero(is_peak[b], as_tuple=True)
            if len(ys) == 0:
                continue
            hh, ww = seed_prob.shape[-2:]
            peaks_xy = torch.stack([xs, ys], -1).float()  # px
            peaks_xy_n = (peaks_xy + 0.5) / torch.tensor([ww, hh], device=device, dtype=torch.float32)
            peak_val = seed_prob[b, 0, ys, xs]

            # 2. coverage: peak inside any FIRED stage-1 star polygon?
            with torch.no_grad():
                fired_xy = final_polys[b, fired[b], :2]
                fired_rays = final_polys[b, fired[b], 2:]
                if fired_xy.shape[0] > 0:
                    inside = _points_inside_star(peaks_xy_n, fired_xy, fired_rays, self.n_rays)  # (P, n_fired)
                    covered = inside.any(1)
                else:
                    covered = torch.zeros(len(peaks_xy_n), dtype=torch.bool, device=device)
            uncov = ~covered
            if not uncov.any():
                continue
            cand_xy = peaks_xy_n[uncov][: self.respawn_peaks]
            cand_val = peak_val[uncov][: self.respawn_peaks]

            # 3. re-anchor the K most-confidently-EMPTY queries onto the peaks
            #    (strongest seed value first).
            order = cand_val.argsort(descending=True)
            cand_xy = cand_xy[order]
            cand_val = cand_val[order]
            kk = min(len(cand_xy), int(self.respawn_peaks), int((~fired[b]).sum().item()))
            if kk <= 0:
                continue
            # victims: emptiest queries first (highest no-object logit)
            victim_idx = empty_conf[b].argsort(descending=True)[:kk]

            # anchors: position=peak; radius=peak EDT value if it looks like an
            # EDT map (values up to 1.0 scale with nucleus size), else half-cell.
            edt_radius = cand_val[:kk].clamp(min=0.02, max=0.5)  # normalized px/256
            anchors = torch.cat([cand_xy[:kk], edt_radius.unsqueeze(-1).expand(-1, self.n_rays)], -1)
            anchor_logits = encode_polygon(anchors)

            # content: sample the projected finest level at the peak positions
            src = proj2d[0][b]  # (hd, h, w)
            content = (
                F.grid_sample(
                    src.unsqueeze(0),
                    cand_xy[:kk].view(1, 1, -1, 2) * 2 - 1,
                    mode='bilinear',
                    align_corners=False,
                    padding_mode='border',
                )
                .flatten(2)
                .permute(0, 2, 1)
            )  # (1, K, hd)
            content = self.enc_output(content)

            p2_embed[b, victim_idx] = content[0]
            p2_refer_logits[b, victim_idx] = anchor_logits
            p2_pos[b, victim_idx] = cand_xy[:kk]
            victim_mask[b, victim_idx] = True

        seed_q2 = None
        if self.seed_in_content:
            seed_q2 = self._sample_seed_at(seed_map, p2_pos)

        # 4. single batched second pass through the shared layers.
        polys2, logits2 = self._raycast_decoder(
            p2_embed,
            p2_refer_logits[:, :, :2].clone(),
            p2_refer_logits.clone(),
            feats,
            shapes,
            attn_mask,
            p0=None,
            confine_s=None,
            seed_q=seed_q2,
            enc_scores=None,
            recast_seed=seed_map,
        )

        # stage-2 final output: victims get their respawned refinement;
        # non-victims KEEP stage-1 finals so the appended layer differs from
        # stage 1 only where the model chose to respawn.
        p2_final_poly = final_polys.clone()
        p2_final_logit = final_logits.clone()
        p2_final_poly[victim_mask] = polys2[-1][victim_mask]
        p2_final_logit[victim_mask] = logits2[-1][victim_mask]

        out_polygons = torch.cat([dec_polygons, p2_final_poly.unsqueeze(0)], 0)
        out_scores = torch.cat([dec_scores, p2_final_logit.unsqueeze(0)], 0)
        self._respawn_layer_idx = nl  # criterion reads this for residual matching
        return out_polygons, out_scores

    def _raycast_decoder(
        self,
        embed,
        refer_centroids,
        refer_polygon_logits,
        feats,
        shapes,
        attn_mask,
        p0=None,
        confine_s=None,
        seed_q=None,
        enc_scores=None,
        recast_features=None,
        recast_seed=None,
    ):
        """Iterative-refinement decoder over the deformable transformer layers.

        Per layer: query_pos from centroids -> decoder layer (self-attn +
        deformable cross-attn on feats/shapes) -> ray polygon logits added to
        the previous refinement (residual in logit space). Training keeps all
        layers for aux loss; eval keeps only the eval_idx layer.

        When ``p0`` and ``confine_s`` are given (grid-query init), refined
        centroids are confined to their origin grid cell via a tanh-bounded
        update (SAP-DETR/LSP-DETR style): p = p0 + s*tanh((p - p0)/s), so a
        query can never drift more than half a cell from where it started.

        When ``self.mds`` is set, the FINAL decoder layer's self-attention is
        masked rank-causally by the previous layer's objectness confidence,
        giving local winner-take-all between adjacent grid cells (MDS-DETR
        duplicate suppression). Earlier layers keep full self-attention for
        feature refinement — mirroring MDS-DETR's O2M-then-O2O design where only
        the last (O2O) layer suppresses duplicates.
        """
        recast_module = getattr(self, 'recast_module', None)
        if recast_module is not None and (recast_seed is None or recast_features is None):
            raise ValueError('recast requires segmentation logits and projected local features')
        output = embed
        dec_polygons = []
        dec_cls = []
        last_refined_logits = None
        refer_centroids_2d = refer_centroids
        prev_scores = enc_scores
        level_masks = getattr(self, '_level_masks', None)

        for i, layer in enumerate(self.decoder.layers):
            if level_masks is not None and hasattr(layer.cross_attn, 'level_mask'):
                layer.cross_attn.level_mask = level_masks[i]
            query_pos = self.query_pos_head(refer_centroids_2d)
            layer_attn_mask = attn_mask
            if self.mds and i == self.decoder.eval_idx and prev_scores is not None:
                layer_attn_mask = self._rank_causal_mask(prev_scores)
            output = layer(
                output,
                refer_centroids_2d,
                feats,
                shapes,
                None,
                layer_attn_mask,
                query_pos,
            )

            polygon_logits = self.dec_bbox_head[i](output)
            if last_refined_logits is not None:
                refined_logits = polygon_logits + last_refined_logits
                refined = decode_polygon(refined_logits)
            else:
                refined_logits = polygon_logits + refer_polygon_logits
                refined = decode_polygon(refined_logits)

            if p0 is not None and confine_s is not None:
                refined = refined.clone()
                refined[..., :2] = p0 + confine_s * torch.tanh((refined[..., :2] - p0) / confine_s)

            # Classification describes the polygon predicted by this layer.
            # Resample the direct seed cue at its current centroid rather than
            # reusing the query's initial grid/peak location.  Detaching the
            # centroid preserves the seed cue as routing evidence, not a path
            # for the classification loss to move the regression output.
            layer_seed_q = seed_q
            if self.seed_in_content and recast_seed is not None:
                layer_seed_q = self._sample_seed_at(recast_seed, refined[..., :2].detach())
            scores_i = self._score(self.dec_score_head[i], output, layer_seed_q)
            if self.training:
                dec_cls.append(scores_i)
                dec_polygons.append(refined)
            elif i == self.decoder.eval_idx:
                dec_cls.append(scores_i)
                dec_polygons.append(refined)
                break

            if recast_module is not None and i + 1 == self.recast_after_layer:
                output = recast_module(output, refined[..., :2], scores_i, recast_features, recast_seed)

            prev_scores = scores_i
            last_refined_logits = refined_logits
            if self.training:
                refer_polygon_logits = refined_logits.detach()
                refer_centroids_2d = refined[..., :2].detach()
            else:
                refer_polygon_logits = refined_logits
                refer_centroids_2d = refined[..., :2]

        return torch.stack(dec_polygons), torch.stack(dec_cls)
