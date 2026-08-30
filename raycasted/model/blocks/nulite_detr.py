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
from ultralytics.nn.modules.transformer import MLP

from raycasted.data.etl.utils import constants as _const


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
        seed_feature: bool = True,
        no_object: bool = True,
        seed_in_content: bool = True,
        mds: bool = True,
        shared_layers: bool = False,
    ):
        self.n_rays = n_rays or _const.N_RAYS
        self.raycast_dim = 2 + self.n_rays
        super().__init__(
            nc=nc,
            ch=ch,
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
        self.seed_threshold = seed_threshold
        self.peak_distance = peak_distance
        self.grid_size = grid_size
        self.query_selection = query_selection
        self.seed_feature = seed_feature
        self.no_object = no_object
        self.seed_in_content = seed_in_content
        self.mds = mds
        if shared_layers:
            # weight-shared transformer layers: reuse ONE DeformableTransformerDecoderLayer
            # instance across all ndl steps (params-only saving, FLOPs unchanged). Per-layer
            # dec_bbox_head/dec_score_head stay distinct so iterative refinement still varies.
            self.decoder.layers = nn.ModuleList([self.decoder.layers[0]] * self.decoder.num_layers)
        self.shared_layers = shared_layers

        # When the seed map is concatenated as an extra feature channel, the
        # input projection must accept one more channel per scale.
        if self.seed_feature:
            self.input_proj = nn.ModuleList(
                nn.Sequential(nn.Conv2d(c + 1, hd, 1, bias=False), nn.BatchNorm2d(hd)) for c in ch
            )

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

    def _get_encoder_input(self, x):
        """Project multi-scale features to hd and flatten for cross-attention.

        Returns (feats, shapes, proj2d): flattened (B, N, hd) tokens, the
        per-scale (h, w) shapes, and the projected 2D maps (proj2d[0] is the
        P2/stride-4 map used for grid-sampling query content).
        """
        proj = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        shapes = [f.shape[-2:] for f in proj]
        feats = torch.cat([f.flatten(2).transpose(1, 2) for f in proj], 1)
        return feats, shapes, proj

    def forward(self, x, batch=None, seed_map=None):
        """Run encoder + query selection + decoder forward.

        Args:
            x: list of multi-scale features [b3, b4, b5] at strides 4/8/16.
            batch: optional targets dict (denoising is disabled, nd=0).
            seed_map: (B, 1, H, W) raw NP seed logits at full resolution.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Inject the seed map as an extra feature channel (SPAR-Det-style seg
        # conditioning): the deformable cross-attention then attends over
        # nucleus-aware features. Downsample the full-res seed prob to each
        # scale and concat along the channel dim.
        if self.seed_feature and seed_map is not None:
            seed_prob = seed_map.sigmoid()
            x = [
                torch.cat(
                    [feat, F.interpolate(seed_prob, size=feat.shape[-2:], mode='bilinear', align_corners=False)],
                    dim=1,
                )
                for feat in x
            ]

        feats, shapes, proj2d = self._get_encoder_input(x)
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
                feats, proj2d, seed_map=seed_map, dn_embed=dn_embed, dn_bbox=dn_bbox
            )
            confine_s = (1.0 / max(1, round(1.0 / self.grid_size))) / 2.0
            if self.seed_in_content and seed_map is not None:
                seed_q = self._sample_seed_at(seed_map, p0)
        elif self.query_selection == 'seed':
            embed, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores = self._seed_decoder_input(
                feats, proj2d, seed_map, dn_embed, dn_bbox
            )
        else:
            embed, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores = self._learned_decoder_input(
                feats, shapes, dn_embed, dn_bbox
            )

        dec_polygons, dec_scores = self._raycast_decoder(
            embed, refer_centroids, refer_polygon_logits, feats, shapes, attn_mask, p0=p0, confine_s=confine_s,
            seed_q=seed_q, enc_scores=enc_scores
        )

        x = dec_polygons, dec_scores, enc_polygons, enc_scores, dn_meta
        if self.training:
            return x
        y = torch.cat((dec_polygons.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
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
            base_radius = self.grid_size * 2 ** i
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

    def _grid_decoder_input(self, feats, proj2d, seed_map=None, dn_embed=None, dn_bbox=None):
        """LSP-DETR-style grid-query initialization (no learned selection).

        Args:
            feats: flattened (B, N, hd) multi-scale tokens for cross-attention.
            proj2d: list of projected 2D feature maps (proj2d[0] = P2, stride 4).
            seed_map: optional (B, 1, H, W) raw seed logits; when
                seed_in_content is set, the seed logit is sampled at p0 and
                concatenated to the enc score head input.
            dn_embed: optional denoising query embeddings (nd=0, unused).
            dn_bbox: optional denoising reference polygons (nd=0, unused).

        Initial nuclei descriptors are placed on a FIXED REGULAR GRID covering
        the whole image: one query per grid cell. Grid cell diameter =
        ``grid_size`` (normalized; ~3.5um at 256px/0.25mpp matches LSP-DETR).
        Position p0 = cell center, content = bilinear-sampled projected P2
        features at p0, rays = half-cell circles. Every nucleus is guaranteed a
        nearby query (N = grid cells >= M); the query-to-cell confinement is
        applied in ``_raycast_decoder``. Returns a 6-tuple with p0 (origin grid
        positions) for that confinement.
        """
        bs = feats.shape[0]
        n_side = max(1, round(1.0 / self.grid_size))
        cell = 1.0 / n_side
        self.num_queries = n_side * n_side
        device, dtype = feats.device, feats.dtype

        coords = (torch.arange(n_side, device=device, dtype=dtype) + 0.5) / n_side
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        p0 = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).unsqueeze(0).repeat(bs, 1, 1)  # (B, nq, 2)

        # content: bilinear sample projected P2 at the grid centers
        grid = p0 * 2 - 1
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

    def _learned_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Select queries via the learned encoder score head (RT-DETR style).

        Scores every multi-scale token with enc_score_head, takes the top-nq by
        max class score, and uses their features as query content + their grid
        anchors as reference polygons. Trained end-to-end by the detection loss
        (unlike seed-map local maxima). Seed map is NOT used for selection here.
        """
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_ray_anchors(shapes, device=feats.device)
        features = self.enc_output(valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)  # (B, N, nc)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(bs, device=feats.device).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_polygon_logits = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_polygons = decode_polygon(refer_polygon_logits)
        refer_centroids = enc_polygons[..., :2]
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        embeddings = (
            self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        )

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

        return embeddings, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores

    def _seed_decoder_input(self, feats, proj2d, seed_map, dn_embed=None, dn_bbox=None):
        """Select queries from seed-map local maxima (InstanSeg-style).

        Args:
            feats: flattened (B, N, hd) multi-scale tokens for cross-attention.
            proj2d: list of projected 2D feature maps (proj2d[0] = P2, stride 4).
            seed_map: (B, 1, H, W) raw seed logits.
            dn_embed: optional denoising query embeddings (nd=0, unused).
            dn_bbox: optional denoising reference polygons (nd=0, unused).
        """
        bs = feats.shape[0]
        nq = self.num_queries
        h, w = proj2d[0].shape[-2:]
        s_h, s_w = seed_map.shape[-2:]

        # --- local maxima on the full-res seed map (torch_peak_local_max) ---
        seed_prob = seed_map.sigmoid()  # (B, 1, H, W)
        kernel = 2 * self.peak_distance + 1
        pooled, max_inds = F.max_pool2d(
            seed_prob,
            kernel_size=kernel,
            stride=1,
            padding=self.peak_distance,
            return_indices=True,
        )
        inds = torch.arange(0, seed_prob.numel(), device=seed_prob.device).reshape(seed_prob.shape)
        is_peak = (max_inds == inds) & (seed_prob > self.seed_threshold)  # (B, 1, H, W)

        # --- top-k peaks by seed value ---
        seed_flat = seed_prob.flatten(2)  # (B, 1, s_h * s_w)
        peak_flat = is_peak.flatten(2)  # (B, 1, s_h * s_w)
        vals = seed_flat.masked_fill(~peak_flat, 0.0)  # zero non-peak positions
        _, topk_idx = torch.topk(vals.squeeze(1), nq, dim=1)  # (B, nq)

        ys = topk_idx // s_w
        xs = topk_idx % s_w
        centroids = torch.stack([(xs.float() + 0.5) / s_w, (ys.float() + 0.5) / s_h], -1)  # (B, nq, 2) in [0, 1]

        # --- query content: bilinear sample projected P2 at the centroids ---
        grid = centroids.to(feats.dtype) * 2 - 1  # [-1, 1] for grid_sample
        content = F.grid_sample(
            proj2d[0],
            grid.unsqueeze(1),
            mode='bilinear',
            align_corners=False,
            padding_mode='border',
        )  # (B, hd, 1, nq)
        content = content.squeeze(2).transpose(1, 2)  # (B, nq, hd)
        content = self.enc_output(content)

        # --- anchors: base-radius circle at each centroid ---
        rays = torch.full((bs, nq, self.n_rays), self.grid_size, device=feats.device, dtype=feats.dtype)
        anchors = encode_polygon(torch.cat([centroids.to(feats.dtype), rays], -1))

        refer_polygon_logits = self.enc_bbox_head(content) + anchors
        enc_polygons = decode_polygon(refer_polygon_logits)
        refer_centroids = enc_polygons[..., :2]
        enc_scores = self.enc_score_head(content)
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

        return embeddings, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores

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

    def _raycast_decoder(
        self, embed, refer_centroids, refer_polygon_logits, feats, shapes, attn_mask, p0=None, confine_s=None,
        seed_q=None, enc_scores=None
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
        output = embed
        dec_polygons = []
        dec_cls = []
        last_refined_logits = None
        refer_centroids_2d = refer_centroids
        prev_scores = enc_scores

        for i, layer in enumerate(self.decoder.layers):
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

            scores_i = self._score(self.dec_score_head[i], output, seed_q)
            if self.training:
                dec_cls.append(scores_i)
                dec_polygons.append(refined)
            elif i == self.decoder.eval_idx:
                dec_cls.append(scores_i)
                dec_polygons.append(refined)
                break

            prev_scores = scores_i
            last_refined_logits = refined_logits
            if self.training:
                refer_polygon_logits = refined_logits.detach()
                refer_centroids_2d = refined[..., :2].detach()
            else:
                refer_polygon_logits = refined_logits
                refer_centroids_2d = refined[..., :2]

        return torch.stack(dec_polygons), torch.stack(dec_cls)
