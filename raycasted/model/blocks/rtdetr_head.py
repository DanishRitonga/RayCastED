"""RayCast RT-DETR Decoder — RTDETRDecoder with ray polygon regression.

Extends ultralytics RTDETRDecoder to predict star-convex polygons (cx, cy, r0,
..., r_{n_rays-1}) instead of axis-aligned bounding boxes (cx, cy, w, h).

Key differences from standard RTDETRDecoder:
- enc_bbox_head / dec_bbox_head output raycast_dim (2+n_rays) instead of 4
- Anchors are n_rays-dimensional circles instead of 4-dim boxes
- Iterative refinement: sigmoid for xy, exp for rays (vs sigmoid for all 4)
- Cross-attention reference points use centroids only (2-dim), not full polygon
- Denoising: raycast-aware _get_cdn_group_raycast applies centroid jitter + ray noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import (
    MLP,
)
from ultralytics.utils.torch_utils import TORCH_1_11

from raycasted.data.etl.utils import constants as _const


class SpatialGAT(nn.Module):
    """Spatial k-NN Graph Attention for decoder query self-attention.

    Replaces dense multi-head self-attention in the RT-DETR decoder with
    spatially-local attention. For each query, builds a spatial k-NN graph
    from predicted centroid positions and restricts attention to the k
    nearest spatial neighbors. This provides the inductive bias that makes
    Conv and STAttention work for scratch training — unlike dense attention
    which produces uniform weights when initialized from scratch.

    Mimics nn.MultiheadAttention interface so it can drop-in replace
    DeformableTransformerDecoderLayer.self_attn.
    """

    def __init__(self, embed_dim: int, num_heads: int, k: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k = k
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rel_pos_proj = nn.Linear(2, num_heads, bias=False)  # Δx,Δy → per-head attn bias
        self.centroids = None

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        average_attn_weights=True,
        is_causal=False,
    ):
        """Spatial k-NN attention. centroids must be set before calling."""
        L, B, _ = query.shape

        q = self.q_proj(query).view(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, n_heads, L, L]

        # Spatial k-NN mask: allow self + K nearest neighbors, mask rest
        spatial_bool = torch.ones(B, 1, L, L, dtype=torch.bool, device=attn_weights.device)
        if self.centroids is not None:
            centroids = self.centroids.detach().clamp(1e-6, 1.0 - 1e-6)
            dists = torch.cdist(centroids, centroids)
            _, knn_idx = dists.topk(min(self.k + 1, L), dim=-1, largest=False)
            knn_idx = knn_idx[:, :, 1:]  # [B, L, K]
            spatial_bool = torch.zeros(B, 1, L, L, dtype=torch.bool, device=attn_weights.device)
            # Allow self-attention
            b_idx = torch.arange(B, device=attn_weights.device).view(B, 1, 1)
            spatial_bool[b_idx, :, torch.arange(L, device=attn_weights.device).view(1, L, 1),
                        torch.arange(L, device=attn_weights.device).view(1, 1, L)] = True
            # Allow K nearest neighbors
            spatial_bool[b_idx, :, torch.arange(L, device=attn_weights.device).view(1, L, 1), knn_idx] = True

            # Relative position bias: Δx,Δy → per-head scalar
            b_exp = torch.arange(B, device=attn_weights.device).view(B, 1, 1).expand(B, L, self.k)
            c_neigh = centroids[b_exp, knn_idx]  # [B, L, K, 2]
            c_query = centroids.unsqueeze(2)   # [B, L, 1, 2]
            rel_pos = c_neigh - c_query         # [B, L, K, 2]
            rel_bias = self.rel_pos_proj(rel_pos)  # [B, L, K, n_heads]
            rel_bias = rel_bias.permute(0, 3, 1, 2)  # [B, n_heads, L, K]

            # Scatter into [B, n_heads, L, L]
            q_idx = torch.arange(L, device=attn_weights.device).view(1, 1, L, 1).expand(B, self.num_heads, L, self.k)
            h_idx = torch.arange(self.num_heads, device=attn_weights.device).view(1, self.num_heads, 1, 1)
            b_exp = b_idx.view(B, 1, 1, 1).expand(B, self.num_heads, L, self.k)
            knn_exp = knn_idx.unsqueeze(1).expand(B, self.num_heads, L, self.k)
            attn_weights[b_exp, h_idx, q_idx, knn_exp] += rel_bias

        attn_weights = attn_weights.masked_fill(~spatial_bool, float('-inf'))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(1, 1, L, L)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(L, B, self.embed_dim)
        return self.out_proj(out), None


def _get_cdn_group_raycast(
    batch,
    num_classes,
    num_queries,
    class_embed,
    num_dn=100,
    cls_noise_ratio=0.5,
    box_noise_scale=1.0,
    training=False,
):
    """Raycast-aware CDN group builder — same logic as ultralytics get_cdn_group.

    Works with n_rays-dim polygon bboxes instead of 4-dim xywh bboxes.
    Noise strategy: additive on centroids, multiplicative on rays.
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    gt_groups = batch['gt_groups']
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    bs = len(gt_groups)
    gt_cls = batch['cls']
    gt_bbox = batch['bboxes']
    b_idx = batch['batch_idx']

    dn_cls = gt_cls.repeat(2 * num_group)
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)

    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        mask = torch.rand(dn_cls.shape, device=dn_cls.device) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        centroid_noise = (torch.rand_like(dn_bbox[:, :2]) * 2 - 1) * box_noise_scale * 0.05
        centroid_noise = centroid_noise.clamp(-0.25, 0.25)

        ray_std = box_noise_scale * 0.1
        ray_noise = torch.randn_like(dn_bbox[:, 2:]) * ray_std
        ray_noise = ray_noise.clamp(-0.5, 0.5)

        rand_sign = torch.randint_like(dn_bbox[:, :1], 0, 2, dtype=torch.float) * 2.0 - 1.0
        rand_scale = torch.rand_like(dn_bbox[:, 2:])
        rand_scale[neg_idx] += 1.0
        ray_noise = ray_noise * rand_sign * rand_scale

        dn_bbox = torch.cat(
            [
                (dn_bbox[:, :2] + centroid_noise).clamp(0.0, 1.0),
                (dn_bbox[:, 2:] + ray_noise).clamp(0.0, 1.0),
            ],
            dim=1,
        )
        dn_bbox = torch.logit(dn_bbox.clamp(1e-6, 1.0 - 1e-6))

    num_dn = int(max_nums * 2 * num_group)
    bbox_dim = gt_bbox.shape[-1]
    dn_cls_embed = class_embed[dn_cls]
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, bbox_dim, device=gt_bbox.device)

    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    attn_mask[num_dn:, :num_dn] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True
    dn_meta = {
        'dn_pos_idx': [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        'dn_num_group': num_group,
        'dn_num_split': [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )


def encode_polygon(polygon: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encode polygon to logit/log space for iterative refinement.

    xy: inverse_sigmoid (logit space)
    rays: log (log space, inverse of exp)
    """
    xy = polygon[..., :2].clamp(eps, 1 - eps)
    rays = polygon[..., 2:].clamp(min=eps)
    return torch.cat([torch.log(xy / (1 - xy)), torch.log(rays)], dim=-1)


def decode_polygon(logits: torch.Tensor) -> torch.Tensor:
    """Decode polygon from logit/log space.

    xy: sigmoid (bounded [0,1])
    rays: exp (strictly positive)
    """
    return torch.cat([logits[..., :2].sigmoid(), logits[..., 2:].exp()], dim=-1)


class RayCastRTDETRDecoder(RTDETRDecoder):
    """RT-DETR Decoder with ray-cast polygon regression.

    Same transformer decoder architecture (deformable cross-attention,
    self-attention, iterative refinement) but predicts star-convex polygons
    instead of bounding boxes.

    Cross-attention uses 2-dim centroids as reference points (same as standard
    RT-DETR). The full polygon is refined in parallel using separate
    encode/decode functions that handle xy (sigmoid) and rays (exp) differently.
    """

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 0,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        n_rays: int | None = None,
        query_stride: int | None = None,
    ):
        self.n_rays = n_rays or _const.N_RAYS
        self.raycast_dim = 2 + self.n_rays
        self.query_stride = query_stride
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
        self.enc_bbox_head = MLP(hd, hd, self.raycast_dim, num_layers=3)
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, self.raycast_dim, num_layers=3) for _ in range(ndl)])
        # Override query_pos_head: parent creates MLP(4, 2*hd, hd) for 4-dim
        # bboxes, but we pass 2-dim centroids as reference points.
        self.query_pos_head = MLP(2, 2 * hd, hd, num_layers=2)
        self._reset_raycast_params()
        self._patch_self_attn(hd, nh, dropout)

    def _patch_self_attn(self, hd, nh, dropout):
        """Replace dense MHA with spatial GAT in all decoder layers."""
        for layer in self.decoder.layers:
            layer.self_attn = SpatialGAT(embed_dim=hd, num_heads=nh, k=8, dropout=dropout)

    def _grid_nq(self, raw_features):
        """Compute number of grid queries from P4 feature map and stride."""
        p4 = raw_features[-1]
        crop_size = p4.shape[2] * 16
        stride = self.query_stride
        return int(crop_size // stride) ** 2

    def _reset_raycast_params(self):
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for reg_ in self.dec_bbox_head:
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)

    @staticmethod
    def _generate_anchors(
        shapes,
        grid_size=0.05,
        dtype=torch.float32,
        device='cpu',
        eps=1e-2,
        n_rays=64,
    ):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            if TORCH_1_11:
                grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')
            else:
                grid_y, grid_x = torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            base_radius = grid_size * (2.0**i)
            rays = torch.ones(1, h * w, n_rays, dtype=dtype, device=device) * base_radius
            anchor = torch.cat([grid_xy.view(1, h * w, 2), rays], dim=-1)
            anchors.append(anchor)
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors[..., :2] > eps) & (anchors[..., :2] < 1 - eps)).all(-1, keepdim=True)
        anchors_encoded = encode_polygon(anchors)
        anchors_encoded = anchors_encoded.masked_fill(~valid_mask, float('inf'))
        return anchors_encoded, valid_mask

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None, raw_features=None):
        bs = feats.shape[0]
        if self.query_stride is not None:
            return self._get_grid_decoder_input(raw_features, bs, dn_embed, dn_bbox)
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(
                shapes, dtype=feats.dtype, device=feats.device, n_rays=self.n_rays
            )
            self.shapes = shapes

        features = self.enc_output(self.valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)
        invalid = ~self.valid_mask.expand(-1, -1, enc_outputs_scores.shape[-1])
        enc_outputs_scores = enc_outputs_scores.masked_fill(invalid, -float('inf'))

        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_polygon_logits = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_polygons = decode_polygon(refer_polygon_logits)
        refer_centroids = enc_polygons[..., :2]

        if dn_bbox is not None:
            dn_centroids = dn_bbox[..., :2] if dn_bbox.shape[-1] > 2 else dn_bbox
            refer_polygon_logits = torch.cat([dn_bbox, refer_polygon_logits], 1)
            refer_centroids = torch.cat([dn_centroids, refer_centroids], 1)

        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features

        if self.training:
            refer_polygon_logits = refer_polygon_logits.detach()
            refer_centroids = refer_centroids.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()

        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores

    def _get_grid_decoder_input(self, raw_features, bs, dn_embed, dn_bbox):
        """Build decoder input from a uniform spatial grid (LSP-DETR style).

        Bypasses encoder score head entirely — uses a fixed grid of query
        positions instead of learned top-K selection. Each query's features
        are sampled from P4 neck features at its grid position via
        grid_sample.
        """
        p4_feat = raw_features[-1]  # [B, C_P4, H_P4, W_P4]
        p4_proj = self.input_proj[-1](p4_feat)  # [B, hd, H, W]
        _, _, H_p4, W_p4 = p4_proj.shape
        crop_size = H_p4 * 16  # P4 stride=16 → ×16 = image size
        stride = self.query_stride
        grid_h = int(crop_size // stride)
        grid_w = int(crop_size // stride)
        nq = grid_h * grid_w

        cy = (torch.arange(grid_h, device=p4_proj.device, dtype=torch.float32) + 0.5) * stride / crop_size
        cx = (torch.arange(grid_w, device=p4_proj.device, dtype=torch.float32) + 0.5) * stride / crop_size
        gy, gx = torch.meshgrid(cy, cx, indexing='ij')
        centroids = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [nq, 2], normalized [0,1]
        centroids = centroids.unsqueeze(0).expand(bs, -1, -1)  # [B, nq, 2]

        grid_sample_input = torch.stack([gx * 2 - 1, gy * 2 - 1], dim=-1).unsqueeze(0).expand(bs, -1, -1, -1).to(dtype=p4_proj.dtype)
        sampled_feats = F.grid_sample(p4_proj, grid_sample_input, align_corners=False)  # [B, hd, grid_h, grid_w]
        top_k_features = sampled_feats.flatten(2).transpose(1, 2)  # [B, nq, hd]

        initial_radius = stride / (2 * crop_size)  # normalized radius in [0,1]
        rays = torch.full((bs, nq, self.n_rays), initial_radius, device=p4_proj.device, dtype=p4_proj.dtype)
        polygons = torch.cat([centroids, rays], dim=-1)  # [B, nq, raycast_dim]
        anchors_encoded = encode_polygon(polygons)

        refer_polygon_logits = self.enc_bbox_head(top_k_features) + anchors_encoded
        enc_polygons = decode_polygon(refer_polygon_logits)
        refer_centroids = enc_polygons[..., :2]

        if dn_bbox is not None:
            dn_centroids = dn_bbox[..., :2] if dn_bbox.shape[-1] > 2 else dn_bbox
            refer_polygon_logits = torch.cat([dn_bbox, refer_polygon_logits], 1)
            refer_centroids = torch.cat([dn_centroids, refer_centroids], 1)

        enc_scores = torch.zeros(bs, nq, self.nc, device=p4_proj.device, dtype=p4_proj.dtype)
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features

        if self.training:
            refer_polygon_logits = refer_polygon_logits.detach()
            refer_centroids = refer_centroids.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()

        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores

    def forward(self, x, batch=None):
        """Run full encoder-decoder forward pass."""
        feats, shapes = self._get_encoder_input(x)
        nq = self._grid_nq(x) if self.query_stride is not None else self.num_queries
        dn_embed, dn_bbox, attn_mask, dn_meta = _get_cdn_group_raycast(
            batch,
            self.nc,
            nq,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        embed, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores = self._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox, raw_features=x
        )

        dec_polygons, dec_scores = self._raycast_decoder(
            embed, refer_centroids, refer_polygon_logits, feats, shapes, attn_mask
        )

        x = dec_polygons, dec_scores, enc_polygons, enc_scores, dn_meta
        if self.training:
            return x
        y = torch.cat((dec_polygons.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _raycast_decoder(self, embed, refer_centroids, refer_polygon_logits, feats, shapes, attn_mask):
        output = embed
        dec_polygons = []
        dec_cls = []
        last_refined_logits = None
        refer_centroids_2d = refer_centroids

        for i, layer in enumerate(self.decoder.layers):
            layer.self_attn.centroids = refer_centroids_2d
            query_pos = self.query_pos_head(refer_centroids_2d)
            output = layer(
                output,
                refer_centroids_2d,
                feats,
                shapes,
                None,
                attn_mask,
                query_pos,
            )

            polygon_logits = self.dec_bbox_head[i](output)
            if last_refined_logits is not None:
                refined_logits = polygon_logits + last_refined_logits
                refined = decode_polygon(refined_logits)
            else:
                refined_logits = polygon_logits + refer_polygon_logits
                refined = decode_polygon(refined_logits)

            if self.training:
                dec_cls.append(self.dec_score_head[i](output))
                dec_polygons.append(refined)
            elif i == self.decoder.eval_idx:
                dec_cls.append(self.dec_score_head[i](output))
                dec_polygons.append(refined)
                break

            last_refined_logits = refined_logits
            if self.training:
                refer_polygon_logits = refined_logits.detach()
                refer_centroids_2d = refined[..., :2].detach()
            else:
                refer_polygon_logits = refined_logits
                refer_centroids_2d = refined[..., :2]

        return torch.stack(dec_polygons), torch.stack(dec_cls)
