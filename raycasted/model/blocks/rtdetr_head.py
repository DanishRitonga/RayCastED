"""RayCast RT-DETR Decoder — RTDETRDecoder with ray polygon regression.

Extends ultralytics RTDETRDecoder to predict star-convex polygons (cx, cy, r0,
..., r_{n_rays-1}) instead of axis-aligned bounding boxes (cx, cy, w, h).

Key differences from standard RTDETRDecoder:
- enc_bbox_head / dec_bbox_head output raycast_dim (2+n_rays) instead of 4
- Anchors are n_rays-dimensional circles instead of 4-dim boxes
- Iterative refinement: sigmoid for xy, exp for rays (vs sigmoid for all 4)
- Cross-attention reference points use centroids only (2-dim), not full polygon
- Denoising disabled by default (nd=0) — ray-polygon noise TBD
"""

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import (
    MLP,
)
from ultralytics.utils.torch_utils import TORCH_1_11

from raycasted.data.etl.utils import constants as _const


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
        self.enc_bbox_head = MLP(hd, hd, self.raycast_dim, num_layers=3)
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, self.raycast_dim, num_layers=3) for _ in range(ndl)])
        # Override query_pos_head: parent creates MLP(4, 2*hd, hd) for 4-dim
        # bboxes, but we pass 2-dim centroids as reference points.
        self.query_pos_head = MLP(2, 2 * hd, hd, num_layers=2)
        self._reset_raycast_params()

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

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(
                shapes, dtype=feats.dtype, device=feats.device, n_rays=self.n_rays
            )
            self.shapes = shapes

        features = self.enc_output(self.valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)
        enc_outputs_scores = enc_outputs_scores.masked_fill(~self.valid_mask.squeeze(-1), -1e9)

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
            enc_polygons = torch.cat([decode_polygon(dn_bbox) if dn_bbox.shape[-1] > 2 else dn_bbox, enc_polygons], 1)

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

    def forward(self, x, batch=None):
        """Run full encoder-decoder forward pass."""
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)
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
        embed, refer_centroids, refer_polygon_logits, enc_polygons, enc_scores = self._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox
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
