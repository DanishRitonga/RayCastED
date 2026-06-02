"""LSP-DETR model wrapper — RayCastED trainer integration.

Uses the full LSP-DETR architecture (SwinV2 backbone + LSPTransformer decoder
with STAttention + CayleySTRING PE) with per-object Hungarian matching and
sigmoid focal loss matching the original LSP-DETR implementation exactly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from raycasted.model.blocks.lsp_detr_arch import (
    FeatureSampling,
    LSPTransformer,
    relative_to_absolute_pos,
)
from raycasted.model.hybrid_loss import HybridHungarianMatcher

if TYPE_CHECKING:
    from collections.abc import Sequence


class LSPSetCriterion(nn.Module):
    """Exact LSP-DETR loss: sigmoid_focal_loss + L1 centroid + L1 log-space radial."""

    def __init__(
        self,
        nc: int,
        matcher: HybridHungarianMatcher,
        weight_dict: dict[str, float] | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        n_rays: int = 64,
        crop_size: int = 256,
    ):
        super().__init__()
        self.nc = nc
        self.matcher = matcher
        self.weight_dict = weight_dict or {'loss_ce': 1.0, 'loss_centroid': 1.0, 'loss_radial': 1.0}
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.n_rays = n_rays
        self.crop_size = crop_size

    def _cls_loss(self, logits, targets, matched):
        num_classes = logits.shape[-1]
        device = logits.device

        tgt_classes = torch.full(logits.shape[:2], num_classes - 1, dtype=torch.int64, device=device)

        for b, (pred_idx, tgt_idx) in enumerate(matched):
            if len(tgt_idx) == 0:
                continue
            labels = targets[b].get('labels')
            if labels is None or len(labels) == 0:
                continue
            matched_labels = labels[tgt_idx].to(device=device)
            valid = matched_labels < num_classes - 1
            tgt_classes[b, pred_idx[valid]] = matched_labels[valid]

        tgt_one_hot = F.one_hot(tgt_classes, num_classes).type_as(logits)
        return sigmoid_focal_loss(
            logits, tgt_one_hot, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='mean'
        )

    def _centroid_loss(self, points, targets, matched):
        total = torch.tensor(0.0, device=points.device)
        count = 0
        for b, (pred_idx, tgt_idx) in enumerate(matched):
            if len(tgt_idx) == 0:
                continue
            boxes = targets[b].get('boxes')
            if boxes is None or boxes.numel() == 0:
                continue
            tgt_pts = boxes[tgt_idx, :2].to(points)
            pred_pts = points[b, pred_idx, :2]
            total = total + F.l1_loss(pred_pts, tgt_pts)
            count += 1
        return total / max(count, 1)

    def _radial_loss(self, radial_log, targets, matched):
        total = torch.tensor(0.0, device=radial_log.device)
        count = 0
        for b, (pred_idx, tgt_idx) in enumerate(matched):
            if len(tgt_idx) == 0:
                continue
            boxes = targets[b].get('boxes')
            if boxes is None or boxes.numel() == 0 or boxes.shape[1] < 2 + self.n_rays:
                continue
            gt_rays_norm = boxes[tgt_idx, 2:]
            gt_log = torch.log(gt_rays_norm * self.crop_size + 1e-7).to(radial_log)
            pred = radial_log[b, pred_idx]
            item = torch.max(F.relu(gt_log - pred), F.relu(pred - gt_log))
            total = total + item.nanmean()
            count += 1
        return total / max(count, 1)

    def forward(self, outputs, targets, crop_size=None):
        _crop_size = crop_size or self.crop_size
        matched = self.matcher(outputs, targets, crop_size=_crop_size)
        loss_dict = {
            'loss_ce': self._cls_loss(outputs['pred_logits'], targets, matched),
            'loss_centroid': self._centroid_loss(outputs['pred_points'], targets, matched),
            'loss_radial': self._radial_loss(outputs['pred_radial'], targets, matched),
        }
        total = sum(loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict)

        if 'aux_outputs' in outputs:
            for aux in outputs['aux_outputs']:
                aux_matched = self.matcher(aux, targets, crop_size=_crop_size)
                total = total + self._cls_loss(aux['pred_logits'], targets, aux_matched) * self.weight_dict.get(
                    'loss_ce', 1.0
                )
                total = total + self._centroid_loss(aux['pred_points'], targets, aux_matched) * self.weight_dict.get(
                    'loss_centroid', 1.0
                )
                total = total + self._radial_loss(
                    aux['pred_radial'], targets, aux_matched
                ) * self.weight_dict.get('loss_radial', 1.0)

        loss_dict['total'] = total
        return loss_dict


class LSPDetrDetectionModel(nn.Module):
    """LSP-DETR model integrated into RayCastED trainer.

    Uses SwinV2-tiny backbone + LSPTransformer decoder with tiled sparse
    attention (STAttention) and CayleySTRING positional encoding.  Loss is
    sigmoid_focal_loss matching the original LSP-DETR paper exactly.
    """

    def __init__(
        self,
        nc: int = 5,
        n_rays: int = 64,
        hidden_dim: int = 384,
        n_heads: int = 12,
        num_layers: int = 6,
        query_block_size: int = 14,
        crop_size: int = 256,
        backbone_name: str = 'microsoft/swinv2-tiny-patch4-window16-256',
    ):
        super().__init__()
        self.nc = nc
        self.n_rays = n_rays
        self.crop_size = crop_size
        self.query_block_size = query_block_size
        self.raycast_dim = 2 + n_rays

        from transformers import AutoBackbone

        self.backbone = AutoBackbone.from_pretrained(backbone_name)
        _, *feature_channels, neck_channels = self.backbone.num_features

        self.feature_sampling = FeatureSampling(neck_channels, hidden_dim)

        self_sta = {'kernel': 3, 'q_tile': 3, 'kv_tile': 3}
        cross_sta: Sequence[dict[str, int]] = (
            {'kernel': 5, 'q_tile': 3, 'kv_tile': 8},
            {'kernel': 5, 'q_tile': 3, 'kv_tile': 4},
            {'kernel': 5, 'q_tile': 3, 'kv_tile': 2},
        )

        self.decode_head = LSPTransformer(
            dim=hidden_dim,
            num_heads=n_heads,
            num_classes=nc,
            query_block_size=query_block_size,
            num_radial_distances=n_rays,
            feature_channels=feature_channels,
            feature_levels=(2, 1, 0, 2, 1, 0),
            self_sta_config=self_sta,
            cross_sta_config=cross_sta,
        )

        matcher = HybridHungarianMatcher(
            cost_class=1.0, cost_centroid=1.0, cost_radial=1.0, cost_inner=0.0,
            focal_alpha=0.25, focal_gamma=2.0, n_rays=n_rays,
        )
        self.criterion = LSPSetCriterion(
            nc=nc, matcher=matcher, focal_alpha=0.25, focal_gamma=2.0, n_rays=n_rays, crop_size=crop_size
        )

        self.names = {i: str(i) for i in range(nc)}

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape

        *features, neck = self.backbone(x).feature_maps

        ref_points = torch.zeros(
            b, math.ceil(h / self.query_block_size), math.ceil(w / self.query_block_size), 2,
            dtype=torch.float32, device=neck.device,
        )
        tgt = self.feature_sampling(
            relative_to_absolute_pos(ref_points, self.query_block_size, self.query_block_size),
            neck,
        )

        out = self.decode_head(tgt, ref_points, features, h, w)
        return {
            'pred_logits': out['logits'],
            'pred_points': out['points'] / self.crop_size,
            'pred_radial': out['radial_distances'],
            'aux_outputs': [
                {'pred_logits': a['logits'], 'pred_points': a['points'] / self.crop_size,
                 'pred_radial': a['radial_distances']}
                for a in out['aux_outputs']
            ],
        }

    def loss(self, batch, preds=None):
        if preds is None:
            preds = self.predict(batch['img'])

        bs = batch['img'].shape[0]
        batch_idx = batch.get('batch_idx', torch.zeros_like(batch['cls']))
        targets = []
        for i in range(bs):
            mask_i = batch_idx == i
            tgt = {'labels': batch['cls'][mask_i].long()}
            if 'bboxes' in batch:
                tgt['boxes'] = batch['bboxes'][mask_i]
            targets.append(tgt)

        loss_dict = self.criterion(preds, targets, crop_size=self.crop_size)
        return loss_dict['total'], torch.as_tensor(
            [loss_dict.get(k, torch.tensor(0.0)).detach() for k in ['loss_ce', 'loss_centroid', 'loss_radial']],
            device=batch['img'].device,
        )

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x)
        return self.predict(x)
