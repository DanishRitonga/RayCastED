"""NuLite DETR Loss — Hungarian 1:1 matching + focal/VFL cls + ray L1 + polar IoU.

Owned by the NuLite DETR head only. Replaces ultralytics DETRLoss's bbox L1 +
GIoU with ray L1 + xy L1 + polar IoU, and HungarianMatcher's bbox/giou costs
with ray/piou costs. Pure 1:1 assignment (no one-to-many branch, no NMS).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch, polar_iou_torch
from raycasted.data.etl.ops.loss import curvature_smoothness_loss_torch
from raycasted.data.etl.utils import constants as _const
from raycasted.model.blocks.star_distances_analytical import (
    _normed_rays_to_vertices,
    analytical_gt_rays,
    build_ray_directions,
)


def _log_space_ray_loss(pred_rays: torch.Tensor, target_rays: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """L1 loss in log-space for ray distances (matches FCN RayCastDetectionLoss).

    Equalizes relative errors across scales: a 10% error on a small ray gives
    the same loss as a 10% error on a large ray.
    """
    log_pred = torch.clamp(pred_rays, min=eps).log()
    log_tgt = torch.clamp(target_rays, min=eps).log()
    return (log_pred - log_tgt).abs().mean(-1)


def _points_inside_star(pred_xy, gt_xy, gt_rays, n_rays):
    """Test whether each predicted centroid lies inside each GT star polygon.

    LSP-DETR inner-mask cost term Wm: a centroid is "inside" a star-convex
    nucleus iff its distance from the GT center along its angular direction is
    <= the ray length interpolated between the two bracketing rays.

    Args:
        pred_xy: (nq, 2) predicted centroids, normalized [0,1].
        gt_xy: (n_gt, 2) GT centers, normalized [0,1].
        gt_rays: (n_gt, n_rays) GT ray lengths, normalized [0,1].
        n_rays: number of radial rays.

    Returns:
        (nq, n_gt) bool tensor, True where the centroid is inside the polygon.
    """
    nq = pred_xy.shape[0]
    n_gt = gt_xy.shape[0]
    dx = pred_xy[:, 0].unsqueeze(1) - gt_xy[:, 0].unsqueeze(0)  # (nq, n_gt)
    dy = pred_xy[:, 1].unsqueeze(1) - gt_xy[:, 1].unsqueeze(0)
    dist = (dx * dx + dy * dy).sqrt()

    phi = torch.remainder(torch.atan2(dy, dx), 2 * math.pi)  # [0, 2pi)
    angle_step = 2 * math.pi / n_rays
    idx = phi / angle_step  # continuous ray index
    i0 = idx.floor().long().clamp(0, n_rays - 1)
    frac = idx - i0.float()
    i1 = (i0 + 1) % n_rays

    gt_rays_exp = gt_rays.unsqueeze(0).expand(nq, n_gt, n_rays)  # (nq, n_gt, n_rays)
    r0 = gt_rays_exp.gather(2, i0.unsqueeze(-1)).squeeze(-1)  # (nq, n_gt)
    r1 = gt_rays_exp.gather(2, i1.unsqueeze(-1)).squeeze(-1)
    d_phi = r0 * (1 - frac) + r1 * frac

    return dist <= d_phi


class RayCastHungarianMatcher(nn.Module):
    """Hungarian matcher using class cost + ray L1 cost + polar IoU cost.

    Same structure as ultralytics HungarianMatcher but replaces bbox L1 and
    GIoU costs with ray L1 and polar IoU costs for star-convex polygons.
    """

    def __init__(
        self,
        cost_gain=None,
        use_fl=True,
        alpha=0.25,
        gamma=2.0,
        n_rays=None,
        cost_inside=10.0,
    ):
        super().__init__()
        if cost_gain is None:
            cost_gain = {'class': 2, 'ray': 5, 'piou': 2, 'inside': cost_inside}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.alpha = alpha
        self.gamma = gamma
        self.n_rays = n_rays or _const.N_RAYS

    def forward(
        self,
        pred_polygons,
        pred_scores,
        gt_polygons,
        gt_cls,
        gt_groups,
    ):
        """Compute Hungarian matching between predictions and GT.

        Cost matrices are computed per image (matching RayCastAssigner's
        per-batch loop) so the pairwise polar-IoU stays at (nq, n_gt_i, R)
        instead of (bs*nq, n_gt_total, R), which would OOM at large batch.
        """
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        pred_scores = pred_scores.detach()
        pred_polygons = pred_polygons.detach()
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)

        gt_poly_split = gt_polygons.split(gt_groups)
        gt_cls_split = gt_cls.split(gt_groups)

        indices = []
        gt_offset = 0
        for bi in range(bs):
            p_scores = pred_scores[bi]  # (nq, nc)
            p_poly = pred_polygons[bi]  # (nq, raycast_dim)
            g_poly = gt_poly_split[bi]  # (n_gt, raycast_dim)
            g_cls = gt_cls_split[bi]  # (n_gt,)
            n_gt = g_poly.shape[0]

            if n_gt == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue

            pred_sel = p_scores[:, g_cls]  # (nq, n_gt)
            if self.use_fl:
                neg_cost = (1 - self.alpha) * (pred_sel**self.gamma) * (-(1 - pred_sel + 1e-8).log())
                pos_cost = self.alpha * ((1 - pred_sel) ** self.gamma) * (-(pred_sel + 1e-8).log())
                cost_class = pos_cost - neg_cost
            else:
                cost_class = -pred_sel

            pred_rays = p_poly[:, 2:]
            gt_rays = g_poly[:, 2:]
            cost_ray = (pred_rays.unsqueeze(1) - gt_rays.unsqueeze(0)).abs().sum(-1)

            pred_xy = p_poly[:, :2]
            gt_xy = g_poly[:, :2]
            cost_xy = (pred_xy.unsqueeze(1) - gt_xy.unsqueeze(0)).abs().sum(-1)

            if nq > 0 and n_gt > 0:
                pred_exp = pred_rays.unsqueeze(1).expand(nq, n_gt, -1)
                gt_exp = gt_rays.unsqueeze(0).expand(nq, n_gt, -1)
                piou = polar_iou_pairwise_flat_torch(pred_exp, gt_exp)
                cost_piou = 1.0 - piou
            else:
                cost_piou = torch.zeros(nq, n_gt, device=pred_polygons.device)

            # LSP-DETR inner-mask cost Wm: 0 if the predicted centroid is
            # inside the GT nucleus, lambda otherwise. Dominates misassignment
            # of displaced centroids (the train17 val->test gap culprit).
            cost_inside = 0.0
            if self.cost_gain.get('inside', 0.0) != 0.0:
                inside = _points_inside_star(pred_xy, gt_xy, gt_rays, self.n_rays)
                cost_inside = self.cost_gain['inside'] * (~inside).float()

            cost = (
                self.cost_gain['class'] * cost_class
                + self.cost_gain['ray'] * (cost_ray + cost_xy)
                + self.cost_gain['piou'] * cost_piou
                + cost_inside
            )
            cost[cost.isnan() | cost.isinf()] = 0.0

            i, j = linear_sum_assignment(cost.cpu())
            indices.append((torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_offset))
            gt_offset += n_gt

        return indices


class NuLiteDETRLoss(nn.Module):
    """Detection loss for the NuLite DETR head: focal/VFL cls + ray L1 + polar IoU.

    Mirrors ultralytics DETRLoss structure but operates on ray polygons instead
    of bounding boxes. Supports auxiliary losses from intermediate decoder
    layers. Query class targets: nc = "no object" for unmatched queries.
    """

    def __init__(
        self,
        nc=80,
        loss_gain=None,
        aux_loss=True,
        use_vfl=True,
        no_object=True,
        n_rays=None,
        cost_inside=10.0,
        per_layer_match=False,
    ):
        super().__init__()
        if loss_gain is None:
            # FCN RayCastDetectionLoss parity: same regression terms + lambdas
            # (xy=500, l1=14, piou=3.0, smooth=0, cls=2.0) plus the DETR-only
            # "no object" column weight.
            loss_gain = {'class': 2.0, 'xy': 500.0, 'l1': 14.0, 'piou': 3.0, 'smooth': 0.0, 'no_object': 1.0}
        self.nc = nc
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.no_object = no_object
        self.per_layer_match = per_layer_match
        self.n_rays = n_rays or _const.N_RAYS
        self.matcher = RayCastHungarianMatcher(
            cost_gain={'class': 2, 'ray': 5, 'piou': 2, 'inside': cost_inside},
            n_rays=self.n_rays,
        )
        from ultralytics.utils.loss import FocalLoss, VarifocalLoss

        self.fl = FocalLoss(2.0, 0.25)
        # LSP-DETR uses pure focal with an explicit "no object" class; VFL is
        # kept only for the legacy non-no_object ablation path.
        self.vfl = VarifocalLoss(2.0, 0.75) if (use_vfl and not no_object) else None
        self.device = torch.device('cpu')

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        bs, nq = pred_scores.shape[:2]
        name = f'loss_class{postfix}'
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        if not self.no_object:
            # legacy path: drop the "no object" column (VFL-style)
            one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if num_gts and self.vfl is not None:
            loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
        elif self.no_object:
            # focal over nc+1 classes with an optional per-column gain on the
            # "no object" class (index nc). gain=1.0 == plain focal (LSP-DETR).
            no_obj_gain = self.loss_gain['no_object']
            per_col = torch.ones(self.nc + 1, device=one_hot.device, dtype=pred_scores.dtype)
            per_col[-1] = no_obj_gain
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_scores, one_hot.float(), weight=per_col.view(1, 1, -1), reduction='none'
            )
            pred_prob = pred_scores.sigmoid()
            p_t = one_hot.float() * pred_prob + (1 - one_hot.float()) * (1 - pred_prob)
            loss_cls = loss_cls * (1.0 - p_t) ** self.fl.gamma
            alpha = self.fl.alpha.to(device=pred_scores.device, dtype=pred_scores.dtype)
            alpha_factor = one_hot.float() * alpha + (1 - one_hot.float()) * (1 - alpha)
            loss_cls = loss_cls * alpha_factor
            loss_cls = loss_cls.mean(1).sum()
        else:
            loss_cls = self.fl(pred_scores, one_hot.float())
        loss_cls /= max(num_gts, 1) / nq

        return {name: loss_cls.squeeze() * self.loss_gain['class']}

    def _get_loss_bbox(self, pred_polygons, gt_polygons, crop_size, postfix=''):
        """FCN-parity polygon regression losses: xy (Huber) + ray L1 (log) + pIoU + smooth."""
        name_xy = f'loss_xy{postfix}'
        name_l1 = f'loss_l1{postfix}'
        name_piou = f'loss_piou{postfix}'
        name_smooth = f'loss_smooth{postfix}'

        loss = {}
        if len(gt_polygons) == 0:
            loss[name_xy] = torch.tensor(0.0, device=pred_polygons.device)
            loss[name_l1] = torch.tensor(0.0, device=pred_polygons.device)
            loss[name_piou] = torch.tensor(0.0, device=pred_polygons.device)
            loss[name_smooth] = torch.tensor(0.0, device=pred_polygons.device)
            return loss

        pred_xy = pred_polygons[:, :2].float()
        gt_xy = gt_polygons[:, :2].float()
        pred_rays = pred_polygons[:, 2:].float()
        gt_rays_static = gt_polygons[:, 2:].float()
        n_gt = len(gt_polygons)

        # L_xy: Huber on centroid (delta=0.05), FCN loss[0].
        loss_xy = F.huber_loss(pred_xy, gt_xy, reduction='none', delta=0.05).mean(-1)
        loss[name_xy] = self.loss_gain['xy'] * loss_xy.sum() / n_gt

        # L_l1: log-space ray MAE on analytical target rays (decoupled from the
        # predicted centroid, FCN loss[2] with log_ray_loss=true).
        crop_size = float(crop_size)
        pred_xy_px = pred_xy * crop_size
        gt_centroids_px = gt_xy * crop_size
        ray_cos, ray_sin = build_ray_directions(self.n_rays, device=pred_rays.device, dtype=pred_rays.dtype)
        gt_vertices = _normed_rays_to_vertices(gt_centroids_px, gt_rays_static, crop_size, ray_cos, ray_sin)
        analytical_target = analytical_gt_rays(pred_xy_px.detach(), gt_vertices, ray_cos, ray_sin, crop_size)
        loss_l1 = _log_space_ray_loss(pred_rays, analytical_target)
        loss[name_l1] = self.loss_gain['l1'] * loss_l1.sum() / n_gt

        # L_piou: -log(polar IoU) vs STATIC GT rays, FCN loss[3].
        piou = polar_iou_torch(pred_rays, gt_rays_static)
        piou_loss = -torch.log(piou.clamp(min=1e-4))
        loss[name_piou] = self.loss_gain['piou'] * piou_loss.sum() / n_gt

        # L_smooth: 2nd-order curvature regularisation, FCN loss[4].
        smooth_loss = curvature_smoothness_loss_torch(pred_rays)
        loss[name_smooth] = self.loss_gain['smooth'] * smooth_loss.sum() / n_gt

        return {k: v.squeeze() for k, v in loss.items()}

    @staticmethod
    def _get_index(match_indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_loss(
        self,
        pred_polygons,
        pred_scores,
        gt_polygons,
        gt_cls,
        gt_groups,
        crop_size,
        postfix='',
        match_indices=None,
    ):
        bs = pred_polygons.shape[0]
        nq = pred_polygons.shape[1]
        num_gts = sum(gt_groups)

        if match_indices is None:
            match_indices = self.matcher(pred_polygons, pred_scores, gt_polygons, gt_cls, gt_groups)

        (batch_idx, src_idx), dst_idx = self._get_index(match_indices)

        # Build per-query targets: nc = "no object" for unmatched queries
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=torch.long)
        targets[batch_idx, src_idx] = gt_cls[dst_idx]

        # Build per-query quality scores (IoU) for VFL
        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)

        # Get matched predictions and GTs for regression loss
        pred_polygons_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=pred_polygons.device)
                for t, (i, _) in zip(pred_polygons, match_indices)
            ]
        )
        gt_polygons_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=pred_polygons.device)
                for t, (_, j) in zip(gt_polygons.unsqueeze(0).repeat(bs, 1, 1), match_indices)
            ]
        )

        # Compute IoU scores for VFL
        if len(pred_polygons_assigned) > 0:
            pred_rays = pred_polygons_assigned[:, 2:].float()
            gt_rays = gt_polygons_assigned[:, 2:].float()
            piou = polar_iou_torch(pred_rays, gt_rays)
            gt_scores[batch_idx, src_idx] = piou.detach()

        loss_cls = self._get_loss_class(pred_scores, targets, gt_scores, num_gts, postfix)
        loss_bbox = self._get_loss_bbox(pred_polygons_assigned, gt_polygons_assigned, crop_size, postfix)

        loss = {}
        loss.update(loss_cls)
        loss.update(loss_bbox)
        return loss

    def _get_loss_aux(
        self,
        pred_polygons,
        pred_scores,
        gt_polygons,
        gt_cls,
        gt_groups,
        crop_size,
        match_indices=None,
        postfix='',
    ):
        loss = torch.zeros(5, device=pred_polygons.device)
        if not self.per_layer_match and match_indices is None:
            # Legacy: match once on the last layer, reuse indices for all aux layers.
            match_indices = self.matcher(pred_polygons[-1], pred_scores[-1], gt_polygons, gt_cls, gt_groups)
        for aux_polygons, aux_scores in zip(pred_polygons, pred_scores):
            loss_ = self._get_loss(
                aux_polygons,
                aux_scores,
                gt_polygons,
                gt_cls,
                gt_groups,
                crop_size,
                postfix=postfix,
                # LSP-DETR "look forward twice": match each layer independently.
                match_indices=None if self.per_layer_match else match_indices,
            )
            loss[0] += loss_[f'loss_class{postfix}']
            loss[1] += loss_[f'loss_xy{postfix}']
            loss[2] += loss_[f'loss_l1{postfix}']
            loss[3] += loss_[f'loss_piou{postfix}']
            loss[4] += loss_[f'loss_smooth{postfix}']

        return {
            f'loss_class_aux{postfix}': loss[0],
            f'loss_xy_aux{postfix}': loss[1],
            f'loss_l1_aux{postfix}': loss[2],
            f'loss_piou_aux{postfix}': loss[3],
            f'loss_smooth_aux{postfix}': loss[4],
        }

    def forward(
        self,
        preds,
        targets,
        dn_polygons=None,
        dn_scores=None,
        dn_meta=None,
    ):
        """Compute total detection loss (cls + xy + ray L1 + polar IoU + smooth + aux)."""
        gt_polygons = targets['bboxes']
        gt_cls = targets['cls']
        gt_groups = targets['gt_groups']
        crop_size = float(targets.get('imgsz', 256))

        dec_polygons, dec_scores = preds
        self.device = dec_polygons.device

        loss = self._get_loss(dec_polygons[-1], dec_scores[-1], gt_polygons, gt_cls, gt_groups, crop_size)

        if self.aux_loss:
            loss_aux = self._get_loss_aux(
                dec_polygons[:-1], dec_scores[:-1], gt_polygons, gt_cls, gt_groups, crop_size
            )
            loss.update(loss_aux)

        return loss
