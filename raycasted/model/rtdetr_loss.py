"""RayCast RT-DETR Loss — Hungarian matching + focal loss + polar IoU for ray polygons.

Replaces ultralytics DETRLoss's bbox L1 + GIoU with ray L1 + xy L1 + polar IoU.
Replaces HungarianMatcher's bbox_cost + giou_cost with ray_cost + piou_cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch, polar_iou_torch
from raycasted.data.etl.utils import constants as _const


class RayCastHungarianMatcher(nn.Module):
    """Hungarian matcher using classification cost + ray L1 cost + polar IoU cost.

    Same structure as ultralytics HungarianMatcher but replaces bbox L1 and GIoU
    costs with ray L1 and polar IoU costs for star-convex polygons.
    """

    def __init__(
        self,
        cost_gain=None,
        use_fl=True,
        alpha=0.25,
        gamma=2.0,
        n_rays=None,
    ):
        super().__init__()
        if cost_gain is None:
            cost_gain = {'class': 2, 'ray': 5, 'piou': 2}
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
        """Compute Hungarian matching between predictions and GT."""
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_polygons = pred_polygons.detach().view(-1, pred_polygons.shape[-1])

        pred_scores = pred_scores[:, gt_cls]

        if self.use_fl:
            neg_cost = (1 - self.alpha) * (pred_scores**self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost - neg_cost
        else:
            cost_class = -pred_scores

        pred_rays = pred_polygons[:, 2:]
        gt_rays = gt_polygons[:, 2:]
        cost_ray = (pred_rays.unsqueeze(1) - gt_rays.unsqueeze(0)).abs().sum(-1)

        pred_xy = pred_polygons[:, :2]
        gt_xy = gt_polygons[:, :2]
        cost_xy = (pred_xy.unsqueeze(1) - gt_xy.unsqueeze(0)).abs().sum(-1)

        n_pred = pred_rays.shape[0]
        n_gt = gt_rays.shape[0]
        if n_pred > 0 and n_gt > 0:
            pred_exp = pred_rays.unsqueeze(1).expand(n_pred, n_gt, -1)
            gt_exp = gt_rays.unsqueeze(0).expand(n_pred, n_gt, -1)
            piou = polar_iou_pairwise_flat_torch(pred_exp, gt_exp)
            cost_piou = 1.0 - piou
        else:
            cost_piou = torch.zeros(n_pred, n_gt, device=pred_polygons.device)

        cost = (
            self.cost_gain['class'] * cost_class
            + self.cost_gain['ray'] * (cost_ray + cost_xy)
            + self.cost_gain['piou'] * cost_piou
        )

        cost[cost.isnan() | cost.isinf()] = 0.0
        cost = cost.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(gt_groups, -1))]
        gt_groups_t = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups_t[k])
            for k, (i, j) in enumerate(indices)
        ]


class RayCastRTDETRDetectionLoss(nn.Module):
    """Detection loss for RayCast RT-DETR: focal/VFL cls + ray L1 + xy L1 + polar IoU.

    Mirrors ultralytics DETRLoss structure but operates on ray polygons instead of
    bounding boxes. Supports auxiliary losses from intermediate decoder layers.
    """

    def __init__(
        self,
        nc=80,
        loss_gain=None,
        aux_loss=True,
        use_vfl=True,
        n_rays=None,
    ):
        super().__init__()
        if loss_gain is None:
            loss_gain = {'class': 1, 'ray': 5, 'piou': 2, 'no_object': 0.1}
        self.nc = nc
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.n_rays = n_rays or _const.N_RAYS
        self.matcher = RayCastHungarianMatcher(
            cost_gain={'class': 2, 'ray': 5, 'piou': 2},
            n_rays=self.n_rays,
        )
        from ultralytics.utils.loss import FocalLoss, VarifocalLoss

        self.fl = FocalLoss(2.0, 0.25)
        self.vfl = VarifocalLoss(2.0, 0.75) if use_vfl else None
        self.device = torch.device('cpu')

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        bs, nq = pred_scores.shape[:2]
        name = f'loss_class{postfix}'
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if num_gts and self.vfl:
            loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
        else:
            loss_cls = self.fl(pred_scores, one_hot.float())
        loss_cls /= max(num_gts, 1) / nq

        return {name: loss_cls.squeeze() * self.loss_gain['class']}

    def _get_loss_polygon(self, pred_polygons, gt_polygons, postfix=''):
        name_ray = f'loss_ray{postfix}'
        name_piou = f'loss_piou{postfix}'

        loss = {}
        if len(gt_polygons) == 0:
            loss[name_ray] = torch.tensor(0.0, device=pred_polygons.device)
            loss[name_piou] = torch.tensor(0.0, device=pred_polygons.device)
            return loss

        pred_rays = pred_polygons[:, 2:]
        gt_rays = gt_polygons[:, 2:]
        pred_xy = pred_polygons[:, :2]
        gt_xy = gt_polygons[:, :2]

        ray_l1 = F.l1_loss(pred_rays, gt_rays, reduction='sum') + F.l1_loss(pred_xy, gt_xy, reduction='sum')
        loss[name_ray] = self.loss_gain['ray'] * ray_l1 / len(gt_polygons)

        piou = polar_iou_torch(pred_rays.float(), gt_rays.float())
        piou_loss = (1.0 - piou).sum() / len(gt_polygons)
        loss[name_piou] = self.loss_gain['piou'] * piou_loss

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
        loss_poly = self._get_loss_polygon(pred_polygons_assigned, gt_polygons_assigned, postfix)

        loss = {}
        loss.update(loss_cls)
        loss.update(loss_poly)
        return loss

    def _get_loss_aux(
        self,
        pred_polygons,
        pred_scores,
        gt_polygons,
        gt_cls,
        gt_groups,
        match_indices=None,
        postfix='',
    ):
        loss = torch.zeros(3, device=pred_polygons.device)
        if match_indices is None:
            match_indices = self.matcher(pred_polygons[-1], pred_scores[-1], gt_polygons, gt_cls, gt_groups)
        for aux_polygons, aux_scores in zip(pred_polygons, pred_scores):
            loss_ = self._get_loss(
                aux_polygons, aux_scores, gt_polygons, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
            )
            loss[0] += loss_[f'loss_class{postfix}']
            loss[1] += loss_[f'loss_ray{postfix}']
            loss[2] += loss_[f'loss_piou{postfix}']

        return {
            f'loss_class_aux{postfix}': loss[0],
            f'loss_ray_aux{postfix}': loss[1],
            f'loss_piou_aux{postfix}': loss[2],
        }

    def forward(
        self,
        preds,
        targets,
        dn_polygons=None,
        dn_scores=None,
        dn_meta=None,
    ):
        """Compute total detection loss (cls + ray L1 + polar IoU + aux)."""
        gt_polygons = targets['bboxes']
        gt_cls = targets['cls']
        gt_groups = targets['gt_groups']

        dec_polygons, dec_scores = preds
        self.device = dec_polygons.device

        loss = self._get_loss(dec_polygons[-1], dec_scores[-1], gt_polygons, gt_cls, gt_groups)

        if self.aux_loss:
            loss_aux = self._get_loss_aux(dec_polygons[:-1], dec_scores[:-1], gt_polygons, gt_cls, gt_groups)
            loss.update(loss_aux)

        return loss
