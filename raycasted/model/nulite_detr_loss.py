"""NuLite DETR Loss — Hungarian 1:1 matching + focal/VFL cls + ray L1 + polar IoU.

Owned by the NuLite DETR head only. Replaces ultralytics DETRLoss's bbox L1 +
GIoU with ray L1 + xy L1 + polar IoU, and HungarianMatcher's bbox/giou costs
with ray/piou costs. Pure 1:1 assignment (no one-to-many branch, no NMS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch, polar_iou_torch
from raycasted.data.etl.utils import constants as _const


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

            cost = (
                self.cost_gain['class'] * cost_class
                + self.cost_gain['ray'] * (cost_ray + cost_xy)
                + self.cost_gain['piou'] * cost_piou
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
    ):
        super().__init__()
        if loss_gain is None:
            loss_gain = {'class': 1, 'ray': 5, 'piou': 2, 'no_object': 1.0}
        self.nc = nc
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.no_object = no_object
        self.n_rays = n_rays or _const.N_RAYS
        self.matcher = RayCastHungarianMatcher(
            cost_gain={'class': 2, 'ray': 5, 'piou': 2},
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
