"""RayCastED — RayCast Detection Loss (Phase 6).

5-term polygon loss:
  L = λ_cls × L_cls + λ_xy × L_xy + λ_L1 × L_L1 + λ_piou × L_PolarIoU + λ_smooth × L_smooth

RayCastDetectionLoss subclasses v8DetectionLoss, replacing bbox/DFL logic
with polygon regression terms.
RayCastE2ELoss subclasses E2ELoss, wiring RayCastDetectionLoss
into both one2many/one2one branches with smoothness annealing.
"""

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from raycasted.data.etl.ops.iou import polar_iou_torch
from raycasted.data.etl.ops.loss import angular_smoothness_loss_torch
from raycasted.model.tal import RayCastAssigner


class RayCastDetectionLoss(v8DetectionLoss):
    """Polygon detection loss with 5 terms.

    Replaces v8DetectionLoss bbox/DFL terms:
      L_cls     — BCE on classification scores
      L_xy      — Huber(δ=0.01) on decoded centroid
      L_L1      — Uniform MAE on 32 rays
      L_PolarIoU — 1 - PolarIoU
      L_smooth  — Angular smoothness regularisation (annealed)

    Inherits assignment framework from v8DetectionLoss, replacing
    the assigner with RayCastAssigner.
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        m = model.model[-1]
        self.raycast_dim = m.raycast_dim  # 2 + n_rays
        self.no = m.nc + self.raycast_dim  # BUG-02 fix (parent sets nc + reg_max*4)
        self.use_dfl = False  # DFL not applicable to polygon regression

        # Assigner swap — read params from the TaskAlignedAssigner super() just created
        self.assigner = RayCastAssigner(
            topk=self.assigner.topk,
            num_classes=self.nc,
            alpha=self.assigner.alpha,
            beta=self.assigner.beta,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
            radius_scale=1.5,
        )

        # Loss weights
        self.lambda_cls = 0.5
        self.lambda_xy = 50.0  # Huber(delta=0.01) on normalised coords produces tiny
        # gradients; high lambda ensures the optimizer sees centroid errors.
        self.lambda_l1 = 1.0
        self.lambda_piou = 2.0
        self.lambda_smooth = 0.05  # annealed by RayCastE2ELoss

    def preprocess(self, targets, batch_size, scale_tensor=None):
        """Preprocess polygon targets.

        Accepts (N, 4+n_rays) collated batch targets:
            [batch_idx, class_id, cx_norm, cy_norm, d_1_norm..d_n_norm]
        Returns [B, N_gt_max, 3+n_rays]:
            [class_id, cx_norm, cy_norm, d_1_norm..d_n_norm]

        No xywh2xyxy conversion, no scaling — all spatial quantities
        arrive normalised from the DataLoader.
        """
        nl, ne = targets.shape  # ne = 36
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]  # drop batch_idx
        # NO xywh2xyxy, NO scaling — data is already normalised
        return out

    @staticmethod
    def decode_pred_xy(xy_raw, anchor_points, stride_tensor, imgsz):
        """Decode grid-cell-relative Sigmoid offsets to absolute normalised coordinates.

        Must match the inference decode in RayCastDetect._inference:
            xy_abs = (xy_offset * 2.0 - 0.5 + anchors) * strides

        Args:
            xy_raw:        [B, N_anchors, 2] — raw head output (pre-sigmoid)
            anchor_points: [N_anchors, 2] — grid cell positions (from make_anchors)
            stride_tensor: [N_anchors, 1] — per-anchor stride
            imgsz:         [H, W] — image size in pixels

        Returns:
            xy_norm: [B, N_anchors, 2] — absolute normalised coords
        """
        xy_offset = xy_raw.sigmoid()  # cell-relative offset in [0, 1]
        xy_pixel = (xy_offset * 2.0 - 0.5 + anchor_points) * stride_tensor  # pixel space
        return xy_pixel / imgsz[[1, 0]]  # normalise

    def get_assigned_targets_and_loss(self, preds, batch):
        """Compute 5-term polygon loss.

        Returns:
            (assignment_info, loss_5vec, loss_detach)
        """
        loss = torch.zeros(5, device=self.device)  # [xy, cls, L1, piou, smooth]

        # --- Prediction parsing ---
        pred_distri = preds['boxes'].permute(0, 2, 1).contiguous()  # [B, N, raycast_dim]
        pred_scores = preds['scores'].permute(0, 2, 1).contiguous()  # [B, N, nc]
        anchor_points, stride_tensor = make_anchors(preds['feats'], self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds['feats'][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # --- Targets ---
        targets = torch.cat(
            (batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1
        )  # [N, 4+n_rays]
        targets = self.preprocess(targets.to(self.device), batch_size)
        gt_labels, gt_bboxes = targets.split((1, self.raycast_dim), 2)  # cls:(B,N,1), poly:(B,N,raycast_dim)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # --- Decode predictions ---
        xy_raw = pred_distri[..., :2]  # [B, N, 2]
        rays_raw = pred_distri[..., 2:]  # [B, N, n_rays]
        pred_xy = self.decode_pred_xy(xy_raw, anchor_points, stride_tensor, imgsz)
        pred_rays = F.softplus(rays_raw)  # [B, N, n_rays]
        pred_poly = torch.cat([pred_xy, pred_rays], dim=-1)  # [B, N, raycast_dim]

        # --- Normalise anchor points to [0, 1] to match GT space ---
        anchor_points_norm = anchor_points * stride_tensor / imgsz[[1, 0]]

        # --- Assignment ---
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_poly.detach(),  # already normalised
            anchor_points_norm,  # normalised to match GT
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # --- L_cls (term 1) ---
        # Cast to float32 for numerical stability under AMP/FP16 validation
        loss[1] = self.bce(pred_scores.float(), target_scores.float()).sum() / target_scores_sum

        # --- Polygon regression losses (foreground only) ---
        if fg_mask.sum():
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # [N_fg, 1]

            fg_pred_xy = pred_xy[fg_mask]  # [N_fg, 2]
            fg_pred_rays = pred_rays[fg_mask]  # [N_fg, n_rays]
            fg_target_xy = target_bboxes[fg_mask][:, :2]  # [N_fg, 2]
            fg_target_rays = target_bboxes[fg_mask][:, 2:]  # [N_fg, n_rays]

            # Cast to float32 for numerical stability under AMP/FP16 validation
            fg_pred_xy = fg_pred_xy.float()
            fg_target_xy = fg_target_xy.float()
            fg_pred_rays = fg_pred_rays.float()
            fg_target_rays = fg_target_rays.float()
            weight = weight.float()

            # L_xy: Huber on decoded centroid
            loss_xy = F.huber_loss(fg_pred_xy, fg_target_xy, reduction='none', delta=1.0).mean(-1)
            loss[0] = (loss_xy.unsqueeze(-1) * weight).sum() / target_scores_sum

            # L_L1: Uniform MAE on 32 rays
            loss_l1 = (fg_pred_rays - fg_target_rays).abs().mean(-1)
            loss[2] = (loss_l1.unsqueeze(-1) * weight).sum() / target_scores_sum

            # L_PolarIoU: 1 - PolarIoU
            fg_piou = polar_iou_torch(fg_pred_rays, fg_target_rays)  # [N_fg] (already float32)
            loss_piou = 1.0 - fg_piou
            loss[3] = (loss_piou * weight.squeeze(-1)).sum() / target_scores_sum

            # L_smooth: Angular smoothness on predicted rays
            fg_smooth = angular_smoothness_loss_torch(fg_pred_rays)  # [N_fg] (already float32)
            loss[4] = (fg_smooth * weight.squeeze(-1)).sum() / target_scores_sum
        else:
            # DDP safety — touch all prediction tensors to avoid unused-gradient errors
            loss[0] += (pred_xy * 0).sum()
            loss[2] += (pred_rays * 0).sum()
            loss[3] += (pred_rays * 0).sum()
            loss[4] += (pred_rays * 0).sum()

        # --- Apply loss weights ---
        loss[0] *= self.lambda_xy
        loss[1] *= self.lambda_cls
        loss[2] *= self.lambda_l1
        loss[3] *= self.lambda_piou
        loss[4] *= self.lambda_smooth

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class RayCastE2ELoss(E2ELoss):
    """E2E dual-assignment loss with smoothness annealing.

    Wires RayCastDetectionLoss into both one2many and one2one branches.
    Smoothness lambda anneals from smooth_start to smooth_end over
    smooth_anneal_epochs. Inherits o2m/o2o weight decay from parent.
    """

    def __init__(self, model):
        super().__init__(model, loss_fn=RayCastDetectionLoss)
        self.smooth_start = 0.05
        self.smooth_end = 0.0
        self.smooth_anneal_epochs = 50

    def update(self):
        """Update o2m/o2o weights (inherited) + anneal smoothness lambda."""
        super().update()
        delta = (self.smooth_start - self.smooth_end) / self.smooth_anneal_epochs
        new_lambda = max(self.smooth_end, self.smooth_start - delta * self.updates)
        self.one2many.lambda_smooth = new_lambda
        self.one2one.lambda_smooth = new_lambda
