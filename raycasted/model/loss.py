"""RayCastED — RayCast Detection Loss (Phase 6).

5-term polygon loss:
  L = λ_xy × L_xy + λ_cls × L_cls + λ_L1 × L_L1 + λ_piou × L_PolarIoU + λ_smooth × L_smooth

RayCastDetectionLoss subclasses v8DetectionLoss, replacing bbox/DFL logic
with polygon regression terms.
RayCastE2ELoss subclasses E2ELoss, wiring RayCastDetectionLoss
into both one2many/one2one branches with smoothness annealing.

Classification loss: BCE only. Focal loss and QFL were tested and found
to be severely harmful for polygon detection (precision collapse, F1 drop
from 0.73 to 0.41). See ablation runs with Run 27 architecture.
"""

from functools import partial

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from raycasted.data.etl.ops.iou import polar_iou_torch
from raycasted.data.etl.ops.loss import angular_smoothness_loss_torch
from raycasted.model.tal import HungarianRayCastAssigner, RayCastAssigner


def _log_space_ray_loss(pred_rays: torch.Tensor, target_rays: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """L1 loss in log-space for ray distances.

    Matches LSP-DETR's observation that linear L1 gives equal gradient for
    absolute errors regardless of ray magnitude, causing small objects (rays
    ~0.01-0.05 normalised) to receive disproportionately weak supervision.
    Log-space converts absolute errors into relative ones: a 10% error on a
    small ray produces the same loss as a 10% error on a large ray.

    Both pred and target are clamped to [eps, ∞) before log to avoid log(0).

    Args:
        pred_rays: [N, n_rays] predicted ray distances (strictly positive via softplus).
        target_rays: [N, n_rays] GT ray distances (normalised by crop_size, in [0, ~1]).
        eps: Minimum clamp value to prevent log(0).

    Returns:
        [N] per-sample mean absolute error in log-space.
    """
    log_pred = torch.clamp(pred_rays, min=eps).log()
    log_tgt = torch.clamp(target_rays, min=eps).log()
    return (log_pred - log_tgt).abs().mean(-1)


class RayCastDetectionLoss(v8DetectionLoss):
    """Polygon detection loss with 5 terms.

    Replaces v8DetectionLoss bbox/DFL terms:
      L_cls     — BCE on classification scores
      L_xy      — Huber on decoded centroid
      L_L1      — Log-space L1 on ray distances
      L_PolarIoU — -log(PolarIoU)
      L_smooth  — Angular smoothness regularisation (annealed)

    Inherits assignment framework from v8DetectionLoss, replacing
    the assigner with RayCastAssigner.
    """

    def __init__(
        self,
        model,
        tal_topk: int = 13,
        tal_topk2: int | None = None,
        assigner_radius_scale: float = 1.5,
        assigner_alpha: float = 0.5,
        assigner_beta: float = 6.0,
        log_ray_loss: bool = False,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        m = model.model[-1]
        self.raycast_dim = m.raycast_dim  # 2 + n_rays
        self.no = m.nc + self.raycast_dim  # BUG-02 fix (parent sets nc + reg_max*4)
        self.use_dfl = False  # DFL not applicable to polygon regression

        # Log-space ray loss configuration
        self.log_ray_loss = log_ray_loss

        # Assigner swap — use tal_topk directly for E2E compatibility
        self.assigner = RayCastAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=assigner_alpha,
            beta=assigner_beta,
            stride=self.stride.tolist(),
            topk2=tal_topk2,  # Pass through topk2 for E2E one2one branch
            radius_scale=assigner_radius_scale,
        )

        # Loss weights (rebalanced so xy and L1 share gradient signal equally)
        # xy and L1 both produce tiny raw values (Huber on normalised coords, log-space rays),
        # so both need high lambda to contribute meaningfully.
        # smooth starts at 0, reverse-anneals to peak over 40% of training (shape prior).
        self.lambda_cls = 2.0    # classification (23-28% of task gradient)
        self.lambda_xy = 15.0    # centroid (45-50% of task gradient)
        self.lambda_l1 = 25.0    # ray accuracy (25-30% of task gradient)
        self.lambda_piou = 2.0   # shape IoU (1-2% of task gradient)
        self.lambda_smooth = 0.0  # reverse-annealed by RayCastE2ELoss (0 → 1.0 over 40% of training)

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

        # --- L_cls: BCE only (focal/QFL tested and found harmful) ---
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

            # L_L1: Uniform MAE on 32 rays (linear or log-space)
            if self.log_ray_loss:
                loss_l1 = _log_space_ray_loss(fg_pred_rays, fg_target_rays)
            else:
                loss_l1 = (fg_pred_rays - fg_target_rays).abs().mean(-1)
            loss[2] = (loss_l1.unsqueeze(-1) * weight).sum() / target_scores_sum

            # L_PolarIoU: -log(PolarIoU) — matches PolarMask formulation
            # Logarithmic loss more strongly penalizes low-IoU predictions than linear (1 - IoU)
            fg_piou = polar_iou_torch(fg_pred_rays, fg_target_rays)  # [N_fg] (already float32)
            loss_piou = -torch.log(fg_piou + 1e-7)
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

    def __init__(
        self,
        model,
        max_epochs: int = 200,
        tal_topk: int = 13,
        assigner_radius_scale: float = 1.5,
        assigner_alpha: float = 0.5,
        assigner_beta: float = 6.0,
        use_hungarian_o2o: bool = True,
        log_ray_loss: bool = False,
        centroid_sigma: float = 0.05,
    ):
        # Bind training config to RayCastDetectionLoss so E2ELoss passes it through
        loss_fn = partial(
            RayCastDetectionLoss,
            tal_topk=tal_topk,
            assigner_radius_scale=assigner_radius_scale,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            log_ray_loss=log_ray_loss,
        )
        super().__init__(model, loss_fn=loss_fn)

        # Fix: parent E2ELoss reads one2one.hyp.epochs for the decay schedule,
        # but RayCastDetectionLoss (mock-based init) doesn't set hyp correctly.
        # Force both branches to use the actual max_epochs.
        for branch in (self.one2many, self.one2one):
            if not hasattr(branch, 'hyp') or branch.hyp is None:
                from ultralytics.cfg import get_cfg

                branch.hyp = get_cfg()
            branch.hyp.epochs = max_epochs

        # CRITICAL: Override parent's hardcoded tal_topk values with our custom values
        # Parent E2ELoss hardcodes tal_topk=10 for one2many and tal_topk=7 for one2one.
        #
        # For one2many: topk=tal_topk, topk2=tal_topk (no secondary filtering)
        # For one2one: topk=tal_topk//2, topk2=1 (select k/2 candidates, filter to 1)
        #
        # Ultralytics' NMS-free mechanism works in two stages:
        #   1. select_topk_candidates picks `topk` anchors per GT
        #   2. select_highest_overlaps checks `topk2 != topk` → keeps only `topk2` best
        # Setting one2one.topk=1 directly SKIPS the candidate pool — the assigner has
        # no choice, so it can't pick the best anchor. Using topk=7,topk2=1 (like stock
        # Ultralytics) gives the assigner 7 candidates and picks the single best overlap.
        self.one2many.assigner.topk = tal_topk
        self.one2many.assigner.topk2 = tal_topk  # no secondary filtering

        if use_hungarian_o2o:
            # Replace the one2one assigner with Hungarian matching for globally
            # optimal 1:1 assignment — critical for dense touching-cell scenes
            # where greedy TAL causes assignment collisions.
            self.one2one.assigner = HungarianRayCastAssigner(
                topk=1,
                num_classes=self.one2one.assigner.num_classes,
                alpha=assigner_alpha,
                beta=assigner_beta,
                stride=self.one2one.assigner.stride if hasattr(self.one2one.assigner, 'stride') else [8, 16, 32],
                topk2=1,
                radius_scale=assigner_radius_scale,
                centroid_sigma=centroid_sigma,
            )
        else:
            one2one_pool = max(tal_topk // 2, 7)  # candidate pool for one2one
            self.one2one.assigner.topk = one2one_pool
            self.one2one.assigner.topk2 = 1  # NMS-free: keep only 1 anchor per GT

        # Validate E2E architecture integrity
        if use_hungarian_o2o:
            assert isinstance(self.one2one.assigner, HungarianRayCastAssigner), (
                'E2E violation: one2one assigner must be HungarianRayCastAssigner'
            )
        else:
            assert self.one2one.assigner.topk2 == 1, (
                f'E2E violation: one2one.topk2={self.one2one.assigner.topk2}, must be 1 for NMS-free'
            )
        assert self.one2many.assigner.topk2 == self.one2many.assigner.topk, (
            f'E2E violation: one2many.topk2 ({self.one2many.assigner.topk2}) != topk ({self.one2many.assigner.topk})'
        )
        # Smooth loss: reverse anneal — starts at 0, ramps up to peak, then holds.
        # Early training: model focuses on detection (xy, cls, L1).
        # After ramp: smoothness pressure helps refine polygon boundaries.
        self.smooth_start = 0.0       # initial value (no smoothness pressure)
        self.smooth_end = 1.0         # peak value (meaningful shape prior)
        self.smooth_anneal_fraction = 0.4  # ramp over first 40% of training
        self.smooth_anneal_epochs = max(1, int(max_epochs * self.smooth_anneal_fraction))

    def update(self):
        """Update o2m/o2o weights (inherited) + anneal smoothness + validate E2E integrity."""
        super().update()

        # Validate E2E integrity on first update (catches config drift)
        if self.updates == 1:
            is_hungarian = isinstance(self.one2one.assigner, HungarianRayCastAssigner)
            if is_hungarian:
                print(
                    f'✓ E2E NMS-free: o2m.topk={self.one2many.assigner.topk}, '
                    f'o2o=Hungarian (globally optimal 1:1 matching)'
                )
            else:
                assert self.one2one.assigner.topk2 == 1, (
                    f'E2E violation: one2one.topk2={self.one2one.assigner.topk2}, must be 1 for NMS-free'
                )
                print(
                    f'✓ E2E NMS-free: o2m.topk={self.one2many.assigner.topk}, '
                    f'o2o.topk={self.one2one.assigner.topk}, o2o.topk2=1'
                )
            assert self.one2many.assigner.topk == self.one2many.assigner.topk2, (
                f'E2E violation: one2many topk={self.one2many.assigner.topk} != topk2={self.one2many.assigner.topk2}'
            )

        # Reverse anneal: ramp from smooth_start → smooth_end over smooth_anneal_epochs
        t = min(self.updates / self.smooth_anneal_epochs, 1.0)
        new_lambda = self.smooth_start + t * (self.smooth_end - self.smooth_start)
        self.one2many.lambda_smooth = new_lambda
        self.one2one.lambda_smooth = new_lambda
