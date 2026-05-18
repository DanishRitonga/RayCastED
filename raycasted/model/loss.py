"""RayCastED — RayCast Detection Loss (Phase 6).

4-term polygon loss:
  L = λ_xy × L_xy + λ_cls × L_cls + λ_L1 × L_L1 + λ_smooth × L_smooth

With GradNorm enabled, the static λ values are replaced by dynamic weights
that equalise gradient norms across all 4 tasks, preventing cls_loss from
dominating the shared backbone gradients.

RayCastDetectionLoss subclasses v8DetectionLoss, replacing bbox/DFL logic
with polygon regression terms.
RayCastE2ELoss subclasses E2ELoss, wiring RayCastDetectionLoss
into both one2many/one2one branches with smoothness annealing.

Classification loss: BCE (default) or Focal loss. Focal loss down-weights
easy negatives and amplifies hard positives, useful for dense cell scenes.
Enabled via focal_gamma > 0.
"""

from __future__ import annotations

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from raycasted.data.etl.ops.iou import polar_iou_torch
from raycasted.data.etl.ops.loss import curvature_smoothness_loss_torch
from raycasted.model.tal import RayCastAssigner

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  GradNorm — Gradient Normalisation for Multi-Task Loss Balancing
# ──────────────────────────────────────────────────────────────────────

_TASK_NAMES = ('xy', 'cls', 'l1', 'piou', 'smooth')


class GradNormManager:
    """Per-task gradient-norm equalisation (Chen et al., 2018).

    Uses the loss-ratio approximation (r_i = L_i(t) / L_i(0)) to estimate
    per-task gradient norms without requiring per-task backward passes.
    After every training step call :meth:`update` to re-balance the loss
    weights so that all tasks contribute equal gradient magnitude to the
    shared backbone.

    The algorithm targets:

        G_i(t) = Ḡ(t) × [ r_i(t) ]^α

    where r_i = L_i(t) / L_i(0) is the inverse training rate and α
    controls how aggressively slow learners are boosted (0 = uniform,
    1 = full inverse-rate scaling).

    Parameters
    ----------
    model : nn.Module
        The full detection model (``model.model`` is the nn.Sequential).
    num_tasks : int
        Number of scalar loss terms (default 5).
    alpha : float
        GradNorm restoring force.  Higher → more aggressive rebalancing.
        Recommended: 0.5 for well-behaved losses, up to 1.0 for extreme
        imbalance.
    initial_weights : list[float] | None
        Starting λ values.  ``None`` uses ``[1.0] * num_tasks``.
    warmup_epochs : int
        Number of epochs before GradNorm activates.  During warmup,
        static weights from ``initial_weights`` are used.
    """

    def __init__(
        self,
        model: nn.Module,
        num_tasks: int = 5,
        alpha: float = 0.5,
        initial_weights: list[float] | None = None,
        warmup_epochs: int = 5,
    ):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.task_names = _TASK_NAMES[:num_tasks]

        # ── learnable log-weights ──
        init = initial_weights or [1.0] * num_tasks
        self.log_weights = nn.Parameter(
            torch.tensor(init, dtype=torch.float32).log(),
            requires_grad=True,
        )

        # ── tracking state ──
        self.initial_losses: torch.Tensor | None = None
        self._current_losses: torch.Tensor | None = None
        self._step_count = 0
        self._warmup_epochs = warmup_epochs

    # ── dynamic enabled (warmup gate) ──────────────────────────────

    @property
    def enabled(self) -> bool:
        """GradNorm is active only after warmup completes."""
        return self._step_count >= self._warmup_epochs

    # ── forward-pass loss storage ──────────────────────────────────

    def store_losses(self, losses: torch.Tensor) -> None:
        """Store per-task losses from the current forward pass.

        Called from RayCastDetectionLoss.__call__ BEFORE weight
        application so that update() can compute loss ratios.

        On the first call after warmup, these become the baseline
        L_i(0) for the inverse training rate r_i = L_i(t) / L_i(0).

        Args:
            losses: [5] tensor of unweighted per-task losses.
        """
        self._current_losses = losses.detach().clone()
        if self.initial_losses is None and self.enabled:
            self.initial_losses = self._current_losses.clone()
            logger.info('GradNorm: initial losses set — %s', self._current_losses.tolist())

    # ── weight computation ─────────────────────────────────────────

    def get_weights(self) -> torch.Tensor:
        """Return unnormalised dynamic weights (clamped to [0.1, 10.0]).

        Unlike softmax-normalised GradNorm, these weights are NOT
        constrained to sum to num_tasks.  This allows slow learners
        (e.g. cls) to receive unconstrained weight growth without
        starving fast learners of gradient signal.
        """
        return self.log_weights.exp().clamp(0.1, 10.0)

    # ── GradNorm update (called from on_train_batch_end callback) ─

    def update(self) -> None:
        """Update log-weights using the loss-ratio method.

        Uses r_i = L_i(t) / L_i(0) as a proxy for per-task gradient
        norms (Chen et al., 2018 §4.2).  Computes target gradient
        norms G_i* = Ḡ × r_i^α and nudges weights so that each
        task contributes equal gradient magnitude.
        """
        if not self.enabled:
            self._step_count += 1
            return
        if self._current_losses is None or self.initial_losses is None:
            return

        with torch.no_grad():
            w = self.log_weights.exp()
            r = self._current_losses / (self.initial_losses + 1e-8)
            r = r.clamp(min=0.01, max=100.0)
            r_avg = r.mean()
            r_rel = r / (r_avg + 1e-8)

            # Target: tasks that learned slowly (high r_rel) get boosted
            grad_w = 1.0 - (r_rel**self.alpha)
            # Decay toward 1.0 to prevent unbounded growth (no normalization)
            grad_w = grad_w + 0.02 * (1.0 - w)
            grad_w = grad_w - grad_w.mean()  # zero-centre

            lr = 0.025
            new_log_w = self.log_weights - lr * grad_w.to(self.log_weights.device)
            self.log_weights.data = 0.9 * self.log_weights.data + 0.1 * new_log_w
            self.log_weights.data.clamp_(-2.3, 2.3)  # e^-2.3≈0.1 to e^2.3≈10.

        self._step_count += 1

    # ── initial loss setter (optional, for resume) ─────────────────

    def set_initial_losses(self, losses: torch.Tensor) -> None:
        """Override initial losses (e.g. on resume from checkpoint)."""
        self.initial_losses = losses.detach().clone()

    # ── logging ────────────────────────────────────────────────────

    def log_state(self) -> dict[str, float]:
        """Return a dict of current state for logging."""
        w = self.get_weights().detach().cpu()
        return {f'gradnorm/{name}': w[i].item() for i, name in enumerate(self.task_names)}


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


def _focal_loss(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard focal loss (Lin et al., 2017) for multi-label classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1, (1-p) if y=0; alpha_t = alpha if y=1, (1-alpha) if y=0.

    Down-weights easy negatives (high p_t) and amplifies hard positives (low p_t),
    critical for dense cell scenes where background anchors vastly outnumber positives.

    NOTE: Previous default alpha=1.0 killed ALL background gradients (bg weight = 0).
    The standard value from Lin et al. 2017 is alpha=0.25 (fg=0.25, bg=0.75).

    Args:
        pred_scores: [B, N, C] raw logits from the detection head.
        target_scores: [B, N, C] classification targets. When ``soft_targets=True``,
            these are quality-weighted alignment scores in [0, 1]. Otherwise binary 0/1.
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Positive sample weight factor. Standard: 0.25 (fg=0.25, bg=0.75).
        class_weights: [C] per-class weight applied to positive samples. When provided,
            each class's positive loss is scaled by its weight — use inverse-frequency
            weights to rebalance rare classes.

    Returns:
        [B, N, C] element-wise focal loss (no reduction).
    """
    # Use logits API for numerical stability (avoids separate sigmoid)
    ce = F.binary_cross_entropy_with_logits(pred_scores.float(), target_scores.float(), reduction='none')
    pred_prob = pred_scores.float().sigmoid()
    p_t = target_scores * pred_prob + (1 - target_scores) * (1 - pred_prob)
    modulating_factor = (1.0 - p_t) ** gamma
    alpha_factor = target_scores * alpha + (1 - target_scores) * (1 - alpha)

    loss = modulating_factor * alpha_factor * ce

    # Per-class weighting for positive samples — rebalances rare classes
    if class_weights is not None:
        # class_weights shape [C] → broadcast over [B, N, C]
        # Scale only where target > 0 (positive class); bg stays as-is
        pos_mask = (target_scores > 0).float()
        loss = loss * (1.0 + pos_mask * (class_weights.unsqueeze(0).unsqueeze(0) - 1.0))

    return loss


class RayCastDetectionLoss(v8DetectionLoss):
    """Polygon detection loss with 5 terms.

    Replaces v8DetectionLoss bbox/DFL terms:
      L_cls      — Focal/BCE on classification scores
      L_xy       — Huber on decoded centroid
      L_L1       — Log-space L1 on ray distances
      L_PolarIoU — -log(PolarIoU) shape-quality loss
      L_smooth   — Angular smoothness regularisation (annealed)

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
        align_threshold: float = 0.0,
        log_ray_loss: bool = False,
        gradnorm_manager: GradNormManager | None = None,
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.25,
        bg_fg_ratio: int = 3,
        plb_enabled: bool = False,
        bg_cls_decay: float = 1.0,
        fg_cls_boost: float = 0.0,
        soft_targets: bool = False,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        m = model.model[-1]
        self.raycast_dim = m.raycast_dim  # 2 + n_rays
        self.no = m.nc + self.raycast_dim  # BUG-02 fix (parent sets nc + reg_max*4)
        self.use_dfl = False  # DFL not applicable to polygon regression

        # Log-space ray loss configuration
        self.log_ray_loss = log_ray_loss

        # GradNorm integration — dynamic loss weight balancing
        self.gradnorm_manager = gradnorm_manager

        # Focal loss configuration (gamma=0 disables focal, uses pure BCE)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Pixel-Level Balancing — area-based fg weighting to boost small nuclei
        self.plb_enabled = plb_enabled

        # Classification loss weighting:
        #   bg_cls_decay: downweight bg anchor cls gradient (e.g., 0.5)
        #   fg_cls_boost: amplify fg by alignment quality (0.0 = disabled)
        #     weight = bg_cls_decay for bg, (1.0 + fg_cls_boost * quality) for fg
        self.bg_cls_decay = bg_cls_decay
        self.fg_cls_boost = fg_cls_boost

        # Soft targets — use assigner's quality-weighted alignment scores directly
        # instead of hard binarising to 0/1. Gives the classifier a graded signal
        # that distinguishes strong matches from weak ones.
        self.soft_targets = soft_targets

        # Per-class inverse-frequency weights [C]. When provided, scales positive
        # classification loss per class to rebalance rare categories.
        # Stored as plain attribute (not register_buffer — parent is not nn.Module).
        self.class_weights: torch.Tensor | None = class_weights

        # Assigner swap — use tal_topk directly for E2E compatibility
        self.assigner = RayCastAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=assigner_alpha,
            beta=assigner_beta,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
            radius_scale=assigner_radius_scale,
            align_threshold=align_threshold,
        )

        # Loss weights — decoded normalised xy space [0,1].
        # High lambda_xy compensates for stride/imgsz gradient attenuation (~0.03x).
        # Raw ~0.088; lambda=500 gives weighted~44, effective grad_mult~7.8.
        # Rebalanced L1/piou (total=27 preserved): more weight to shape-quality loss
        # so PolarIoU drives polygon shape rather than per-ray pixel distance alone.
        self.lambda_cls = 2.0
        self.lambda_xy = 500.0
        self.lambda_l1 = 14.0
        self.lambda_piou = 13.0
        self.lambda_smooth = 0.0  # reverse-annealed by RayCastE2ELoss
        self.bg_fg_ratio = bg_fg_ratio

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

        # --- Pixel-Level Balancing: area-based fg weight (boost small nuclei) ---
        plb_weights = None
        if self.plb_enabled:
            gt_ray_sum = gt_bboxes[:, :, 2:].sum(dim=-1)  # [B, N_gt_max]
            gt_ray_sum = gt_ray_sum * mask_gt.squeeze(-1).float()  # zero out padding
            total_area = gt_ray_sum.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
            plb_per_gt = 2.0 * (1.0 - gt_ray_sum / total_area)  # [B, N_gt_max]
            plb_weights = torch.gather(plb_per_gt, 1, target_gt_idx)  # [B, N_anchors]
            plb_weights = plb_weights * fg_mask.float()  # zero for bg

        # --- L_cls: BCE or Focal loss with bg subsampling ---
        # Save raw quality before binarization for fg_cls_boost.
        fg_quality = target_scores.float().clone()
        if self.soft_targets:
            # Keep assigner's quality-weighted alignment scores (e.g., 0.87 for a
            # strong match, 0.31 for a weak one). This gives the classifier a graded
            # signal — the model learns "this is a confident class-2 prediction" vs
            # "this is a marginal class-1 prediction." Prevents class collapse by
            # preserving the quality discriminability that hard 0/1 targets destroy.
            cls_targets = fg_quality.clone()
        else:
            # Hard binarize — legacy behaviour. Alignment quality belongs in the
            # centerness branch, not the classification branch.
            cls_targets = fg_quality.clone()
            cls_targets[cls_targets > 0] = 1.0

        # Subsample background anchors to cap bg:fg ratio per batch element.
        # Without this, P2's 4096 anchors overwhelm the ~1200 fg anchors,
        # producing cls_loss ~15 that drowns the regression gradient signal.
        ignore_mask = torch.zeros_like(cls_targets, dtype=torch.bool)
        if self.bg_fg_ratio > 0:
            fg_per_batch = fg_mask.sum(dim=1)  # [B]
            bg_mask = ~fg_mask  # [B, N]
            for b in range(batch_size):
                n_fg_b = fg_per_batch[b].item()
                if n_fg_b == 0:
                    continue
                bg_indices = bg_mask[b].nonzero(as_tuple=True)[0]
                n_bg_keep = min(bg_indices.shape[0], int(n_fg_b * self.bg_fg_ratio))
                if n_bg_keep < bg_indices.shape[0]:
                    perm = torch.randperm(bg_indices.shape[0], device=self.device)
                    drop_idx = perm[n_bg_keep:]
                    ignore_mask[b, bg_indices[drop_idx]] = True
                    cls_targets[b, bg_indices[drop_idx]] = 0

        # Class weights — scale positive cls loss per class to rebalance rare categories
        cw = self.class_weights  # [C] or None
        if cw is not None and cw.device != pred_scores.device:
            self.class_weights = cw.to(pred_scores.device)
            cw = self.class_weights

        if self.focal_gamma > 0:
            loss_cls = _focal_loss(
                pred_scores.float(),
                cls_targets,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
                class_weights=cw,
            )
        else:
            loss_cls = self.bce(pred_scores.float(), cls_targets)
            if cw is not None:
                pos_mask = (cls_targets > 0).float()
                loss_cls = loss_cls * (1.0 + pos_mask * (cw.unsqueeze(0).unsqueeze(0) - 1.0))

        if self.bg_fg_ratio > 0:
            loss_cls = loss_cls.masked_fill(ignore_mask, 0.0)

        if self.bg_cls_decay < 1.0 or self.fg_cls_boost > 0:
            if self.fg_cls_boost > 0:
                # Quality-aware fg boosting: weight = 1 + fg_cls_boost * quality
                # Perfect match (quality=1) gets 1+fg_cls_boost, marginal gets ~1.0
                fg_weight = 1.0 + self.fg_cls_boost * fg_quality
                bg_weight = torch.where(cls_targets > 0, fg_weight, self.bg_cls_decay)
            else:
                bg_weight = torch.where(cls_targets > 0, 1.0, self.bg_cls_decay)
            loss_cls = loss_cls * bg_weight

        target_scores_sum = max(cls_targets.sum(), 1)
        loss[1] = loss_cls.sum() / target_scores_sum

        # Track fg/bg cls contributions for diagnostics (normalized same as loss[1])
        _cls_fg_sum = (loss_cls * cls_targets).sum().item() / max(target_scores_sum, 1)
        _cls_bg_sum = (loss_cls * (1 - cls_targets)).sum().item() / max(target_scores_sum, 1)

        # --- Polygon regression losses (foreground only) ---
        n_fg = max(fg_mask.sum(), 1)
        if n_fg > 0:
            fg_plb = plb_weights[fg_mask] if plb_weights is not None else None
            fg_pred_rays = pred_rays[fg_mask]
            fg_target_xy = target_bboxes[fg_mask][:, :2]
            fg_target_rays = target_bboxes[fg_mask][:, 2:]

            fg_target_xy = fg_target_xy.float()
            fg_pred_rays = fg_pred_rays.float()
            fg_target_rays = fg_target_rays.float()

            fg_pred_xy = pred_xy[fg_mask]
            loss_xy = F.huber_loss(fg_pred_xy.float(), fg_target_xy, reduction='none', delta=0.05).mean(-1)
            if fg_plb is not None:
                loss_xy = loss_xy * fg_plb
            loss[0] = loss_xy.sum() / n_fg

            # L_L1: Uniform MAE on n_rays (linear or log-space)
            if self.log_ray_loss:
                loss_l1 = _log_space_ray_loss(fg_pred_rays, fg_target_rays)
            else:
                loss_l1 = (fg_pred_rays - fg_target_rays).abs().mean(-1)
            if fg_plb is not None:
                loss_l1 = loss_l1 * fg_plb
            loss[2] = loss_l1.sum() / n_fg

            # L_PolarIoU: -log(PolarIoU) — matches PolarMask formulation
            fg_piou = polar_iou_torch(fg_pred_rays, fg_target_rays)
            piou_loss = -torch.log(fg_piou + 1e-7)
            if fg_plb is not None:
                piou_loss = piou_loss * fg_plb
            loss[3] = piou_loss.sum() / n_fg

            # L_smooth: Curvature (2nd-order) regularisation on predicted rays
            smooth_loss = curvature_smoothness_loss_torch(fg_pred_rays)
            if fg_plb is not None:
                smooth_loss = smooth_loss * fg_plb
            loss[4] = smooth_loss.sum() / n_fg
        else:
            # DDP safety — touch all prediction tensors to avoid unused-gradient errors
            loss[0] += (pred_xy * 0).sum()
            loss[2] += (pred_rays * 0).sum()
            loss[3] += (pred_rays * 0).sum()
            loss[4] += (pred_rays * 0).sum()

        # --- Diagnostic: log raw (unweighted) losses + fg count every 100 steps ---
        if hasattr(self, '_diag_step'):
            self._diag_step += 1
        else:
            self._diag_step = 0
        if self._diag_step % 100 == 0:
            _raw = loss.detach().clone()
            n_fg_actual = fg_mask.sum().item() if fg_mask.sum() > 0 else 0
            _branch = getattr(self, 'branch_name', '???')
            LOGGER.info(
                '\nDIAG %s step=%d | fg=%d/%d | raw: xy=%.4f cls=%.4f(fg=%.3f bg=%.3f) l1=%.4f piou=%.4f smooth=%.5f',
                _branch,
                self._diag_step,
                n_fg_actual,
                fg_mask.numel(),
                _raw[0].item(),
                _raw[1].item(),
                _cls_fg_sum,
                _cls_bg_sum,
                _raw[2].item(),
                _raw[3].item(),
                _raw[4].item(),
            )

        # --- Store unweighted per-task losses for GradNorm ---
        if self.gradnorm_manager is not None:
            self.gradnorm_manager.store_losses(loss.detach())

        # --- Apply loss weights ---
        # If GradNorm is active, use its dynamic weights; otherwise use static lambdas.
        if self.gradnorm_manager is not None and self.gradnorm_manager.enabled:
            w = self.gradnorm_manager.get_weights()  # [5]
            loss[0] *= w[0]
            loss[1] *= w[1]
            loss[2] *= w[2]
            loss[3] *= w[3]
            loss[4] *= w[4]
        else:
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
    """E2E dual-assignment loss with smoothness annealing and GradNorm.

    Wires RayCastDetectionLoss into both one2many and one2one branches.
    Smoothness lambda anneals from smooth_start to smooth_end over
    smooth_anneal_epochs. Inherits o2m/o2o weight decay from parent.

    When ``gradnorm=True``, replaces static λ weights with GradNorm
    (Chen et al., 2018) dynamic weights that equalise gradient norms
    across all 5 tasks, preventing cls_loss from dominating the shared
    backbone.
    """

    def __init__(
        self,
        model,
        max_epochs: int = 200,
        tal_topk: int = 13,
        assigner_radius_scale: float = 1.5,
        assigner_alpha: float = 0.5,
        assigner_beta: float = 6.0,
        log_ray_loss: bool = False,
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.25,
        align_threshold: float = 0.0,
        gradnorm: bool = False,
        gradnorm_alpha: float = 0.5,
        gradnorm_warmup_epochs: int = 5,
        lambda_aux_xy: float = 0.0,
        aux_xy_ramp_epochs: int = 100,
        bg_fg_ratio: int = 3,
        plb_enabled: bool = False,
        bg_cls_decay: float = 1.0,
        fg_cls_boost: float = 0.0,
        soft_targets: bool = False,
        class_weights: torch.Tensor | None = None,
        o2o_topk2_start: int = 1,
        o2o_topk2_anneal_epoch: int = 0,
    ):
        # --- GradNorm manager (created before loss_fn so branches can reference it) ---
        self.gradnorm_manager: GradNormManager | None = None
        if gradnorm:
            self.gradnorm_manager = GradNormManager(
                model=model,
                num_tasks=5,
                alpha=gradnorm_alpha,
                warmup_epochs=gradnorm_warmup_epochs,
            )
            logger.info('GradNorm enabled: α=%.2f, warmup=%d epochs', gradnorm_alpha, gradnorm_warmup_epochs)

        # Bind training config to RayCastDetectionLoss so E2ELoss passes it through
        loss_fn = partial(
            RayCastDetectionLoss,
            tal_topk=tal_topk,
            assigner_radius_scale=assigner_radius_scale,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            log_ray_loss=log_ray_loss,
            gradnorm_manager=self.gradnorm_manager,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            align_threshold=align_threshold,
            bg_fg_ratio=bg_fg_ratio,
            plb_enabled=plb_enabled,
            bg_cls_decay=bg_cls_decay,
            fg_cls_boost=fg_cls_boost,
            soft_targets=soft_targets,
            class_weights=class_weights,
        )
        super().__init__(model, loss_fn=loss_fn)

        # Tag branches so DIAG logging can label output
        self.one2many.branch_name = 'o2m'
        self.one2one.branch_name = 'o2o'

        # Fix: parent E2ELoss reads one2one.hyp.epochs for the decay schedule,
        # but RayCastDetectionLoss (mock-based init) doesn't set hyp correctly.
        # Force both branches to use the actual max_epochs.
        for branch in (self.one2many, self.one2one):
            if not hasattr(branch, 'hyp') or branch.hyp is None:
                from ultralytics.cfg import get_cfg

                branch.hyp = get_cfg()
            branch.hyp.epochs = max_epochs

        # Override parent's hardcoded tal_topk values with our custom values.
        # Stock Ultralytics: o2m topk=10/topk2=10, o2o topk=7/topk2=1.
        # We use the same assigner class (RayCastAssigner) for both branches,
        # differing only in topk/topk2 — exactly like YOLO26 dual-TAL.
        #
        # o2m: topk=tal_topk, topk2=tal_topk → dense multi-anchor supervision
        # o2o: topk=pool, topk2=1 → single best anchor per GT (NMS-free)
        #
        # Ultralytics' NMS-free mechanism (select_highest_overlaps):
        #   1. select_topk_candidates picks `topk` anchors per GT
        #   2. if topk2 != topk → keeps only `topk2` best per GT
        self.one2many.assigner.topk = tal_topk
        self.one2many.assigner.topk2 = tal_topk  # no secondary filtering

        one2one_pool = max(tal_topk // 2, 7)
        self.one2one.assigner.topk = one2one_pool
        self.one2one.assigner.topk2 = o2o_topk2_start  # annealed to 1 in update()

        # o2o topk2 annealing: start with few positives per GT for stable gradients,
        # then tighten to topk2=1 for strict NMS-free inference. Inspired by
        # One-to-Few (Li et al., CVPR 2023) — small objects benefit from multiple
        # positives early in training.
        self._o2o_topk2_start = o2o_topk2_start
        self._o2o_topk2_anneal_epoch = o2o_topk2_anneal_epoch

        # Validate E2E architecture integrity
        assert self.one2one.assigner.topk2 >= 1, (
            f'E2E violation: one2one.topk2={self.one2one.assigner.topk2}, must be >= 1'
        )
        assert self.one2many.assigner.topk2 == self.one2many.assigner.topk, (
            f'E2E violation: one2many topk={self.one2many.assigner.topk} != topk2={self.one2many.assigner.topk2}'
        )
        assert self.one2many.assigner.topk2 == self.one2many.assigner.topk, (
            f'E2E violation: one2many.topk2 ({self.one2many.assigner.topk2}) != topk ({self.one2many.assigner.topk})'
        )

        # Steps per epoch (for epoch estimation in update())
        self.steps_per_epoch = 133

        # Smooth loss (curvature): reverse anneal — starts at 0, ramps up to peak, then holds.
        # 2nd-order difference penalises sharp kinks while allowing smooth irregular shapes.
        # Early training: model focuses on detection (xy, cls, L1, piou).
        # After ramp: curvature regularisation eliminates zigzag edge artefacts.
        self.smooth_start = 0.0  # initial value (no curvature pressure)
        self.smooth_end = 3.0  # meaningful curvature weight (complementary to L1/piou)
        self.smooth_anneal_fraction = 0.4  # ramp over first 40% of training
        self.smooth_anneal_epochs = max(1, int(max_epochs * self.smooth_anneal_fraction))

        # Auxiliary xy head: direct backbone→xy bypass for feature poverty
        self._aux_xy_base = lambda_aux_xy
        self.aux_xy_lambda = lambda_aux_xy
        self._max_epochs = max_epochs
        self.aux_xy_decay_epoch = max(1, aux_xy_ramp_epochs)

    def update(self):
        """Update o2m/o2o weights (inherited) + anneal smoothness + topk2 + validate E2E integrity."""
        super().update()

        current_epoch = self.updates / max(self.steps_per_epoch, 1)

        # Validate E2E integrity on first update (catches config drift)
        if self.updates == 1:
            topk2_cur = self.one2one.assigner.topk2
            print(
                f'✓ E2E NMS-free (dual-TAL): o2m.topk={self.one2many.assigner.topk}, '
                f'o2o.topk={self.one2one.assigner.topk}, o2o.topk2={topk2_cur}'
            )
            if self._o2o_topk2_start > 1:
                print(
                    f'  o2o topk2 anneal: {self._o2o_topk2_start}→1 starting at epoch {self._o2o_topk2_anneal_epoch}'
                )

            assert self.one2many.assigner.topk == self.one2many.assigner.topk2, (
                f'E2E violation: one2many topk={self.one2many.assigner.topk} != topk2={self.one2many.assigner.topk2}'
            )

            if self._aux_xy_base > 0:
                print(f'  Auxiliary XY head: weight={self._aux_xy_base}, decay_epoch={self.aux_xy_decay_epoch}')

        # Loss weight annealing
        current_epoch = self.updates / max(self.steps_per_epoch, 1)

        # Smooth: reverse-anneal (0 → 1) over first 40% of training
        t_smooth = min(self.updates / self.smooth_anneal_epochs, 1.0)
        lambda_smooth = self.smooth_start + t_smooth * (self.smooth_end - self.smooth_start)

        # Static lambdas — no warmup ramp (total=27 preserved)
        lambda_l1 = 14.0
        lambda_piou = 13.0

        for branch in (self.one2many, self.one2one):
            branch.lambda_smooth = lambda_smooth
            branch.lambda_l1 = lambda_l1
            branch.lambda_piou = lambda_piou

        # o2o topk2 annealing: few positives → strict 1:1 for NMS-free inference
        # Linear decay from o2o_topk2_start to 1 over the annealing window.
        # Before anneal_epoch: keep topk2_start (multi-positive for stable gradients)
        # After anneal_epoch: linear decay to 1 (tighten to strict o2o)
        if self._o2o_topk2_start > 1 and self._o2o_topk2_anneal_epoch > 0:
            if current_epoch < self._o2o_topk2_anneal_epoch:
                new_topk2 = self._o2o_topk2_start
            else:
                remaining = max(self._max_epochs - self._o2o_topk2_anneal_epoch, 1)
                progress = min((current_epoch - self._o2o_topk2_anneal_epoch) / remaining, 1.0)
                new_topk2 = max(int(round(self._o2o_topk2_start - progress * (self._o2o_topk2_start - 1))), 1)
            self.one2one.assigner.topk2 = new_topk2

        # Auxiliary XY: decay after aux_xy_decay_epoch
        if self._aux_xy_base > 0 and current_epoch >= self.aux_xy_decay_epoch:
            decay_progress = (current_epoch - self.aux_xy_decay_epoch) / max(
                self._max_epochs - self.aux_xy_decay_epoch, 1
            )
            self.aux_xy_lambda = self._aux_xy_base * max(1.0 - decay_progress, 0.0)
        elif self._aux_xy_base > 0:
            self.aux_xy_lambda = self._aux_xy_base

    def __call__(self, preds, batch):
        """Compute E2E losses + auxiliary xy loss on neck features."""
        parsed = self.one2many.parse_output(preds)
        one2many_preds = parsed['one2many']
        one2one_preds = parsed['one2one']

        loss_one2many = self.one2many.loss(one2many_preds, batch)
        loss_one2one = self.one2one.loss(one2one_preds, batch)
        total_loss = loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o
        loss_detach = loss_one2one[1]

        has_aux = self.aux_xy_lambda > 0 and 'aux_xy_raw' in one2many_preds

        if has_aux:
            aux_raw = one2many_preds['aux_xy_raw'].permute(0, 2, 1).contiguous()
            feats = one2many_preds['feats']
            anchor_points, stride_tensor = make_anchors(feats, self.one2many.stride, 0.5)
            imgsz = torch.tensor(feats[0].shape[2:], device=aux_raw.device, dtype=aux_raw.dtype) * stride_tensor[0]

            aux_xy_offset = aux_raw.sigmoid()
            aux_xy_pixel = (aux_xy_offset * 2.0 - 0.5 + anchor_points) * stride_tensor
            aux_pred_xy = aux_xy_pixel / imgsz[[1, 0]]

            targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.one2many.preprocess(targets.to(self.one2many.device), aux_raw.shape[0])
            _, gt_bboxes = targets.split((1, self.one2many.raycast_dim), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            _, target_bboxes, _, fg_mask, _ = self.one2many.assigner(
                one2many_preds['scores'].permute(0, 2, 1).contiguous().detach().sigmoid(),
                torch.cat(
                    [
                        aux_pred_xy.detach(),
                        F.softplus(one2many_preds['boxes'].permute(0, 2, 1).contiguous()[..., 2:].detach()),
                    ],
                    dim=-1,
                ),
                anchor_points * stride_tensor / imgsz[[1, 0]],
                targets.split((1, self.one2many.raycast_dim), 2)[0],
                gt_bboxes,
                mask_gt,
            )

            n_fg = max(fg_mask.sum(), 1)
            if n_fg > 0:
                fg_target_xy = target_bboxes[fg_mask][:, :2].float()
                fg_aux_xy = aux_pred_xy[fg_mask].float()
                aux_loss = F.huber_loss(fg_aux_xy, fg_target_xy, reduction='none', delta=0.05).mean(-1)
                aux_loss_val = aux_loss.sum() / n_fg * self.aux_xy_lambda
                # Add as a 6th element to avoid broadcasting into the 5-element loss tensor.
                # Ultralytics calls .sum() on total_loss for backward, so a scalar aux loss
                # added to a 5-element tensor would be counted 5x via broadcast.
                total_loss = torch.cat([total_loss, aux_loss_val.unsqueeze(0)])

                loss_detach = torch.cat([loss_detach, aux_loss_val.detach().unsqueeze(0)])
            else:
                loss_detach = torch.cat([loss_detach, torch.zeros(1, device=loss_detach.device)])
        else:
            loss_detach = torch.cat([loss_detach, torch.zeros(1, device=loss_detach.device)])

        return total_loss, loss_detach
