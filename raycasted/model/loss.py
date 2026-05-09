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
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from raycasted.data.etl.ops.loss import angular_smoothness_loss_torch
from raycasted.model.tal import HungarianRayCastAssigner, RayCastAssigner

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  GradNorm — Gradient Normalisation for Multi-Task Loss Balancing
# ──────────────────────────────────────────────────────────────────────

_TASK_NAMES = ('xy', 'cls', 'l1', 'smooth')


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
    alpha: float = 1.0,
) -> torch.Tensor:
    """Focal loss for multi-label classification.

    Down-weights easy negatives and amplifies hard positives, which helps
    in dense cell scenes where background anchors vastly outnumber positive ones.

    When gamma=0 and alpha=1.0, this reduces to standard BCE.

    Args:
        pred_scores: [B, N, C] raw logits from the detection head.
        target_scores: [B, N, C] soft classification targets from the assigner.
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Positive sample weight factor. 1.0 means no extra weighting.

    Returns:
        [B, N, C] element-wise focal loss (no reduction).
    """
    pred = pred_scores.sigmoid()
    ce = F.binary_cross_entropy(pred, target_scores, reduction='none')
    focal_weight = alpha * (1 - pred) ** gamma * target_scores + pred**gamma * (1 - target_scores)
    return focal_weight * ce


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
        align_threshold: float = 0.0,
        log_ray_loss: bool = False,
        gradnorm_manager: GradNormManager | None = None,
        focal_gamma: float = 0.0,
        focal_alpha: float = 1.0,
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

        # Loss weights (rebalanced so xy and L1 share gradient signal equally)
        # xy and L1 both produce tiny raw values (Huber on normalised coords, log-space rays),
        # so both need high lambda to contribute meaningfully.
        # smooth starts at 0, reverse-anneals to peak over 40% of training (shape prior).
        self.lambda_cls = 2.0
        self.lambda_xy = 15.0
        self.lambda_l1 = 25.0
        self.lambda_smooth = 0.0  # reverse-annealed by RayCastE2ELoss

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
        """Compute 4-term polygon loss.

        Returns:
            (assignment_info, loss_4vec, loss_detach)
        """
        loss = torch.zeros(4, device=self.device)  # [xy, cls, L1, smooth]

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

        # --- L_cls: BCE or Focal loss normalised by target_scores_sum ---
        target_scores_sum = max(target_scores.sum(), 1)
        if self.focal_gamma > 0:
            cls_targets = target_scores.float().clone()
            cls_targets[cls_targets > 0] = 1.0
            loss_cls = _focal_loss(
                pred_scores.float(),
                cls_targets,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
            )
        else:
            loss_cls = self.bce(pred_scores.float(), target_scores.float())
        loss[1] = loss_cls.sum() / target_scores_sum

        # --- Polygon regression losses (foreground only, uniform weight) ---
        if fg_mask.sum():
            n_fg = max(fg_mask.sum(), 1)

            fg_pred_xy = pred_xy[fg_mask]
            fg_pred_rays = pred_rays[fg_mask]
            fg_target_xy = target_bboxes[fg_mask][:, :2]
            fg_target_rays = target_bboxes[fg_mask][:, 2:]

            fg_pred_xy = fg_pred_xy.float()
            fg_target_xy = fg_target_xy.float()
            fg_pred_rays = fg_pred_rays.float()
            fg_target_rays = fg_target_rays.float()

            # L_xy: Huber on decoded centroid
            loss_xy = F.huber_loss(fg_pred_xy, fg_target_xy, reduction='none', delta=1.0).mean(-1)
            loss[0] = loss_xy.sum() / n_fg

            # L_L1: Uniform MAE on 32 rays (linear or log-space)
            if self.log_ray_loss:
                loss_l1 = _log_space_ray_loss(fg_pred_rays, fg_target_rays)
            else:
                loss_l1 = (fg_pred_rays - fg_target_rays).abs().mean(-1)
            loss[2] = loss_l1.sum() / n_fg

            # L_smooth: Angular smoothness on predicted rays
            loss[3] = angular_smoothness_loss_torch(fg_pred_rays).sum() / n_fg
        else:
            # DDP safety — touch all prediction tensors to avoid unused-gradient errors
            loss[0] += (pred_xy * 0).sum()
            loss[2] += (pred_rays * 0).sum()
            loss[3] += (pred_rays * 0).sum()

        # --- Store unweighted per-task losses for GradNorm ---
        if self.gradnorm_manager is not None:
            self.gradnorm_manager.store_losses(loss.detach())

        # --- Apply loss weights ---
        # If GradNorm is active, use its dynamic weights; otherwise use static lambdas.
        if self.gradnorm_manager is not None and self.gradnorm_manager.enabled:
            w = self.gradnorm_manager.get_weights()  # [4]
            loss[0] *= w[0]
            loss[1] *= w[1]
            loss[2] *= w[2]
            loss[3] *= w[3]
        else:
            loss[0] *= self.lambda_xy
            loss[1] *= self.lambda_cls
            loss[2] *= self.lambda_l1
            loss[3] *= self.lambda_smooth

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
    across all 4 tasks, preventing cls_loss from dominating the shared
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
        use_hungarian_o2o: bool = True,
        log_ray_loss: bool = False,
        cost_class: float = 1.0,
        cost_centroid: float = 1.0,
        cost_ray: float = 1.0,
        focal_gamma: float = 0.0,
        focal_alpha: float = 1.0,
        align_threshold: float = 0.0,
        gradnorm: bool = False,
        gradnorm_alpha: float = 0.5,
        gradnorm_warmup_epochs: int = 5,
    ):
        # --- GradNorm manager (created before loss_fn so branches can reference it) ---
        self.gradnorm_manager: GradNormManager | None = None
        if gradnorm:
            self.gradnorm_manager = GradNormManager(
                model=model,
                num_tasks=4,
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
                cost_class=cost_class,
                cost_centroid=cost_centroid,
                cost_ray=cost_ray,
                stride=self.one2one.assigner.stride if hasattr(self.one2one.assigner, 'stride') else [8, 16, 32],
                topk2=1,
                radius_scale=assigner_radius_scale,
                align_threshold=align_threshold,
            )
        else:
            one2one_pool = max(tal_topk // 2, 7)
            self.one2one.assigner = RayCastAssigner(
                topk=one2one_pool,
                num_classes=self.one2one.assigner.num_classes,
                alpha=assigner_alpha,
                beta=assigner_beta,
                stride=self.one2one.assigner.stride if hasattr(self.one2one.assigner, 'stride') else [8, 16, 32],
                topk2=1,
                radius_scale=assigner_radius_scale,
                align_threshold=align_threshold,
            )

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
        self.smooth_start = 0.0  # initial value (no smoothness pressure)
        self.smooth_end = 1.0  # peak value (meaningful shape prior)
        self.smooth_anneal_fraction = 0.4  # ramp over first 40% of training
        self.smooth_anneal_epochs = max(1, int(max_epochs * self.smooth_anneal_fraction))

    def update(self):
        """Update o2m/o2o weights (inherited) + anneal smoothness + validate E2E integrity."""
        super().update()

        # Validate E2E integrity on first update (catches config drift)
        if self.updates == 1:
            is_hungarian = isinstance(self.one2one.assigner, HungarianRayCastAssigner)
            if is_hungarian:
                a = self.one2one.assigner
                print(
                    f'✓ E2E NMS-free: o2m.topk={self.one2many.assigner.topk}, '
                    f'o2o=Hungarian L1-cost (cls={a.cost_class}, xy={a.cost_centroid}, ray={a.cost_ray})'
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
