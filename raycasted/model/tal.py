"""RayCastED — RayCast Assigner (Phase 5).

RayCastAssigner subclasses TaskAlignedAssigner to replace box-based
assignment with polygon-aware logic:
  - get_box_metrics: Polar-IoU geometric overlap for alignment metric
  - get_pos_mask: Gaussian spatial decay (no hard containment filter)
  - select_topk_candidates: dynamic topk cap per GT to avoid garbage padding

All anchors are candidates — no hard containment filter. The Gaussian
decay in get_pos_mask provides soft spatial weighting so distant anchors
receive near-zero alignment metric and are naturally excluded by topk.

The two branches use different ranking strategies:

  one2many:  align = cls_score^α × PolarIoU^β  (multiplicative, geometric dominance)
  one2one:   cost = w_cls*focal + w_xy*L2 + w_ray*log_l1  (additive, Hungarian)

Polar-IoU in the one2many branch provides absolute geometric discrimination
critical for dense touching-cell scenes: even a 1-pixel boundary overshoot
between adjacent nuclei is harshly penalised by β=6 exponentiation,
preventing anchor assignment collisions.

HungarianRayCastAssigner extends RayCastAssigner with globally optimal
bipartite matching (scipy.linear_sum_assignment) for the one2one branch.

get_targets is NOT overridden — the parent implementation is dimension-
agnostic (uses gt_bboxes.shape[-1] dynamically) and works for any
polygon dimensionality (2 + N_RAYS) without modification.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics.utils.tal import TaskAlignedAssigner


class RayCastAssigner(TaskAlignedAssigner):
    """Polygon-aware assigner with Polar-IoU + Gaussian spatial decay.

    Alignment metric (multiplicative):
      align = cls_score^alpha * PolarIoU^beta * exp(-d^2 / 2sigma^2)

    Gaussian decay gives soft spatial proximity scoring — anchors near the
    GT centroid get full weight, distant ones fade smoothly. No hard
    containment filter: all anchors are candidates. sigma = radius_scale * R75.

    Both o2m and o2o branches use this assigner with different topk/topk2:
      - o2m: topk=N, topk2=N -> dense multi-anchor supervision
      - o2o: topk=M, topk2=1 -> single best anchor per GT (NMS-free)

    Overrides three methods from TaskAlignedAssigner:
      - get_box_metrics: Polar-IoU overlap for alignment metric
      - get_pos_mask: Gaussian spatial decay on align_metric
      - select_topk_candidates: dynamic topk cap per GT

    get_targets is inherited as-is — dimension-agnostic.
    """

    def __init__(
        self,
        topk=13,
        num_classes=80,
        alpha=0.5,
        beta=6.0,
        stride=None,
        eps=1e-9,
        topk2=None,
        radius_scale=1.5,
        align_threshold=0.0,
        prefilter_k=0,
    ):
        """Initialize RayCastAssigner.

        Args:
            topk: Number of top-k candidate anchors per GT.
            num_classes: Number of object classes.
            alpha: Exponent on cls_score in alignment metric.
            beta: Exponent on Polar-IoU in alignment metric.
            stride: Feature map strides (default [8, 16, 32]).
            eps: Small value to prevent division by zero.
            topk2: Secondary topk for additional filtering.
            radius_scale: Gaussian sigma multiplier. sigma = radius_scale * R75.
                Controls spatial decay width. Hard cutoff is at 3sigma.
                Lower = tighter assignment. Higher = wider spatial influence.
            align_threshold: Minimum overlap proxy for a candidate to be
                considered a positive. Anchors below this threshold are
                zeroed out before topk selection. Range [0, 1), default 0.
            prefilter_k: Number of nearest anchors to prefilter for PolarIoU.
                Must be > topk for proper IoU-based ranking.
                Default (0): auto-computed as max(topk*5, 100).
        """
        super().__init__(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride or [8, 16, 32],
            eps=eps,
            topk2=topk2,
        )
        self.radius_scale = radius_scale
        self.align_threshold = align_threshold
        self.stal_min_positives = 0  # set by RayCastE2ELoss if enabled
        # Number of nearest anchors to prefilter for PolarIoU.
        # Must be > topk for proper IoU-based ranking. Default: max(topk*5, 100).
        self.prefilter_k = prefilter_k if prefilter_k > 0 else max(topk * 5, 100)

    # -----------------------------------------------------------------
    # Shared helper: per-GT radius computation
    # -----------------------------------------------------------------

    def _compute_gt_radii(self, gt_rays):
        """Compute per-GT radius for containment and Gaussian sigma.

        Uses 75th-percentile non-zero ray as radius. Falls back to max ray
        when fewer than 8 non-zero rays (small polygons), zero for empty GTs.

        Args:
            gt_rays: (N_valid_gt, N_RAYS) ray distances.

        Returns:
            radii: (N_valid_gt,) per-GT radius in normalised coords.
        """
        sorted_rays, _ = gt_rays.sort(dim=1)
        n_non_zero = (gt_rays > 0).sum(dim=1)
        n_rays = gt_rays.shape[-1]
        n_zero = n_rays - n_non_zero
        pct75_idx = (n_non_zero.float() * 0.75).long().clamp(min=0)
        pct75_flat_idx = (n_zero + pct75_idx).clamp(max=n_rays - 1)
        pct75_radii = sorted_rays.gather(1, pct75_flat_idx.unsqueeze(1)).squeeze(1)
        max_radii = sorted_rays[:, -1]
        return torch.where(
            n_non_zero >= 8,
            pct75_radii,
            torch.where(n_non_zero > 0, max_radii, torch.zeros(1, device=gt_rays.device)),
        )

    # -----------------------------------------------------------------
    # get_box_metrics — Polar-IoU alignment metric
    # -----------------------------------------------------------------

    def _prefilter_by_distance(self, gt_bboxes, anc_points, mask_gt):
        """Compute top-K nearest anchors + Gaussian weights, fully batched.

        Operates on the full (bs, n_max, ...) tensors without a per-batch loop.
        Invalid GTs (zero bboxes, zero radii) naturally get sigma=0 → gauss=0,
        so they contribute nothing downstream.

        Returns:
            topk_idx: (bs, n_max, K) anchor indices nearest to each GT.
            gauss:    (bs, n_max, K) Gaussian weights for those anchors.
        """
        bs, n_max = gt_bboxes.shape[:2]
        k_prefilt = self.prefilter_k

        flat_rays = gt_bboxes[:, :, 2:].reshape(-1, gt_bboxes.shape[-1] - 2)
        radii = self._compute_gt_radii(flat_rays).reshape(bs, n_max)
        sigma = radii * self.radius_scale

        dist = torch.cdist(gt_bboxes[:, :, :2].float(), anc_points.float())
        _, topk_idx = dist.topk(k_prefilt, dim=2, largest=False)

        sigma_k = sigma.unsqueeze(-1)
        topk_dist = dist.gather(2, topk_idx)
        gauss = torch.exp(-topk_dist.pow(2) / (2.0 * sigma_k.pow(2) + self.eps))

        return topk_idx, gauss

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, gauss_data=None):
        """Compute alignment metric using Polar-IoU, fully batched.

        Formula: align = cls_score^alpha * PolarIoU^beta

        Two paths:
          - With gauss_data (topk_idx, gauss): block-sparse PolarIoU on top-K
            nearest anchors. Fully batched — single gather + polar_iou_torch +
            scatter across all batch elements. No Python loop.
          - Without: all-pairs fallback with per-batch loop (backward compat).
        """
        na = pd_bboxes.shape[-2]
        mask_gt_bool = mask_gt.bool().expand(-1, -1, na)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=torch.float32, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt_bool] = pd_scores[ind[0], :, ind[1]][mask_gt_bool]

        if gauss_data is not None:
            topk_idx, _gauss = gauss_data
            k_prefilt = topk_idx.shape[-1]
            n_rays = pd_bboxes.shape[-1] - 2
            from raycasted.data.etl.ops.iou import polar_iou_torch

            bs_idx = torch.arange(self.bs, device=topk_idx.device)
            bs_idx = bs_idx.view(-1, 1, 1).expand(-1, self.n_max_boxes, k_prefilt)
            pd_block = pd_bboxes[bs_idx, topk_idx, 2:]
            gt_rays = gt_bboxes[:, :, 2:].unsqueeze(2).expand(-1, -1, k_prefilt, -1)

            iou_block = polar_iou_torch(pd_block.reshape(-1, n_rays), gt_rays.reshape(-1, n_rays)).reshape(
                self.bs, self.n_max_boxes, k_prefilt
            )
            overlaps.scatter_(2, topk_idx, iou_block.to(overlaps.dtype))
        else:
            from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch

            for b in range(self.bs):
                valid_gt_mask = mask_gt_bool[b].any(dim=1)
                valid_gt_idx = valid_gt_mask.nonzero(as_tuple=False).squeeze(-1)
                n_valid_gt = valid_gt_idx.shape[0]
                if n_valid_gt == 0:
                    continue

                pd_rays = pd_bboxes[b, :, 2:]
                gt_rays = gt_bboxes[b, valid_gt_idx, 2:]
                pd_exp = pd_rays[:, None, :].expand(-1, n_valid_gt, -1)
                gt_exp = gt_rays[None, :, :].expand(na, -1, -1)
                iou = polar_iou_pairwise_flat_torch(pd_exp, gt_exp)
                overlaps[b, valid_gt_idx] = iou.T.to(overlaps.dtype)

        if self.align_threshold > 0:
            overlaps = overlaps * (overlaps >= self.align_threshold).float()

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    # -----------------------------------------------------------------
    # Override 2b: get_pos_mask — Gaussian spatial decay on align_metric
    # -----------------------------------------------------------------

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Override parent to add Gaussian spatial decay to alignment metric.

        Formula:  align_metric = cls^alpha * PolarIoU^beta * exp(-d^2 / 2*sigma^2)

        The Gaussian decay naturally zeros out distant anchors (<0.011 at 3*sigma),
        so they never survive topk selection. This gives soft spatial weighting
        without hard containment filtering. Fully batched — no Python loops.
        """
        gauss_data = self._prefilter_by_distance(gt_bboxes, anc_points, mask_gt) if self.radius_scale > 0 else None

        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, gauss_data=gauss_data
        )

        if gauss_data is not None:
            topk_idx, gauss = gauss_data
            current = align_metric.gather(2, topk_idx)
            updated = current * gauss
            align_metric.scatter_(2, topk_idx, updated)

        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())

        mask_pos = mask_topk * mask_gt

        # STAL: enforce minimum positive assignments per GT (YOLO26)
        if self.stal_min_positives > 0:
            pos_per_gt = mask_pos.sum(dim=-1)  # [bs, n_max_boxes]
            valid_gt = mask_gt.any(dim=-1)  # [bs, n_max_boxes]
            missing = (pos_per_gt < self.stal_min_positives) & valid_gt
            if missing.any():
                distances = torch.cdist(gt_bboxes[:, :, :2].float(), anc_points.float())
                _, nearest_idx = distances.min(dim=-1)  # [bs, n_max_boxes]
                for b in range(self.bs):
                    missing_gts = missing[b].nonzero(as_tuple=True)[0]
                    for g in missing_gts:
                        mask_pos[b, g, nearest_idx[b, g]] = 1

        return mask_pos, align_metric, overlaps

    # -----------------------------------------------------------------
    # Override 3: dynamic topk
    # -----------------------------------------------------------------

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select top-k candidates with dynamic per-GT cap.

        Overrides parent to clamp topk per GT to the number of actual
        candidates (non-zero metric entries). When a GT has fewer than
        ``self.topk`` anchors within its containment radius, the parent
        pads with the first-k entries — which injects garbage anchors
        into the positive set. This override prevents that by masking
        zero-metric entries before topk selection.
        """
        bs, n_max_boxes, n_anchors = metrics.shape
        safe_k = min(self.topk, n_anchors)

        topk_metrics, topk_idxs = torch.topk(metrics, safe_k, dim=-1, largest=True)

        metric_mask = topk_metrics > self.eps
        if topk_mask is not None:
            metric_mask = metric_mask & topk_mask
        topk_idxs.masked_fill_(~metric_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(safe_k):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        count_tensor.clamp_(0, 1)
        return count_tensor


class HungarianRayCastAssigner(RayCastAssigner):
    """Globally optimal bipartite matching assigner for the one2one branch.

    Solves with scipy.linear_sum_assignment for globally optimal 1:1 matching.
    Uses its own cost matrix (not shared with dual-TAL assigner).

    Inherits select_candidates_in_gts from RayCastAssigner. Overrides _forward
    to replace the TAL selection mechanism with Hungarian matching.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_centroid: float = 1.0,
        cost_ray: float = 1.0,
        **kwargs,
    ):
        """Initialize HungarianRayCastAssigner.

        Args:
            cost_class: Weight for focal classification cost.
            cost_centroid: Weight for L2 centroid distance cost.
            cost_ray: Weight for log-space L1 ray cost.
            **kwargs: Passed to parent RayCastAssigner (topk, num_classes
                etc.).
        """
        super().__init__(**kwargs)
        self.cost_class = cost_class
        self.cost_centroid = cost_centroid
        self.cost_ray = cost_ray

    def _compute_cost_matrix(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt_bool):
        """Compute unified cost matrix for Hungarian assignment.

        cost = w_cls * focal_cls_cost + w_xy * L2_centroid + w_ray * log_l1_rays
        """
        bs = pd_scores.shape[0]
        n_max_boxes = pd_scores.shape[1]
        na = pd_bboxes.shape[-2]
        cost = torch.zeros([bs, n_max_boxes, na], dtype=torch.float32, device=pd_bboxes.device)

        for b in range(bs):
            valid_gt_mask = mask_gt_bool[b].any(dim=1)
            valid_gt_idx = valid_gt_mask.nonzero(as_tuple=False).squeeze(-1)
            n_valid_gt = valid_gt_idx.shape[0]
            if n_valid_gt == 0:
                continue

            cand_idx = torch.arange(na, device=pd_bboxes.device)

            # Focal classification cost (DETR-style, fixed params)
            gt_cls = gt_labels[b, valid_gt_idx, 0].long().clamp(min=0)
            out_prob = pd_scores[b, cand_idx].sigmoid()
            alpha_focal, gamma_focal = 0.25, 2.0
            neg_cost = (1 - alpha_focal) * (out_prob**gamma_focal) * (-(1 - out_prob + 1e-8).log())
            pos_cost = alpha_focal * ((1 - out_prob) ** gamma_focal) * (-(out_prob + 1e-8).log())
            cost_cls = pos_cost[:, gt_cls] - neg_cost[:, gt_cls]

            # L2 centroid cost
            pd_xy = pd_bboxes[b, cand_idx, :2].float()
            gt_xy = gt_bboxes[b, valid_gt_idx, :2].float()
            cost_xy = torch.cdist(pd_xy, gt_xy, p=2)

            # Log-space L1 ray cost
            pd_rays = pd_bboxes[b, cand_idx, 2:].float()
            gt_rays = gt_bboxes[b, valid_gt_idx, 2:].float()
            log_pd = pd_rays[:, None, :].clamp(min=1e-4).log()
            log_gt = gt_rays[None, :, :].clamp(min=1e-4).log()
            cost_ray = (log_pd - log_gt).abs().mean(dim=-1)

            total = self.cost_class * cost_cls + self.cost_centroid * cost_xy + self.cost_ray * cost_ray
            total = total.nan_to_num(nan=1e8, posinf=1e8, neginf=-1e8)

            cost[b, valid_gt_idx[:, None], cand_idx[None, :]] = total.T

        return cost

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Hungarian matching assignment with unified cost matrix.

        Uses the same _compute_cost_matrix as the one2many branch, ensuring
        both branches rank candidates identically. Solves with
        scipy.linear_sum_assignment for globally optimal 1:1 matching.

        Args:
            pd_scores: Predicted classification scores, shape (B, N_anchors, nc).
            pd_bboxes: Predicted polygon targets, shape (B, N_anchors, 2+N_RAYS).
            anc_points: Anchor grid positions, shape (N_anchors, 2).
            gt_labels: GT class labels, shape (B, N_max_gt, 1).
            gt_bboxes: GT polygon targets, shape (B, N_max_gt, 2+N_RAYS).
            mask_gt: Valid GT mask, shape (B, N_max_gt, 1).

        Returns:
            Tuple of (target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx).
        """
        mask_gt_bool = mask_gt.bool()

        # All anchors are candidates (no hard containment)
        bs = pd_scores.shape[0]
        na = pd_scores.shape[1]
        device = gt_bboxes.device

        target_labels = torch.full((bs, na), self.num_classes, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(bs, na, gt_bboxes.shape[-1], dtype=gt_bboxes.dtype, device=device)
        target_scores = torch.zeros_like(pd_scores)
        fg_mask = torch.zeros(bs, na, dtype=torch.bool, device=device)
        target_gt_idx = torch.zeros(bs, na, dtype=torch.long, device=device)

        # Use unified cost matrix
        cost_matrix = self._compute_cost_matrix(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt_bool)

        for b in range(bs):
            valid_gt_mask = mask_gt_bool[b, :, 0]
            n_valid_gt = valid_gt_mask.sum().item()
            if n_valid_gt == 0:
                continue

            valid_gt_idx = valid_gt_mask.nonzero(as_tuple=False).squeeze(-1)

            # All anchors are candidates
            cand_idx = torch.arange(na, device=device)
            n_cand = na
            if n_cand == 0:
                continue

            # Extract (n_valid_gt, n_cand) sub-matrix for Hungarian
            cost_np = cost_matrix[b, valid_gt_idx][:, cand_idx].cpu().numpy()
            cost_np = np.nan_to_num(cost_np, nan=1e8, posinf=1e8, neginf=-1e8)

            # Solve bipartite matching
            row_idx, col_idx = linear_sum_assignment(cost_np)

            for gi, ci in zip(row_idx, col_idx):
                gt_i = valid_gt_idx[gi].item()
                anc_i = cand_idx[ci].item()

                target_labels[b, anc_i] = gt_labels[b, gt_i, 0].long()
                target_bboxes[b, anc_i] = gt_bboxes[b, gt_i]
                fg_mask[b, anc_i] = True
                target_gt_idx[b, anc_i] = gt_i

                # Quality score from cost: 1/(1+cost) ∈ (0, 1]
                target_scores[b, anc_i, gt_labels[b, gt_i, 0].long()] = 1.0 / (1.0 + cost_np[gi, ci])

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx
