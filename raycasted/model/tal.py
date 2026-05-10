"""RayCastED — RayCast Assigner (Phase 5).

RayCastAssigner subclasses TaskAlignedAssigner to replace box-based
assignment with polygon-aware logic:
  - select_candidates_in_gts: 75th-percentile radius containment
  - get_box_metrics: unified cost matrix (focal cls + L2 centroid + log-space L1 rays)
  - select_topk_candidates: dynamic topk cap per GT to avoid garbage padding

Both one2many and one2one branches share the identical cost formulation:
  cost = w_cls * focal_cls_cost + w_xy * L2(pred_xy, gt_xy) + w_ray * mean(|log(pred) - log(gt)|)

One2many inverts the cost to a similarity metric: align = 1 / (1 + cost)
Hungarian uses the raw cost for linear_sum_assignment (minimisation).

HungarianRayCastAssigner extends RayCastAssigner with globally optimal
bipartite matching (scipy.linear_sum_assignment) for the one2one branch.

get_targets is NOT overridden — the parent implementation is dimension-
agnostic (uses gt_bboxes.shape[-1] dynamically) and works for 34-dim
polygon targets without modification.

Spec reference: docs/project.md §11
"""

import torch
from scipy.optimize import linear_sum_assignment
from ultralytics.utils.tal import TaskAlignedAssigner


class RayCastAssigner(TaskAlignedAssigner):
    """Polygon-aware assigner with unified cost matrix for both branches.

    Uses identical cost terms for one2many and one2one assignment:
      cost = w_cls * focal_cls + w_xy * L2_centroid + w_ray * log_l1_rays

    Overrides three methods from TaskAlignedAssigner:
      - select_candidates_in_gts: radius containment instead of box containment
      - get_box_metrics: unified cost → inverted similarity for topk selection
      - select_topk_candidates: dynamic topk cap per GT to avoid garbage padding

    get_targets is inherited as-is — the parent implementation is dimension-
    agnostic (uses gt_bboxes.shape[-1] dynamically) and works for 34-dim
    polygon targets without modification.
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
        cost_class=1.0,
        cost_centroid=1.0,
        cost_ray=1.0,
    ):
        """Initialize RayCastAssigner.

        Args:
            topk: Number of top-k candidate anchors per GT.
            num_classes: Number of object classes.
            alpha: Unused (kept for API compatibility).
            beta: Unused (kept for API compatibility).
            stride: Feature map strides (default [8, 16, 32]).
            eps: Small value to prevent division by zero.
            topk2: Secondary topk for additional filtering.
            radius_scale: Multiplier on 75th-percentile containment radius.
                Monitor mean positive assignments per GT cell for first 100
                batches. Target: 1-4. Below 1 → too small. Above 10 → too large.
            align_threshold: Minimum overlap proxy for a candidate to be
                considered a positive. Anchors below this threshold are
                zeroed out before topk selection. Range [0, 1), default 0.
            cost_class: Weight for focal classification cost.
            cost_centroid: Weight for L2 centroid distance cost.
            cost_ray: Weight for log-space L1 ray cost.
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
        self.cost_class = cost_class
        self.cost_centroid = cost_centroid
        self.cost_ray = cost_ray

    # -----------------------------------------------------------------
    # Override 1: radius-based containment
    # -----------------------------------------------------------------

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """Select anchors within containment radius of each GT polygon centroid.

        Replaces the parent's box-containment check (xyxy corner test) with
        a polar-radius containment test using the 75th-percentile GT ray value.

        Args:
            xy_centers: Anchor grid positions, shape (N_anchors, 2).
            gt_bboxes: GT polygon targets, shape (B, N_max_gt, 34).
                       Columns: [cx, cy, d_1..d_32] in normalised coords.
            mask_gt: Valid GT mask, shape (B, N_max_gt, 1).
            eps: Unused (kept for API compatibility).

        Returns:
            Boolean mask of shape (B, N_max_gt, N_anchors).
        """
        _ = eps
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        mask = torch.zeros(bs, n_boxes, n_anchors, dtype=torch.bool, device=gt_bboxes.device)

        for b in range(bs):
            valid = mask_gt[b, :, 0].bool()
            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue

            gt_xy = gt_bboxes[b, valid_idx, :2]
            gt_rays = gt_bboxes[b, valid_idx, 2:]

            sorted_rays, _ = gt_rays.sort(dim=1)

            n_non_zero = (gt_rays > 0).sum(dim=1)
            n_rays = gt_rays.shape[-1]
            n_zero = n_rays - n_non_zero

            pct75_idx = (n_non_zero.float() * 0.75).long().clamp(min=0)
            pct75_flat_idx = (n_zero + pct75_idx).clamp(max=n_rays - 1)
            pct75_radii = sorted_rays.gather(1, pct75_flat_idx.unsqueeze(1)).squeeze(1)

            max_radii = sorted_rays[:, -1]

            radii = torch.where(
                n_non_zero >= 8,
                pct75_radii,
                torch.where(n_non_zero > 0, max_radii, torch.zeros(1, device=gt_bboxes.device)),
            )

            containment = radii * self.radius_scale

            dist = torch.cdist(gt_xy.float(), xy_centers.float())
            valid_mask = (dist <= containment[:, None]) & (containment[:, None] > 0)

            mask[b, valid_idx] = valid_mask

        return mask

    # -----------------------------------------------------------------
    # Unified cost matrix — shared by both branches
    # -----------------------------------------------------------------

    def _compute_cost_matrix(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt_bool):
        """Compute unified cost matrix for assignment.

        Uses identical terms for both one2many and one2one branches:
          cost = w_cls * focal_cls_cost + w_xy * L2_centroid + w_ray * log_l1_rays

        Args:
            pd_scores: Classification scores, shape (B, N_anchors, nc).
            pd_bboxes: Decoded polygon predictions, shape (B, N_anchors, 34).
                       Columns: [decoded_xy(2), softplus_rays(32)].
            gt_labels: GT class labels, shape (B, N_max_gt, 1).
            gt_bboxes: GT polygon targets, shape (B, N_max_gt, 34).
                       Columns: [cx, cy, d_1..d_32].
            mask_gt_bool: Boolean GT mask, shape (B, N_max_gt, 1).

        Returns:
            cost: (B, N_max_gt, N_anchors) — lower is better.
            overlaps: (B, N_max_gt, N_anchors) — ray similarity 1/(1+log_l1).
        """
        na = pd_bboxes.shape[-2]
        cost = torch.zeros([self.bs, self.n_max_boxes, na], dtype=torch.float32, device=pd_bboxes.device)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=torch.float32, device=pd_bboxes.device)

        for b in range(self.bs):
            candidate_mask = mask_gt_bool[b].any(dim=0)
            cand_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
            n_cand = cand_idx.shape[0]
            if n_cand == 0:
                continue

            valid_gt_mask = mask_gt_bool[b].any(dim=1)
            valid_gt_idx = valid_gt_mask.nonzero(as_tuple=False).squeeze(-1)

            # --- Focal classification cost ---
            gt_cls = gt_labels[b, valid_gt_idx, 0].long().clamp(min=0)
            out_prob = pd_scores[b, cand_idx].sigmoid()
            alpha_focal = 0.25
            gamma_focal = 2.0
            neg_cost = (1 - alpha_focal) * (out_prob**gamma_focal) * (-(1 - out_prob + 1e-8).log())
            pos_cost = alpha_focal * ((1 - out_prob) ** gamma_focal) * (-(out_prob + 1e-8).log())
            cost_cls = pos_cost[:, gt_cls] - neg_cost[:, gt_cls]  # (n_cand, n_valid_gt)

            # --- L2 centroid cost ---
            pd_xy = pd_bboxes[b, cand_idx, :2].float()
            gt_xy = gt_bboxes[b, valid_gt_idx, :2].float()
            cost_xy = torch.cdist(pd_xy, gt_xy, p=2)  # (n_cand, n_valid_gt)

            # --- Log-space L1 ray cost: mean(|log(pred) - log(gt)|) ---
            pd_rays = pd_bboxes[b, cand_idx, 2:].float()
            gt_rays = gt_bboxes[b, valid_gt_idx, 2:].float()
            log_pd = pd_rays[:, None, :].clamp(min=1e-4).log()
            log_gt = gt_rays[None, :, :].clamp(min=1e-4).log()
            log_l1 = (log_pd - log_gt).abs().mean(dim=-1)  # (n_cand, n_valid_gt)
            cost_ray = log_l1

            # --- Normalize each cost term to [0, 1] for scale-independent weighting ---
            def _norm(c):
                c_min, c_max = c.min(), c.max()
                rng = c_max - c_min
                if rng < 1e-8:
                    return torch.zeros_like(c)
                return (c - c_min) / rng

            cost_cls_n = _norm(cost_cls)
            cost_xy_n = _norm(cost_xy)
            cost_ray_n = _norm(cost_ray)

            # --- Combined cost (all terms in [0, 1]) ---
            total = (
                self.cost_class * cost_cls_n + self.cost_centroid * cost_xy_n + self.cost_ray * cost_ray_n
            )  # (n_cand, n_valid_gt)

            # Ray similarity for overlaps (used by parent's normalisation)
            ray_sim = 1.0 / (1.0 + log_l1)

            pair_mask = mask_gt_bool[b][valid_gt_idx[:, None], cand_idx[None, :]]
            cost[b, valid_gt_idx[:, None], cand_idx[None, :]] = total.T * pair_mask.float()
            overlaps[b, valid_gt_idx[:, None], cand_idx[None, :]] = ray_sim.T * pair_mask.float()

        return cost, overlaps

    # -----------------------------------------------------------------
    # Override 2: get_box_metrics (one2many similarity from unified cost)
    # -----------------------------------------------------------------

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric from unified cost matrix.

        Calls _compute_cost_matrix to get the same cost used by Hungarian,
        then inverts to a similarity metric: align = 1 / (1 + cost).

        The parent's select_topk_candidates picks anchors with largest
        align_metric, which correctly prefers lowest-cost anchors.

        Args:
            pd_scores: Classification scores, shape (B, N_anchors, nc).
            pd_bboxes: Decoded polygon predictions, shape (B, N_anchors, 34).
            gt_labels: GT class labels, shape (B, N_max_gt, 1).
            gt_bboxes: GT polygon targets, shape (B, N_max_gt, 34).
            mask_gt: Valid GT mask, shape (B, N_max_gt, 1).

        Returns:
            align_metric: (B, N_max_gt, N_anchors).
            overlaps: (B, N_max_gt, N_anchors).
        """
        mask_gt_bool = mask_gt.bool()
        cost, overlaps = self._compute_cost_matrix(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt_bool)

        if self.align_threshold > 0:
            overlaps = overlaps * (overlaps >= self.align_threshold).float()

        align_metric = 1.0 / (1.0 + cost)
        return align_metric, overlaps

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

    Uses the same _compute_cost_matrix as the one2many branch for identical
    cost formulation. Solves with scipy.linear_sum_assignment for globally
    optimal 1:1 matching.

    Inherits select_candidates_in_gts and _compute_cost_matrix from
    RayCastAssigner. Only overrides _forward to replace the selection
    mechanism with Hungarian matching.
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
        self._cost_class_init = cost_class
        self._cost_centroid_init = cost_centroid
        self._cost_ray_init = cost_ray
        super().__init__(
            cost_class=cost_class,
            cost_centroid=cost_centroid,
            cost_ray=cost_ray,
            **kwargs,
        )

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Hungarian matching assignment with unified cost matrix.

        Uses the same _compute_cost_matrix as the one2many branch, ensuring
        both branches rank candidates identically. Solves with
        scipy.linear_sum_assignment for globally optimal 1:1 matching.

        Args:
            pd_scores: Predicted classification scores, shape (B, N_anchors, nc).
            pd_bboxes: Predicted polygon targets, shape (B, N_anchors, 34).
            anc_points: Anchor grid positions, shape (N_anchors, 2).
            gt_labels: GT class labels, shape (B, N_max_gt, 1).
            gt_bboxes: GT polygon targets, shape (B, N_max_gt, 34).
            mask_gt: Valid GT mask, shape (B, N_max_gt, 1).

        Returns:
            Tuple of (target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)

        bs = pd_scores.shape[0]
        na = pd_scores.shape[1]
        device = gt_bboxes.device

        target_labels = torch.full((bs, na), self.num_classes, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(bs, na, gt_bboxes.shape[-1], dtype=gt_bboxes.dtype, device=device)
        target_scores = torch.zeros_like(pd_scores)
        fg_mask = torch.zeros(bs, na, dtype=torch.bool, device=device)
        target_gt_idx = torch.zeros(bs, na, dtype=torch.long, device=device)

        mask_gt_bool = mask_gt.bool()

        # Use unified cost matrix
        cost_matrix, overlaps = self._compute_cost_matrix(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt_bool)

        for b in range(bs):
            valid_gt_mask = mask_gt_bool[b, :, 0]
            n_valid_gt = valid_gt_mask.sum().item()
            if n_valid_gt == 0:
                continue

            valid_gt_idx = valid_gt_mask.nonzero(as_tuple=False).squeeze(-1)

            candidate_mask = mask_in_gts[b].any(dim=0)
            cand_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
            n_cand = cand_idx.shape[0]
            if n_cand == 0:
                continue

            # Extract (n_valid_gt, n_cand) sub-matrix for Hungarian
            cost_np = cost_matrix[b, valid_gt_idx][:, cand_idx].cpu().numpy()

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
