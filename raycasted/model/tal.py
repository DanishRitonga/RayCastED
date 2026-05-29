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
  one2one:   topk2=1 for strict 1:1 matching (NMS-free inference)

Polar-IoU in the one2many branch provides absolute geometric discrimination
critical for dense touching-cell scenes: even a 1-pixel boundary overshoot
between adjacent nuclei is harshly penalised by β=6 exponentiation,
preventing anchor assignment collisions.

get_targets is NOT overridden — the parent implementation is dimension-
agnostic (uses gt_bboxes.shape[-1] dynamically) and works for any
polygon dimensionality (2 + N_RAYS) without modification.
"""

import torch
from ultralytics.utils.tal import TaskAlignedAssigner


def _nwd_similarity_torch(pd_rays, gt_rays, pd_centroids, gt_centroids, c_val=0.001):
    """Normalised Wasserstein Distance similarity between star-convex polygons.

    Converts rays → vertices, fits a 2D Gaussian to each polygon, then computes
    the Wasserstein-2 distance between the two Gaussians.

    W₂² = ||μ₁−μ₂||² + Tr(Σ₁ + Σ₂ − 2(Σ₁^½ Σ₂ Σ₁^½)^½)
    NWD = exp(−W₂² / C)   in [0, 1]

    Unlike pIoU (sharp cliff for small objects), NWD drops smoothly for
    misaligned small nuclei — a 4px nucleus 2px off gets pIoU≈0.5 but NWD≈0.8.

    Args:
        pd_rays:      (N, n_rays) predicted rays in normalised coords.
        gt_rays:      (N, n_rays) ground-truth rays in normalised coords.
        pd_centroids: (N, 2) predicted centroids in normalised coords.
        gt_centroids: (N, 2) GT centroids in normalised coords.
        c_val:        Normalisation constant (smaller = sharper drop).

    Returns:
        nwd: (N,) similarity scores in [0, 1].
    """
    import raycasted.data.etl.utils.constants as _const

    n_rays = pd_rays.shape[-1]
    device = pd_rays.device
    dtype = torch.float32

    cos = torch.tensor(_const.RAY_COS[:n_rays], device=device, dtype=dtype)
    sin = torch.tensor(_const.RAY_SIN[:n_rays], device=device, dtype=dtype)

    pd_rays_f = pd_rays.float()
    gt_rays_f = gt_rays.float()
    pd_centroids_f = pd_centroids.float()
    gt_centroids_f = gt_centroids.float()

    # --- Rays → vertices ---
    pd_vx = pd_centroids_f[:, 0:1] + pd_rays_f * cos  # (N, n_rays)
    pd_vy = pd_centroids_f[:, 1:2] + pd_rays_f * sin
    pd_vertices = torch.stack([pd_vx, pd_vy], dim=-1)  # (N, n_rays, 2)

    gt_vx = gt_centroids_f[:, 0:1] + gt_rays_f * cos
    gt_vy = gt_centroids_f[:, 1:2] + gt_rays_f * sin
    gt_vertices = torch.stack([gt_vx, gt_vy], dim=-1)

    # --- Covariance matrices from vertices relative to centroids ---
    pd_centered = pd_vertices - pd_centroids_f.unsqueeze(1)  # (N, n_rays, 2)
    gt_centered = gt_vertices - gt_centroids_f.unsqueeze(1)
    pd_cov = pd_centered.transpose(-2, -1) @ pd_centered / n_rays  # (N, 2, 2)
    gt_cov = gt_centered.transpose(-2, -1) @ gt_centered / n_rays

    # --- Wasserstein distance ---
    mu_diff_sq = ((pd_centroids_f - gt_centroids_f) ** 2).sum(-1)  # (N,)

    tr_pd = pd_cov.diagonal(dim1=-2, dim2=-1).sum(-1)  # (N,)
    tr_gt = gt_cov.diagonal(dim1=-2, dim2=-1).sum(-1)

    eigvals_pd, eigvecs_pd = torch.linalg.eigh(pd_cov)
    sqrt_eigvals_pd = eigvals_pd.clamp(min=0).sqrt()
    sqrt_pd = eigvecs_pd @ torch.diag_embed(sqrt_eigvals_pd) @ eigvecs_pd.transpose(-2, -1)

    b = sqrt_pd @ gt_cov @ sqrt_pd
    eigvals_b, _ = torch.linalg.eigh(b)
    eigvals_b = eigvals_b.clamp(min=0)
    tr_sqrt_b = eigvals_b.sqrt().sum(-1)  # (N,)

    w2_sq = (mu_diff_sq + tr_pd + tr_gt - 2.0 * tr_sqrt_b).clamp(min=0.0)
    nwd = torch.exp(-w2_sq / max(c_val, 1e-8))

    return nwd.clamp(0.0, 1.0).to(pd_rays.dtype)


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
        use_nwd=False,
        nwd_c=0.001,
    ):
        """Initialize RayCastAssigner.

        Args:
            topk: Number of top-k candidate anchors per GT.
            num_classes: Number of object classes.
            alpha: Exponent on cls_score in alignment metric.
            beta: Exponent on Polar-IoU or NWD in alignment metric.
            stride: Feature map strides (default [8, 16, 32]).
            eps: Small value to prevent division by zero.
            topk2: Secondary topk for additional filtering.
            radius_scale: Gaussian sigma multiplier. sigma = radius_scale * R75.
            align_threshold: Minimum overlap proxy below which anchors are zeroed.
            prefilter_k: Number of nearest anchors to prefilter. Default: auto.
            use_nwd: If True, use NWD similarity instead of PolarIoU.
            nwd_c: NWD normalisation constant. Smaller = sharper drop.
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
        # Number of nearest anchors to prefilter for PolarIoU / NWD.
        # Must be > topk for proper ranking. Default: max(topk*5, 100).
        self.prefilter_k = prefilter_k if prefilter_k > 0 else max(topk * 5, 100)
        self.use_nwd = use_nwd
        self.nwd_c = nwd_c

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

            bs_idx = torch.arange(self.bs, device=topk_idx.device)
            bs_idx = bs_idx.view(-1, 1, 1).expand(-1, self.n_max_boxes, k_prefilt)
            pd_block = pd_bboxes[bs_idx, topk_idx, 2:]
            gt_rays = gt_bboxes[:, :, 2:].unsqueeze(2).expand(-1, -1, k_prefilt, -1)

            if self.use_nwd:
                pd_centroids = pd_bboxes[bs_idx, topk_idx, :2]
                gt_centroids = gt_bboxes[:, :, :2].unsqueeze(2).expand(-1, -1, k_prefilt, -1)
                nwd_block = _nwd_similarity_torch(
                    pd_block.reshape(-1, n_rays),
                    gt_rays.reshape(-1, n_rays),
                    pd_centroids.reshape(-1, 2),
                    gt_centroids.reshape(-1, 2),
                    c_val=self.nwd_c,
                ).reshape(self.bs, self.n_max_boxes, k_prefilt)
                overlaps.scatter_(2, topk_idx, nwd_block.to(overlaps.dtype))
            else:
                from raycasted.data.etl.ops.iou import polar_iou_torch

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
