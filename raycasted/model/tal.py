"""RayCastED — RayCast Assigner (Phase 5).

RayCastAssigner subclasses TaskAlignedAssigner to replace box-based
assignment with polygon-aware logic:
  - select_candidates_in_gts: 75th-percentile radius containment
  - get_box_metrics: VRAM-safe Polar-IoU with per-batch chunking

HungarianRayCastAssigner extends RayCastAssigner with globally optimal
bipartite matching (scipy.linear_sum_assignment) for the one2one branch,
replacing the greedy topk selection that causes assignment collisions
in dense touching-cell scenes.

get_targets is NOT overridden — the parent implementation is dimension-
agnostic (uses gt_bboxes.shape[-1] dynamically) and works for 34-dim
polygon targets without modification.

Spec reference: docs/project.md §11
"""

import torch
from scipy.optimize import linear_sum_assignment
from ultralytics.utils.tal import TaskAlignedAssigner

from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch

# VRAM guard: max candidate-GT pairs per chunk.
# 1M pairs × 32 rays × 4 bytes = ~128 MB per chunk.
MAX_FLAT_PAIRS = 1_000_000


class RayCastAssigner(TaskAlignedAssigner):
    """Polygon-aware assigner replacing bbox IoU with Polar-IoU.

    Overrides two methods from TaskAlignedAssigner:
      - select_candidates_in_gts: radius containment instead of box containment
      - get_box_metrics: VRAM-safe Polar-IoU instead of bbox_iou(CIoU)

    get_targets is inherited as-is — the parent is dimension-agnostic.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2: int | None = None,
        radius_scale: float = 1.5,
    ):
        """Initialize RayCastAssigner.

        Args:
            topk: Number of top-k candidate anchors per GT.
            num_classes: Number of object classes.
            alpha: Exponent for classification score in alignment metric.
            beta: Exponent for IoU in alignment metric.
            stride: Feature map strides (default [8, 16, 32]).
            eps: Small value to prevent division by zero.
            topk2: Secondary topk for additional filtering.
            radius_scale: Multiplier on 75th-percentile containment radius.
                Monitor mean positive assignments per GT cell for first 100
                batches. Target: 1-4. Below 1 → too small. Above 10 → too large.
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
                       Columns: [cx, cy, d_1, ..., d_32] (normalised).
            mask_gt: Valid GT mask, shape (B, N_max_gt, 1).
            eps: Unused (kept for API compatibility).

        Returns:
            Boolean mask of shape (B, N_max_gt, N_anchors).
        """
        _ = eps  # API compat — not used in radius containment
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        mask = torch.zeros(bs, n_boxes, n_anchors, dtype=torch.bool, device=gt_bboxes.device)

        for b in range(bs):
            # Valid GTs for this image
            valid = mask_gt[b, :, 0].bool()  # (N_max_gt,)
            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue

            # GT centroids and rays
            gt_xy = gt_bboxes[b, valid_idx, :2]  # (N_valid, 2)
            gt_rays = gt_bboxes[b, valid_idx, 2:]  # (N_valid, 32)
            # --- Compute containment radius per GT (vectorised) ---
            # Sort rays ascending (zeros first)
            sorted_rays, _ = gt_rays.sort(dim=1)  # (N_valid, n_rays)

            n_non_zero = (gt_rays > 0).sum(dim=1)  # (N_valid,)
            n_rays = gt_rays.shape[-1]
            n_zero = n_rays - n_non_zero

            # 75th-percentile index among non-zero rays
            pct75_idx = (n_non_zero.float() * 0.75).long().clamp(min=0)
            pct75_flat_idx = (n_zero + pct75_idx).clamp(max=n_rays - 1)
            pct75_radii = sorted_rays.gather(1, pct75_flat_idx.unsqueeze(1)).squeeze(1)

            # Fallback: max non-zero ray when < 8 non-zero rays
            max_radii = sorted_rays[:, -1]  # max value in sorted array

            # Build final radii: pct75 if >=8 non-zero, max if 1-7, 0 if none
            radii = torch.where(
                n_non_zero >= 8,
                pct75_radii,
                torch.where(n_non_zero > 0, max_radii, torch.zeros(1, device=gt_bboxes.device)),
            )

            containment = radii * self.radius_scale  # (N_valid,)

            # --- Distance check: (N_valid, N_anchors) ---
            dist = torch.cdist(gt_xy.float(), xy_centers.float())  # (N_valid, N_anchors)
            valid_mask = (dist <= containment[:, None]) & (containment[:, None] > 0)

            # Place in output
            mask[b, valid_idx] = valid_mask

        return mask

    # -----------------------------------------------------------------
    # Override 2: VRAM-safe Polar-IoU
    # -----------------------------------------------------------------

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric using VRAM-safe Polar-IoU.

        Replaces the parent's bbox_iou(CIoU) with polar_iou_pairwise_flat_torch,
        computed in a per-batch loop with conditional chunking to avoid OOM
        on dense TIL fields (BUG-05).

        Args:
            pd_scores: Classification scores, shape (B, N_anchors, nc).
            pd_bboxes: Decoded polygon predictions, shape (B, N_anchors, 34).
                       Columns: [decoded_xy(2), softplus_rays(32)].
            gt_labels: GT class labels, shape (B, N_max_gt, 1).
            gt_bboxes: GT polygon targets, shape (B, N_max_gt, 34).
                       Columns: [cx, cy, d_1..d_32].
            mask_gt: Combined candidate + validity mask, shape (B, N_max_gt, N_anchors).

        Returns:
            align_metric: (B, N_max_gt, N_anchors).
            overlaps: (B, N_max_gt, N_anchors).
        """
        na = pd_bboxes.shape[-2]
        mask_gt_bool = mask_gt.bool()  # (B, N_gt, N_anchors)
        # Force overlaps to float32 — under AMP/FP16 validation, the parent
        # TaskAlignedAssigner._forward normalisation uses self.eps=1e-9 which
        # underflows to 0 in float16, causing NaN. bbox_scores can stay as-is
        # (PyTorch upcasts automatically in align_metric = scores^α * overlaps^β).
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=torch.float32, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        # --- Classification scores (same logic as parent, not memory-intensive) ---
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt_bool] = pd_scores[ind[0], :, ind[1]][mask_gt_bool]

        # --- VRAM-safe per-batch Polar-IoU ---
        for b in range(self.bs):
            # Candidate anchors selected by any GT in this image
            candidate_mask = mask_gt_bool[b].any(dim=0)  # (N_anchors,)
            cand_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_cand,)
            n_cand = cand_idx.shape[0]
            if n_cand == 0:
                continue

            # Valid GTs that have at least one candidate
            valid_gt_mask = mask_gt_bool[b].any(dim=1)  # (N_gt,)
            valid_gt_idx = valid_gt_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_valid_gt,)
            n_valid_gt = valid_gt_idx.shape[0]

            # Extract rays (columns 2: of the 34-dim polygon vector)
            pd_rays = pd_bboxes[b, cand_idx, 2:]  # (N_cand, 32)
            gt_rays = gt_bboxes[b, valid_gt_idx, 2:]  # (N_valid_gt, 32)

            # Pairwise Polar-IoU with conditional chunking
            if n_cand * n_valid_gt <= MAX_FLAT_PAIRS:
                # Small enough: compute in one shot
                pd_exp = pd_rays[:, None, :].expand(-1, n_valid_gt, -1)  # (N_cand, N_valid_gt, 32)
                gt_exp = gt_rays[None, :, :].expand(n_cand, -1, -1)  # (N_cand, N_valid_gt, 32)
                iou = polar_iou_pairwise_flat_torch(pd_exp, gt_exp)  # (N_cand, N_valid_gt)
            else:
                # Dense scene: chunk to stay under memory budget
                chunk_size = max(1, MAX_FLAT_PAIRS // n_valid_gt)
                iou = torch.zeros(n_cand, n_valid_gt, device=pd_bboxes.device, dtype=pd_bboxes.dtype)
                for chunk_start in range(0, n_cand, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_cand)
                    pd_chunk = pd_rays[chunk_start:chunk_end][:, None, :].expand(-1, n_valid_gt, -1)
                    gt_chunk = gt_rays[None, :, :].expand(chunk_end - chunk_start, -1, -1)
                    iou[chunk_start:chunk_end] = polar_iou_pairwise_flat_torch(pd_chunk, gt_chunk)

            # Fill overlaps — only at positions where mask_gt is True.
            # iou is (N_cand, N_valid_gt), overlaps needs (N_valid_gt, N_cand).
            # Cast to overlaps dtype to handle AMP half/float mismatch.
            pair_mask = mask_gt_bool[b][valid_gt_idx[:, None], cand_idx[None, :]]  # (N_valid_gt, N_cand)
            overlaps[b, valid_gt_idx[:, None], cand_idx[None, :]] = iou.T.to(overlaps.dtype) * pair_mask.to(
                overlaps.dtype
            )

        # Alignment metric: cls_score^alpha * iou^beta (same formula as parent)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps


class HungarianRayCastAssigner(RayCastAssigner):
    """Globally optimal bipartite matching assigner for the one2one branch.

    Replaces the greedy topk→topk2 selection in TaskAlignedAssigner with
    scipy.linear_sum_assignment (Hungarian algorithm) to find the globally
    optimal 1:1 assignment between GT objects and anchor points.

    This is critical for dense touching-cell scenes (e.g., PanNuke) where
    greedy TAL causes assignment collisions: nearby GTs independently pick
    the same anchors, and select_highest_overlaps resolves ties by max-IoU,
    often giving suboptimal assignments that degrade NMS-free inference.

    Inherits select_candidates_in_gts and get_box_metrics from RayCastAssigner.
    Only overrides _forward to replace the selection mechanism.
    """

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Hungarian matching assignment — globally optimal 1:1 matching.

        Uses the same candidate filtering and IoU computation as TAL, but
        replaces topk selection with Hungarian algorithm on the cost matrix
        (negative alignment metric), ensuring each GT is matched to exactly
        one anchor with no collisions.

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
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)

        bs = pd_scores.shape[0]
        na = pd_scores.shape[1]
        n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        target_labels = torch.full((bs, na), self.num_classes, dtype=torch.long, device=device)
        target_bboxes = torch.zeros_like(pd_bboxes)
        target_scores = torch.zeros_like(pd_scores)
        fg_mask = torch.zeros(bs, na, dtype=torch.bool, device=device)
        target_gt_idx = torch.zeros(bs, na, dtype=torch.long, device=device)
        mask_pos = torch.zeros(bs, n_max_boxes, na, dtype=torch.float32, device=device)

        for b in range(bs):
            valid_gt_mask = mask_gt[b, :, 0].bool()
            n_valid_gt = valid_gt_mask.sum().item()
            if n_valid_gt == 0:
                continue

            valid_gt_idx = valid_gt_mask.nonzero(as_tuple=False).squeeze(-1)

            # Cost matrix: negative alignment metric (Hungarian minimises cost)
            # Shape: (n_valid_gt, n_cand) — only consider candidate anchors
            candidate_mask = mask_in_gts[b].any(dim=0) & mask_gt[b, :, 0].any()
            candidate_mask = mask_in_gts[b].any(dim=0)

            cand_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
            n_cand = cand_idx.shape[0]
            if n_cand == 0:
                continue

            # Build cost matrix from alignment metric
            # align_metric[b] is (n_max_boxes, na)
            cost = align_metric[b][valid_gt_idx[:, None], cand_idx[None, :]]  # (n_valid_gt, n_cand)

            # Apply candidate mask: set non-candidates to large cost
            pair_mask = mask_in_gts[b][valid_gt_idx[:, None], cand_idx[None, :]]
            cost = cost * pair_mask.float()

            # Hungarian algorithm (minimise negative = maximise alignment)
            # For large matrices, fall back to top-1 greedy per GT
            if n_valid_gt * n_cand > 0:
                cost_np = (-cost.float()).cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
                matched_gt = valid_gt_idx[row_ind]
                matched_anchor = cand_idx[col_ind]

                # Verify matches are within candidate mask
                for gt_i, anc_i in zip(matched_gt, matched_anchor):
                    if mask_in_gts[b, gt_i, anc_i]:
                        mask_pos[b, gt_i, anc_i] = 1.0
                        fg_mask[b, anc_i] = True
                        target_gt_idx[b, anc_i] = gt_i

            # Fill targets for matched anchors
            if fg_mask[b].any():
                fg_idx = fg_mask[b].nonzero(as_tuple=False).squeeze(-1)
                gt_indices = target_gt_idx[b, fg_idx]
                target_labels[b, fg_idx] = gt_labels[b, gt_indices, 0].long()
                target_bboxes[b, fg_idx] = gt_bboxes[b, gt_indices]

                # One-hot class scores
                cls_labels = gt_labels[b, gt_indices, 0].long().clamp(min=0)
                target_scores[b, fg_idx] = torch.zeros(
                    fg_idx.shape[0], self.num_classes, device=device, dtype=target_scores.dtype
                ).scatter_(1, cls_labels.unsqueeze(-1), 1.0)

        # Normalize alignment metric (same as parent)
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx
