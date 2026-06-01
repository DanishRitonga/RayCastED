import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HybridHungarianMatcher:
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_centroid: float = 1.0,
        cost_radial: float = 1.0,
        cost_inner: float = 9999.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        n_rays: int = 64,
    ):
        self.cost_class = cost_class
        self.cost_centroid = cost_centroid
        self.cost_radial = cost_radial
        self.cost_inner = cost_inner
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.n_rays = n_rays

    def _focal_cost(self, prob: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        prob = prob.unsqueeze(1)
        targets = targets.unsqueeze(0)
        neg_cost = alpha * (1 - prob) ** gamma * prob.log()
        pos_cost = (1 - alpha) * prob**gamma * (1 - prob).log()
        cost = torch.where(targets == 1, -pos_cost, -neg_cost)
        return cost.mean(dim=-1)

    @torch.no_grad()
    def __call__(
        self, outputs: dict, targets: list[dict], crop_size: int = 256
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        B = outputs['pred_logits'].shape[0]
        device = outputs['pred_logits'].device
        indices = []

        for b in range(B):
            tgt = targets[b]
            tgt_labels = tgt.get('labels', torch.zeros(0, dtype=torch.long, device=device))
            num_tgt = len(tgt_labels)

            if num_tgt == 0:
                indices.append(
                    (
                        torch.zeros(0, dtype=torch.long, device=device),
                        torch.zeros(0, dtype=torch.long, device=device),
                    )
                )
                continue

            out_prob = outputs['pred_logits'][b].sigmoid()
            num_classes = out_prob.shape[-1]
            tgt_class = torch.zeros(num_tgt, num_classes, device=device)
            valid = tgt_labels < num_classes - 1
            tgt_class[valid, tgt_labels[valid]] = 1

            cost_matrix = self.cost_class * self._focal_cost(out_prob, tgt_class)

            if 'boxes' in tgt and tgt['boxes'].numel() > 0:
                tgt_boxes = tgt['boxes'].to(device=device, dtype=torch.float32)

                # Cost 2: centroid L1 in normalized [0,1] space
                out_points = outputs['pred_points'][b, :, :2].float()
                tgt_points = tgt_boxes[:, :2]
                cost_matrix = cost_matrix + self.cost_centroid * torch.cdist(out_points, tgt_points, p=1).to(
                    cost_matrix.dtype
                )

                # Cost 3: radial distances in log-pixel space
                if self.cost_radial > 0 and tgt_boxes.shape[1] >= 2 + self.n_rays:
                    pred_radial = outputs['pred_radial'][b].float()  # [Q, n_rays]
                    gt_rays_norm = tgt_boxes[:, 2:]  # [T, n_rays] normalized [0,1]
                    gt_rays_pixel_log = torch.log(gt_rays_norm * crop_size + 1e-7)
                    cost_radial_mtx = torch.cdist(pred_radial, gt_rays_pixel_log, p=1) / self.n_rays
                    cost_matrix = cost_matrix + self.cost_radial * cost_radial_mtx.to(cost_matrix.dtype)

                # Cost 4: inner — heavy penalty if query centroid is outside nucleus boundary
                if self.cost_inner > 0 and tgt_boxes.shape[1] >= 2 + self.n_rays:
                    dist_mtx = torch.cdist(out_points, tgt_points, p=2) * crop_size
                    gt_max_radius = tgt_boxes[:, 2:].max(dim=-1).values * crop_size
                    is_outside = (dist_mtx > gt_max_radius.unsqueeze(0)).to(cost_matrix.dtype)
                    cost_matrix = cost_matrix + self.cost_inner * is_outside

            cost_np = cost_matrix.cpu().detach().numpy()
            cost_np = np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=-1e6)
            try:
                pred_idx, tgt_idx = linear_sum_assignment(cost_np)
            except ValueError:
                pred_idx, tgt_idx = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            indices.append(
                (
                    torch.as_tensor(pred_idx, dtype=torch.long, device=device),
                    torch.as_tensor(tgt_idx, dtype=torch.long, device=device),
                )
            )

        return indices


class HybridSetCriterion(nn.Module):
    def __init__(
        self,
        nc: int,
        matcher: HybridHungarianMatcher,
        weight_dict: dict | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        n_rays: int = 64,
        crop_size: int = 256,
    ):
        super().__init__()
        self.nc = nc
        self.matcher = matcher
        self.weight_dict = weight_dict or {
            'loss_ce': 1.0,
            'loss_centroid': 1.0,
            'loss_radial': 1.0,
        }
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.n_rays = n_rays
        self.crop_size = crop_size

    def _focal_loss(self, logits, targets, matched):
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        src_logits = logits
        num_classes = src_logits.shape[-1]
        device = src_logits.device

        tgt_classes = torch.full(src_logits.shape[:2], num_classes - 1, dtype=torch.int64, device=device)

        for b, (pred_idx, tgt_idx) in enumerate(matched):
            if len(tgt_idx) == 0:
                continue
            tgt_labels = targets[b].get('labels')
            if tgt_labels is None or len(tgt_labels) == 0:
                continue
            tgt_labels = tgt_labels.to(device=device)
            matched_labels = tgt_labels[tgt_idx]
            valid = matched_labels < num_classes - 1
            tgt_classes[b, pred_idx[valid]] = matched_labels[valid]

        tgt_one_hot = F.one_hot(tgt_classes, num_classes).type_as(src_logits)

        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, tgt_one_hot, reduction='none')
        p_t = prob * tgt_one_hot + (1 - prob) * (1 - tgt_one_hot)
        modulating = (1 - p_t) ** gamma
        alpha_weight = tgt_one_hot * alpha + (1 - tgt_one_hot) * (1 - alpha)
        loss = (alpha_weight * modulating * ce_loss).mean()
        return loss

    def _centroid_loss(self, points, targets, matched):
        total_loss = torch.tensor(0.0, device=points.device, dtype=points.dtype)
        count = 0
        for b, (pred_idx, tgt_idx) in enumerate(matched):
            if len(tgt_idx) == 0:
                continue
            tgt_boxes = targets[b].get('boxes')
            if tgt_boxes is None or tgt_boxes.numel() == 0:
                continue
            tgt_boxes = tgt_boxes.to(device=points.device, dtype=points.dtype)
            tgt_pts = tgt_boxes[tgt_idx, :2]
            pred_pts = points[b, pred_idx, :2]
            total_loss = total_loss + F.l1_loss(pred_pts, tgt_pts)
            count += 1
        return total_loss / max(count, 1)

    def _radial_loss(self, radial_log, points, targets, matched):
        total_loss = torch.tensor(0.0, device=radial_log.device, dtype=radial_log.dtype)
        count = 0
        for b, (pred_idx, tgt_idx) in enumerate(matched):
            if len(tgt_idx) == 0:
                continue
            tgt_boxes = targets[b].get('boxes')
            if tgt_boxes is None or tgt_boxes.numel() == 0:
                continue
            if tgt_boxes.shape[1] < 2 + self.n_rays:
                continue
            tgt_boxes = tgt_boxes.to(device=radial_log.device, dtype=torch.float32)
            gt_rays_norm = tgt_boxes[tgt_idx, 2:]  # [M, n_rays] normalized [0,1]
            gt_rays_pixel_log = torch.log(gt_rays_norm * self.crop_size + 1e-7)
            pred = radial_log[b, pred_idx]  # [M, n_rays] log-pixel

            # LSP-DETR interval loss: min=max=gt_rays for non-overlapping PanNuke
            loss_min = F.relu(gt_rays_pixel_log - pred)
            loss_max = F.relu(pred - gt_rays_pixel_log)
            item_loss = torch.max(loss_min, loss_max)
            total_loss = total_loss + item_loss.nanmean()
            count += 1
        return total_loss / max(count, 1)

    def forward(self, outputs, targets, crop_size=None):
        _crop_size = crop_size or self.crop_size
        matched = self.matcher(outputs, targets, crop_size=_crop_size)
        loss_dict = {
            'loss_ce': self._focal_loss(outputs['pred_logits'], targets, matched),
            'loss_centroid': self._centroid_loss(outputs['pred_points'], targets, matched),
            'loss_radial': self._radial_loss(outputs['pred_radial'], outputs['pred_points'], targets, matched),
        }
        total_loss = sum(loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict)

        if 'aux_outputs' in outputs:
            for aux in outputs['aux_outputs']:
                aux_matched = self.matcher(aux, targets, crop_size=_crop_size)
                total_loss = total_loss + self._focal_loss(
                    aux['pred_logits'], targets, aux_matched
                ) * self.weight_dict.get('loss_ce', 1.0)
                total_loss = total_loss + self._centroid_loss(
                    aux['pred_points'], targets, aux_matched
                ) * self.weight_dict.get('loss_centroid', 1.0)
                total_loss = total_loss + self._radial_loss(
                    aux['pred_radial'], aux['pred_points'], targets, aux_matched
                ) * self.weight_dict.get('loss_radial', 1.0)

        loss_dict['total'] = total_loss
        return loss_dict
