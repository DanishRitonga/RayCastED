from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .gpu_augment import GPUAugment
from .masks2centroids import masks2centroids


class LSPCollateFn:
    """CPU-only collate: stacks images, pads masks. GPU work is done in the training loop."""

    def __init__(self, n_rays: int = 64, allow_overlaps: bool = True) -> None:
        self.n_rays = n_rays
        self.allow_overlaps = allow_overlaps

    def __call__(self, batch: list[tuple[Tensor, dict]]) -> dict[str, Tensor | list[dict]]:
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]

        N_max = max(len(t['masks']) for t in targets) if targets else 0
        B, C, H, W = images.shape

        padded_masks = torch.zeros(B, N_max, H, W, dtype=torch.float32)
        for i, t in enumerate(targets):
            n = len(t['masks'])
            if n > 0:
                padded_masks[i, :n] = t['masks']

        return {
            'img': images,
            'targets': targets,
            'padded_masks': padded_masks,
        }


def gpu_prepare_batch(
    batch: dict,
    augment: GPUAugment | None = None,
    n_rays: int = 64,
    allow_overlaps: bool = True,
) -> dict:
    """Apply GPU augment, star_distances, centroids. Called in main process only."""
    images = batch['img']
    targets = batch['targets']
    padded_masks = batch['padded_masks']

    device = torch.device('cuda') if torch.cuda.is_available() else images.device
    B, C, H, W = images.shape

    images = images.to(device)
    padded_masks = padded_masks.to(device)

    if augment is not None:
        images, padded_masks = augment(images, padded_masks)
        images = images.clamp(0, 255)
        padded_masks = padded_masks.clamp(0, 1)
        padded_masks = (padded_masks > 0.5).float()

    for i, t in enumerate(targets):
        n = len(t['labels'])
        if n == 0:
            continue
        mask_sum = padded_masks[i, :n].sum(dim=(1, 2))
        keep = mask_sum > 0
        if not keep.all():
            t['labels'] = t['labels'][keep.cpu()]
            t['masks'] = padded_masks[i, :n][keep]

    N_max = max(len(t['labels']) for t in targets) if targets else 0
    new_padded = torch.zeros(B, N_max, H, W, dtype=torch.float32, device=device)
    for i, t in enumerate(targets):
        n = len(t['labels'])
        if n > 0:
            new_padded[i, :n] = t['masks']
    padded_masks = new_padded

    from .star_distances_triton import star_distances_triton

    for i, t in enumerate(targets):
        n = len(t['labels'])
        if n == 0:
            t['radial_distances'] = torch.zeros(2, n_rays, H, W, device=device)
            t['centroids'] = torch.empty(0, 2, device=device)
            t['labels'] = t['labels'].to(device)
            continue

        masks_i = padded_masks[i, :n]
        lower, upper = star_distances_triton(masks_i, n_rays)
        if not allow_overlaps:
            upper = lower
        t['radial_distances'] = torch.stack([lower, upper], dim=0)
        t['centroids'] = masks2centroids(masks_i, normalize=True)
        t['labels'] = t['labels'].to(device)

    for i, t in enumerate(targets):
        n = len(t['labels'])
        if n == 0:
            t['boxes'] = torch.empty(0, 2 + n_rays, device=device)
            continue
        upper_map = t['radial_distances'][1]  # (n_rays, H, W) in pixel space
        grid = t['centroids'] * 2 - 1  # (n, 2) → [-1, 1]
        grid = grid.view(1, 1, n, 2)
        sampled = F.grid_sample(
            upper_map.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True,
        )  # (1, n_rays, 1, n)
        rays_norm = sampled.squeeze(0).squeeze(1).T / H  # (n, n_rays) normalized [0, 1]
        t['boxes'] = torch.cat([t['centroids'], rays_norm], dim=-1)  # (n, 2 + n_rays)

    return {'img': images, 'targets': targets}
