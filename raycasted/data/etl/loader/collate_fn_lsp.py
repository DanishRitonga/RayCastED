from __future__ import annotations

import torch
from torch import Tensor

from .gpu_augment import GPUAugment
from .masks2centroids import masks2centroids


class LSPCollateFn:
    """Collate function with batch-level GPU augmentation.

    Applies GPUAugment to stacked batch, then computes star_distances via
    Triton kernel (GPU-resident, zero CPU round-trip) and centroids on transformed masks.
    """

    def __init__(self, augment: GPUAugment | None = None, n_rays: int = 64, allow_overlaps: bool = True) -> None:
        self.augment = augment
        self.n_rays = n_rays
        self.allow_overlaps = allow_overlaps

    def __call__(self, batch: list[tuple[Tensor, dict]]) -> dict[str, Tensor | list[dict]]:
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]

        N_max = max(len(t['masks']) for t in targets) if targets else 0
        B, C, H, W = images.shape

        device = torch.device('cuda') if torch.cuda.is_available() else images.device

        padded_masks = torch.zeros(B, N_max, H, W, dtype=torch.float32)
        for i, t in enumerate(targets):
            n = len(t['masks'])
            if n > 0:
                padded_masks[i, :n] = t['masks']

        images = images.to(device)
        padded_masks = padded_masks.to(device)

        if self.augment is not None:
            images, padded_masks = self.augment(images, padded_masks)
            images = images.clamp(0, 255)
            padded_masks = padded_masks.clamp(0, 1)
            padded_masks = (padded_masks > 0.5).float()

        # GPU star_distances: one call per image in the batch
        from .star_distances_triton import star_distances_triton

        for i, t in enumerate(targets):
            n = len(t['labels'])
            if n == 0:
                t['radial_distances'] = torch.zeros(2, self.n_rays, H, W, device=device)
                t['centroids'] = torch.empty(0, 2, device=device)
                t['labels'] = t['labels'].to(device)
                continue

            masks_i = padded_masks[i, :n]
            lower, upper = star_distances_triton(masks_i, self.n_rays)
            if not self.allow_overlaps:
                upper = lower
            t['radial_distances'] = torch.stack([lower, upper], dim=0)
            t['centroids'] = masks2centroids(masks_i, normalize=True)
            t['labels'] = t['labels'].to(device)

        return {
            'img': images,
            'targets': targets,
        }
