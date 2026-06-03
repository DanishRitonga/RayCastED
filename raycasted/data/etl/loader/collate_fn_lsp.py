from __future__ import annotations

import torch
from torch import Tensor

from .gpu_augment import GPUAugment
from .masks2centroids import masks2centroids


class LSPCollateFn:
    """Collate function with batch-level GPU augmentation.

    Applies GPUAugment to stacked batch, then computes star_distances (CPU
    round-trip via Rust) and centroids on the transformed masks.
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
        device = images.device

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

        # Batch star_distances: one call for entire batch
        from stardist import star_distances
        import numpy as np

        counts = [len(t['labels']) for t in targets]
        has_any = [n for n in counts if n > 0]
        if has_any:
            all_masks = torch.cat(
                [padded_masks[i, : counts[i]].cpu().numpy().astype(np.uint8) for i in range(B) if counts[i] > 0],
                axis=0,
            )
            lower_bound, upper_bound = star_distances(all_masks, self.n_rays)
            if not self.allow_overlaps:
                upper_bound = lower_bound
            all_radial = np.stack((lower_bound, upper_bound), axis=0)

            offset = 0
            for i, t in enumerate(targets):
                n = counts[i]
                if n == 0:
                    t['radial_distances'] = torch.zeros(2, self.n_rays, H, W)
                    t['centroids'] = torch.empty(0, 2)
                else:
                    t['radial_distances'] = torch.from_numpy(all_radial[:, offset : offset + n]).float()
                    t['centroids'] = masks2centroids(padded_masks[i, :n], normalize=True)
                    offset += n
                t['labels'] = t['labels'].to(device)
        else:
            for t in targets:
                t['radial_distances'] = torch.zeros(2, self.n_rays, H, W)
                t['centroids'] = torch.empty(0, 2)
                t['labels'] = t['labels'].to(device)

        return {
            'img': images,
            'targets': targets,
        }
