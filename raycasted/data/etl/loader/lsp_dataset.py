from __future__ import annotations

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from datasets import Dataset, concatenate_datasets
from stardist import star_distances
from torch import Tensor

from .masks2centroids import masks2centroids


def load_pannuke_folds(data_paths: list[str], folds: list[int] | None = None) -> Dataset:
    """Load PanNuke parquet files, optionally filtering by fold.

    Each parquet file must have columns: image (RGB bytes), instances (list of mask bytes),
    categories (list of int labels), tissue (int). Fold is extracted from the
    filename pattern 'fold{N}-...parquet'.
    """
    import re
    from pathlib import Path

    datasets_list = []
    for p in data_paths:
        ds = Dataset.from_parquet(p)
        match = re.search(r'fold(\d+)', Path(p).name)
        if match is None:
            raise ValueError(f'Cannot extract fold from filename: {p}')
        fold_num = int(match[1])
        if folds is None or fold_num in folds:
            datasets_list.append(ds)
    return concatenate_datasets(datasets_list) if datasets_list else Dataset.from_dict({})


class LSPDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """PanNuke dataset matching LSP-DETR exactly.

    Loads PanNuke parquet files, applies albumentations transforms to image+masks,
    computes radial distance maps via star_distances (Rust), and returns
    normalized tensors for training.
    """

    def __init__(
        self,
        data: Dataset,
        transforms: A.Compose | None = None,
        n_rays: int = 64,
        allow_overlaps: bool = True,
    ) -> None:
        self.data = data
        self.transforms = transforms or A.Compose([])
        self.n_rays = n_rays
        self.allow_overlaps = allow_overlaps
        self._to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _pil_masks_to_np(masks: list) -> np.ndarray:
        """Convert list of PIL mask images to (H, W, N) uint8 ndarray."""
        if not masks:
            return np.empty((256, 256, 0), dtype=np.uint8)
        return np.stack([np.array(m, dtype=np.uint8) for m in masks], axis=-1)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        sample = self.data[idx]
        image = np.array(sample['image'], dtype=np.uint8)
        masks = self._pil_masks_to_np(sample['instances'])
        labels = np.array(sample['categories'], dtype=np.int64)
        tissue_id = sample['tissue']

        transformed = self.transforms(image=image, mask=masks)
        image = transformed['image']
        masks = transformed['mask'].transpose(2, 0, 1)

        keep = masks.any(axis=(1, 2))
        masks = masks[keep]
        labels = labels[keep]

        lower_bound, upper_bound = star_distances(masks, self.n_rays)
        if not self.allow_overlaps:
            upper_bound = lower_bound
        radial_distances = np.stack((lower_bound, upper_bound), axis=0)

        image = self._to_tensor(image=image)['image']
        masks = torch.from_numpy(masks)

        return image, {
            'masks': masks,
            'labels': torch.from_numpy(labels).long(),
            'radial_distances': torch.from_numpy(radial_distances),
            'centroids': masks2centroids(masks, normalize=True),
            'tissue': tissue_id,
        }
