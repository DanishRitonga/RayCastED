from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch import Tensor


def _decode_pil_to_numpy(batch):
    """Dataset transform: pre-decode PIL images/masks to numpy arrays."""
    import numpy as np

    batch['image'] = [np.array(img, dtype=np.float32) for img in batch['image']]
    batch['instances'] = [[np.array(mask, dtype=np.float32) for mask in masks] for masks in batch['instances']]
    return batch


def load_pannuke_folds(data_paths: list[str], folds: list[int] | None = None) -> Dataset:
    """Load PanNuke parquet files, optionally filtering by fold."""
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


class LSPDataset(torch.utils.data.Dataset[tuple[Tensor, dict]]):
    """PanNuke dataset for LSP-DETR training with GPU augmentation.

    Loads raw PIL images + masks, converts to tensors without any transforms.
    Augmentation is applied at batch level in the collate_fn via GPUAugment.
    star_distances + centroids are computed after augmentation in the collate_fn.
    """

    def __init__(
        self,
        data: Dataset,
        n_rays: int = 64,
    ) -> None:
        data.set_transform(_decode_pil_to_numpy, columns=['image', 'instances'], output_all_columns=True)
        self.data = data
        self.n_rays = n_rays

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        sample = self.data[idx]
        image = torch.from_numpy(sample['image']).permute(2, 0, 1)
        masks = (
            torch.from_numpy(np.stack(sample['instances'], axis=0))
            if sample['instances']
            else torch.empty((0, 256, 256), dtype=torch.float32)
        )
        labels = torch.tensor(sample['categories'], dtype=torch.long)
        tissue_id = sample['tissue']

        return image, {
            'masks': masks,
            'labels': labels,
            'tissue': tissue_id,
        }
