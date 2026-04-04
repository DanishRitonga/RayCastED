"""RayCastED — RayCastTileDataset and collate_fn.

Reads tiled .npz files from TransformOrchestrator output, applies online
augmentation and random cropping, normalises to [0, 1], and emits tensors
ready for training.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..ops.augment import flip_horizontal, flip_vertical, rotate_90
from ..ops.filter import filter_and_clip_annotations
from ..utils.constants import CX_IDX, CY_IDX, RAY_END_IDX, RAY_START_IDX


class RayCastTileDataset(Dataset):
    """PyTorch dataset for raycast polygon tiles.

    Reads tiled .npz files produced by TransformOrchestrator. Each tile contains
    an image, raycast annotations in pixel space, and content dimensions. The
    dataset applies random cropping (constrained to tissue area), geometric
    augmentation, and normalisation to [0, 1].
    """

    def __init__(
        self,
        data_dir: str,
        crop_size: int = 640,
        augment: bool = True,
        min_rays_after_clip: float = 0.3,
    ):
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.min_rays_after_clip = min_rays_after_clip
        self.rng = np.random.default_rng()

        self.tile_paths = sorted(self.data_dir.glob('*.npz'))
        if not self.tile_paths:
            raise FileNotFoundError(f'No .npz files found in {self.data_dir}')

    def __len__(self) -> int:
        """Return the number of tiles in the dataset."""
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray]:
        """Load, crop, augment, normalise, and return a single tile."""
        data = np.load(self.tile_paths[idx])

        image = data['image']
        annotations = data.get('annotations', data.get('bboxes'))
        content_h = int(data['content_h'])
        content_w = int(data['content_w'])

        if annotations is None or len(annotations) == 0:
            annotations = np.zeros((0, 35), dtype=np.float32)

        # 1. Random crop (pixel space)
        image, annotations = self._random_crop(image, annotations, content_h, content_w)

        # 2. Augment (pixel space)
        if self.augment:
            image, annotations = self._augment(image, annotations)

        # 3. Normalise to [0, 1]
        annotations = self._normalise(annotations)

        # 4. Validate
        self._validate_batch(annotations)

        # 5. Convert image to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image_tensor, annotations

    def _random_crop(
        self, image: np.ndarray, annotations: np.ndarray, content_h: int, content_w: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop image and filter annotations to a random crop_size x crop_size region.

        Crop origin is constrained to [0, content_w - crop_size] x [0, content_h - crop_size]
        so the crop stays within tissue area (not white padding).

        Returns:
            (cropped_image, filtered_annotations) in pixel space.
        """
        max_x = max(0, content_w - self.crop_size)
        max_y = max(0, content_h - self.crop_size)

        origin_x = int(self.rng.integers(0, max_x + 1))
        origin_y = int(self.rng.integers(0, max_y + 1))

        # Crop image
        cropped = image[origin_y : origin_y + self.crop_size, origin_x : origin_x + self.crop_size]

        # Pad if image is smaller than crop_size (edge case: target_size < crop_size)
        h, w = cropped.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            cropped = np.pad(
                cropped,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=255,
            )

        # Filter and clip annotations
        filtered = filter_and_clip_annotations(
            annotations, origin_x, origin_y, self.crop_size, self.crop_size, self.min_rays_after_clip
        )

        return cropped, filtered

    def _augment(self, image: np.ndarray, annotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply random geometric augmentations to image and annotations in lockstep.

        Augmentations: horizontal flip (50%), vertical flip (50%), rotation (0/90/180/270).
        """
        # Horizontal flip
        if self.rng.random() < 0.5:
            image = image[:, ::-1, :].copy()
            annotations = flip_horizontal(annotations, self.crop_size)

        # Vertical flip
        if self.rng.random() < 0.5:
            image = image[::-1, :, :].copy()
            annotations = flip_vertical(annotations, self.crop_size)

        # Rotation
        k = int(self.rng.integers(0, 4))
        if k > 0:
            image = np.rot90(image, k=k).copy()
            annotations = rotate_90(annotations, k, self.crop_size)

        return image, annotations

    def _normalise(self, annotations: np.ndarray) -> np.ndarray:
        """Normalise spatial quantities by crop_size. Only place normalisation happens.

        Divides cx, cy, and all rays by crop_size. Class_id is left untouched.
        """
        if len(annotations) == 0:
            return annotations

        annotations = annotations.copy()
        annotations[:, CX_IDX] /= self.crop_size
        annotations[:, CY_IDX] /= self.crop_size
        annotations[:, RAY_START_IDX:RAY_END_IDX] /= self.crop_size
        return annotations

    def _validate_batch(self, labels: np.ndarray) -> None:
        """Assert normalisation bounds. Implements BUG-08 fix.

        Rays must be <= 1.0 (not 1.5). Centroids must be in [0, 1].
        """
        if len(labels) == 0:
            return

        rays = labels[:, RAY_START_IDX:RAY_END_IDX]
        assert rays.max() <= 1.0, f'Ray > 1.0: clipping or normalisation bug (max={rays.max()})'

        cx_cy = labels[:, CX_IDX : CY_IDX + 1]
        assert cx_cy.min() >= 0.0, f'Centroid < 0: {cx_cy.min()}'
        assert cx_cy.max() <= 1.0, f'Centroid > 1.0: {cx_cy.max()}'


def collate_fn(
    batch: list[tuple[torch.Tensor, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate a batch of (image, labels) tuples.

    Args:
        batch: List of (image_tensor [3, H, W], labels_array [N, 35]) tuples.
            Labels are normalised float32.

    Returns:
        images: [B, 3, H, W] float32
        targets: [sum_M, 36] float32 — collated with batch_idx column prepended
    """
    images = torch.stack([item[0] for item in batch])

    target_list = []
    for batch_idx, (_, labels) in enumerate(batch):
        if labels.shape[0] == 0:
            continue
        batch_col = torch.full((labels.shape[0], 1), batch_idx, dtype=torch.float32)
        target_list.append(torch.cat([batch_col, torch.from_numpy(labels)], dim=1))

    targets = torch.cat(target_list, dim=0) if target_list else torch.zeros((0, 36), dtype=torch.float32)

    return images, targets
