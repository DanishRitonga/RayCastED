"""RayCastED — RayCastTileDataset and collate_fn.

Reads tiled .npz files from TransformOrchestrator output, applies online
augmentation and random cropping, normalises to [0, 1], and emits tensors
ready for training.

Annotation column layout: [class_id, cx, cy, d_1, ..., d_n]
Ray count is derived from the actual .npz annotation shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from ..ops.augment import flip_horizontal, flip_vertical, random_scale, random_translate, rotate_90, stain_jitter
from ..ops.filter import filter_and_clip_annotations

if TYPE_CHECKING:
    from .sampler import WeightedClassSampler


class RayCastTileDataset(Dataset):
    """PyTorch dataset for raycast polygon tiles.

    Reads tiled .npz files produced by TransformOrchestrator. Each tile contains
    an image, raycast annotations in pixel space, and content dimensions. The
    dataset applies random cropping (constrained to tissue area), geometric
    augmentation, photometric augmentation, and normalisation to [0, 1].
    """

    def __init__(
        self,
        data_dir: str,
        crop_size: int = 640,
        augment: bool = True,
        min_rays_after_clip: float = 0.3,
        augment_config: dict | None = None,
        num_classes: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.min_rays_after_clip = min_rays_after_clip
        self.augment_config = augment_config or {}
        self.num_classes = num_classes
        self.rng = np.random.default_rng()

        self.tile_paths = sorted(self.data_dir.glob('*.npz'))
        if not self.tile_paths:
            raise FileNotFoundError(f'No .npz files found in {self.data_dir}')

        # Detect n_rays from the first tile's annotation shape
        self._detect_n_rays()

        # Compatibility: Ultralytics plot_training_labels reads dataset.labels
        self.labels = self._build_labels()

        # Class and tissue distributions for weighted sampling (lazy — built on first access)
        self._tile_classes: list[np.ndarray] | None = None
        self._tissues: np.ndarray | None = None

    def __len__(self) -> int:
        """Return the number of tiles in the dataset."""
        return len(self.tile_paths)

    def _detect_n_rays(self) -> None:
        """Detect ray count from the first tile's annotations.

        Compares with the globally configured N_RAYS and warns on mismatch.
        The annotation format is [class_id, cx, cy, d_1..d_n], so n_rays = ncols - 3.
        """
        from ..utils import constants as _const

        for path in self.tile_paths:
            data = np.load(path)
            anns = data.get('annotations', data.get('bboxes'))
            if anns is not None and len(anns) > 0:
                self.n_rays = anns.shape[1] - 3
                configured = _const.N_RAYS
                if self.n_rays != configured:
                    print(
                        f'WARNING: Tiles have {self.n_rays} rays but N_RAYS={configured}. '
                        f'Re-ingest with --n-rays {configured} to match, or use --n-rays {self.n_rays}.'
                    )
                return
        self.n_rays = _const.N_RAYS  # fallback if all tiles are empty

    def _build_labels(self) -> list[dict]:
        """Build Ultralytics-compatible labels list for plot_training_labels.

        Ultralytics' DetectionTrainer.plot_training_labels() reads
        dataset.labels and expects a list of dicts with 'bboxes' and 'cls'
        keys. We provide cx/cy as a pseudo-bbox so the plotting code
        doesn't crash, even though we don't use bboxes for training.
        """
        labels = []
        for path in self.tile_paths:
            data = np.load(path)
            anns = data.get('annotations', data.get('bboxes'))
            if anns is None or len(anns) == 0:
                labels.append({'bboxes': np.zeros((0, 4), dtype=np.float32), 'cls': np.zeros((0,), dtype=np.float32)})
                continue
            # Filter Ignore class
            valid = anns[anns[:, 0] != 255]
            if len(valid) == 0:
                labels.append({'bboxes': np.zeros((0, 4), dtype=np.float32), 'cls': np.zeros((0,), dtype=np.float32)})
                continue
            # Use cx/cy as pseudo xyxy bbox for compatibility
            cx = valid[:, 1]
            cy = valid[:, 2]
            bboxes = np.stack([cx, cy, cx, cy], axis=1).astype(np.float32)
            cls = valid[:, 0].astype(np.float32)
            labels.append({'bboxes': bboxes, 'cls': cls})
        return labels

    def _build_class_tissue_index(self) -> None:
        """Scan all tiles once to build per-tile class lists and tissue IDs.

        Populates self._tile_classes and self._tissues for WeightedClassSampler.
        Called lazily on first call to get_sampler().
        """
        from ..utils import constants as _const

        tile_classes: list[np.ndarray] = []
        tissues: list[int] = []

        for path in self.tile_paths:
            data = np.load(path)
            anns = data.get('annotations', data.get('bboxes'))
            tissue_val = data.get('tissue', None)

            if anns is not None and len(anns) > 0:
                valid = anns[anns[:, _const.CLASS_IDX] != 255]
                if len(valid) > 0:
                    classes = np.unique(valid[:, _const.CLASS_IDX].astype(np.intp))
                else:
                    classes = np.array([], dtype=np.intp)
            else:
                classes = np.array([], dtype=np.intp)

            tile_classes.append(classes)
            tissues.append(int(tissue_val) if tissue_val is not None else 0)

        self._tile_classes = tile_classes
        self._tissues = np.array(tissues, dtype=np.intp)

    def get_sampler(self, gamma: float = 0.85) -> WeightedClassSampler:
        """Build a WeightedClassSampler from this dataset's class/tissue distribution.

        Args:
            gamma: Re-weighting strength in [0, 1]. 0 = uniform, 1 = full inverse-frequency.

        Returns:
            WeightedClassSampler instance ready to pass to DataLoader(sampler=...).

        Raises:
            ValueError: If num_classes was not provided at construction time.
        """
        from .sampler import WeightedClassSampler

        if self.num_classes is None:
            raise ValueError(
                'num_classes must be provided to RayCastTileDataset to use weighted sampling. '
                'Pass num_classes=<N> when constructing the dataset.'
            )

        if self._tile_classes is None:
            self._build_class_tissue_index()

        return WeightedClassSampler(
            tile_classes=self._tile_classes,
            tissues=self._tissues,
            num_classes=self.num_classes,
            gamma=gamma,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray]:
        """Load, crop, augment, normalise, and return a single tile."""
        data = np.load(self.tile_paths[idx])

        image = data['image']
        annotations = data.get('annotations', data.get('bboxes'))
        content_h = int(data['content_h'])
        content_w = int(data['content_w'])

        if annotations is None or len(annotations) == 0:
            annotations = np.zeros((0, 0), dtype=np.float32)
        else:
            # Filter out Ignore class (class_id=255) — these must not reach the loss
            valid_mask = annotations[:, 0] != 255
            annotations = annotations[valid_mask]

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

        return image_tensor, annotations, str(self.tile_paths[idx])

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
        """Apply random augmentations to image and annotations in lockstep.

        Augmentations applied in order:
        1. Stain jitter (color-only, HSV perturbation)
        2. Random scale (resize + scale annotations)
        3. Random translation (shift image + shift cx/cy)
        4. Horizontal flip (50%)
        5. Vertical flip (50%)
        6. Rotation (0/90/180/270)
        """
        if len(annotations) == 0:
            return image, annotations

        cfg = self.augment_config

        # Stain jitter (color-only, no annotation transform needed)
        if cfg.get('stain_jitter', False):
            image = stain_jitter(
                image,
                self.rng,
                hsv_h=cfg.get('stain_hsv_h', 0.05),
                hsv_s=cfg.get('stain_hsv_s', 0.3),
                hsv_v=cfg.get('stain_hsv_v', 0.2),
                blur_prob=cfg.get('stain_blur_prob', 0.2),
                blur_sigma=cfg.get('stain_blur_sigma', 1.0),
            )

        # Random scale
        if cfg.get('scale_augment', False):
            image, annotations = random_scale(
                image,
                annotations,
                scale_range=cfg.get('scale_range', (0.7, 1.3)),
                crop_size=self.crop_size,
                rng=self.rng,
            )
            if len(annotations) == 0:
                return image, annotations

        # Random translation
        if cfg.get('translate_augment', False):
            image, annotations = random_translate(
                image,
                annotations,
                translate_range=cfg.get('translate_range', 0.1),
                crop_size=self.crop_size,
                rng=self.rng,
            )
            if len(annotations) == 0:
                return image, annotations

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
        annotations[:, 1] /= self.crop_size  # CX
        annotations[:, 2] /= self.crop_size  # CY
        annotations[:, 3:] /= self.crop_size  # all rays
        return annotations

    def _validate_batch(self, labels: np.ndarray) -> None:
        """Assert normalisation bounds. Implements BUG-08 fix.

        Rays must be <= 1.0 (not 1.5). Centroids must be in [0, 1].
        """
        if len(labels) == 0:
            return

        rays = labels[:, 3:]
        assert rays.max() <= 1.0, f'Ray > 1.0: clipping or normalisation bug (max={rays.max()})'

        cx_cy = labels[:, 1:3]
        assert cx_cy.min() >= 0.0, f'Centroid < 0: {cx_cy.min()}'
        assert cx_cy.max() <= 1.0, f'Centroid > 1.0: {cx_cy.max()}'


def collate_fn(batch):
    """Collate a batch of (image, labels, path) tuples.

    Args:
        batch: List of (image_tensor [3, H, W], labels_array [N, 3+n_rays], path) tuples.

    Returns:
        images: [B, 3, H, W] float32
        targets: [sum_M, 4+n_rays] float32 — collated with batch_idx column prepended
    """
    images = torch.stack([item[0] for item in batch])

    target_list = []
    for batch_idx, (_, labels, _) in enumerate(batch):
        if labels.shape[0] == 0:
            continue
        batch_col = np.full((labels.shape[0], 1), batch_idx, dtype=np.float32)
        target_list.append(np.concatenate([batch_col, labels], axis=1))

    if target_list:
        targets = torch.from_numpy(np.concatenate(target_list, axis=0))
    else:
        from raycasted.data.etl.utils.constants import N_RAYS

        targets = torch.zeros((0, 4 + N_RAYS), dtype=torch.float32)

    return images, targets
