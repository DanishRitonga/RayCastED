"""RayCastED — Augmentation Operations.

Geometric and photometric augmentations for star-convex polygon annotations.
Geometric transforms use precomputed permutation indices for efficient ray reordering.
Photometric transforms are color-only and do not affect polygon annotations.

Ray count is derived from the annotation array shape (N, 3+n_rays)
so the same code works with any ray count.
"""

import numpy as np

# Lazy import for LSP-DETR transforms (not needed by ETL pipeline)
try:
    import albumentations as A
except ImportError:
    A = None  # type: ignore[assignment]


def _get_permutation_indices(n_rays: int) -> dict:
    """Get flip/rotation permutation indices for the given ray count.

    Uses precomputed indices from constants if available, otherwise computes on the fly.
    """
    from ..utils import constants as _const

    if n_rays == _const.N_RAYS:
        return {
            'flip_h': _const.FLIP_H_IDX,
            'flip_v': _const.FLIP_V_IDX,
            'rot': _const.ROT_INDICES,
        }

    # Compute on the fly
    flip_h = np.array([(n_rays // 2 - i) % n_rays for i in range(n_rays)], dtype=np.int64)
    flip_v = np.array([(n_rays - i) % n_rays for i in range(n_rays)], dtype=np.int64)
    rot = {}
    for k in (1, 2, 3):
        shift = (k * n_rays) // 4
        rot[k] = np.array([(i + shift) % n_rays for i in range(n_rays)], dtype=np.int64)

    return {'flip_h': flip_h, 'flip_v': flip_v, 'rot': rot}


def flip_horizontal(
    annotations: np.ndarray,
    canvas_w: int,
) -> np.ndarray:
    """Apply horizontal flip to annotations.

    Reflects x-coordinate and permutes ray ordering.

    Args:
        annotations: Array of shape (N, 3+n_rays) in ETL format (crop-relative).
        canvas_w: Width of the canvas (crop size).

    Returns:
        flipped: Array of shape (N, 3+n_rays) with flipped annotations.
    """
    if annotations is None or len(annotations) == 0:
        return annotations if annotations is not None else np.zeros((0, 0), dtype=np.float32)

    annotations = annotations.copy()

    # Flip x-coordinate: x' = canvas_w - x
    annotations[:, 1] = canvas_w - annotations[:, 1]  # CX

    # Permute rays
    n_rays = annotations.shape[1] - 3
    perm = _get_permutation_indices(n_rays)
    rays = annotations[:, 3:]
    annotations[:, 3:] = rays[:, perm['flip_h']]

    return annotations


def flip_vertical(
    annotations: np.ndarray,
    canvas_h: int,
) -> np.ndarray:
    """Apply vertical flip to annotations.

    Reflects y-coordinate and permutes ray ordering.

    Args:
        annotations: Array of shape (N, 3+n_rays) in ETL format (crop-relative).
        canvas_h: Height of the canvas (crop size).

    Returns:
        flipped: Array of shape (N, 3+n_rays) with flipped annotations.
    """
    if annotations is None or len(annotations) == 0:
        return annotations if annotations is not None else np.zeros((0, 0), dtype=np.float32)

    annotations = annotations.copy()

    # Flip y-coordinate: y' = canvas_h - y
    annotations[:, 2] = canvas_h - annotations[:, 2]  # CY

    # Permute rays
    n_rays = annotations.shape[1] - 3
    perm = _get_permutation_indices(n_rays)
    rays = annotations[:, 3:]
    annotations[:, 3:] = rays[:, perm['flip_v']]

    return annotations


def rotate_90(
    annotations: np.ndarray,
    k: int,
    canvas_size: int,
) -> np.ndarray:
    """Apply 90° rotation(s) to annotations.

    Rotation is counter-clockwise. After rotation, coordinates are
    transformed to the new coordinate system.

    Args:
        annotations: Array of shape (N, 3+n_rays) in ETL format (crop-relative).
        k: Number of 90° rotations (1, 2, or 3).
        canvas_size: Size of the square canvas (assumes square crop).

    Returns:
        rotated: Array of shape (N, 3+n_rays) with rotated annotations.
    """
    if annotations is None or len(annotations) == 0:
        return annotations if annotations is not None else np.zeros((0, 0), dtype=np.float32)

    if k not in [1, 2, 3]:
        raise ValueError(f'k must be 1, 2, or 3, got {k}')

    annotations = annotations.copy()

    cx = annotations[:, 1].copy()  # CX
    cy = annotations[:, 2].copy()  # CY

    # Apply rotation to centroid
    # k=1 (90° CCW): (x, y) → (y, canvas_size - x)
    # k=2 (180°): (x, y) → (canvas_size - x, canvas_size - y)
    # k=3 (270° CCW): (x, y) → (canvas_size - y, x)

    if k == 1:
        new_cx = cy
        new_cy = canvas_size - cx
    elif k == 2:
        new_cx = canvas_size - cx
        new_cy = canvas_size - cy
    else:  # k == 3
        new_cx = canvas_size - cy
        new_cy = cx

    annotations[:, 1] = new_cx
    annotations[:, 2] = new_cy

    # Permute rays
    n_rays = annotations.shape[1] - 3
    perm = _get_permutation_indices(n_rays)
    rays = annotations[:, 3:]
    annotations[:, 3:] = rays[:, perm['rot'][k]]

    return annotations


def stain_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    hsv_h: float = 0.05,
    hsv_s: float = 0.3,
    hsv_v: float = 0.2,
    blur_prob: float = 0.2,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """Apply stochastic stain-like color jitter (HSV perturbation + optional blur).

    Color-only transform — does **not** modify polygon annotations.
    Inspired by Tellez et al. (2019): quantifying data augmentation
    effects in histopathology.

    Args:
        image: HWC uint8 RGB image.
        rng: NumPy random generator.
        hsv_h: Hue shift range (0-1 scale).
        hsv_s: Saturation scale range (±).
        hsv_v: Value/brightness scale range (±).
        blur_prob: Probability of applying Gaussian blur.
        blur_sigma: Maximum Gaussian blur sigma.

    Returns:
        Augmented image (HWC uint8 RGB).
    """
    import cv2

    # HSV jitter
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Hue: shift by random amount in [-hsv_h, hsv_h] (OpenCV H is 0-179)
    image_hsv[:, :, 0] += rng.uniform(-hsv_h * 179, hsv_h * 179, size=1).astype(np.float32)

    # Saturation: scale by random factor in [1-hsv_s, 1+hsv_s]
    image_hsv[:, :, 1] *= rng.uniform(1.0 - hsv_s, 1.0 + hsv_s, size=1).astype(np.float32)

    # Value: scale by random factor in [1-hsv_v, 1+hsv_v]
    image_hsv[:, :, 2] *= rng.uniform(1.0 - hsv_v, 1.0 + hsv_v, size=1).astype(np.float32)

    # Clip to valid ranges and convert back
    image_hsv[:, :, 0] = np.clip(image_hsv[:, :, 0], 0, 179)
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1], 0, 255)
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2], 0, 255)

    image_out = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Optional Gaussian blur
    if rng.random() < blur_prob:
        sigma = rng.uniform(0.5, blur_sigma)
        ksize = int(6 * sigma) | 1  # ensure odd kernel size
        image_out = cv2.GaussianBlur(image_out, (ksize, ksize), sigma)

    return image_out


def random_scale(
    image: np.ndarray,
    annotations: np.ndarray,
    scale_range: tuple[float, float],
    crop_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random scale augmentation to image and polygon annotations.

    Scales the image and all spatial annotation quantities (cx, cy, rays)
    by a random factor, then crops/pads back to crop_size x crop_size.

    Args:
        image: HWC uint8 RGB image of shape (crop_size, crop_size, 3).
        annotations: Array of shape (N, 3+n_rays) in pixel space (ETL format).
        scale_range: (min_scale, max_scale) tuple.
        crop_size: Target canvas size (assumes square).
        rng: NumPy random generator.

    Returns:
        (scaled_image, scaled_annotations) — same shapes as input.
    """
    import cv2

    from .filter import filter_and_clip_annotations

    if annotations is None or len(annotations) == 0:
        return image, annotations

    scale = rng.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # Scale image
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)

    # Scale annotations: multiply cx, cy, and all rays by scale factor
    scaled_ann = annotations.copy()
    scaled_ann[:, 1] *= scale  # cx
    scaled_ann[:, 2] *= scale  # cy
    scaled_ann[:, 3:] *= scale  # all rays

    # Crop or pad back to crop_size x crop_size
    if scale > 1.0:
        # Image is larger — random crop
        max_x = max(0, new_w - crop_size)
        max_y = max(0, new_h - crop_size)
        ox = int(rng.integers(0, max_x + 1))
        oy = int(rng.integers(0, max_y + 1))
        cropped = scaled_image[oy : oy + crop_size, ox : ox + crop_size]

        # Shift annotation coordinates to crop-relative
        scaled_ann[:, 1] -= ox  # cx
        scaled_ann[:, 2] -= oy  # cy

        # Filter out-of-bounds and clip rays
        scaled_ann = filter_and_clip_annotations(scaled_ann, 0, 0, crop_size, crop_size, min_rays_after_clip=0.3)
    else:
        # Image is smaller — pad with white (255)
        pad_h = crop_size - new_h
        pad_w = crop_size - new_w
        # Random offset for padding
        pad_top = int(rng.integers(0, pad_h + 1)) if pad_h > 0 else 0
        pad_left = int(rng.integers(0, pad_w + 1)) if pad_w > 0 else 0
        cropped = np.full((crop_size, crop_size, 3), 255, dtype=np.uint8)
        cropped[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = scaled_image

        # Shift annotation coordinates by padding offset
        scaled_ann[:, 1] += pad_left  # cx
        scaled_ann[:, 2] += pad_top  # cy

        # Filter any that ended up out of bounds
        scaled_ann = filter_and_clip_annotations(scaled_ann, 0, 0, crop_size, crop_size, min_rays_after_clip=0.3)

    return cropped, scaled_ann


def random_translate(
    image: np.ndarray,
    annotations: np.ndarray,
    translate_range: float,
    crop_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random translation augmentation to image and polygon annotations.

    Shifts the image and annotation centroids by a random offset.
    Rays (distances from centroid) are unchanged.

    Args:
        image: HWC uint8 RGB image of shape (crop_size, crop_size, 3).
        annotations: Array of shape (N, 3+n_rays) in pixel space (ETL format).
        translate_range: Fraction of crop_size for max shift.
        crop_size: Canvas size (assumes square).
        rng: NumPy random generator.

    Returns:
        (shifted_image, shifted_annotations) — same shapes as input.
    """
    import cv2

    from .filter import filter_and_clip_annotations

    if annotations is None or len(annotations) == 0:
        return image, annotations

    max_shift = int(crop_size * translate_range)
    if max_shift == 0:
        return image, annotations

    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))

    if dx == 0 and dy == 0:
        return image, annotations

    # Shift image using affine transform
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Shift annotation centroids (rays unchanged — they're distances from centroid)
    shifted_ann = annotations.copy()
    shifted_ann[:, 1] += dx  # cx
    shifted_ann[:, 2] += dy  # cy

    # Filter out-of-bounds and clip rays
    shifted_ann = filter_and_clip_annotations(shifted_ann, 0, 0, crop_size, crop_size, min_rays_after_clip=0.3)

    return shifted_image, shifted_ann


# =============================================================================
# LSP-DETR albumentations transforms (guard against empty masks after augment)
# =============================================================================

if A is not None:

    class LSPElasticTransform(A.ElasticTransform):  # noqa: D101
        def apply_to_mask(self, mask, map_x, map_y, **params):
            if mask.size == 0:
                return mask
            return super().apply_to_mask(mask, map_x, map_y, **params)

    class LSPHorizontalFlip(A.HorizontalFlip):  # noqa: D101
        def apply_to_mask(self, mask, *args, **params):
            if mask.size == 0:
                return mask
            return super().apply_to_mask(mask, *args, **params)

    class LSPVerticalFlip(A.VerticalFlip):  # noqa: D101
        def apply_to_mask(self, mask, *args, **params):
            if mask.size == 0:
                return mask
            return super().apply_to_mask(mask, *args, **params)

    class LSPRandomRotate90(A.RandomRotate90):  # noqa: D101
        def apply_to_mask(self, mask, *args, **params):
            if mask.size == 0:
                return mask
            return super().apply_to_mask(mask, *args, **params)

    class LSPRandomSizedCrop(A.RandomSizedCrop):  # noqa: D101
        def apply_to_mask(self, mask, crop_coords, **params):
            if mask.size == 0:
                return mask
            return super().apply_to_mask(mask, crop_coords, **params)


def build_train_augmentations():
    """Build the LSP-DETR albumentations training pipeline.

    12 transforms, applied to both image and mask simultaneously.
    Matches the original PanNuke.yaml config.
    """
    import albumentations as A

    return A.Compose(
        [
            LSPRandomRotate90(p=1.0),
            LSPHorizontalFlip(p=0.5),
            LSPVerticalFlip(p=0.5),
            A.Downscale(scale_range=(0.5, 0.5), p=0.15),
            A.Blur(blur_limit=11, p=0.2),
            A.GaussNoise(std_range=(0.0, 0.44), p=0.25),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2),
            A.Superpixels(p_replace=0.1, n_segments=200, max_size=128, p=0.1),
            A.ZoomBlur(max_factor=1.05, p=0.1),
            LSPRandomSizedCrop(min_max_height=(128, 256), size=(256, 256), p=0.1),
            LSPElasticTransform(sigma=25, alpha=0.5, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_eval_augmentations():
    """Build the LSP-DETR albumentations eval pipeline (just normalize)."""
    import albumentations as A

    return A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
