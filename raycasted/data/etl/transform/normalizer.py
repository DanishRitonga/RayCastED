import json
from typing import Any

import cv2
import numpy as np

from .stain_estimator_gpu import StainEstimatorGPU
from .stainEstimator import StainEstimator


class NormalizerAndPadder:
    """A memory-only functional transformer that applies population-based
    stain normalization and bottom-right constant padding.
    """

    def __init__(self, config: dict[str, Any], profile_path: str | None = None, use_gpu: bool = False):
        self.target_size = config.get('max_size', 1024)
        self.use_gpu = use_gpu

        self.use_normalization = False
        if profile_path:
            with open(profile_path) as f:
                profile = json.load(f)

            self.target_matrix = np.array(profile['stain_matrix'])
            self.target_concentrations = np.array(profile['max_concentrations'])
            self.method = profile.get('method', 'macenko')
            self.use_normalization = True

    def process_roi(self, image: np.ndarray, annotations: Any) -> tuple[np.ndarray, Any, int, int]:
        """Executes the Stage 3 transformation sequentially in memory.

        Returns:
            (transformed_image, untouched_annotations, content_h, content_w)
            where content_h/content_w are the original pre-padding dimensions.
        """
        # Record original dimensions before any padding
        content_h, content_w = image.shape[:2]

        # 1. Normalize
        if self.use_normalization and self.method == 'macenko':
            image = self._apply_macenko_gpu(image) if self.use_gpu else self._apply_macenko(image)

        # 2. Pad (if smaller than target size)
        if content_h < self.target_size or content_w < self.target_size:
            image = self._pad_bottom_right(image)
            # Annotations require NO changes because (0,0) remains top-left!

        return image, annotations, content_h, content_w

    def _apply_macenko(self, image: np.ndarray, Io: int = 240) -> np.ndarray:
        """Applies the canonical Macenko transformation to a single image."""
        # Note: Assumes StainEstimator is imported from your Stage 2 module
        source_matrix, source_concentrations = StainEstimator._estimate_macenko(image, Io=Io)

        # Failsafe: If the image is entirely white background, math collapses.
        # We silently return the original white image.
        if source_matrix is None:
            return image

        # 1. Convert to OD space
        img_reshaped = image.reshape((-1, 3)).astype(np.float64)
        OD = -np.log10((img_reshaped + 1) / Io)

        # 2. Calculate pixel concentrations using the pseudo-inverse of the source matrix
        C_source = np.dot(OD, np.linalg.pinv(source_matrix))

        # 3. Scale concentrations to the population target
        # Add epsilon to prevent division by zero in empty channels
        source_concentrations = np.where(source_concentrations == 0, 1e-6, source_concentrations)
        C_norm = C_source * (self.target_concentrations / source_concentrations)

        # 4. Reconstruct OD using the TARGET stain matrix
        OD_norm = np.dot(C_norm, self.target_matrix)

        # 5. Convert back to RGB space
        img_norm = Io * (10**-OD_norm) - 1
        img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)

        return img_norm.reshape(image.shape)

    def _apply_macenko_gpu(self, image: np.ndarray, Io: int = 240) -> np.ndarray:
        """Applies Macenko transformation with GPU-accelerated source estimation."""
        import torch

        source_matrix_np, source_concentrations_np = StainEstimatorGPU._estimate_macenko(image, Io=Io)
        if source_matrix_np is None:
            return image

        dev = torch.device('cuda')
        source_matrix = torch.from_numpy(source_matrix_np).to(device=dev, dtype=torch.float64)
        source_concentrations = torch.from_numpy(source_concentrations_np).to(device=dev, dtype=torch.float64)

        img_t = torch.from_numpy(image.reshape(-1, 3).astype(np.float64)).to(device=dev, dtype=torch.float64)
        od = -torch.log10((img_t + 1.0) / Io)

        c_source = od @ torch.linalg.pinv(source_matrix)
        src_conc = torch.where(source_concentrations == 0, torch.tensor(1e-6, device=dev), source_concentrations)
        c_norm = c_source * (torch.from_numpy(self.target_concentrations).to(dev) / src_conc)

        target_t = torch.from_numpy(self.target_matrix).to(dev)
        od_norm = c_norm @ target_t

        img_norm = Io * torch.pow(10.0, -od_norm) - 1
        img_norm = torch.clamp(img_norm, 0, 255).to(torch.uint8)

        return img_norm.reshape(image.shape[0], image.shape[1], 3).cpu().numpy()

    def _pad_bottom_right(self, image: np.ndarray) -> np.ndarray:
        """Pads the image with white space on the bottom and right edges."""
        h, w = image.shape[:2]

        pad_bottom = max(0, self.target_size - h)
        pad_right = max(0, self.target_size - w)

        if pad_bottom == 0 and pad_right == 0:
            return image

        # cv2.copyMakeBorder is highly optimized in C++
        return cv2.copyMakeBorder(
            image,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],  # White background for H&E
        )
