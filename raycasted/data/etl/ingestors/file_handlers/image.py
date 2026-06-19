import cv2
import numpy as np


class ImageHandler:
    """Stateless toolkit for image I/O operations."""

    @staticmethod
    def load_rgb(path: str) -> np.ndarray:
        """Load an image from disk and convert BGR to RGB."""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f'Failed to read image at {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
