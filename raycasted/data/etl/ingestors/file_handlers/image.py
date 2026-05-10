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

    @staticmethod
    def decode_bytes(byte_string: bytes, is_mask: bool = False) -> np.ndarray:
        """Decode raw bytes into a numpy array via OpenCV."""
        np_arr = np.frombuffer(byte_string, np.uint8)
        flags = cv2.IMREAD_UNCHANGED if is_mask else cv2.IMREAD_COLOR
        decoded = cv2.imdecode(np_arr, flags)
        if decoded is None:
            raise ValueError('OpenCV failed to decode the byte array.')
        if not is_mask and len(decoded.shape) == 3:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return decoded
