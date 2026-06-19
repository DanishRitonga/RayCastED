import cv2
import numpy as np


class ParquetHandler:
    """Stateless toolkit for Parquet file I/O and byte decoding."""

    @staticmethod
    def decode_image_bytes(byte_string: bytes, is_mask: bool = False) -> np.ndarray:
        """Decode image bytes from a Parquet binary column."""
        np_arr = np.frombuffer(byte_string, np.uint8)
        flags = cv2.IMREAD_UNCHANGED if is_mask else cv2.IMREAD_COLOR
        decoded = cv2.imdecode(np_arr, flags)
        if decoded is None:
            raise ValueError('OpenCV failed to decode the byte array.')
        if not is_mask and len(decoded.shape) == 3:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return decoded
