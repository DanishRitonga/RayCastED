import cv2
import numpy as np
import polars as pl


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

    @staticmethod
    def identify_columns(schema: pl.Schema) -> tuple[str, str, str, str]:
        """Identify RGB, mask, category, and tissue columns from a Parquet schema."""
        rgb_col, mask_col, cat_col, tissue_col = None, None, None, None

        for col_name, dtype in schema.items():
            if isinstance(dtype, pl.Struct) or dtype == pl.Binary:
                rgb_col = col_name
            elif isinstance(dtype, pl.List) and (isinstance(dtype.inner, pl.Struct) or dtype.inner == pl.Binary):
                mask_col = col_name
            elif isinstance(dtype, pl.List) and dtype.inner in [pl.Int64, pl.Int32, pl.UInt32, pl.Int8]:
                cat_col = col_name
            elif dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.Int8] and not isinstance(dtype, pl.List):
                tissue_col = col_name

        return rgb_col, mask_col, cat_col, tissue_col
