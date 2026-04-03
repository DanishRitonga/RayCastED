import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_parquet_as_df(parquet_dir: Path) -> pd.DataFrame:
    """Loads all parquet files from the specified directory and concatenates them into a single DataFrame."""
    dfs = [pd.read_parquet(f) for f in sorted(parquet_dir.glob('*.parquet'))]

    return pd.concat(dfs, ignore_index=True)


def decode_image_bytes(byte_data) -> np.ndarray:
    """Decodes image bytes into a NumPy array."""
    image = Image.open(io.BytesIO(byte_data))
    return np.array(image).astype(np.uint8)


def _get_bbox(mask: np.ndarray, format: str = 'xyxy') -> tuple[int, int, int, int] | None:
    y, x = np.where(mask > 0)

    if x.size and y.size:
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        if format == 'xyxy':
            return (x_min, y_min, x_max, y_max)
        elif format == 'xywh':
            return (x_min, y_min, x_max - x_min, y_max - y_min)

    return None


def get_yolo_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    bbox = _get_bbox(mask, format='xywh')
    if bbox is not None:
        x_min, y_min, w, h = bbox
        h_img, w_img = mask.shape[:2]

        return (
            (x_min + w / 2.0) / w_img,
            (y_min + h / 2.0) / h_img,
            w / w_img,
            h / h_img,
        )

    return None
