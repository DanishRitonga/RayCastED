# %%
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from raycasted.data.utils import load_parquet_as_df

# %%
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'dataset' / 'PanNuke' / 'data'
config_path = script_dir / 'pannuke_utils' / 'config.json'


# %%

dfs = load_parquet_as_df(data_dir)


def _get_config_map(config_path: Path) -> tuple[dict[int, str], dict[int, str]]:
    with open(config_path) as f:
        config = json.load(f)
    return config['category'], config['tissue']


CATEGORY_MAP, TISSUE_MAP = _get_config_map(config_path)


# %%
def _decode_image_bytes(byte_data) -> np.ndarray:
    image = Image.open(io.BytesIO(byte_data))
    return np.array(image).astype(np.uint8)


def decode_roi_bytes(df: pd.DataFrame, row_index: int) -> np.ndarray:
    """Decodes the ROI image bytes for a given row index in the DataFrame."""
    row = df.iloc[row_index]
    byte_data = row['image']['bytes']

    return _decode_image_bytes(byte_data)


def decode_ins_bytes(df: pd.DataFrame, row_index: int, ins_index: int = 0) -> np.ndarray:
    """Decodes the instance image bytes for a given row index and instance index in the DataFrame."""
    row = df.iloc[row_index]
    byte_data = row['instances'][ins_index]['bytes']

    return _decode_image_bytes(byte_data)


# %%
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


def _get_yolo_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
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


# %%
def _get_gt_df(df: pd.DataFrame) -> pd.DataFrame:
    gt_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        gts = []
        for ins, cat in zip(row['instances'], row['categories']):
            mask = _decode_image_bytes(ins['bytes'])
            bbox = _get_yolo_bbox(mask)
            if bbox is not None:
                gt = (cat, *bbox)
                gts.append(gt)
        gt_list.append(gts)

    df['yolo_gt'] = gt_list

    return df


df = _get_gt_df(dfs)
# %%
# We explode the dataframe once to use it for the category and pair plots
exploded_df = df.explode('categories').reset_index(drop=True)

# Optional: Set a clean visual style for the plots
sns.set_theme(style='whitegrid')

# %%
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='tissue', palette='viridis')

plt.title('Distribution of Tissue')
plt.xlabel('Tissue ID')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.countplot(data=exploded_df, x='categories', palette='magma')

plt.title('Distribution of Individual Categories')
plt.xlabel('Category ID')
plt.ylabel('Count')
plt.show()

# %% Create a matrix of Category vs Tissue counts
pair_matrix = pd.crosstab(exploded_df['categories'], exploded_df['tissue'])

plt.figure(figsize=(8, 6))
sns.heatmap(pair_matrix, annot=False, fmt='d', cmap='Blues')

plt.title('Heatmap of Category-Tissue Pairs')
plt.xlabel('Tissue ID')
plt.ylabel('Category ID')
plt.show()
