# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from raycasted.data.utils import decode_image_bytes, get_yolo_bbox, load_parquet_as_df

# %%
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'dataset' / 'MoNuSAC' / 'data'

dfs = load_parquet_as_df(data_dir)


def _get_gt_df(df: pd.DataFrame) -> pd.DataFrame:
    gt_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        gts = []
        for ins, cat in zip(row['instances'], row['categories']):
            mask = decode_image_bytes(ins['bytes'])
            bbox = get_yolo_bbox(mask)
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

# %%
row = df.iloc[0]

# Decode the image
img = decode_image_bytes(row['image']['bytes'])
h_img, w_img = img.shape[:2]

# Create figure and axis
fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(img)

# Draw each bounding box from yolo_gt
for gt in row['yolo_gt']:
    cat, x_center, y_center, width, height = gt

    # Skip class 0
    if cat == 0:
        continue

    # Convert from YOLO format (normalized) to pixel coordinates
    x_center_px = x_center * w_img
    y_center_px = y_center * h_img
    width_px = width * w_img
    height_px = height * h_img

    # Convert to top-left corner coordinates
    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2

    # Draw rectangle
    rect = plt.Rectangle((x_min, y_min), width_px, height_px, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

    # # Add category label
    # ax.text(
    #     x_min,
    #     y_min - 5,
    #     f'Cat: {cat}',
    #     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
    #     fontsize=10,
    #     color='black',
    # )

ax.axis('off')
plt.tight_layout()
plt.show()

# %%
# Create combined binary mask from all instances
img = decode_image_bytes(row['image']['bytes'])
h_img, w_img = img.shape[:2]

# Initialize empty binary mask
combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)

# Combine all instance masks (excluding class 0)
for ins, cat in zip(row['instances'], row['categories']):
    if cat == 0:
        continue
    mask = decode_image_bytes(ins['bytes'])
    combined_mask = np.maximum(combined_mask, mask)

# Plot the combined binary mask
fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(combined_mask, cmap='gray')
ax.axis('off')
plt.tight_layout()
plt.show()

# %%
