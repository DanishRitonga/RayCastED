# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

script_dir = Path(__file__).parent
segpath_dir = script_dir.parent / 'dataset' / 'SegPath'
data_dir = segpath_dir / 'CD3CD20_Lymphocyte'
mask_path = data_dir / 'CD3CD20_Lymphocyte_388_140288_041984_mask.png'


# %%
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
binary_mask = (mask * 255).astype(np.uint8)

contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# %%
bounding_boxes = []
for contour in contours:
    # cv2.boundingRect returns the top-left x,y and the width/height
    x, y, w, h = cv2.boundingRect(contour)

    # Optional: Filter out single-pixel noise
    # if w > 2 and h > 2:

    bounding_boxes.append((x, y, w, h))

# %%
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(binary_mask, cmap='gray')

for x, y, w, h in bounding_boxes:
    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

ax.set_title('Bounding Boxes')
plt.tight_layout()
plt.show()

# %%
docs_path = segpath_dir / 'SegPath.csv'
docs_df = pd.read_csv(docs_path)
sliced_df = docs_df[['TMA number', 'Antibody target']]

result = (
    sliced_df.groupby('TMA number')
    .agg(
        {
            'Antibody target': list  # Wraps separate elements into a single list
        }
    )
    .reset_index()
)

# %%
data = []

# Recursively loop through all .png files in the directory and subdirectories
for file_path in segpath_dir.rglob('*.png'):
    # .stem gets the filename without the '.png' extension
    filename = file_path.stem

    # Split the filename based on the underscores
    parts = filename.split('_')

    # Safety check: ensure the file matches your exact 6-part pattern
    if len(parts) == 6:
        antigen, celltype, slide, posx, posy, img_type = parts

        # Append the extracted info as a dictionary
        data.append(
            {
                'antigen': antigen,
                'celltype': celltype,
                'slide': slide,
                'posx': posx,
                'posy': posy,
                'img_type': img_type,
            }
        )
    else:
        # Optional: Print a warning for files that don't match the naming convention
        print(f'Skipping file with unexpected format: {file_path.name}')

# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(data)

# %%
df[['posx', 'posy']] = df[['posx', 'posy']].astype(int)

# %%
prev = 0
for i, val in enumerate(sorted(df[(df['slide'] == '167')]['posx'].unique())):
    print(prev - val)
    prev = val
