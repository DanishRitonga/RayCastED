# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from raycasted.data.utils import get_yolo_bbox

# %% Load a single .mat file
script_dir = Path(__file__).parent
consep_dir = script_dir.parent / 'dataset' / 'CoNSeP'

mat_path = consep_dir / 'Train' / 'Labels' / 'train_1.mat'
data = scipy.io.loadmat(str(mat_path))

# %% Access the different components
inst_map = data['inst_map']  # 1000x1000 array, unique ID per nucleus
type_map = data['type_map']  # 1000x1000 array, class per pixel (0-7)
inst_type = data['inst_type']  # Nx1 array, type of each instance
inst_centroid = data['inst_centroid']  # Nx2 array, (x, y) coordinates

print(f'Shape of inst_map: {inst_map.shape}')
print(f'Number of nuclei: {len(inst_type)}')
print(f'Unique classes: {np.unique(type_map)}')

# %%
plt.imshow(inst_map)
plt.tight_layout()
plt.axis('off')
plt.show()

# %%

yolo_gt = ''
for i in range(1, np.max(inst_map).astype(int) + 1):
    ins = inst_map == i
    bbox = get_yolo_bbox(ins)
    cat = int(inst_type[i - 1, 0])  # instance IDs start at 1, so we use i-1 for indexing

    if bbox is not None:
        yolo_gt += f'{cat} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n'
