# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'dataset' / 'PanopTILs'

csv_filename = 'TCGA-S3-AA15-DX1_xmin55486_ymin28926_MPP-0.2500_xmin-0_ymin-1024_xmax-1024_ymax-2048'

csv_dir = data_dir.joinpath(
    'BootstrapNucleiManualRegions_TCGA',
    'tcga',
    'csv',
)

csv_dir_2 = data_dir.joinpath(
    'ManualNucleiManualRegions',
    'csv',
)

mask_dir = data_dir.joinpath(
    'BootstrapNucleiManualRegions_TCGA',
    'tcga',
    'masks',
)

csv_path = csv_dir / f'{csv_filename}.csv'

img_path = data_dir.joinpath(
    'BootstrapNucleiManualRegions_TCGA_1',
    'tcga',
    'masks',
    'TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_xmin-0_ymin-0_xmax-1024_ymax-1024.png',
)
# %%
df = pd.read_csv(csv_path)

# %%
img = cv2.imread(
    str(img_path),
    cv2.IMREAD_GRAYSCALE,
)

img = np.array(img).astype(np.uint8)

plt.imshow(img == 1)
plt.show()

# %%
dfs = []
for files in csv_dir.rglob('*.csv'):
    df = pd.read_csv(files)
    dfs.append(df)

for files in csv_dir_2.rglob('*.csv'):
    df = pd.read_csv(files)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# %%
for file in mask_dir.glob('*.png'):
    print(file.stem)
    if file.stem == 'TCGA-A2-A04T-DX1_xmin72145_ymin39078_MPP-0.2500_xmin-2048_ymin-2048_xmax-3072_ymax-3072':
        img = cv2.imread(str(file))
        img_array = np.array(img)

        slices = [img_array[:, :, i] for i in range(3)]
        for slice in slices:
            plt.imshow(slice)
            plt.axis('off')
            plt.show()
        break
