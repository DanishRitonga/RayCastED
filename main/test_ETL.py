# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle

from raycasted.data.etl import ETLConfig, ParquetIngestor

pl.Config.set_fmt_str_lengths(100)


config_dir = Path(__file__).parent.joinpath('dataset.yaml')
print('1. Parsing Configuration...')
try:
    config_manager = ETLConfig(config_dir)
    pannuke_config = config_manager.get_dataset_config('MoNuSAC')
    print(f'✅ PanNuke config loaded. Resolved Root: {pannuke_config["root_dir"]}')
except Exception as e:
    print(f'❌ Config parsing failed: {e}')

print('\n2. Initializing Ingestor & Building Registry...')
try:
    ingestor = ParquetIngestor(config=pannuke_config)
    registry = ingestor.get_registry()

    print('✅ Registry built successfully. Preview:')
    print(registry.head(3))

    if registry.is_empty():
        print('❌ Registry is empty! Check your directory paths.')

except Exception as e:
    print(f'❌ Ingestor initialization failed: {e}')
# %%

print('\n3. Testing Pixel Extraction on the First Parquet File...')

# Grab the very first row from the registry as a Python dictionary
first_parquet_row = registry.row(1, named=True)
print(f'Processing archive: {first_parquet_row["image_path"]}')

# Call process_item (this returns the generator)
roi_generator = ingestor.process_item(first_parquet_row)

# Iterate through the generator, but break after the first ROI to keep it quick
i = 0
for roi_id, image_array, bbox, origin in roi_generator:
    print(f'\n✅ Successfully Extracted ROI: {roi_id}')
    print(f'   -> Image Array Shape: {image_array.shape}, dtype: {image_array.dtype}')
    print(f'   -> Bounding Boxes: {bbox.shape[0]}')
    print(f'   -> Tissue Origin: {origin}')

    # 4. Optional Visual Sanity Check
    print('\n4. Plotting visual sanity check...')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_array)
    axes[0].set_title(f'RGB Image ({roi_id})')
    axes[0].axis('off')

    axes[1].imshow(image_array)
    axes[1].set_title(f'Bounding Boxes ({roi_id})')
    axes[1].axis('off')

    # Plot each bounding box
    for box in bbox:
        x1, y1, x2, y2, class_id = box
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)

    plt.tight_layout()
    plt.show()

    print('=' * 50)

    i += 1
    if i == 5:
        break


# %%
parquet_path = registry.row(0, named=True)['image_path']
lf = pl.scan_parquet(parquet_path)
schema = lf.collect_schema()
for col_name, dtype in schema.items():
    print(dtype)

# %%
npz_path = Path(__file__).parent.joinpath('debug_fold1-00000-of-00001_roi_0.npz')

with np.load(npz_path, allow_pickle=True) as data:
    img = data['image']
    ins_mat = data['instance_matrix']
    cat = data['category_dict']

# %%
first_parquet_row = registry.row(0, named=True)
print(f'Processing archive: {first_parquet_row["image_path"]}')

# Call process_item (this returns the generator)
roi_generator = ingestor.process_item(first_parquet_row)

cats = []
for _, _, _, category_dict in roi_generator:
    cats.extend(category_dict)
