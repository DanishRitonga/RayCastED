# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Rectangle

from raycasted.data.etl import CSVPolygonIngestor, ETLConfig

pl.Config.set_fmt_str_lengths(100)


yaml_path = Path(__file__).parent.joinpath('dataset.yaml')

print('1. Parsing Configuration...')
try:
    config_manager = ETLConfig(yaml_path)

    # Fetch PanopTILs config
    dataset_name = 'PanopTILs'
    panoptils_config = config_manager.get_dataset_config(dataset_name)

    print(f'✅ {dataset_name} config loaded.')
    print(f'   Column Map: {panoptils_config.get("csv_column_map")}')
except Exception as e:
    print(f'❌ Config parsing failed: {e}')

print('\n2. Initializing Ingestor & Building Registry...')
try:
    ingestor = CSVPolygonIngestor(config=panoptils_config)
    registry = ingestor.get_registry()

    print('✅ Registry built successfully. Preview:')
    print(registry.head(3))

    if registry.is_empty():
        print('❌ Registry is empty! Check your directory paths and regex/extension rules.')

except Exception as e:
    print(f'❌ Ingestor initialization failed: {e}')

# %%
print('\n3. Testing Pixel Extraction on the First CSV File...')

first_row = registry.row(2, named=True)
print(f'Processing ROI: {first_row["roi_id"]}')

roi_id, image_array, bbox, origin = ingestor.process_item(first_row)
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
