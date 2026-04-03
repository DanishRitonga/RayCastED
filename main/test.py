# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from raycasted.data.etl import ETLConfig, GeoJSONIngestor

pl.Config.set_fmt_str_lengths(100)


yaml_path = Path(__file__).parent.joinpath('dataset.yaml')

print('1. Parsing Configuration...')
try:
    config_manager = ETLConfig(yaml_path)
    # Fetch PUMA config and its specific namespace map
    dataset_name = 'PUMA'
    puma_config = config_manager.get_dataset_config(dataset_name)

    # Inject the namespace map into the config dict so BaseDataIngestor can find it
    print(f'✅ {dataset_name} config loaded.')
    print(f'   Resolved Root: {puma_config.get("resolved_root_dir")}')
    print(f'   Namespace Map: {puma_config.get("namespace_map")}')
except Exception as e:
    print(f'❌ Config parsing failed: {e}')

print('\n2. Initializing Ingestor & Building Registry...')
try:
    # Pass the injected config to the ingestor
    ingestor = GeoJSONIngestor(config=puma_config)
    registry = ingestor.get_registry()

    print('✅ Registry built successfully. Preview:')
    print(registry.head(3))

    if registry.is_empty():
        print('❌ Registry is empty! Check your directory paths and regex/extension rules.')

except Exception as e:
    print(f'❌ Ingestor initialization failed: {e}')

# %%
print('\n3. Testing Pixel Extraction on the First GeoJSON File...')
row = 5
# Grab the very first row from the registry as a Python dictionary
first_row = registry.row(row, named=True)
print(f'Processing ROI: {first_row["roi_id"]}')
print(f'Image: {first_row["image_path"]}')
print(f'Mask: {first_row["mask_path"]}')

try:
    # Call process_item (returns the tuple directly, not a generator!)
    roi_id, image_array, instance_matrix, cats_array = ingestor.process_item(first_row)

    print(f'\n✅ Successfully Extracted ROI: {roi_id}')
    print(f'   -> Image Array Shape: {image_array.shape}, dtype: {image_array.dtype}')
    print(f'   -> Instance Matrix Shape: {instance_matrix.shape}, dtype: {instance_matrix.dtype}')
    print(f'   -> Categories Array Shape: {cats_array.shape}, dtype: {cats_array.dtype}')

    # Validate the ontology extraction
    unique_instances = np.unique(instance_matrix)
    num_instances = len(unique_instances) - 1  # Subtract 1 for background (0)
    print(f'   -> Extracted {num_instances} polygon instances.')

    print(f'   -> First 5 Categories: {cats_array[:5]}')

    # Ensure array indices match exactly
    assert len(cats_array) == num_instances + 1, 'Category array length mismatch!'

except ValueError as e:
    print(f'\n❌ Pipeline stopped by Fail-Loud Gatekeeper: {e}')
except Exception as e:
    print(f'\n❌ Extraction failed: {e}')

print('\n4. Plotting visual sanity check...')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image_array)
axes[0].set_title(f'RGB Image ({roi_id})')
axes[0].axis('off')

# We use a masked array to make the background (0) transparent or distinct
masked_instance = np.ma.masked_where(instance_matrix == 0, instance_matrix)

axes[1].imshow(np.zeros(image_array.shape[:2]), cmap='gray')  # Show original image in background
# Overlay the polygons with a semi-transparent colormap
axes[1].imshow(masked_instance, cmap='jet', interpolation='nearest')
axes[1].set_title('Rasterized GeoJSON Polygons')
axes[1].axis('off')

plt.tight_layout()
plt.show()
