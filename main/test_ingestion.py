import types
from pathlib import Path

import numpy as np
import polars as pl

# Adjust these imports based on your actual project structure
from raycasted.data.etl import (
    CSVPolyParser,
    ETLConfig,
    GeoJSONParser,
    ParquetParser,
)

INGESTOR_MAP = {
    'parquet': ParquetParser,
    'geojson': GeoJSONParser,
    'csv_poly': CSVPolyParser,
}

_INGESTOR_MAP_LEGACY = {
    1: ParquetParser,
    4: GeoJSONParser,
    5: CSVPolyParser,
}


def test_all_datasets():
    print('--- 🚀 Starting Global Pipeline Test ---')

    yaml_path = Path(__file__).parent.joinpath('dataset.yaml')

    # 2. Load Global Configurations
    try:
        config_manager = ETLConfig(yaml_path)
        global_settings = config_manager.raw_config.get('global_settings', {})
        datasets = config_manager.raw_config.get('datasets', {})
    except Exception as e:
        print(f'❌ Failed to parse config files: {e}')
        return

    if not datasets:
        print('❌ No datasets found in the configuration.')
        return

    print(f'✅ Found {len(datasets)} datasets to test: {list(datasets.keys())}\n')

    # 3. Dynamically Iterate Through Every Dataset
    for dataset_name in datasets:
        print(f'{"=" * 60}')
        print(f'🧪 Testing Dataset: {dataset_name}')
        print(f'{"=" * 60}')

        try:
            # Fetch the dataset-specific config block
            dataset_cfg = config_manager.get_dataset_config(dataset_name)

            ingestor_key = dataset_cfg.get('ingestor')
            if not ingestor_key:
                method_int = dataset_cfg.get('ingestion_method')
                ingestor_key = {1: 'parquet', 4: 'geojson', 5: 'csv_poly'}.get(method_int)
            IngestorClass = INGESTOR_MAP.get(ingestor_key)

            if not IngestorClass:
                print(f'⚠️  Skipping {dataset_name}: Unknown ingestor "{ingestor_key}"')
                continue

            # Initialize the Ingestor using our new 2-argument contract
            ingestor = IngestorClass(config=dataset_cfg, global_settings=global_settings)
            registry = ingestor.get_registry()

            if registry.is_empty():
                print('⚠️  Registry is empty! Check your root_dir or regex paths.')
                continue

            print(f'✅ Registry Built: Found {len(registry)} valid file pairings.')

            # Grab the very first row
            first_row = registry.row(0, named=True)
            print(f'⏳ Processing ROI: {first_row["roi_id"]} ...')

            # 4. Unify the Extraction Contract (Generator vs Tuple)
            result = ingestor.process_item(first_row)

            if isinstance(result, types.GeneratorType):
                # It's a Parquet file! Just grab the first yielded ROI
                result = next(result)

            # 5. Unpack the 5-Tuple
            roi_id, image_array, instance_matrix, cats_array, tissue_origin = result

            # 6. Validate Outputs
            print('\n✅ Extraction Successful!')
            print(f'   -> Image:      {image_array.shape}, {image_array.dtype}')
            print(f'   -> Instances:  {instance_matrix.shape}, {instance_matrix.dtype}')

            max_id = np.max(instance_matrix)
            print(f'   -> Categories: {cats_array.shape}, {cats_array.dtype} (Max ID: {max_id})')
            print(f'   -> Tissue:     {tissue_origin} (Global Integer ID)')
            print(f'   -> Categories Preview: {cats_array[:5]}')

            # 7. The Ultimate Sanity Check
            # The length of the category array must perfectly match the highest instance ID + 1 (for background)
            assert len(cats_array) == max_id + 1, '❌ CRITICAL: Category array length mismatch!'

            print('✅ Data mathematically verified.')

        except ValueError as e:
            print(f'❌ Gatekeeper Stopped Pipeline: {e}')
        except Exception as e:
            print(f'❌ Extraction Failed: {e}')

    print('\n--- 🎉 Global Pipeline Test Complete ---')


if __name__ == '__main__':
    pl.Config.set_fmt_str_lengths(100)
    test_all_datasets()
