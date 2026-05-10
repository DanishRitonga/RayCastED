import types
from pathlib import Path

import numpy as np
import polars as pl

from raycasted.data.etl import (
    CSVPolygonIngestor,
    ETLConfig,
    GeoJSONIngestor,
    ParquetIngestor,
)

INGESTOR_MAP = {
    'parquet': ParquetIngestor,
    'geojson': GeoJSONIngestor,
    'csv_poly': CSVPolygonIngestor,
}

_INGESTOR_MAP_LEGACY = {
    1: ParquetIngestor,
    4: GeoJSONIngestor,
    5: CSVPolygonIngestor,
}


def cache_ingested_data():
    print('--- 📦 Starting Global Ingestion & Caching ---')

    # Using the new ETLConfig (schema defaults internally)
    yaml_path = Path(__file__).parent.joinpath('dataset.yaml')

    try:
        config_manager = ETLConfig(yaml_path)
        global_settings = config_manager.raw_config.get('global_settings', {})
        datasets = config_manager.raw_config.get('datasets', {})
    except Exception as e:
        print(f'❌ Failed to parse config: {e}')
        return

    if not datasets:
        print('❌ No datasets found in the configuration.')
        return

    # Set up the intermediate cache directory
    # Defaults to './ingested_cache' if not specified in global_settings
    cache_root = Path(global_settings.get('cache_dir', './cache/L1/'))
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f'📁 Cache directory set to: {cache_root.resolve()}')

    for dataset_name in datasets:
        print(f'\n{"=" * 60}')
        print(f'🧪 Processing Dataset: {dataset_name}')
        print(f'{"=" * 60}')

        try:
            dataset_cfg = config_manager.get_dataset_config(dataset_name)
            ingestor_key = dataset_cfg.get('ingestor')
            if not ingestor_key:
                method_int = dataset_cfg.get('ingestion_method')
                ingestor_key = {1: 'parquet', 4: 'geojson', 5: 'csv_poly'}.get(method_int)
            IngestorClass = INGESTOR_MAP.get(ingestor_key)

            if not IngestorClass:
                print(f'⚠️  Skipping {dataset_name}: Unknown ingestor "{ingestor_key}"')
                continue

            # Initialize the Ingestor
            ingestor = IngestorClass(config=dataset_cfg, global_settings=global_settings)
            registry = ingestor.get_registry()

            if registry.is_empty():
                print('⚠️  Registry is empty! Skipping.')
                continue

            # Create a dedicated cache folder for this specific dataset
            dataset_cache_dir = cache_root / dataset_name
            dataset_cache_dir.mkdir(exist_ok=True)

            print(f'✅ Registry Built: Found {len(registry)} valid file pairings.')

            for row in registry.iter_rows(named=True):
                print(f'⏳ Opening archive/file: {row["roi_id"]} ...')

                # This returns EITHER a single tuple OR a generator
                result = ingestor.process_item(row)

                # Unify them! If it's a single tuple, wrap it in a list so we can loop it.
                # If it's a generator, we just loop through it naturally.
                items_to_process = result if isinstance(result, types.GeneratorType) else [result]

                # Now extract EVERY patch from the file
                for item in items_to_process:
                    try:
                        roi_id, image_array, instance_matrix, cats_array, tissue_origin = item

                        save_path = dataset_cache_dir / f'{roi_id}.npz'
                        print(f'   💾 Saving ROI: {roi_id}...')

                        np.savez_compressed(
                            save_path,
                            image=image_array,
                            instance_matrix=instance_matrix,
                            cats_array=cats_array,
                            tissue_origin=np.array([tissue_origin], dtype=np.int16),
                        )
                        print('   ✅ Saved successfully!')
                    except Exception as e:
                        print(f'   ❌ Failed to process item: {e}')
                        continue

        except ValueError as e:
            print(f'❌ Gatekeeper Stopped Pipeline: {e}')
        except Exception as e:
            print(f'❌ Extraction Failed: {e}')

    print('\n--- 🎉 Caching Run Complete ---')


if __name__ == '__main__':
    pl.Config.set_fmt_str_lengths(100)
    cache_ingested_data()
