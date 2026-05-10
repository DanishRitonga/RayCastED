# %%
import types
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Adjust these imports to match your project structure
from raycasted.data.etl import CSVPolygonIngestor, ETLConfig, GeoJSONIngestor, ParquetIngestor


def get_ingestor(dataset_name: str, config: dict):
    """Factory to instantiate the correct ingestor based on config."""
    ingestor_key = config.get('ingestor')
    if not ingestor_key:
        method = config.get('ingestion_method')
        ingestor_key = {1: 'parquet', 2: 'parquet', 3: 'parquet', 4: 'geojson', 5: 'csv_poly'}.get(method)

    registry = {
        'parquet': ParquetIngestor,
        'geojson': GeoJSONIngestor,
        'csv_poly': CSVPolygonIngestor,
    }
    cls = registry.get(ingestor_key)
    if cls is None:
        raise ValueError(f'Unknown ingestor "{ingestor_key}" for {dataset_name}')
    return cls(config=config)


def extract_categories_from_registry(dataset_name: str, ingestor, limit: int = 10):
    """Iterates through the registry and collects all standardized labels."""
    registry = ingestor.get_registry()

    if registry.is_empty():
        print(f'  [!] Registry empty for {dataset_name}')
        return Counter()

    # Limit the number of files processed for speed during EDA
    total_rows = len(registry)
    if limit is not None:
        total_rows = min(limit, total_rows)

    print(f'  -> Scanning {total_rows} files for {dataset_name}...')

    cat_counter = Counter()

    for idx, row in enumerate(registry.head(total_rows).iter_rows(named=True)):
        result = ingestor.process_item(row)

        # Handle 1-to-Many (Parquet Generators) vs 1-to-1 (GeoJson/CSV Tuples)
        if isinstance(result, types.GeneratorType):
            for roi_id, img, mask, cats in result:
                # Add all labels to counter, excluding the 'background'
                valid_cats = [c for c in cats if c != 'background']
                cat_counter.update(valid_cats)
        else:
            roi_id, img, mask, cats = result
            valid_cats = [c for c in cats if c != 'background']
            cat_counter.update(valid_cats)

    return cat_counter


def plot_stacked_ratios(all_counts: dict):
    """Generates a 100% stacked bar chart of category distributions."""
    datasets = list(all_counts.keys())

    # Find all unique categories across the entire pipeline
    all_categories = set()
    for counter in all_counts.values():
        all_categories.update(counter.keys())
    all_categories = sorted(list(all_categories))

    # Calculate ratios (percentages)
    category_ratios = defaultdict(list)

    for dataset in datasets:
        total_cells = sum(all_counts[dataset].values())

        for cat in all_categories:
            if total_cells > 0:
                ratio = (all_counts[dataset][cat] / total_cells) * 100
            else:
                ratio = 0.0
            category_ratios[cat].append(ratio)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(datasets))

    # Use a nice colormap to differentiate classes
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(all_categories))]

    for idx, cat in enumerate(all_categories):
        ratios = category_ratios[cat]
        ax.bar(datasets, ratios, label=cat, bottom=bottom, color=colors[idx], edgecolor='white')
        bottom += np.array(ratios)

    ax.set_ylabel('Percentage of Total Instances (%)')
    ax.set_title('Standardized Category Distribution Ratio by Dataset')
    ax.legend(title='Standardized Classes', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    yaml_path = Path(__file__).parent.joinpath('dataset.yaml')

    print('1. Parsing Configuration...')
    config_manager = ETLConfig(yaml_path)

    # We will store the counts here: {'PanNuke': Counter({'tumor': 500, ...}), ...}
    dataset_counts = {}

    print('\n2. Extracting Ontologies...')

    # Iterate dynamically through whatever datasets are active in your YAML
    # Adjust config_manager.get_dataset_names() to match your actual method
    active_datasets = config_manager.raw_config.get('datasets', {}).keys()

    for ds_name in active_datasets:
        print(f'\nProcessing Dataset: {ds_name}')

        # Load config and inject namespace
        ds_config = config_manager.get_dataset_config(ds_name)
        ds_config['namespace_map'] = config_manager.get_namespace_map(ds_name)

        try:
            ingestor = get_ingestor(ds_name, ds_config)

            # Change limit=None to process the entire dataset
            counts = extract_categories_from_registry(ds_name, ingestor, limit=5)

            dataset_counts[ds_name] = counts
            print(f'  -> Found {sum(counts.values())} total instances.')

        except Exception as e:
            print(f'  [X] Failed to process {ds_name}: {e}')

    print('\n3. Generating Distribution Plot...')

    plot_stacked_ratios(dataset_counts)

# %%
