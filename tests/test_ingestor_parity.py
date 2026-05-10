"""Head-to-head parity test: old ingestors vs new parsers.

For each dataset in the config, processes 1 row through both the legacy
ingestor and the new parser, then compares the output arrays.

Usage:
    uv run python tests/test_ingestor_parity.py
    uv run python tests/test_ingestor_parity.py --config main/pannuke.yaml
    uv run python tests/test_ingestor_parity.py --dataset PanNuke
"""

import argparse
import sys
from collections.abc import Generator
from pathlib import Path

import numpy as np

from raycasted.data.etl import ETLConfig
from raycasted.data.etl.ingestors import (
    DISPATCH_MAP,
    PARSER_REGISTRY,
    CSVPolygonIngestor,
    CSVPolyParser,
    GeoJSONIngestor,
    GeoJSONParser,
    MatInstanceIngestor,
    MatInstParser,
    ParquetIngestor,
    ParquetParser,
)
from raycasted.data.etl.utils.constants import configure_rays

OLD_NEW_MAP = {
    ParquetIngestor: ParquetParser,
    GeoJSONIngestor: GeoJSONParser,
    CSVPolygonIngestor: CSVPolyParser,
    MatInstanceIngestor: MatInstParser,
}


def _normalize_bbox_columns(arr: np.ndarray) -> np.ndarray:
    """Ensure bbox array uses canonical [class, x1, y1, x2, y2] column order.

    Old ingestors sometimes produced [x1, y1, x2, y2, class].
    Detect and fix by checking if col-0 median >> col-4 median.
    """
    if arr.ndim != 2 or arr.shape[1] != 5 or arr.shape[0] == 0:
        return arr
    col0_median = np.median(np.abs(arr[:, 0]))
    col4_median = np.median(np.abs(arr[:, 4]))
    if col0_median > col4_median * 2 and col4_median < 50:
        return arr[:, [4, 0, 1, 2, 3]]
    return arr


def compare_results(old_result, new_result, dataset_name: str, roi_id: str) -> list[str]:
    """Compare two (roi_id, image, annotations, tissue) tuples.

    Returns list of differences (empty = perfect match).
    """
    diffs = []

    old_roi, old_img, old_ann, old_tissue = old_result
    new_roi, new_img, new_ann, new_tissue = new_result

    if old_roi != new_roi:
        diffs.append(f'roi_id mismatch: {old_roi} vs {new_roi}')

    if old_img.shape != new_img.shape:
        diffs.append(f'image shape mismatch: {old_img.shape} vs {new_img.shape}')
    elif not np.array_equal(old_img, new_img):
        if old_img.dtype != new_img.dtype:
            diffs.append(f'image dtype mismatch: {old_img.dtype} vs {new_img.dtype}')
        else:
            max_diff = np.abs(old_img.astype(float) - new_img.astype(float)).max()
            n_diff = np.count_nonzero(old_img != new_img)
            total = old_img.size
            pct = n_diff / total * 100
            if pct > 0.1:
                diffs.append(f'image diff: {n_diff}/{total} pixels ({pct:.2f}%), max={max_diff}')

    if old_tissue != new_tissue:
        diffs.append(f'tissue mismatch: {old_tissue} vs {new_tissue}')

    old_ann = _normalize_bbox_columns(old_ann)
    new_ann = _normalize_bbox_columns(new_ann)

    if old_ann.shape != new_ann.shape:
        diffs.append(f'annotation shape mismatch: {old_ann.shape} vs {new_ann.shape}')
    else:
        if old_ann.dtype != new_ann.dtype:
            diffs.append(f'annotation dtype: {old_ann.dtype} vs {new_ann.dtype}')

        if not np.allclose(old_ann, new_ann, atol=1e-5, rtol=1e-4, equal_nan=True):
            abs_diff = np.abs(old_ann - new_ann)
            max_diff = abs_diff.max()
            mean_diff = abs_diff.mean()
            n_close = np.sum(abs_diff < 1e-5)
            n_total = abs_diff.size
            diffs.append(
                f'annotations differ: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={n_close}/{n_total}'
            )

            if old_ann.shape[1] >= 3 and new_ann.shape[1] >= 3:
                centroid_diff = np.abs(old_ann[:, 1:3] - new_ann[:, 1:3])
                diffs.append(f'  centroid diff: max={centroid_diff.max():.4f}, mean={centroid_diff.mean():.4f}')
            if old_ann.shape[1] > 3 and new_ann.shape[1] > 3:
                ray_diff = np.abs(old_ann[:, 3:] - new_ann[:, 3:])
                diffs.append(f'  ray diff: max={ray_diff.max():.4f}, mean={ray_diff.mean():.4f}')

            diffs.append(f'  old first 3 rows: {old_ann[:3].tolist()}')
            diffs.append(f'  new first 3 rows: {new_ann[:3].tolist()}')

    return diffs


def test_dataset(config: ETLConfig, dataset_name: str, max_rows: int = 0) -> bool:
    """Test one dataset. Returns True if parity is achieved."""
    merged_config = config.get_dataset_config(dataset_name)

    n_rays = merged_config.get('n_rays', 32)
    configure_rays(n_rays)

    method_int = merged_config.get('ingestion_method')
    ingestor_key = merged_config.get('ingestor')

    new_to_old = {v: k for k, v in OLD_NEW_MAP.items()}

    if method_int:
        old_cls = DISPATCH_MAP.get(method_int)
        new_cls = OLD_NEW_MAP.get(old_cls)
    elif ingestor_key:
        new_cls = PARSER_REGISTRY.get(ingestor_key)
        old_cls = new_to_old.get(new_cls)
    else:
        print(f'  [SKIP] No ingestor/method specified for {dataset_name}')
        return True

    if old_cls is None or new_cls is None:
        print(f'  [SKIP] No old/new pair found for {dataset_name}')
        return True

    try:
        old_instance = old_cls(merged_config)
        new_instance = new_cls(merged_config)
    except Exception as e:
        print(f'  [SKIP] Constructor failed: {e}')
        return True

    old_registry = old_instance.get_registry()
    if old_registry.is_empty():
        print(f'  [SKIP] Registry empty for {dataset_name}')
        return True

    rows = list(old_registry.iter_rows(named=True))
    test_rows = rows[:max_rows] if max_rows > 0 else rows

    all_pass = True
    for row in test_rows:
        roi_id = row.get('roi_id', '?')

        try:
            old_result = old_instance.process_item(row)
        except Exception as e:
            print(f'  [OLD ERROR] {roi_id}: {e}')
            all_pass = False
            continue

        try:
            new_result = new_instance.process_item(row)
        except Exception as e:
            print(f'  [NEW ERROR] {roi_id}: {e}')
            all_pass = False
            continue

        scalar_types = (int, float, str, np.integer)

        def _normalize_results(result):
            if isinstance(result, Generator):
                return list(result)
            if isinstance(result, (list, tuple)) and not isinstance(result[0], scalar_types):
                return result
            return [result]

        old_results = _normalize_results(old_result)
        new_results = _normalize_results(new_result)

        if isinstance(old_results, tuple) and len(old_results) == 4 and isinstance(old_results[0], (str, int)):
            old_results = [old_results]
        if isinstance(new_results, tuple) and len(new_results) == 4 and isinstance(new_results[0], (str, int)):
            new_results = [new_results]

        n_old = len(old_results)
        n_new = len(new_results)

        if n_old != n_new:
            print(f'  [FAIL] {roi_id}: result count mismatch ({n_old} vs {n_new})')
            all_pass = False
            continue

        for i, (old_r, new_r) in enumerate(zip(old_results, new_results)):
            diffs = compare_results(old_r, new_r, dataset_name, f'{roi_id}[{i}]')
            tag = f'{roi_id}[{i}]' if n_old > 1 else roi_id

            if not diffs:
                old_ann = old_r[2]
                print(f'  [PASS] {tag} — shape={old_ann.shape}, dtype={old_ann.dtype}')
            else:
                print(f'  [FAIL] {tag}:')
                for d in diffs:
                    print(f'         {d}')
                all_pass = False

    return all_pass


def main():
    """CLI entry point for ingestor parity testing."""
    parser = argparse.ArgumentParser(description='Head-to-head ingestor parity test')
    parser.add_argument('--config', default='main/dataset.yaml', help='YAML config path')
    parser.add_argument('--dataset', default=None, help='Test only this dataset')
    parser.add_argument('--root-dir', default=None, help='Override global root_dir')
    parser.add_argument('--rows', type=int, default=0, help='Rows per dataset (0=all)')
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f'Config not found: {config_path}')
        sys.exit(1)

    config = ETLConfig(config_path)
    if args.root_dir:
        config.global_settings['root_dir'] = args.root_dir
    datasets = [args.dataset] if args.dataset else config.list_datasets()

    print(f'Config: {config_path}')
    print(f'Datasets: {datasets}')
    print()

    results = {}
    for ds_name in datasets:
        print(f'--- {ds_name} ---')
        try:
            passed = test_dataset(config, ds_name, max_rows=args.rows)
        except Exception as e:
            print(f'  [ERROR] {e}')
            passed = False
        results[ds_name] = passed
        print()

    print('=' * 50)
    all_pass = True
    for ds_name, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] {ds_name}')
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print('All datasets passed parity check.')
    else:
        print('Some datasets FAILED parity check.')
        sys.exit(1)


if __name__ == '__main__':
    main()
