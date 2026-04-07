"""IngestionOrchestrator — drives all ingestors from YAML config.

Reads config via ETLConfig → instantiates the correct ingestor per dataset →
iterates the file registry → saves each ROI as a .npz file.

Output layout: <output_dir>/<dataset_name>/<split>/<roi_id>.npz

Each .npz contains:
    image        — uint8 HWC array
    annotations  — float32 (N, 35) raycast array (or bboxes, depending on type)
    tissue       — int32 tissue origin ID
"""

import collections
import re
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.config import ETLConfig
from .csv_poly_ingestor import CSVPolygonIngestor
from .geojson_ingestor import GeoJSONIngestor
from .mat_inst_ingestor import MatInstanceIngestor
from .parquet_ingestor import ParquetIngestor

# Dispatch map: ingestion_method → Ingestor class
# Method 3 (MatInstanceIngestor) registered per GAP-01 — must be present even
# if CoNSeP is commented out in the YAML config.
DISPATCH_MAP: dict[int, type] = {
    1: ParquetIngestor,
    3: MatInstanceIngestor,
    4: GeoJSONIngestor,
    5: CSVPolygonIngestor,
}

# Matches path traversal: "..", leading "/" or "\", any backslash
_TRAVERSAL_RE = re.compile(r'(\.\.|^[/\\]|\\)')


def _safe_path_component(value: str) -> str:
    """Sanitise a string used as a single filesystem path component.

    Rejects path traversal patterns to prevent writes outside output_dir.
    """
    if _TRAVERSAL_RE.search(value):
        raise ValueError(f'Unsafe path component rejected: {value!r}')
    return value


def _worker_process_row(
    merged_config: dict,
    method: int,
    row: dict,
    output_dir: str,
    dataset_name: str,
) -> tuple[collections.Counter, str | None]:
    """Process a single row in a worker subprocess.

    Returns (stats_counter, error_message). If error_message is not None,
    the row failed and stats will be empty.
    """
    import collections as _collections

    ingestor_cls = DISPATCH_MAP[method]
    ingestor = ingestor_cls(merged_config)
    output_path = Path(output_dir)

    stats = _collections.Counter()
    split = row.get('split', 'unassigned')

    try:
        result = ingestor.process_item(row)
    except Exception as e:
        return stats, f'{row.get("roi_id", "?")}: {e}'

    results = result if isinstance(result, Generator) else [result]

    for item in results:
        roi_id = item[0]
        image = item[1]
        annotations = item[2]
        tissue = item[3]

        safe_dataset = _safe_path_component(dataset_name)
        safe_split = _safe_path_component(split)
        safe_roi = _safe_path_component(str(roi_id))

        out_dir = output_path / safe_dataset / safe_split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{safe_roi}.npz'

        np.savez(out_path, image=image, annotations=annotations, tissue=np.int32(tissue))
        stats['roi_saved'] += 1

    return stats, None


class IngestionOrchestrator:
    """Orchestrates the ingestion of all datasets defined in the ETL config.

    Usage:
        orchestrator = IngestionOrchestrator('main/dataset.yaml', output_dir='output/')
        orchestrator.run()
    """

    def __init__(self, config_path: str, output_dir: str | None = None, workers: int = 1):
        self.config = ETLConfig(config_path)
        if output_dir is None:
            raise ValueError('output_dir is required — specify where .npz files should be written.')
        self.output_dir = Path(output_dir)
        self.workers = max(1, workers)

    def run(self, dataset_name: str | None = None) -> None:
        """Run ingestion for one or all datasets.

        Args:
            dataset_name: If provided, ingest only this dataset.
                Otherwise ingest all datasets in the config.
        """
        datasets = [dataset_name] if dataset_name else self.config.list_datasets()

        for ds_name in datasets:
            self._ingest_dataset(ds_name)

    def _ingest_dataset(self, dataset_name: str) -> None:
        """Ingest a single dataset end-to-end.

        Datasets are processed sequentially, but rows within a dataset
        are processed in parallel when workers > 1.
        """
        merged_config = self.config.get_dataset_config(dataset_name)

        method = merged_config.get('ingestion_method')
        if method not in DISPATCH_MAP:
            raise ValueError(
                f"Unknown ingestion_method={method} for dataset '{dataset_name}'. "
                f'Supported: {list(DISPATCH_MAP.keys())}'
            )

        ingestor_cls = DISPATCH_MAP[method]
        ingestor = ingestor_cls(merged_config)

        registry = ingestor.get_registry()
        if registry.is_empty():
            print(f'[{dataset_name}] No files found in registry — skipping.')
            return

        total = len(registry)
        print(f'[{dataset_name}] Processing {total} items with {ingestor_cls.__name__} ({self.workers} worker(s))...')

        rows = list(registry.iter_rows(named=True))

        if self.workers == 1:
            stats = self._process_rows_sequential(dataset_name, ingestor, rows, total)
        else:
            stats = self._process_rows_parallel(dataset_name, merged_config, method, rows, total)

        print(
            f'[{dataset_name}] Done. '
            f'roi_saved={stats["roi_saved"]}, '
            f'skipped={stats["skipped"]}, '
            f'errors={stats["errors"]}'
        )

    def _process_rows_sequential(
        self, dataset_name: str, ingestor: Any, rows: list[dict], total: int
    ) -> collections.Counter:
        """Process rows sequentially (workers=1)."""
        stats = collections.Counter()
        for idx, row in enumerate(rows):
            self._process_row(dataset_name, ingestor, row, stats)
            if (idx + 1) % 50 == 0 or idx + 1 == total:
                print(f'  [{idx + 1}/{total}] processed')
        return stats

    def _process_rows_parallel(
        self,
        dataset_name: str,
        merged_config: dict,
        method: int,
        rows: list[dict],
        total: int,
    ) -> collections.Counter:
        """Process rows in parallel across multiple worker processes.

        Each worker gets a fresh ingestor instance (ingestors hold config
        state that isn't fork-safe). Results are written to disk in the
        worker; stats are aggregated back in the main process.
        """
        stats = collections.Counter()
        n_workers = min(self.workers, len(rows))

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for idx, row in enumerate(rows):
                future = pool.submit(
                    _worker_process_row,
                    merged_config,
                    method,
                    row,
                    str(self.output_dir),
                    dataset_name,
                )
                futures[future] = idx

            for done_count, future in enumerate(as_completed(futures), 1):
                row_stats, row_error = future.result()

                if row_error:
                    print(f'  Error: {row_error}')
                    stats['errors'] += 1
                else:
                    stats.update(row_stats)

                if done_count % 50 == 0 or done_count == total:
                    print(f'  [{done_count}/{total}] processed')

        return stats

    def _process_row(
        self,
        dataset_name: str,
        ingestor: Any,
        row: dict,
        stats: collections.Counter,
    ) -> None:
        """Process a single registry row and save the result."""
        split = row.get('split', 'unassigned')

        try:
            result = ingestor.process_item(row)
        except Exception as e:
            print(f'  Error processing {row.get("roi_id", "?")}: {e}')
            stats['errors'] += 1
            return

        # ParquetIngestor.process_item() is a generator; others return a tuple.
        # Normalise both to an iterable of result tuples.
        results = result if isinstance(result, Generator) else [result]

        for item in results:
            self._save_item(dataset_name, split, item, stats)

    def _save_item(
        self,
        dataset_name: str,
        split: str,
        item: tuple,
        stats: collections.Counter,
    ) -> None:
        """Save a single processed item to .npz."""
        roi_id = item[0]
        image = item[1]
        annotations = item[2]
        tissue = item[3]

        # Sanitise all path components to prevent directory traversal
        safe_dataset = _safe_path_component(dataset_name)
        safe_split = _safe_path_component(split)
        safe_roi = _safe_path_component(str(roi_id))

        out_dir = self.output_dir / safe_dataset / safe_split
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f'{safe_roi}.npz'

        np.savez(
            out_path,
            image=image,
            annotations=annotations,
            tissue=np.int32(tissue),
        )
        stats['roi_saved'] += 1
