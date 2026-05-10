import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .normalizer import NormalizerAndPadder
from .spatialChunker import SpatialChunker


class TransformOrchestrator:
    """Orchestrates the ETL transform pipeline: chunking, profiling, normalization."""

    def __init__(self, config_manager, ingested_dir: str, final_output_dir: str, use_gpu: bool = False):
        self.config = config_manager.get_global_config()
        self.ingested_dir = Path(ingested_dir)
        self.final_output_dir = Path(final_output_dir)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu

        # Memory-only transformers
        self.chunker = SpatialChunker(self.config)
        self.registry: pl.DataFrame = None
        self._chunked_dir: str | None = None

    def run_pipeline(self):
        """The Master Execution Flow."""
        # 1. Run Chunking
        self.registry = self._chunk_and_index()

        # 2. Run Profiling
        profile_path = self._build_population_profile()

        # 3. Instantiate Stage 3
        self.normalizer = NormalizerAndPadder(self.config, profile_path, use_gpu=self.use_gpu)

        print('\n--- Stage 3: Normalization, Padding, & Final Cache ---')

        # Iterate over the perfectly chunked registry
        for row in self.registry.iter_rows(named=True):
            npz_path = Path(row['path'])
            data = np.load(npz_path)

            # Load into RAM
            img = data['image']
            annotations = data.get('annotations', data.get('bboxes'))
            tissue = data['tissue']

            # Run the memory-only Stage 3 transformer
            final_img, final_annotations, content_h, content_w = self.normalizer.process_roi(img, annotations)

            # Save to the Final PyTorch-Ready Directory
            save_path = self.final_output_dir / npz_path.name
            np.savez_compressed(
                save_path,
                image=final_img,
                annotations=final_annotations,
                tissue=tissue,
                content_h=np.int32(content_h),
                content_w=np.int32(content_w),
            )

        # Cleanup temp chunked directory
        if self._chunked_dir:
            shutil.rmtree(self._chunked_dir, ignore_errors=True)

        print('\nGlobal Transformation Pipeline Complete! Data is ready for PyTorch.')

    def _chunk_and_index(self) -> pl.DataFrame:
        """Stage 1: Chunk ingested ROIs spatially and build a registry.

        Discovers all .npz files from ingestion, applies SpatialChunker,
        saves chunks to a temp directory, and returns a Polars registry.

        Returns:
            Polars DataFrame with columns: path, split, roi_id, dataset, chunk_id
        """
        self._chunked_dir = tempfile.mkdtemp(prefix='raycasted_chunked_')
        chunked_path = Path(self._chunked_dir)

        records = []

        # Layout: <ingested_dir>/<dataset>/<split>/<roi_id>.npz
        for npz_path in sorted(self.ingested_dir.rglob('*.npz')):
            split = npz_path.parent.name
            roi_id = npz_path.stem
            dataset = npz_path.parent.parent.name

            data = np.load(npz_path)
            img = data['image']
            annotations = data.get('annotations', data.get('bboxes'))
            tissue = int(data['tissue'])

            for chunk_id, img_chunk, ann_chunk, tissue_val in self.chunker.process_roi(
                roi_id, img, annotations, tissue
            ):
                out_path = chunked_path / f'{chunk_id}.npz'
                np.savez_compressed(
                    out_path,
                    image=img_chunk,
                    annotations=ann_chunk,
                    tissue=np.int32(tissue_val),
                )

                records.append(
                    {
                        'path': str(out_path),
                        'split': split,
                        'roi_id': roi_id,
                        'dataset': dataset,
                        'chunk_id': chunk_id,
                    }
                )

        if not records:
            raise RuntimeError(f'No .npz files found in {self.ingested_dir}. Run IngestionOrchestrator first.')

        print(f'Stage 1 complete: {len(records)} chunks from {len(set(r["roi_id"] for r in records))} ROIs.')
        return pl.DataFrame(records)

    def _build_population_profile(self) -> str | None:
        """Stage 2: Compute population-level stain normalization profile.

        Iterates through chunked tiles, estimates stain profiles, computes
        population statistics, and saves as JSON.

        Returns:
            Path to the saved profile JSON, or None if no valid profiles found.
        """
        estimator = self._get_estimator()
        stain_matrices = []
        max_concentrations = []

        for row in self.registry.iter_rows(named=True):
            data = np.load(row['path'])
            img = data['image']

            matrix, concentrations = estimator.get_profile(img, method='macenko')

            if matrix is None or concentrations is None:
                continue

            stain_matrices.append(matrix)
            max_concentrations.append(concentrations)

        if not stain_matrices:
            print('WARNING: No valid stain profiles extracted. Normalization will be skipped.')
            return None

        all_matrices = np.stack(stain_matrices, axis=0)
        all_concentrations = np.stack(max_concentrations, axis=0)

        mean_matrix = all_matrices.mean(axis=0)
        pop_concentrations = np.percentile(all_concentrations, 99, axis=0)

        profile = {
            'stain_matrix': mean_matrix.tolist(),
            'max_concentrations': pop_concentrations.tolist(),
            'method': 'macenko',
            'n_tiles_used': len(stain_matrices),
        }

        profile_path = self.final_output_dir / 'stain_profile.json'
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)

        print(f'Stage 2 complete: stain profile built from {len(stain_matrices)} tiles.')
        return str(profile_path)
