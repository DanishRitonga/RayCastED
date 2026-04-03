from pathlib import Path

import numpy as np
import polars as pl

from .normalizer import NormalizerAndPadder
from .spatialChunker import SpatialChunker


class TransformOrchestrator:
    def __init__(self, config_manager, ingested_dir: str, final_output_dir: str):
        self.config = config_manager.get_global_config()
        self.ingested_dir = Path(ingested_dir)
        self.final_output_dir = Path(final_output_dir)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)

        # Memory-only transformers
        self.chunker = SpatialChunker(self.config)
        self.registry: pl.DataFrame = None

    def run_pipeline(self):
        """The Master Execution Flow"""
        # 1. Run Chunking (Your logic from earlier, saving to a temp dir, returning a registry)
        self.registry = self._chunk_and_index()

        # 2. Run Profiling (Your Stage 2 logic)
        profile_path = self._build_population_profile()

        # 3. Instantiate Stage 3
        self.normalizer = NormalizerAndPadder(self.config, profile_path)

        print('\n--- 🎨 Stage 3: Normalization, Padding, & Final Cache ---')

        # Iterate over the perfectly chunked registry
        for row in self.registry.iter_rows(named=True):
            npz_path = Path(row['path'])
            data = np.load(npz_path)

            # Load into RAM
            img = data['image']
            annotations = data.get('bboxes', data.get('annotations'))
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

        print('\n✅ Global Transformation Pipeline Complete! Data is ready for PyTorch.')
