"""Phase 11 GPU test — End-to-end pipeline orchestrator.

Validates:
  - Full pipeline: ingest -> transform -> train completes on real PUMA data
  - Ingestion produces correct <output>/ingested/<dataset>/<split>/ layout
  - Transform produces correct <output>/transformed/ layout
  - Training runs on GPU with auto-generated data YAML

The PUMA dataset only has training data (no official val split), so this test
runs ingest + transform via the pipeline, then manually splits tiles into
train/val before launching training.

Run with: PYTHONPATH=. uv run python tests/phase_11/test_pipeline_gpu.py
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PUMA_DATA_ROOT = '/mnt/data/projects/RayCastED/raycasted/data/dataset'


def _check_cuda():
    if not torch.cuda.is_available():
        print('SKIP: no CUDA device available')
        return False
    return True


def _write_puma_only_config(tmp_dir: str) -> str:
    """Write a minimal dataset.yaml with only PUMA and annotation_type=raycast."""
    config = {
        'global_settings': {
            'root_dir': PUMA_DATA_ROOT,
            'max_size': 1024,
            'crop_size': 640,
            'output_mpp': 0.25,
            'patching_overlap_pct': 10,
            'annotation_type': 'raycast',
            'global_cell_map': {
                'Background': 0,
                'Lymphocyte': 1,
                'Immune Cells': 2,
                'Epithelial': 3,
                'Stroma': 4,
                'Ignore': 255,
            },
            'global_tissue_map': {
                'Melanoma': 4,
            },
        },
        'datasets': {
            'PUMA': {
                'root_dir': 'PUMA',
                'ingestion_method': 4,
                'native_mpp': 0.22,
                'split_separation': 'filename_regex',
                'split_args': {
                    'regex': '(training_set)',
                },
                'modality_separation': 'physical_parallel',
                'modality_dirs': {
                    'image_dir': '01_training_dataset_tif_ROIs',
                    'mask_dir': '01_training_dataset_geojson_nuclei',
                },
                'modality_pairing_rule': {
                    'match_extension': '.geojson',
                    'add_suffix': '_nuclei',
                },
                'namespace_map': {
                    'nuclei_tumor': 'Epithelial',
                    'nuclei_lymphocyte': 'Lymphocyte',
                    'nuclei_plasma_cell': 'Immune Cells',
                    'nuclei_histiocyte': 'Immune Cells',
                    'nuclei_melanophage': 'Immune Cells',
                    'nuclei_neutrophil': 'Immune Cells',
                    'nuclei_stroma': 'Stroma',
                    'nuclei_endothelium': 'Stroma',
                    'nuclei_epithelium': 'Epithelial',
                    'nuclei_apoptosis': 'Ignore',
                },
                'tissue_type': 'Melanoma',
            },
        },
    }
    config_path = os.path.join(tmp_dir, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def _reorganize_to_train_val(transformed_dir: Path, val_fraction: float = 0.15):
    """Move tiles from whatever split dirs into train/ and val/.

    PUMA only has training data, so all tiles end up in one directory.
    This manually creates the train/val split that RayCastTrainer expects.
    """
    train_dir = transformed_dir / 'train'
    val_dir = transformed_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Collect all .npz tiles from any subdirectory
    all_tiles = []
    for subdir in transformed_dir.iterdir():
        if subdir.is_dir() and subdir.name not in ('train', 'val'):
            all_tiles.extend(subdir.glob('*.npz'))

    # Also check root level
    all_tiles.extend(transformed_dir.glob('*.npz'))

    if not all_tiles:
        print(f'WARNING: no tiles found in {transformed_dir}')
        return 0, 0

    # Split
    n_val = max(1, int(len(all_tiles) * val_fraction))
    val_tiles = all_tiles[:n_val]
    train_tiles = all_tiles[n_val:]

    for t in train_tiles:
        shutil.move(str(t), str(train_dir / t.name))
    for t in val_tiles:
        shutil.move(str(t), str(val_dir / t.name))

    return len(train_tiles), len(val_tiles)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_pipeline():
    """Full pipeline: ingest -> transform -> train on real PUMA data with GPU."""
    if not _check_cuda():
        return

    puma_dir = Path(PUMA_DATA_ROOT) / 'PUMA'
    if not puma_dir.exists():
        print(f'SKIP: PUMA data not found at {puma_dir}')
        return

    from raycasted.model.head import RayCastDetect
    from raycasted.model.train import RayCastTrainer
    from raycasted.pipeline import RayCastPipeline

    start = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = _write_puma_only_config(tmp_dir)
        output_dir = os.path.join(tmp_dir, 'output')

        # --- Stage 1 + 2: Ingest + Transform via pipeline ---
        pipeline = RayCastPipeline(
            config_path=config_path,
            output_dir=output_dir,
        )
        pipeline.run(stage='ingest')
        pipeline.run(stage='transform')

        # --- Verify ingestion output ---
        ingested_dir = Path(output_dir) / 'ingested'
        assert ingested_dir.exists(), f'Ingested dir not found: {ingested_dir}'
        puma_ingested = ingested_dir / 'PUMA'
        assert puma_ingested.exists(), f'PUMA ingested dir not found: {puma_ingested}'
        npz_files = list(puma_ingested.rglob('*.npz'))
        assert len(npz_files) > 0, 'No .npz files in ingested output'
        print(f'\n  Ingestion: {len(npz_files)} .npz ROI files')

        # --- Reorganize transformed tiles into train/val ---
        transformed_dir = Path(output_dir) / 'transformed'
        n_train, n_val = _reorganize_to_train_val(transformed_dir)
        print(f'  Transform: {n_train} train + {n_val} val tiles')
        assert n_train > 0, 'No training tiles after reorganization'
        assert n_val > 0, 'No validation tiles after reorganization'

        # --- Generate data.yaml ---
        cell_map = pipeline.config.global_settings.get('global_cell_map', {})
        names = {v: k for k, v in cell_map.items() if v != 255}
        nc = max(names.keys()) + 1 if names else 1
        data_yaml = {
            'train': str(transformed_dir / 'train'),
            'val': str(transformed_dir / 'val'),
            'nc': nc,
            'names': names,
        }
        yaml_path = transformed_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        print(f'  data.yaml: nc={nc}, names={names}')

        # --- Stage 3: Train on GPU ---
        trainer = RayCastTrainer(
            overrides={
                'model': 'yolo11n.yaml',
                'data': str(yaml_path),
                'epochs': 2,
                'batch': 4,
                'imgsz': 640,
                'mosaic': 0.0,
                'mixup': 0.0,
                'cache': False,
                'pretrained': False,
                'device': '0',
                'workers': 0,
                'verbose': False,
                'plots': False,
                'save': False,
                'val': True,
            }
        )
        trainer.train()

        elapsed = time.time() - start

        head = trainer.model.model[-1]
        assert isinstance(head, RayCastDetect), f'Expected RayCastDetect, got {type(head).__name__}'
        ta = trainer.model.training_args
        assert ta['nc'] == nc
        assert ta['n_rays'] == 32

        print(
            f'\nPASS: end-to-end pipeline — ingest({len(npz_files)} ROIs) -> transform({n_train}+{n_val} tiles) -> train(2 epochs GPU) in {elapsed:.1f}s'
        )


if __name__ == '__main__':
    print('Phase 11 GPU tests')
    print('=' * 60)
    test_end_to_end_pipeline()
    print('\nPhase 11 GPU tests complete.')
