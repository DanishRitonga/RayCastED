r"""RayCastED — End-to-End Pipeline Orchestrator.

Chains ETL (ingestion -> transform) with training into a single configurable
pipeline. Reads one dataset.yaml and auto-generates everything needed for
training, including the split-aware data YAML with class mappings.

Usage:
    # Full pipeline
    uv run python -m raycasted.pipeline \
        --config main/dataset.yaml --output output/ \
        --epochs 100 --batch 16 --imgsz 640

    # Individual stages
    uv run python -m raycasted.pipeline \
        --config main/dataset.yaml --output output/ --stage ingest --dataset PUMA
    uv run python -m raycasted.pipeline \
        --config main/dataset.yaml --output output/ --stage transform
    uv run python -m raycasted.pipeline \
        --config main/dataset.yaml --output output/ --stage train --epochs 50
"""

import argparse
import shutil
from pathlib import Path

import yaml

from raycasted.data.etl.ingestors.ingestion_orchestrator import IngestionOrchestrator
from raycasted.data.etl.transform.transform_orchestrator import TransformOrchestrator
from raycasted.data.etl.utils.config import ETLConfig
from raycasted.data.etl.utils.constants import configure_rays
from raycasted.model.train import RayCastTrainer


class RayCastPipeline:
    """Orchestrates the full RayCastED pipeline: ingest -> transform -> train.

    Reads a single ETL YAML config and manages output directories for each
    stage. Auto-generates the training data YAML (with nc, names, train/val
    paths) from the ETL config's global_cell_map.

    Attributes:
        config: ETLConfig parsed from the dataset YAML.
        output_dir: Base output directory.
        ingested_dir: Stage 1 output (raw ROIs per dataset/split).
        transformed_dir: Stage 2 output (tiles organised by train/val split).
    """

    def __init__(
        self,
        config_path: str,
        output_dir: str,
        training_overrides: dict | None = None,
        ingest_workers: int | None = None,
        n_rays: int | None = None,
    ):
        self.config = ETLConfig(config_path)
        self.output_dir = Path(output_dir)
        self.ingested_dir = self.output_dir / 'ingested'
        self.transformed_dir = self.output_dir / 'transformed'
        self.training_overrides = training_overrides or {}
        self.ingest_workers = ingest_workers

        # Resolve n_rays: CLI > config YAML > default 32
        if n_rays is None:
            n_rays = int(self.config.global_settings.get('n_rays', 32))
        self.n_rays = int(n_rays)
        configure_rays(self.n_rays)

        # Default imgsz to crop_size from ETL config unless explicitly overridden
        if 'imgsz' not in self.training_overrides:
            self.training_overrides['imgsz'] = self.config.global_settings.get('crop_size', 640)

        # Default project to output_dir so training results land under --output, not runs/
        if 'project' not in self.training_overrides:
            self.training_overrides['project'] = str(self.output_dir.resolve())

    def run(self, stage: str = 'all', dataset: str | None = None) -> None:
        """Run pipeline stages up to and including the specified one.

        Args:
            stage: 'all', 'ingest', 'transform', or 'train'.
            dataset: Restrict ingestion to a single dataset name.
        """
        if stage in ('all', 'ingest'):
            self.ingest(dataset)
        if stage in ('all', 'transform'):
            self.transform()
        if stage in ('all', 'train'):
            self.train()

    # ------------------------------------------------------------------
    # Stage 1: Ingestion
    # ------------------------------------------------------------------

    def ingest(self, dataset: str | None = None) -> None:
        """Run IngestionOrchestrator to convert raw data to .npz ROIs."""
        print('\n' + '=' * 60)
        print('Stage 1: Ingestion')
        print('=' * 60)
        orch = IngestionOrchestrator(
            config_path=str(self.config.config_path),
            output_dir=str(self.ingested_dir),
            workers=self.ingest_workers,
        )
        orch.run(dataset)
        print(f'Ingestion output: {self.ingested_dir}')

    # ------------------------------------------------------------------
    # Stage 2: Transform (chunk + normalise + organise by split)
    # ------------------------------------------------------------------

    def transform(self) -> None:
        """Run TransformOrchestrator, then reorganise tiles by split."""
        print('\n' + '=' * 60)
        print('Stage 2: Transform')
        print('=' * 60)

        t = TransformOrchestrator(
            config_manager=self.config,
            ingested_dir=str(self.ingested_dir),
            final_output_dir=str(self.transformed_dir),
        )
        t.run_pipeline()

        # TransformOrchestrator saves tiles flat — reorganise by split
        self._organize_by_split(t.registry)

        print(f'Transform output: {self.transformed_dir}')

    def _organize_by_split(self, registry) -> None:
        """Move flat tiles into train/ and val/ subdirectories.

        TransformOrchestrator saves all tiles to a single directory but
        tracks the split label in the registry. RayCastTileDataset needs
        separate directories for train and val.
        """
        if registry is None:
            print('WARNING: No registry from transform — skipping split reorganisation.')
            return

        known_splits = {'train', 'val', 'test'}
        orphaned = 0
        moved = 0
        for row in registry.iter_rows(named=True):
            # Registry path points to the (now-deleted) temp chunked file.
            # The final tile has the same filename in transformed_dir.
            tile_name = Path(row['path']).name
            src = self.transformed_dir / tile_name
            if not src.exists():
                continue
            split = row['split']
            if split not in known_splits:
                orphaned += 1
                continue
            dst_dir = self.transformed_dir / split
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / tile_name
            shutil.move(str(src), str(dst))
            moved += 1

        if orphaned:
            print(
                f'WARNING: {orphaned} tiles with unrecognized split names were NOT moved. '
                f'Expected one of {known_splits}. Add a split_map or change split_separation.'
            )
        print(f'Reorganised {moved} tiles into train/ and val/ subdirectories.')

    # ------------------------------------------------------------------
    # Stage 3: Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Auto-generate training YAML and launch RayCastTrainer."""
        print('\n' + '=' * 60)
        print('Stage 3: Training')
        print('=' * 60)

        yaml_path = self._generate_training_yaml()

        overrides = {
            'data': yaml_path,
            **self.training_overrides,
        }
        trainer = RayCastTrainer(overrides=overrides)
        trainer.train()

    def _generate_training_yaml(self) -> str:
        """Auto-generate data.yaml from ETL config for RayCastTrainer.

        Reads global_cell_map from the ETL config to build:
          - train/val paths (pointing to transformed tiles)
          - nc (number of classes, excluding Ignore=255)
          - names (class id -> class name mapping, excluding Ignore)
        """
        cell_map = self.config.global_settings.get('global_cell_map', {})

        # Filter out Ignore(255) only — Background(0) and all other valid IDs stay
        names = {v: k for k, v in cell_map.items() if v != 255}
        nc = max(names.keys()) + 1 if names else 1
        # Fill gaps — Ultralytics expects contiguous 0-indexed names dict of length nc
        names = {i: names.get(i, f'class_{i}') for i in range(nc)}

        train_dir = self.transformed_dir / 'train'
        val_dir = self.transformed_dir / 'val'

        # Validate that split directories exist and contain tiles
        for label, d in [('train', train_dir), ('val', val_dir)]:
            if not d.exists() or not any(d.glob('*.npz')):
                raise RuntimeError(
                    f'No {label} tiles found in {d}. '
                    f'Check split_map or split_separation config — '
                    f'all tiles may have been assigned to unrecognized split names.'
                )

        yaml_data = {
            'train': str(train_dir.resolve()),
            'val': str(val_dir.resolve()),
            'nc': nc,
            'names': names,
        }

        yaml_path = self.transformed_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        print(f'Auto-generated training config: {yaml_path}')
        print(f'  Classes ({nc}): {names}')
        print(f'  Train dir: {train_dir}')
        print(f'  Val dir: {val_dir}')
        return str(yaml_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='RayCastED end-to-end pipeline')
    parser.add_argument('--config', required=True, help='Path to dataset.yaml ETL config')
    parser.add_argument('--output', required=True, help='Base output directory')
    parser.add_argument(
        '--stage',
        default='all',
        choices=['all', 'ingest', 'transform', 'train'],
        help='Pipeline stage to run (default: all)',
    )
    parser.add_argument('--dataset', default=None, help='Restrict ingestion to a single dataset')
    parser.add_argument(
        '--ingest-workers',
        type=int,
        default=None,
        help='Parallel ingestion workers per dataset (default: os.cpu_count()-1)',
    )

    # Training overrides
    parser.add_argument('--model', default=None, help='Model architecture YAML (overrides --variant)')
    parser.add_argument(
        '--variant',
        default='s',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO26 variant: n=2.6M, s=10M, m=22M, l=26M, x=59M params (default: s)',
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None, help='Input image size (default: from config crop_size)')
    parser.add_argument('--n-rays', type=int, default=None, help='Number of radial rays (default: from config or 32)')
    parser.add_argument('--device', default='', help='Device (cpu, 0, 0,1)')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--lr0', type=float, default=None, help='Initial learning rate')
    parser.add_argument('--project', default=None, help='Project directory for saves')
    parser.add_argument('--name', default=None, help='Experiment name')

    args = parser.parse_args()

    # Read variant from config if not overridden by CLI
    variant = args.variant
    if args.model is None:
        try:
            import yaml as _yaml

            with open(args.config) as f:
                cfg = _yaml.safe_load(f)
            variant = cfg.get('global_settings', {}).get('variant', args.variant)
        except Exception:
            pass

    # Build training overrides from CLI args
    model_yaml = args.model if args.model else f'yolo26{variant}.yaml'
    training_overrides = {
        'model': model_yaml,
        'optimizer': 'MuSGD',
        'epochs': args.epochs,
        'batch': args.batch,
        'workers': args.workers,
        'pretrained': False,
        'cache': False,
    }
    if args.imgsz is not None:
        training_overrides['imgsz'] = args.imgsz
    if args.device:
        training_overrides['device'] = args.device
    if args.lr0 is not None:
        training_overrides['lr0'] = args.lr0
    if args.project:
        training_overrides['project'] = args.project
    if args.name:
        training_overrides['name'] = args.name

    pipeline = RayCastPipeline(
        config_path=args.config,
        output_dir=args.output,
        training_overrides=training_overrides,
        ingest_workers=args.ingest_workers,
        n_rays=args.n_rays,
    )
    pipeline.run(stage=args.stage, dataset=args.dataset)


if __name__ == '__main__':
    main()
