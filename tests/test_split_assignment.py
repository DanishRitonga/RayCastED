"""Tests for split_map mapping and stratified random sampling.

Covers:
  - split_map: regex extraction + dict mapping (fold1→train, fold2→val, fold3→test)
  - split_map: unmapped values preserved as-is
  - stratified sampling: correct ratios, determinism with seed
  - stratified sampling: edge cases (empty registry, single ROI)
  - config validation: ratio sum check, split_map value check
"""

import polars as pl
import pytest

from raycasted.data.etl.ingestors._base import BaseDataIngestor
from raycasted.data.etl.utils.config import DatasetConfig, SplitArgs


# ---------------------------------------------------------------------------
# Concrete stub — BaseDataIngestor is abstract
# ---------------------------------------------------------------------------


class _StubIngestor(BaseDataIngestor):
    """Minimal concrete subclass for testing split assignment methods."""

    def process_item(self, row):
        pass


def _make_ingestor(config: dict) -> _StubIngestor:
    """Create a _StubIngestor with mocked __init__ (skips file discovery)."""
    obj = object.__new__(_StubIngestor)
    obj.root_dir = config.get('root_dir', '/tmp')
    obj.config = config
    obj.file_registry = None
    obj.namespace_map = config.get('namespace_map', {})
    obj.tissue_map = config.get('tissue_map', {})
    obj.global_cell_map = config.get('global_cell_map', {})
    obj.global_tissue_map = config.get('global_tissue_map', {})
    obj.target_mpp = config.get('output_mpp')
    obj.native_mpp = config.get('native_mpp')
    obj.scale_factor = 1.0
    obj.annotation_type = config.get('annotation_type', 'bbox').lower()
    return obj


def _make_registry_df(n: int) -> pl.DataFrame:
    """Create a minimal registry DataFrame with n entries."""
    return pl.DataFrame(
        {
            'roi_id': [f'roi_{i:03d}' for i in range(n)],
            'image_path': [f'/data/roi_{i:03d}.parquet' for i in range(n)],
            'mask_path': [f'/data/roi_{i:03d}.parquet' for i in range(n)],
            'split': [None] * n,
        },
        schema={'roi_id': pl.Utf8, 'image_path': pl.Utf8, 'mask_path': pl.Utf8, 'split': pl.Utf8},
    )


# ---------------------------------------------------------------------------
# SplitArgs validation tests
# ---------------------------------------------------------------------------


class TestSplitArgs:
    """Tests for the SplitArgs config model."""

    def test_defaults(self):
        args = SplitArgs()
        assert args.train_ratio == 0.8
        assert args.val_ratio == 0.1
        assert args.test_ratio == 0.1
        assert args.seed == 42
        assert args.split_map is None
        assert args.regex is None

    def test_custom_values(self):
        args = SplitArgs(
            regex='(fold[1-3])',
            split_map={'fold1': 'train', 'fold2': 'val', 'fold3': 'test'},
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=123,
        )
        assert args.split_map['fold1'] == 'train'
        assert args.train_ratio == 0.7
        assert args.seed == 123


class TestDatasetConfigSplitValidation:
    """Tests for split_separation validation in DatasetConfig."""

    def test_filename_regex_with_split_map_valid(self):
        cfg = DatasetConfig(
            root_dir='/tmp',
            ingestion_method=1,
            native_mpp=0.25,
            split_separation='filename_regex',
            modality_separation='bundled_archive',
            split_args=SplitArgs(
                regex='(fold[1-3])',
                split_map={'fold1': 'train', 'fold2': 'val', 'fold3': 'test'},
            ),
        )
        assert cfg.split_args.split_map['fold1'] == 'train'

    def test_filename_regex_with_invalid_split_map_value(self):
        with pytest.raises(ValueError, match='not a recognized split name'):
            DatasetConfig(
                root_dir='/tmp',
                ingestion_method=1,
                native_mpp=0.25,
                split_separation='filename_regex',
                modality_separation='bundled_archive',
                split_args=SplitArgs(
                    regex='(fold[1-3])',
                    split_map={'fold1': 'train', 'fold2': 'INVALID'},
                ),
            )

    def test_none_separation_validates_ratio_sum(self):
        with pytest.raises(ValueError, match='sum to ~1.0'):
            DatasetConfig(
                root_dir='/tmp',
                ingestion_method=1,
                native_mpp=0.25,
                split_separation='none',
                modality_separation='bundled_archive',
                split_args=SplitArgs(train_ratio=0.5, val_ratio=0.1, test_ratio=0.1),
            )

    def test_none_separation_valid_ratios(self):
        cfg = DatasetConfig(
            root_dir='/tmp',
            ingestion_method=1,
            native_mpp=0.25,
            split_separation='none',
            modality_separation='bundled_archive',
            split_args=SplitArgs(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1),
        )
        assert cfg.split_args.train_ratio == 0.8

    def test_physical_separation_requires_split_dirs(self):
        with pytest.raises(ValueError, match='split_dirs required'):
            DatasetConfig(
                root_dir='/tmp',
                ingestion_method=1,
                native_mpp=0.25,
                split_separation='physical',
                modality_separation='bundled_archive',
            )


# ---------------------------------------------------------------------------
# Stratified sampling tests
# ---------------------------------------------------------------------------


class TestStratifiedSampling:
    """Tests for _assign_stratified_splits."""

    def test_basic_split_ratios(self):
        """20 ROIs → ~80% train, ~10% val, ~10% test."""
        ingestor = _make_ingestor({'tissue_type': 'Breast'})
        df = _make_registry_df(20)

        result = ingestor._assign_stratified_splits(df, 0.8, 0.1, 0.1, 42)
        splits = result['split'].to_list()

        assert set(splits) <= {'train', 'val', 'test'}
        n_train = splits.count('train')
        n_val = splits.count('val')
        n_test = splits.count('test')
        assert n_train + n_val + n_test == 20
        assert n_train >= 1
        assert n_val >= 1
        assert n_test >= 1

    def test_deterministic_with_seed(self):
        """Same seed → same splits."""
        ingestor = _make_ingestor({})
        df = _make_registry_df(30)

        r1 = ingestor._assign_stratified_splits(df, 0.8, 0.1, 0.1, 42)
        r2 = ingestor._assign_stratified_splits(df, 0.8, 0.1, 0.1, 42)
        assert r1['split'].to_list() == r2['split'].to_list()

    def test_different_seed_different_splits(self):
        """Different seed → different assignment (with high probability)."""
        ingestor = _make_ingestor({})
        df = _make_registry_df(30)

        r1 = ingestor._assign_stratified_splits(df, 0.8, 0.1, 0.1, 42)
        r2 = ingestor._assign_stratified_splits(df, 0.8, 0.1, 0.1, 99)
        assert r1['split'].to_list() != r2['split'].to_list()

    def test_single_roi(self):
        """1 ROI → at least train."""
        ingestor = _make_ingestor({})
        df = _make_registry_df(1)

        result = ingestor._assign_stratified_splits(df, 0.8, 0.1, 0.1, 42)
        assert result['split'].to_list() == ['train']

    def test_empty_registry(self):
        """0 ROIs → empty result with split column."""
        ingestor = _make_ingestor({})
        df = _make_registry_df(0)

        result = ingestor._assign_stratified_splits(df, 0.8, 0.1, 0.1, 42)
        assert len(result) == 0

    def test_all_three_splits_present(self):
        """With enough ROIs, all three splits should appear."""
        ingestor = _make_ingestor({})
        df = _make_registry_df(50)

        result = ingestor._assign_stratified_splits(df, 0.8, 0.1, 0.1, 42)
        splits = set(result['split'].to_list())
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits


# ---------------------------------------------------------------------------
# split_map integration test
# ---------------------------------------------------------------------------


class TestSplitMap:
    """Tests for split_map applied after regex extraction."""

    def test_split_map_mapping(self):
        """Polars replace maps fold1→train, fold2→val, fold3→test."""
        df = pl.DataFrame({'split': ['fold1', 'fold2', 'fold3', 'fold1']})
        split_map = {'fold1': 'train', 'fold2': 'val', 'fold3': 'test'}
        result = df.with_columns(pl.col('split').replace(split_map).alias('split'))
        assert result['split'].to_list() == ['train', 'val', 'test', 'train']

    def test_split_map_preserves_unmapped(self):
        """Values not in split_map are kept as-is."""
        df = pl.DataFrame({'split': ['fold1', 'fold2', 'unknown']})
        split_map = {'fold1': 'train', 'fold2': 'val'}
        result = df.with_columns(pl.col('split').replace(split_map).alias('split'))
        assert result['split'].to_list() == ['train', 'val', 'unknown']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
