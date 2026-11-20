"""Fold-provenance tests: oracle checkpoints must record and verify their fold."""

import argparse
from pathlib import Path

import pytest
import torch

from raycasted.model.oracle_provenance import (
    OracleFoldError,
    check_oracle_fold_binding,
    dataset_folds,
    fold_name,
    infer_fold,
    read_oracle_provenance,
    resolve_folds,
)
from raycasted.model.train_nulite_seg import oracle_provenance_config


def _tiles(root: Path, split: str, fold: int, n: int = 2) -> Path:
    """Create a split directory holding n fold-tagged ROI tiles."""
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f'fold{fold}-item{i}_roi_{i}.npz').write_bytes(b'')
    return d


def _ckpt(path: Path, folds=None, fold_names=None, arch='NuLite-T(fastvit_t8) fold1-only') -> Path:
    """Write a minimal oracle checkpoint with the given provenance metadata."""
    config = {'data.dataset': 'PanNuke'}
    if folds is not None:
        config['folds'] = folds
    if fold_names is not None:
        config['fold_names'] = fold_names
    torch.save({'model_state_dict': {}, 'arch': arch, 'config': config}, str(path))
    return path


def test_infer_fold_reads_parquet_tile_and_missing_paths(tmp_path):
    """Folds come from the parquet name, the tile names, or nowhere."""
    assert infer_fold('data/PanNuke/fold2-00000-of-00001.parquet') == 2
    assert infer_fold(_tiles(tmp_path, 'train', 3)) == 3
    assert infer_fold(tmp_path / 'transformed' / 'train') is None
    assert infer_fold(None) is None
    assert fold_name(2) == 'fold2' and fold_name(None) == 'unknown'


def test_infer_fold_does_not_match_bare_numbers(tmp_path):
    """``fold`` must be followed by exactly one digit, not a longer number."""
    assert infer_fold('runs/fold20/model.pt') is None
    assert infer_fold('/data/fold1-extra/fold3-x.npz') == 1


def test_resolve_folds_prefers_overrides_then_parquet(tmp_path):
    """Explicit fold arguments win; otherwise the parquet beats the tile dir."""
    tiles = _tiles(tmp_path, 'train', 2)
    resolved = resolve_folds(
        train_tiles=tiles,
        train_parquet='data/fold2-00000-of-00001.parquet',
        val_tiles=_tiles(tmp_path, 'val', 3),
        val_parquet='data/fold3-00000-of-00001.parquet',
    )
    assert resolved['train'] == (2, 'train parquet')
    assert resolved['val'] == (3, 'val parquet')

    overridden = resolve_folds(
        train_tiles=tiles,
        train_parquet='data/fold2-00000-of-00001.parquet',
        val_tiles=_tiles(tmp_path, 'val', 3),
        val_parquet='data/fold3-00000-of-00001.parquet',
        overrides={'train': 3, 'val': 1},
    )
    assert overridden['train'] == (3, '--train-fold')
    assert overridden['val'] == (1, '--val-fold')


def test_dataset_folds_reads_split_directories_from_data_yaml(tmp_path):
    """The detector's data.yaml names the folds it reads for train and val."""
    _tiles(tmp_path, 'train', 2)
    _tiles(tmp_path, 'val', 3)
    yaml_path = tmp_path / 'data.yaml'
    yaml_path.write_text('train: train\nval: val\nnc: 5\nnames:\n  0: Neoplastic\n')

    folds = dataset_folds(yaml_path)
    assert folds['train'] == 2
    assert folds['val'] == 3
    assert folds['test'] is None


def test_dataset_folds_resolves_path_key(tmp_path):
    """A ``path:`` root is honoured for relative split names."""
    root = tmp_path / 'v2-r3'
    _tiles(root, 'transformed/train', 3)
    _tiles(root, 'transformed/val', 1)
    yaml_path = tmp_path / 'data.yaml'
    yaml_path.write_text(f'path: {root}\ntrain: transformed/train\nval: transformed/val\n')

    folds = dataset_folds(yaml_path)
    assert folds['train'] == 3 and folds['val'] == 1


def test_read_oracle_provenance_prefers_fold_names(tmp_path):
    """``fold_names`` wins, and legacy 0-based ``folds`` is still understood."""
    modern = _ckpt(
        tmp_path / 'modern.pt',
        folds={'train': 1, 'val': 2},
        fold_names={'train': 'fold2', 'val': 'fold3'},
    )
    prov = read_oracle_provenance(modern)
    assert (prov['train_fold'], prov['val_fold']) == (2, 3)

    legacy = _ckpt(tmp_path / 'legacy.pt', folds={'train': 0, 'val': 1})
    prov = read_oracle_provenance(legacy)
    assert (prov['train_fold'], prov['val_fold']) == (1, 2)


def test_check_reports_ok_when_oracle_matches_training_fold(tmp_path):
    """The normal case: oracle trained on the run's own training fold."""
    run_root = tmp_path / 'run'
    _tiles(run_root, 'train', 2)
    _tiles(run_root, 'val', 3)
    yaml_path = run_root / 'data.yaml'
    yaml_path.write_text('train: train\nval: val\n')
    oracle = _ckpt(tmp_path / 'oracle.pt', fold_names={'train': 'fold2', 'val': 'fold3'})

    result = check_oracle_fold_binding(oracle, data_yaml=yaml_path)
    assert result['status'] == 'ok'
    assert result['oracle']['train_fold'] == 2


@pytest.mark.parametrize(
    ('oracle_fold', 'run_train', 'run_val', 'status'),
    [
        ('fold1', 2, 3, 'test-contaminated'),
        ('fold3', 2, 3, 'val-contaminated'),
    ],
)
def test_check_raises_when_oracle_saw_a_held_out_fold(tmp_path, oracle_fold, run_train, run_val, status):
    """An oracle trained on the run's val or test fold is a hard error."""
    run_root = tmp_path / 'run'
    _tiles(run_root, 'train', run_train)
    _tiles(run_root, 'val', run_val)
    yaml_path = run_root / 'data.yaml'
    yaml_path.write_text('train: train\nval: val\n')
    oracle = _ckpt(tmp_path / 'oracle.pt', fold_names={'train': oracle_fold, 'val': 'fold2'})

    with pytest.raises(OracleFoldError) as excinfo:
        check_oracle_fold_binding(oracle, data_yaml=yaml_path)
    assert oracle_fold in str(excinfo.value)
    assert 'nulite_ckpt' in str(excinfo.value)

    allowed = check_oracle_fold_binding(oracle, data_yaml=yaml_path, allow_leak=True)
    assert allowed['status'] == status


def test_check_warns_when_checkpoint_records_no_fold(tmp_path):
    """A checkpoint without fold metadata is unverifiable, not silently accepted."""
    run_root = tmp_path / 'run'
    _tiles(run_root, 'train', 2)
    _tiles(run_root, 'val', 3)
    yaml_path = run_root / 'data.yaml'
    yaml_path.write_text('train: train\nval: val\n')
    oracle = _ckpt(tmp_path / 'oracle.pt')

    result = check_oracle_fold_binding(oracle, data_yaml=yaml_path)
    assert result['status'] == 'unknown'


def test_oracle_provenance_config_stamps_resolved_folds(tmp_path):
    """The trainer writes the resolved folds; unresolved folds stay None."""
    args = argparse.Namespace(
        train_tiles=str(_tiles(tmp_path, 'train', 2)),
        train_parquet='data/fold2-00000-of-00001.parquet',
        val_tiles=str(_tiles(tmp_path, 'val', 3)),
        val_parquet='data/fold3-00000-of-00001.parquet',
    )
    folds = {'train': (2, 'train parquet'), 'val': (3, 'val parquet')}
    meta = oracle_provenance_config(args, folds, 500)
    assert meta['folds'] == {'train': 1, 'val': 2}
    assert meta['fold_names'] == {'train': 'fold2', 'val': 'fold3'}
    assert meta['provenance']['inferred']['train'] == 'train parquet'
    assert meta['provenance']['val_tiles_used'] == 500

    unknown = oracle_provenance_config(args, {'train': (None, None), 'val': (None, None)}, 500)
    assert unknown['folds'] == {'train': None, 'val': None}
    assert unknown['fold_names'] == {'train': 'unknown', 'val': 'unknown'}
