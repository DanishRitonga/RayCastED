"""Fold provenance for the frozen NuLite-T segmentation oracles.

The oracle's foreground map is an input to the detector, so an oracle trained
on a fold that the detector uses for validation or for testing silently leaks
labels: its frozen map already saw the nuclei the run is supposed to predict.
Before this module existed the oracle trainer stamped every checkpoint as
``fold1-only`` regardless of what it actually read, so the only record of the
real binding was the training log.

The helpers here (a) derive the fold from the paths that are actually read,
(b) let the trainer write that derivation into the checkpoint, and (c) re-check
it where the oracle is loaded into a detector.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import yaml

#: ``fold1``..``fold9`` as they appear in PanNuke parquet names and ROI tiles.
FOLD_RE = re.compile(r'fold([1-9])(?!\d)')

#: Detector training-config key that downgrades a fold leak to a warning.
ALLOW_LEAK_KEY = 'nulite_allow_fold_leak'


class OracleFoldError(ValueError):
    """Raised when a frozen oracle was trained on a fold the run does not train on."""


def fold_name(fold: int | None) -> str:
    """Render a 1-based fold index as ``fold2`` (``unknown`` when unresolved)."""
    return f'fold{fold}' if fold else 'unknown'


def infer_fold(path) -> int | None:
    """Return the 1-based PanNuke fold a path or tile directory belongs to.

    The path itself is checked first (``.../fold2-00000-of-00001.parquet``),
    then the first few tiles inside a directory (``fold2-..._roi_....npz``).
    Returns ``None`` when the path carries no fold; callers must treat that as
    unknown instead of assuming a default.
    """
    if path is None:
        return None
    p = Path(path)
    for part in p.parts:
        match = FOLD_RE.search(part)
        if match:
            return int(match.group(1))
    if p.is_dir():
        for tile in sorted(p.glob('*.npz'))[:8]:
            match = FOLD_RE.search(tile.name)
            if match:
                return int(match.group(1))
    return None


def resolve_folds(train_tiles=None, train_parquet=None, val_tiles=None, val_parquet=None, overrides=None) -> dict:
    """Resolve the train/val folds from explicit overrides, then from the paths.

    Returns ``{'train': (fold, source), 'val': (fold, source)}`` where fold is
    1-based or ``None`` and source names where the value came from.
    """
    overrides = overrides or {}
    resolved = {}
    for role in ('train', 'val'):
        override = overrides.get(role)
        if override:
            resolved[role] = (int(override), f'--{role}-fold')
            continue
        candidates = (
            (f'{role} parquet', train_parquet if role == 'train' else val_parquet),
            (f'{role} tiles', train_tiles if role == 'train' else val_tiles),
        )
        fold, source = None, None
        for label, value in candidates:
            fold = infer_fold(value)
            if fold:
                source = label
                break
        resolved[role] = (fold, source)
    return resolved


def read_oracle_provenance(ckpt_path, ckpt=None) -> dict:
    """Read the fold binding recorded inside an oracle checkpoint.

    ``folds`` holds 0-based indices (the historical convention); ``fold_names``
    holds ``fold2``-style names and takes precedence when present.
    """
    if ckpt is None:
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    config = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
    folds = config.get('folds') or {}
    names = config.get('fold_names') or {}

    def _one(role):
        named = names.get(role)
        if named:
            match = FOLD_RE.search(str(named))
            return int(match.group(1)) if match else None
        index = folds.get(role)
        return None if index is None else int(index) + 1

    return {
        'path': str(ckpt_path),
        'arch': ckpt.get('arch') if isinstance(ckpt, dict) else None,
        'train_fold': _one('train'),
        'val_fold': _one('val'),
        'oracle_kind': config.get('oracle_kind'),
        'note': config.get('note'),
    }


def dataset_folds(data_yaml) -> dict:
    """Infer the fold behind each split directory listed in a detector data.yaml.

    Split values are resolved relative to the yaml's ``path`` key when present,
    otherwise relative to the yaml's own directory (the pipeline writes bare
    ``train:``/``val:`` names next to the split directories).
    """
    folds = {'train': None, 'val': None, 'test': None}
    if not data_yaml:
        return folds
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        return folds
    try:
        spec = yaml.safe_load(yaml_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return folds
    root = spec.get('path')
    for split in folds:
        value = spec.get(split)
        if not value:
            continue
        target = Path(str(value))
        if not target.is_absolute():
            base = Path(str(root)) if root else yaml_path.parent
            if root and not base.is_absolute():
                base = yaml_path.parent / base
            target = base / target
        folds[split] = infer_fold(target)
    return folds


def check_oracle_fold_binding(
    ckpt_path,
    data_yaml=None,
    train_dir=None,
    val_dir=None,
    allow_leak=False,
    ckpt=None,
) -> dict:
    """Verify that a frozen oracle was trained on the run's own training fold.

    Prints a ``[oracle-provenance]`` line describing the binding and raises
    :class:`OracleFoldError` when the oracle's training fold is one the run
    holds out (validation or test). Set ``nulite_allow_fold_leak: true`` in the
    training config to downgrade that to a warning.

    Returns ``{'status', 'oracle', 'dataset', 'message'}`` with status one of
    ``ok``, ``unknown`` (checkpoint records no fold), ``unverified`` (the run's
    own fold cannot be inferred), ``val-contaminated``, ``test-contaminated``.
    """
    provenance = read_oracle_provenance(ckpt_path, ckpt=ckpt)
    folds = dataset_folds(data_yaml)
    if train_dir is not None:
        folds['train'] = infer_fold(train_dir)
    if val_dir is not None:
        folds['val'] = infer_fold(val_dir)

    oracle_train = provenance['train_fold']
    run_train = folds['train']
    run_val = folds['val']

    def _result(status, message):
        prefix = '[oracle-provenance]'
        if status in ('val-contaminated', 'test-contaminated'):
            print(f'{prefix} ORACLE FOLD LEAK: {message}')
            if not allow_leak:
                raise OracleFoldError(message)
        elif status == 'ok':
            print(f'{prefix} ok: {message}')
        else:
            print(f'{prefix} WARNING: {message}')
        return {'status': status, 'oracle': provenance, 'dataset': folds, 'message': message}

    if oracle_train is None:
        return _result(
            'unknown',
            f'{Path(str(ckpt_path)).name} records no training fold; its map may cover any fold. '
            'Retrain it (or pass --train-fold/--val-fold) so the binding is checkable.',
        )
    if run_train is None:
        return _result(
            'unverified',
            f"oracle was trained on {fold_name(oracle_train)}, but the run's training fold could not be "
            f'inferred from {data_yaml or train_dir or "the data config"}; check the binding by hand.',
        )
    if oracle_train == run_train:
        note = ''
        if provenance['val_fold'] is not None and run_val is not None and provenance['val_fold'] != run_val:
            oracle_val = fold_name(provenance['val_fold'])
            note = f' (oracle was selected on {oracle_val}, run validates on {fold_name(run_val)})'
        return _result(
            'ok',
            f'oracle trained on {fold_name(oracle_train)}, run trains on {fold_name(run_train)}'
            f' and holds out {fold_name(run_val)}{note}.',
        )
    if run_val is not None and oracle_train == run_val:
        return _result(
            'val-contaminated',
            f"{Path(str(ckpt_path)).name} was trained on {fold_name(oracle_train)}, which is this run's "
            f'VALIDATION fold. Its frozen foreground map already saw those labels, so validation-based '
            f'model selection is contaminated. Point nulite_ckpt at an oracle trained on '
            f'{fold_name(run_train)}.',
        )
    return _result(
        'test-contaminated',
        f'{Path(str(ckpt_path)).name} was trained on {fold_name(oracle_train)}, but this run trains on '
        f'{fold_name(run_train)} and holds out {fold_name(run_val)}; {fold_name(oracle_train)} is therefore '
        f'part of the evaluation data and the frozen map leaks held-out labels. Point nulite_ckpt at an '
        f'oracle trained on {fold_name(run_train)}.',
    )
