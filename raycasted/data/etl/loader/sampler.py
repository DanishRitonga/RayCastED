"""RayCastED — Weighted class-and-tissue sampler for balanced training.

Computes per-tile sampling weights from class and tissue distributions stored
in .npz tiles. Tiles containing rare classes or rare tissue types are
up-sampled so the model sees balanced class frequencies over each epoch.

Adapted from LSP-DETR's WeightedClassAndTissueSampler (Pekár, LKCell).
Uses the same gamma-smoothed inverse-frequency formula:

    w = N / (gamma * count + (1 - gamma) * N)

where gamma controls the strength of re-weighting (0 = uniform, 1 = full
inverse-frequency).
"""

import numpy as np
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(
    tile_classes: list[np.ndarray],
    num_classes: int,
    gamma: float = 0.85,
) -> np.ndarray:
    """Compute per-tile weights based on cell-class distribution.

    Args:
        tile_classes: List of 1-D arrays, each containing the class IDs
            present in one tile (deduplicated per tile).
        num_classes: Total number of valid classes.
        gamma: Re-weighting strength in [0, 1].

    Returns:
        Per-tile weight array of shape (num_tiles,).
    """
    binary = np.zeros((len(tile_classes), num_classes), dtype=np.bool_)
    for i, classes in enumerate(tile_classes):
        binary[i, np.unique(classes)] = True

    class_presence = binary.sum(axis=0).astype(np.float64)
    total = class_presence.sum()
    if total == 0:
        return np.ones(len(tile_classes), dtype=np.float64)

    weight_per_class = total / (gamma * class_presence + (1 - gamma) * total)
    tile_weights = (1 - gamma) * binary.max(axis=-1).astype(np.float64) + gamma * (binary * weight_per_class).sum(
        axis=-1
    )

    nonzero = tile_weights != 0
    if not nonzero.all():
        tile_weights[~nonzero] = tile_weights[nonzero].min()

    return tile_weights


def compute_tissue_weights(
    tissues: np.ndarray,
    gamma: float = 0.85,
) -> np.ndarray:
    """Compute per-tile weights based on tissue-type distribution.

    Args:
        tissues: 1-D integer array of tissue IDs, one per tile.
        gamma: Re-weighting strength in [0, 1].

    Returns:
        Per-tile weight array of shape (num_tiles,).
    """
    _, counts = np.unique(tissues, return_counts=True)
    n = len(tissues)
    weights = n / (gamma * counts + (1 - gamma) * n)
    return weights[tissues].astype(np.float64)


class WeightedClassSampler(WeightedRandomSampler):
    """WeightedRandomSampler that balances both cell-class and tissue-type.

    Reads per-tile class histograms and tissue IDs, computes inverse-frequency
    weights for each axis, normalises, and sums them.
    """

    def __init__(
        self,
        tile_classes: list[np.ndarray],
        tissues: np.ndarray,
        num_classes: int,
        gamma: float = 0.85,
        replacement: bool = True,
    ) -> None:
        assert 0 <= gamma <= 1, f'gamma must be in [0, 1], got {gamma}'

        tw = compute_tissue_weights(tissues, gamma)
        cw = compute_class_weights(tile_classes, num_classes, gamma)
        weights = tw / tw.max() + cw / cw.max()

        super().__init__(weights.tolist(), len(tissues), replacement)
