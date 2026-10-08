import numpy as np
import torch

from raycasted.model.train_nulite_seg import (
    _center_focal_loss,
    _centroid_potential,
    _soft_dice_loss,
    _TileDS,
    parse_args,
)


def test_fg_center_cli_selects_dual_target(tmp_path):
    """The new CLI mode selects the two-output target contract."""
    args = parse_args(
        [
            '--train-tiles',
            'train',
            '--train-parquet',
            'train.parquet',
            '--val-tiles',
            'val',
            '--val-parquet',
            'val.parquet',
            '--out',
            str(tmp_path / 'oracle.pt'),
            '--fg-center-target',
        ]
    )
    assert args.fg_center_target
    assert not args.center_potential_target
    assert args.val_center_weight == 0.5
    assert args.center_loss_weight == 0.1


def test_dual_target_dataset_returns_independent_fg_and_center_maps():
    """The dataset returns broad foreground and sparse centers separately."""
    images = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    masks = np.zeros((1, 16, 16), dtype=np.uint8)
    masks[0, 3:13, 4:12] = 1
    centers = np.zeros((1, 16, 16), dtype=np.float32)
    centers[0, 8, 8] = 1.0
    valleys = np.zeros((1, 16, 16), dtype=np.uint8)
    dataset = _TileDS(images, masks, train=False, centers=centers, valleys=valleys, dual_target=True)

    image, foreground, center, valley = dataset[0]

    assert image.shape == (3, 16, 16)
    assert foreground.dtype == torch.float32
    assert foreground.sum() == 80
    assert center.sum() == 1
    assert valley.sum() == 0


def test_dual_losses_backpropagate_to_both_output_channels():
    """Both oracle logits receive a nonzero training gradient."""
    logits = torch.zeros((2, 2, 16, 16), requires_grad=True)
    foreground = torch.zeros((2, 16, 16))
    foreground[:, 4:12, 4:12] = 1.0
    center = torch.zeros((2, 16, 16))
    center[:, 8, 8] = 1.0

    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], foreground)
        + _soft_dice_loss(logits[:, 0], foreground)
        + _center_focal_loss(logits[:, 1], center)
    )
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[:, 0].abs().sum() > 0
    assert logits.grad[:, 1].abs().sum() > 0


def test_centroid_target_keeps_touching_instances_as_distinct_peaks():
    """Touching masks retain distinct peaks and a supervised midpoint valley."""
    first = np.zeros((32, 32), dtype=bool)
    second = np.zeros((32, 32), dtype=bool)
    first[8:24, 3:15] = True
    second[8:24, 15:27] = True

    target, valley = _centroid_potential([first, second], height=32, width=32)

    assert np.count_nonzero(target == 1.0) == 2
    assert valley.sum() > 0
    assert target[16, 15] < 0.5
