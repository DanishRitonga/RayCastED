"""Focused checks for geometry-gated foreground supervision of Hungarian owners."""

from types import SimpleNamespace

import pytest
import torch

from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel


def _polygon(x, y, radius=0.08):
    return torch.tensor([x, y] + [radius] * 64)


def _loss(owners, pred, scores, gt, classes, threshold=0.5):
    stub = SimpleNamespace(
        criterion=SimpleNamespace(matcher=lambda *_: owners),
        viable_owner_iou=threshold,
        viable_owner_margin=1.0,
        _paired_star_iou_sampled=NuLiteRayCastDETRModel._paired_star_iou_sampled,
    )
    return NuLiteRayCastDETRModel._viable_owner_foreground_loss(
        stub,
        pred[None],
        scores[None],
        {'bboxes': gt, 'cls': classes},
        [len(gt)],
    )


def test_sampled_iou_identical_and_displaced():
    """The IoU gate accepts coincident stars and rejects distant ones."""
    target = torch.stack((_polygon(0.3, 0.3), _polygon(0.7, 0.7)))
    pred = torch.stack((_polygon(0.3, 0.3), _polygon(0.2, 0.7)))
    iou = NuLiteRayCastDETRModel._paired_star_iou_sampled(pred, target)
    assert iou[0].item() == pytest.approx(1.0)
    assert iou[1].item() == pytest.approx(0.0)


def test_only_viable_owner_gets_extra_foreground_gradient():
    """No gradient reaches poor owners or unassigned duplicate queries."""
    target = torch.stack((_polygon(0.3, 0.3), _polygon(0.7, 0.7)))
    pred = torch.stack((_polygon(0.3, 0.3), _polygon(0.2, 0.7), _polygon(0.3, 0.3)))
    scores = torch.zeros(3, 6, requires_grad=True)
    owners = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    value = _loss(owners, pred, scores, target, torch.tensor([2, 3]))
    value.backward()
    assert scores.grad[0, 2] < 0
    assert scores.grad[0, -1] > 0
    assert scores.grad[0, [0, 1, 3, 4]].abs().sum() == 0
    assert scores.grad[1:].abs().sum() == 0  # poor owner and duplicate are untouched


def test_empty_ownership_has_zero_loss_and_gradient():
    """Empty images keep a valid zero-gradient loss."""
    pred = _polygon(0.3, 0.3)[None]
    scores = torch.zeros(1, 6, requires_grad=True)
    empty = torch.empty(0, dtype=torch.long)
    value = _loss([(empty, empty)], pred, scores, pred, torch.tensor([2]))
    value.backward()
    assert value.item() == 0
    assert scores.grad.abs().sum() == 0


def test_enabled_model_loss_reports_new_term(monkeypatch):
    """The enabled setting reaches the training loss and adds one telemetry slot."""
    model = NuLiteRayCastDETRModel(
        nc=5,
        n_rays=64,
        variant='fastvit_t8',
        pretrained=False,
        nq=16,
        ndl=2,
        hd=32,
        d_ffn=64,
        grid_size=0.25,
        viable_owner_fg_weight=0.5,
        verbose=False,
    )
    monkeypatch.setattr(
        model,
        '_paired_star_iou_sampled',
        lambda pred, target, samples=20: pred.new_ones(len(pred)),
    )
    model.train()
    polygon = _polygon(0.5, 0.5)[None]
    batch = {
        'img': torch.rand(1, 3, 256, 256),
        'batch_idx': torch.zeros(1, dtype=torch.long),
        'cls': torch.tensor([2]),
        'bboxes': polygon,
        'seed_map': torch.zeros(1, 1, 256, 256),
    }
    total, items = model.loss(batch)
    assert torch.isfinite(total)
    assert items.shape == (6,)
    assert items[-1] > 0
    total.backward()
    assert model.detr.dec_score_head[-1].weight.grad is not None
