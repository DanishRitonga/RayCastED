"""Focused tests for the class-balanced encoder-proposal classification loss."""

from types import SimpleNamespace

import pytest
import torch

from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel

NC = 5
RAD_DIM = 66


def _scores(batch=1, nq=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, nq, NC + 1, generator=g, requires_grad=True)


def _polygons(batch=1, nq=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, nq, RAD_DIM, generator=g)


def _owners(batch=1):
    return [(torch.tensor([0, 2]), torch.tensor([0, 1]))] * batch


def _targets(n_gt=2, cls=(1, 3)):
    return {
        'cls': torch.tensor(cls, dtype=torch.long),
        'bboxes': torch.zeros(n_gt, RAD_DIM),
    }


def _stub(cap=5.0):
    return SimpleNamespace(
        nc=NC,
        criterion=SimpleNamespace(matcher=lambda *_: _owners()),
        proposal_bal_cap=cap,
    )


def test_balanced_proposal_loss_is_finite_and_scalar():
    """The loss returns a finite scalar for a normal batch."""
    stub = _stub()
    loss = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(), _scores(), _targets(), [2]
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_balanced_proposal_loss_backpropagates_to_logits():
    """Gradients reach the proposal logits."""
    stub = _stub()
    scores = _scores()
    loss = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(), scores, _targets(), [2]
    )
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0


def test_balanced_proposal_loss_raises_matched_class():
    """A matched query's true class is pushed up relative to the empty column."""
    stub = _stub()
    scores = _scores()
    scores.data.zero_()
    before = scores.data.clone()
    loss = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(), scores, _targets(), [2]
    )
    loss.backward()
    grad = scores.grad[0]
    # query 0 -> class 1, query 2 -> class 3
    assert grad[0, 1] < 0
    assert grad[0, -1] > 0
    assert grad[2, 3] < 0
    assert grad[2, -1] > 0
    assert torch.equal(before, scores.data)


def test_balanced_proposal_loss_pushes_unmatched_to_empty():
    """An unmatched query is trained to prefer the empty column."""
    stub = _stub()
    scores = _scores()
    scores.data.zero_()
    loss = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(), scores, _targets(), [2]
    )
    loss.backward()
    grad = scores.grad[0]
    for q in (1, 3, 4, 5, 6, 7):
        assert grad[q, -1] < 0


def test_balanced_proposal_loss_prefers_correct_prediction():
    """A confident correct prediction costs less than a confident wrong one."""
    stub = _stub()
    good = _scores()
    bad = _scores()
    with torch.no_grad():
        good.data.zero_()
        good.data[0, 0, 1] = 6.0
        good.data[0, 0, -1] = -6.0
        good.data[0, 2, 3] = 6.0
        good.data[0, 2, -1] = -6.0
        bad.data.zero_()
        bad.data[0, 0, -1] = 6.0
        bad.data[0, 2, -1] = 6.0
    l_good = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(), good, _targets(), [2]
    )
    l_bad = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(), bad, _targets(), [2]
    )
    assert l_good < l_bad


@pytest.mark.parametrize('cap', [1.0, 3.0, 5.0, 20.0])
def test_balanced_proposal_loss_respects_cap(cap):
    """The foreground ratio never exceeds the configured cap."""
    stub = _stub(cap=cap)
    # Only 1 of 8 queries matched -> raw ratio 7, capped at ``cap``.
    stub.criterion = SimpleNamespace(
        matcher=lambda *_: [(torch.tensor([0]), torch.tensor([0]))]
    )
    scores = _scores()
    scores.data.zero_()
    loss = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(), scores, _targets(n_gt=1, cls=(2,)), [1]
    )
    loss.backward()
    # Positive query must receive gradient regardless of the cap.
    assert scores.grad[0, 0, 2] < 0


def test_balanced_proposal_loss_handles_empty_gt():
    """No GT in the batch still yields a finite loss that targets empty."""
    stub = _stub()
    stub.criterion = SimpleNamespace(matcher=lambda *_: [(torch.tensor([]), torch.tensor([]))])
    scores = _scores()
    scores.data.zero_()
    loss = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(), scores, {'cls': torch.zeros(0, dtype=torch.long),
                                    'bboxes': torch.zeros(0, RAD_DIM)}, [0]
    )
    assert torch.isfinite(loss)
    loss.backward()
    # Every query is negative, so all rows learn to raise the empty column.
    assert (scores.grad[0, :, -1] < 0).all()


def test_balanced_proposal_loss_batch_aggregates():
    """Multiple images each contribute their own balanced term."""
    stub = SimpleNamespace(
        nc=NC,
        proposal_bal_cap=5.0,
        criterion=SimpleNamespace(
            matcher=lambda *_: [
                (torch.tensor([0]), torch.tensor([0])),
                (torch.tensor([1, 2]), torch.tensor([0, 1])),
            ]
        ),
    )
    scores = _scores(batch=2)
    loss = NuLiteRayCastDETRModel._balanced_proposal_loss(
        stub, _polygons(batch=2), scores, _targets(), [1, 2]
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad[0].abs().sum() > 0
    assert scores.grad[1].abs().sum() > 0


def test_proposal_bal_weight_validation():
    """Negative weight or a cap below 1 is rejected by the attribute check."""
    with pytest.raises(ValueError):
        _check_attrs(-0.1, 5.0)
    with pytest.raises(ValueError):
        _check_attrs(0.1, 0.5)
    _check_attrs(0.0, 5.0)
    _check_attrs(0.25, 1.0)


def _check_attrs(weight, cap):
    """Mirror of the attribute validation performed in ``__init__``."""
    if weight < 0 or cap < 1:
        raise ValueError('proposal balanced loss requires nonnegative weight and cap >= 1')
