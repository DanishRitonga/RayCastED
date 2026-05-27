"""E2E Dual-Assignment Fix Validation.

Tests that the critical bugs are fixed:
- one2many branch uses configurable topk (default 13)
- one2one branch uses TAL with topk=max(topk//2,7), topk2=1 (or annealed)
- Hungarian assigner is not created (removed — dead end for FCN)
- Both branches use RayCastAssigner by default
- O2M/O2O decay schedule spans full training (not ~1.5 epochs)

Run with: uv run python tests/phase_6/test_e2e_fix.py
"""

from unittest.mock import MagicMock

import torch

from raycasted.model.loss import RayCastDetectionLoss, RayCastE2ELoss
from raycasted.model.tal import RayCastAssigner


def _make_mock_model(nc=4, reg_max=1):
    """Create a mock model for testing."""
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.randn(1)])
    model.args = MagicMock()
    model.args.box = 7.5
    model.args.cls = 0.5
    model.args.dfl = 1.5
    model.args.epochs = 100

    m = MagicMock()
    m.nc = nc
    m.reg_max = reg_max
    m.stride = torch.tensor([8.0, 16.0, 32.0])
    m.end2end = True
    m.raycast_dim = 34
    model.model = [MagicMock(), m]
    model.model[-1] = m
    return model


def test_e2e_dual_tal_default():
    """Verify default: both branches use TAL assigner, Hungarian is not created."""
    model = _make_mock_model()
    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=13)

    assert isinstance(e2e.one2many, RayCastDetectionLoss)
    assert isinstance(e2e.one2one, RayCastDetectionLoss)

    assert isinstance(e2e.one2many.assigner, RayCastAssigner)
    assert isinstance(e2e.one2one.assigner, RayCastAssigner)

    assert e2e.one2many.assigner.topk == 13
    assert e2e.one2many.assigner.topk2 == 13
    assert e2e.one2one.assigner.topk == 7
    assert e2e.one2one.assigner.topk2 == 1

    assert not hasattr(e2e, 'hungarian_assigner') or e2e.hungarian_assigner is None
    print('PASS: E2E dual-TAL — o2m.topk=13, o2o.topk=7, o2o.topk2=1')


def test_e2e_hungarian_not_created():
    """Verify Hungarian assigner is not created (removed — dead end for FCN)."""
    model = _make_mock_model()
    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=13)

    assert not hasattr(e2e, 'hungarian_assigner') or e2e.hungarian_assigner is None
    assert isinstance(e2e.one2one.assigner, RayCastAssigner)
    print('PASS: Hungarian assigner not created (removed — dead end for FCN)')


def test_e2e_custom_tal_topk():
    """Verify custom tal_topk is respected for one2many."""
    model = _make_mock_model()

    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=20)

    assert e2e.one2many.assigner.topk == 20
    assert e2e.one2many.assigner.topk2 == 20
    assert e2e.one2one.assigner.topk == 10  # max(20//2, 7) = 10
    assert e2e.one2one.assigner.topk2 == 1
    print('PASS: Custom tal_topk=20 — o2m.topk=20, o2o.topk=10')


def test_e2e_assertions_trigger_on_bad_config():
    """Verify assertions catch misconfigurations."""
    model = _make_mock_model()

    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=13)

    e2e.one2many.assigner.topk = 0

    try:
        e2e.update()
        assert False, 'Expected assertion error for one2many.topk=0'
    except AssertionError as e:
        assert 'E2E violation' in str(e), f'Wrong error message: {e}'
        print('PASS: Assertion catches one2many.topk=0 violation')


def test_e2e_alpha_beta_configurable():
    """Verify alpha/beta are passed through to assigners."""
    model = _make_mock_model()
    e2e = RayCastE2ELoss(model, max_epochs=200, assigner_alpha=0.25, assigner_beta=3.0)

    assert e2e.one2many.assigner.alpha == 0.25
    assert e2e.one2many.assigner.beta == 3.0
    assert e2e.one2one.assigner.alpha == 0.25
    assert e2e.one2one.assigner.beta == 3.0
    print('PASS: alpha=0.25, beta=3.0 configured on both branches')


def test_e2e_default_parameters_match_baseline():
    """Verify defaults match proven ablation baseline."""
    model = _make_mock_model()

    e2e = RayCastE2ELoss(model)

    assert e2e.one2many.assigner.radius_scale == 1.5
    assert e2e.one2many.assigner.topk == 13
    assert e2e.one2many.lambda_l1 == 14.0
    assert e2e.one2many.lambda_xy == 500.0
    assert e2e.one2many.lambda_cls == 2.0
    print('PASS: Default parameters match current loss configuration')


def test_e2e_decay_schedule_spans_full_training():
    """Verify o2m/o2o decay schedule spans full training using epoch units."""
    model = _make_mock_model()
    spe = 133
    max_epochs = 200
    e2e = RayCastE2ELoss(model, max_epochs=max_epochs, tal_topk=13, steps_per_epoch=spe)

    assert e2e.one2one.hyp.epochs == max_epochs

    e2e.update()
    assert e2e.o2m > 0.79, f'After 1 epoch o2m should be ~0.8, got {e2e.o2m}'

    for _ in range(max_epochs - 2):
        e2e.update()
    assert e2e.o2m < 0.11, f'After {max_epochs} epochs o2m should be ~0.1, got {e2e.o2m}'

    print(f'PASS: O2M decay spans full {max_epochs} epochs')


if __name__ == '__main__':
    test_e2e_dual_tal_default()
    test_e2e_hungarian_not_created()
    test_e2e_custom_tal_topk()
    test_e2e_assertions_trigger_on_bad_config()
    test_e2e_alpha_beta_configurable()
    test_e2e_default_parameters_match_baseline()
    test_e2e_decay_schedule_spans_full_training()
    print('\n✓ All E2E fix validation passed!')
