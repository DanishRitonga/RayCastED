"""E2E Dual-Assignment Fix Validation.

Tests that the critical bug is fixed:
- one2many branch uses topk=13 (configurable)
- one2one branch uses topk=1 (enforced)
- Both branches use RayCastAssigner

Run with: uv run python tests/phase_6/test_e2e_fix.py
"""

import torch
from unittest.mock import MagicMock

from raycasted.model.loss import RayCastE2ELoss, RayCastDetectionLoss
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


def test_e2e_topk_configuration():
    """Verify one2one uses topk=1, one2many uses topk=13."""
    model = _make_mock_model()
    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=13)

    # Check branch types
    assert isinstance(e2e.one2many, RayCastDetectionLoss), (
        f'one2many should be RayCastDetectionLoss, got {type(e2e.one2many).__name__}'
    )
    assert isinstance(e2e.one2one, RayCastDetectionLoss), (
        f'one2one should be RayCastDetectionLoss, got {type(e2e.one2one).__name__}'
    )

    # Check assigner types
    assert isinstance(e2e.one2many.assigner, RayCastAssigner), 'one2many should use RayCastAssigner'
    assert isinstance(e2e.one2one.assigner, RayCastAssigner), 'one2one should use RayCastAssigner'

    # CRITICAL: Check topk/topk2 values
    assert e2e.one2many.assigner.topk == 13, f'one2many.topk should be 13, got {e2e.one2many.assigner.topk}'
    assert e2e.one2many.assigner.topk2 == 13, f'one2many.topk2 should be 13 (no secondary filter), got {e2e.one2many.assigner.topk2}'
    assert e2e.one2one.assigner.topk == 7, f'one2one.topk should be 7 (candidate pool), got {e2e.one2one.assigner.topk}'
    assert e2e.one2one.assigner.topk2 == 1, f'one2one.topk2 should be 1 (NMS-free), got {e2e.one2one.assigner.topk2}'

    print('PASS: E2E NMS-free — o2m.topk=13, o2o.topk=7→1')


def test_e2e_custom_tal_topk():
    """Verify custom tal_topk is respected for one2many."""
    model = _make_mock_model()

    # Test with custom topk=20
    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=20)

    assert e2e.one2many.assigner.topk == 20, f'one2many.topk should be 20, got {e2e.one2many.assigner.topk}'
    assert e2e.one2many.assigner.topk2 == 20, f'one2many.topk2 should be 20, got {e2e.one2many.assigner.topk2}'
    assert e2e.one2one.assigner.topk == 10, f'one2one.topk should be 10 (candidate pool), got {e2e.one2one.assigner.topk}'
    assert e2e.one2one.assigner.topk2 == 1, f'one2one.topk2 should be 1 (NMS-free), got {e2e.one2one.assigner.topk2}'

    print('PASS: Custom tal_topk=20 — o2m.topk=20, o2o.topk=10→1')


def test_e2e_assertions_trigger_on_bad_config():
    """Verify assertions catch misconfigurations."""
    model = _make_mock_model()

    # Manually break the configuration to test assertions
    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=13)

    # Simulate a bug: set one2many.topk to 0 (invalid)
    e2e.one2many.assigner.topk = 0

    try:
        e2e.update()  # Should trigger assertion
        assert False, 'Expected assertion error for one2many.topk=0'
    except AssertionError as e:
        assert 'E2E violation' in str(e), f'Wrong error message: {e}'
        print('PASS: Assertion catches one2many.topk=0 violation')


def test_e2e_default_parameters_match_baseline():
    """Verify defaults match proven ablation baseline (Run 7)."""
    model = _make_mock_model()

    # Test with defaults (no parameters specified)
    e2e = RayCastE2ELoss(model)

    # Check loss defaults (updated for dense cell optimization)
    assert e2e.one2many.focal_loss == False, f'Default focal_loss should be False, got {e2e.one2many.focal_loss}'
    assert e2e.one2many.assigner.radius_scale == 1.5, (
        f'Default radius_scale should be 1.5, got {e2e.one2many.assigner.radius_scale}'
    )
    assert e2e.one2many.assigner.topk == 13, f'Default tal_topk should be 13, got {e2e.one2many.assigner.topk}'
    assert e2e.one2many.lambda_l1 == 5.0, f'Default lambda_l1 should be 5.0 (LSP-DETR), got {e2e.one2many.lambda_l1}'
    assert e2e.one2many.lambda_piou == 0.5, (
        f'Default lambda_piou should be 0.5 (minimal for topk=1), got {e2e.one2many.lambda_piou}'
    )

    print('PASS: Default parameters match NMS-free + LSP-DETR loss configuration')


if __name__ == '__main__':
    test_e2e_topk_configuration()
    test_e2e_custom_tal_topk()
    test_e2e_assertions_trigger_on_bad_config()
    test_e2e_default_parameters_match_baseline()
    print('\n✓ All E2E fix validation passed!')
