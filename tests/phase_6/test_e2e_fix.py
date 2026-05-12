"""E2E Dual-Assignment Fix Validation.

Tests that the critical bugs are fixed:
- one2many branch uses topk=13 (configurable)
- one2one branch uses Hungarian matching (default) or topk2=1 (legacy TAL)
- Both branches use RayCastAssigner or HungarianRayCastAssigner

Run with: uv run python tests/phase_6/test_e2e_fix.py
"""

from unittest.mock import MagicMock

import torch

from raycasted.model.loss import RayCastDetectionLoss, RayCastE2ELoss
from raycasted.model.tal import HungarianRayCastAssigner, RayCastAssigner


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


def test_e2e_hungarian_default():
    """Verify default: one2one uses Hungarian matching."""
    model = _make_mock_model()
    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=13)

    # Check branch types
    assert isinstance(e2e.one2many, RayCastDetectionLoss)
    assert isinstance(e2e.one2one, RayCastDetectionLoss)

    # Check assigner types
    assert isinstance(e2e.one2many.assigner, RayCastAssigner)
    assert isinstance(e2e.one2one.assigner, HungarianRayCastAssigner), (
        f'one2one should use HungarianRayCastAssigner by default, got {type(e2e.one2one.assigner).__name__}'
    )

    # Check one2many topk
    assert e2e.one2many.assigner.topk == 13
    assert e2e.one2many.assigner.topk2 == 13

    print('PASS: E2E NMS-free — o2m.topk=13, o2o=Hungarian')


def test_e2e_tal_fallback():
    """Verify use_hungarian_o2o=False falls back to TAL topk2=1."""
    model = _make_mock_model()
    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=13, use_hungarian_o2o=False)

    assert isinstance(e2e.one2one.assigner, RayCastAssigner)
    assert not isinstance(e2e.one2one.assigner, HungarianRayCastAssigner)
    assert e2e.one2one.assigner.topk == 7
    assert e2e.one2one.assigner.topk2 == 1

    print('PASS: TAL fallback — o2o.topk=7, o2o.topk2=1')


def test_e2e_custom_tal_topk():
    """Verify custom tal_topk is respected for one2many."""
    model = _make_mock_model()

    # Test with custom topk=20
    e2e = RayCastE2ELoss(model, max_epochs=200, tal_topk=20)

    assert e2e.one2many.assigner.topk == 20
    assert e2e.one2many.assigner.topk2 == 20
    assert isinstance(e2e.one2one.assigner, HungarianRayCastAssigner)

    print('PASS: Custom tal_topk=20 — o2m.topk=20, o2o=Hungarian')


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


def test_e2e_alpha_beta_configurable():
    """Verify alpha/beta are passed through to assigners."""
    model = _make_mock_model()
    e2e = RayCastE2ELoss(model, max_epochs=200, assigner_alpha=0.25, assigner_beta=3.0)

    assert e2e.one2many.assigner.alpha == 0.25, f'one2many alpha should be 0.25, got {e2e.one2many.assigner.alpha}'
    assert e2e.one2many.assigner.beta == 3.0, f'one2many beta should be 3.0, got {e2e.one2many.assigner.beta}'
    assert e2e.one2one.assigner.alpha == 0.25, f'one2one alpha should be 0.25, got {e2e.one2one.assigner.alpha}'
    assert e2e.one2one.assigner.beta == 3.0, f'one2one beta should be 3.0, got {e2e.one2one.assigner.beta}'

    print('PASS: alpha=0.25, beta=3.0 configured on both branches')


def test_e2e_default_parameters_match_baseline():
    """Verify defaults match proven ablation baseline (Run 7)."""
    model = _make_mock_model()

    # Test with defaults (no parameters specified)
    e2e = RayCastE2ELoss(model)

    # Check loss defaults (rebalanced: xy and L1 share gradient equally)
    assert e2e.one2many.assigner.radius_scale == 1.5, (
        f'Default radius_scale should be 1.5, got {e2e.one2many.assigner.radius_scale}'
    )
    assert e2e.one2many.assigner.topk == 13, f'Default tal_topk should be 13, got {e2e.one2many.assigner.topk}'
    assert e2e.one2many.lambda_l1 == 25.0, f'Default lambda_l1 should be 25.0, got {e2e.one2many.lambda_l1}'
    assert e2e.one2many.lambda_xy == 1500.0, f'Default lambda_xy should be 1500.0, got {e2e.one2many.lambda_xy}'
    assert e2e.one2many.lambda_cls == 2.0, f'Default lambda_cls should be 2.0, got {e2e.one2many.lambda_cls}'
    assert e2e.one2many.assigner.cost_class == 1.0, (
        f'Default cost_class should be 1.0, got {e2e.one2many.assigner.cost_class}'
    )
    assert e2e.one2many.assigner.cost_centroid == 1.0, (
        f'Default cost_centroid should be 1.0, got {e2e.one2many.assigner.cost_centroid}'
    )
    assert e2e.one2many.assigner.cost_ray == 1.0, (
        f'Default cost_ray should be 1.0, got {e2e.one2many.assigner.cost_ray}'
    )

    print('PASS: Default parameters match NMS-free + LSP-DETR loss configuration')


if __name__ == '__main__':
    test_e2e_hungarian_default()
    test_e2e_tal_fallback()
    test_e2e_custom_tal_topk()
    test_e2e_assertions_trigger_on_bad_config()
    test_e2e_alpha_beta_configurable()
    test_e2e_default_parameters_match_baseline()
    print('\n✓ All E2E fix validation passed!')
