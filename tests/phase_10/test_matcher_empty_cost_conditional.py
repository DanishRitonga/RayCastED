"""Phase 10 tests — conditional (logit-renormalised) empty-column cost.

In ``conditional`` mode the class probability handed to the focal term is
renormalised over {true class, no-object}::

    p_cond = p_class / (p_class + p_empty + eps)

So a query that splits its probability mass with the empty column scores worse
than one that is confidently foreground at the same raw ``p_class``. Unlike the
additive term this is not row-constant: it changes which query a GT prefers.

Run with: uv run python -m pytest tests/phase_10/test_matcher_empty_cost_conditional.py -x -q
"""

import pytest
import torch

import raycasted.model.nulite_detr_loss as loss_mod
from raycasted.model.nulite_detr_loss import RayCastHungarianMatcher

N_RAYS = 8


def _capture_cost(matcher, pred_poly, pred_scores, gt_poly, gt_cls):
    """Run the matcher but record the cost matrix it hands to the solver."""
    captured = {}
    orig = loss_mod.linear_sum_assignment

    def fake(cost):
        captured['cost'] = cost.clone()
        return orig(cost)

    loss_mod.linear_sum_assignment = fake
    try:
        matcher(
            pred_poly.unsqueeze(0),
            pred_scores.unsqueeze(0),
            gt_polygons=gt_poly,
            gt_cls=gt_cls,
            gt_groups=[len(gt_cls)],
        )
    finally:
        loss_mod.linear_sum_assignment = orig
    return captured['cost']


def _circle_gt(radius=0.1):
    gt_xy = torch.tensor([[0.5, 0.5]])
    gt_poly = torch.cat([gt_xy, torch.full((1, N_RAYS), radius)], dim=-1)
    return gt_poly


def _query(offset, class_logit, empty_logit):
    pred_poly = torch.zeros(1, 2 + N_RAYS)
    pred_poly[0, :2] = torch.tensor([0.5 + offset, 0.5])
    pred_poly[0, 2:] = 0.1
    pred_scores = torch.full((1, 6), -6.0)
    pred_scores[0, 0] = class_logit
    pred_scores[0, 5] = empty_logit
    return pred_poly, pred_scores


def _matcher(empty_cost, mode='conditional', lam=1.0, has_no_object=True):
    return RayCastHungarianMatcher(
        n_rays=N_RAYS,
        lsp_mode=True,
        empty_cost=empty_cost,
        empty_cost_lambda=lam,
        empty_cost_mode=mode,
        has_no_object=has_no_object,
    )


def test_invalid_mode_raises():
    """An unknown mode string must fail loudly, not silently fall back."""
    with pytest.raises(ValueError):
        _matcher(True, mode='softmax')


def test_flag_off_matches_legacy_path():
    """empty_cost=False must reproduce the legacy cost matrix exactly."""
    poly, scores = _query(0.02, 6.0, -6.0)
    gt_poly, gt_cls = _circle_gt(), torch.tensor([0])
    legacy = RayCastHungarianMatcher(n_rays=N_RAYS, lsp_mode=True)
    off = _capture_cost(_matcher(False), poly, scores, gt_poly, gt_cls)
    ref = _capture_cost(legacy, poly, scores, gt_poly, gt_cls)
    assert torch.equal(off, ref)


def test_conditional_differs_from_additive():
    """The two modes must not be numerically equivalent."""
    poly, scores = _query(0.02, 4.0, 4.0)
    gt_poly, gt_cls = _circle_gt(), torch.tensor([0])
    additive = _capture_cost(_matcher(True, mode='additive'), poly, scores, gt_poly, gt_cls)
    conditional = _capture_cost(_matcher(True, mode='conditional'), poly, scores, gt_poly, gt_cls)
    assert not torch.allclose(additive, conditional)


def test_conditional_penalises_mass_shared_with_empty():
    """Same p_class, higher p_empty must cost more in conditional mode."""
    gt_poly, gt_cls = _circle_gt(), torch.tensor([0])
    low_empty_cost = _capture_cost(
        _matcher(True), *_query(0.02, 4.0, -8.0), gt_poly, gt_cls
    )
    high_empty_cost = _capture_cost(
        _matcher(True), *_query(0.02, 4.0, 8.0), gt_poly, gt_cls
    )
    assert float(high_empty_cost[0, 0]) > float(low_empty_cost[0, 0])


def test_conditional_penalty_monotonic_in_p_empty():
    """Cost must increase strictly with the empty-column probability."""
    gt_poly, gt_cls = _circle_gt(), torch.tensor([0])
    costs = []
    for empty_logit in (-8.0, -2.0, 0.0, 4.0, 8.0):
        cost = _capture_cost(_matcher(True), *_query(0.02, 4.0, empty_logit), gt_poly, gt_cls)
        costs.append(float(cost[0, 0]))
    assert all(b > a for a, b in zip(costs, costs[1:]))


def test_conditional_prefers_confident_fg_over_better_class_score_with_empty_mass():
    """A slightly worse p_class with no empty mass beats a better one that shares mass."""
    gt_poly, gt_cls = _circle_gt(), torch.tensor([0])
    # q0: higher class logit but competitive empty logit (p_cond ~ 0.5)
    # q1: lower class logit, empty logit heavily negative (p_cond ~ 1.0)
    q0_poly, q0_scores = _query(0.02, 4.0, 4.0)
    q1_poly, q1_scores = _query(0.03, 2.0, -8.0)
    pred_poly = torch.cat([q0_poly, q1_poly], dim=0)
    pred_scores = torch.cat([q0_scores, q1_scores], dim=0)

    def assigned(matcher):
        captured = {}
        orig = loss_mod.linear_sum_assignment

        def fake(cost):
            captured['rows'], captured['cols'] = orig(cost)
            return captured['rows'], captured['cols']

        loss_mod.linear_sum_assignment = fake
        try:
            matcher(
                pred_poly.unsqueeze(0),
                pred_scores.unsqueeze(0),
                gt_polygons=gt_poly,
                gt_cls=gt_cls,
                gt_groups=[1],
            )
        finally:
            loss_mod.linear_sum_assignment = orig
        return int(captured['rows'][0])

    assert assigned(_matcher(False)) == 0
    assert assigned(_matcher(True)) == 1


def test_conditional_is_skipped_without_no_object():
    """Without the no-object column there is nothing to renormalise against."""
    poly, scores = _query(0.02, 4.0, 4.0)
    gt_poly, gt_cls = _circle_gt(), torch.tensor([0])
    off = _capture_cost(_matcher(False, has_no_object=False), poly, scores, gt_poly, gt_cls)
    on = _capture_cost(_matcher(True, has_no_object=False), poly, scores, gt_poly, gt_cls)
    assert torch.equal(off, on)
