"""Phase 10 tests — soft inside gate for the LSP Hungarian matcher.

The soft gate replaces the constant 9999 veto on outside query-GT pairs with a
distance-scaled penalty ``w * excess**p``, where ``excess = dist/d_phi - 1`` is
zero inside the GT star and grows outward. These tests pin the geometry, the
flag-off equivalence, and the calibration direction.

Run with: uv run python -m pytest tests/phase_10/test_matcher_soft_inside.py -x -q
"""

import torch

import raycasted.model.nulite_detr_loss as loss_mod
from raycasted.model.nulite_detr_loss import (
    RayCastHungarianMatcher,
    _points_inside_star,
    _points_inside_star_excess,
)

N_RAYS = 8


def _circle_gt(radius=0.1, n_rays=N_RAYS):
    gt_xy = torch.tensor([[0.5, 0.5]])
    gt_rays = torch.full((1, n_rays), radius)
    gt_poly = torch.cat([gt_xy, gt_rays], dim=-1)
    return gt_xy, gt_rays, gt_poly


def _capture_cost(matcher, pred_poly, pred_scores, gt_poly):
    """Run the matcher but record the cost matrix it hands to the solver."""
    captured = {}
    orig = loss_mod.linear_sum_assignment

    def fake(cost):
        captured['cost'] = cost.clone()
        return orig(cost)

    loss_mod.linear_sum_assignment = fake
    try:
        indices = matcher(
            pred_poly.unsqueeze(0),
            pred_scores.unsqueeze(0),
            gt_polygons=gt_poly,
            gt_cls=torch.tensor([0]),
            gt_groups=[1],
        )[0]
    finally:
        loss_mod.linear_sum_assignment = orig
    return captured['cost'], indices


def _toy_batch(dist_out=0.16, logit_out=6.0, logit_in=-6.0, empty_in=6.0):
    """Build a one-GT, two-query toy batch.

    q0 is OUTSIDE by ``dist_out`` but confidently class-0; q1 is inside/centred
    but confidently EMPTY (i.e. its class-0 logit is very low).
    """
    pred_poly = torch.zeros(2, 2 + N_RAYS)
    pred_poly[0, :2] = torch.tensor([0.5 + dist_out, 0.5])
    pred_poly[1, :2] = torch.tensor([0.51, 0.5])
    pred_poly[:, 2:] = 0.1

    pred_scores = torch.full((2, 6), -6.0)  # 5 real classes + empty column
    pred_scores[0, 0] = logit_out
    pred_scores[1, 0] = logit_in
    pred_scores[1, 5] = empty_in
    return pred_poly, pred_scores


def test_excess_matches_inside_mask():
    """The boolean mask must agree with the legacy helper exactly."""
    torch.manual_seed(0)
    pred_xy = torch.rand(200, 2)
    _, gt_rays, gt_poly = _circle_gt()
    inside, excess = _points_inside_star_excess(pred_xy, gt_poly[:, :2], gt_rays, N_RAYS)
    legacy = _points_inside_star(pred_xy, gt_poly[:, :2], gt_rays, N_RAYS)
    assert torch.equal(inside, legacy)
    assert (excess[~inside] > 0).all()
    assert excess[inside].eq(0).all()


def test_excess_is_zero_on_boundary_and_monotonic():
    """Excess is 0 inside, follows dist/r - 1 outside, and increases."""
    _, gt_rays, gt_poly = _circle_gt(radius=0.1)
    dists = torch.tensor([0.0, 0.05, 0.09, 0.12, 0.2, 0.4])
    pred_xy = torch.stack([0.5 + dists, torch.full_like(dists, 0.5)], dim=-1)
    inside, excess = _points_inside_star_excess(pred_xy, gt_poly[:, :2], gt_rays, N_RAYS)

    assert inside[:3].all()
    assert excess[:3].eq(0).all()
    assert not inside[3:].any()
    expected = dists[3:] / 0.1 - 1.0
    assert torch.allclose(excess[3:].squeeze(1), expected, atol=1e-4)
    assert (excess[3:].squeeze(1).diff() > 0).all()


def test_hard_gate_makes_outside_pair_unmatchable():
    """Flag off: the outside query pays >= 9999 and loses the GT."""
    _, _, gt_poly = _circle_gt()
    pred_poly, pred_scores = _toy_batch()
    matcher = RayCastHungarianMatcher(n_rays=N_RAYS, lsp_mode=True, soft_inside=False)
    cost, indices = _capture_cost(matcher, pred_poly, pred_scores, gt_poly)

    assert cost[0, 0] > 9990.0
    assert cost[1, 0] < 10.0
    assert indices[0].tolist() == [1]


def test_soft_gate_cost_is_finite_and_monotonic_in_width():
    """Flag on: outside cost scales with w, and w=0 removes the outside veto."""
    _, _, gt_poly = _circle_gt()
    pred_poly, pred_scores = _toy_batch()

    costs = {}
    for w in (0.0, 5.0, 20.0):
        matcher = RayCastHungarianMatcher(
            n_rays=N_RAYS, lsp_mode=True, soft_inside=True, soft_inside_w=w
        )
        costs[w], _ = _capture_cost(matcher, pred_poly, pred_scores, gt_poly)

    assert costs[20.0][0, 0] > costs[5.0][0, 0] > costs[0.0][0, 0]
    assert (costs[20.0][0, 0] - costs[0.0][0, 0]).item() > 20 * 0.6**2 - 1e-3
    assert costs[20.0][0, 0] < 100.0


def test_soft_gate_flips_assignment_with_calibration():
    """Same geometry, different weight: small admits, calibrated rejects.

    A small weight admits the confident outside query; the calibrated weight
    (20) sends the assignment to the inside query instead.
    """
    _, _, gt_poly = _circle_gt()
    pred_poly, pred_scores = _toy_batch()

    small = RayCastHungarianMatcher(
        n_rays=N_RAYS, lsp_mode=True, soft_inside=True, soft_inside_w=0.5
    )
    calibrated = RayCastHungarianMatcher(
        n_rays=N_RAYS, lsp_mode=True, soft_inside=True, soft_inside_w=20.0
    )
    _, idx_small = _capture_cost(small, pred_poly, pred_scores, gt_poly)
    _, idx_cal = _capture_cost(calibrated, pred_poly, pred_scores, gt_poly)

    assert idx_small[0].tolist() == [0]
    assert idx_cal[0].tolist() == [1]


def test_inside_pairs_are_never_penalised():
    """The gate must be exactly 0 for inside pairs at any weight."""
    _, gt_rays, gt_poly = _circle_gt()
    pred_xy = torch.tensor([[0.5, 0.5], [0.51, 0.5], [0.55, 0.5]])
    inside, excess = _points_inside_star_excess(pred_xy, gt_poly[:, :2], gt_rays, N_RAYS)
    assert inside.all()
    assert excess.eq(0).all()


def test_soft_gate_clamps_far_pairs_to_legacy_veto():
    """Far outside pairs are bounded by 9999 instead of escalating unbounded.

    With r=0.1 and a query 5.0 away, excess is ~49 and the raw penalty would be
    20 * 49**2 ~= 48k. The clamp keeps the matrix at the legacy veto magnitude so
    it stays finite under fp16 and for degenerate zero-length GT rays.
    """
    _, _, gt_poly = _circle_gt(radius=0.1)
    pred_poly, pred_scores = _toy_batch(dist_out=5.0)
    matcher = RayCastHungarianMatcher(
        n_rays=N_RAYS, lsp_mode=True, soft_inside=True, soft_inside_w=20.0
    )
    cost, indices = _capture_cost(matcher, pred_poly, pred_scores, gt_poly)

    raw_penalty = 20.0 * (5.0 / 0.1 - 1.0) ** 2
    assert raw_penalty > 9999.0
    assert cost[0, 0] < 10001.0
    assert cost[1, 0] < 10.0
    assert indices[0].tolist() == [1]
