"""Phase 10 tests — empty-column cost for the Hungarian matcher.

The empty-column cost adds ``lambda * -log(1 - p_empty)`` to every match cost of
a query, where ``p_empty`` is the explicit no-object column. The term is constant
across GT columns, so it cannot reorder a fixed-size assignment, but it does
change which queries the solver leaves unused. These tests pin the flag-off
equivalence, the monotonicity, and the intended steering.

Run with: uv run python -m pytest tests/phase_10/test_matcher_empty_cost.py -x -q
"""

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


def _single_query(offset=0.02, class_logit=6.0, empty_logit=-6.0):
    pred_poly = torch.zeros(1, 2 + N_RAYS)
    pred_poly[0, :2] = torch.tensor([0.5 + offset, 0.5])
    pred_poly[0, 2:] = 0.1
    pred_scores = torch.full((1, 6), -6.0)
    pred_scores[0, 0] = class_logit
    pred_scores[0, 5] = empty_logit
    return pred_poly, pred_scores


def _matcher(empty_cost, lam=1.0, has_no_object=True):
    return RayCastHungarianMatcher(
        n_rays=N_RAYS,
        lsp_mode=True,
        empty_cost=empty_cost,
        empty_cost_lambda=lam,
        has_no_object=has_no_object,
    )


def test_flag_off_is_identical_to_legacy_path():
    """Default construction must reproduce the pre-existing cost exactly."""
    torch.manual_seed(0)
    pred_poly = torch.rand(4, 2 + N_RAYS) * 0.2 + 0.4
    pred_scores = torch.randn(4, 6)
    gt_poly = _circle_gt()
    gt_cls = torch.tensor([0])

    legacy_matcher = RayCastHungarianMatcher(n_rays=N_RAYS, lsp_mode=True)
    legacy = _capture_cost(legacy_matcher, pred_poly, pred_scores, gt_poly, gt_cls)
    off = _capture_cost(_matcher(False), pred_poly, pred_scores, gt_poly, gt_cls)
    assert torch.equal(legacy, off)


def test_lambda_zero_is_a_no_op():
    """lambda=0 must not change any cost entry."""
    torch.manual_seed(1)
    pred_poly = torch.rand(4, 2 + N_RAYS) * 0.2 + 0.4
    pred_scores = torch.randn(4, 6)
    gt_poly = _circle_gt()
    gt_cls = torch.tensor([0])

    off = _capture_cost(_matcher(False), pred_poly, pred_scores, gt_poly, gt_cls)
    zero = _capture_cost(_matcher(True, lam=0.0), pred_poly, pred_scores, gt_poly, gt_cls)
    assert torch.allclose(off, zero, atol=1e-6)


def test_term_is_row_constant_across_gt_columns():
    """Adding the empty cost must shift every GT column of a row equally."""
    torch.manual_seed(2)
    pred_poly = torch.rand(3, 2 + N_RAYS) * 0.2 + 0.4
    pred_scores = torch.randn(3, 6)
    gt_poly = torch.tensor(
        [
            [0.5, 0.5] + [0.1] * N_RAYS,
            [0.8, 0.5] + [0.1] * N_RAYS,
        ]
    )
    gt_cls = torch.tensor([0, 0])

    off = _capture_cost(_matcher(False), pred_poly, pred_scores, gt_poly, gt_cls)
    on = _capture_cost(_matcher(True, lam=2.5), pred_poly, pred_scores, gt_poly, gt_cls)
    delta = on - off
    # Gated pairs sit near 1e4, where float32 spacing is ~1e-3, so the
    # row-constant shift can only be recovered to about that resolution.
    assert torch.allclose(delta, delta[:, :1].expand_as(delta), atol=2e-3)


def test_penalty_grows_with_p_empty():
    """A more confident-empty query must pay more, and the cost is monotonic."""
    costs = []
    for empty_logit in (-8.0, -2.0, 0.0, 4.0, 8.0):
        pred_poly, pred_scores = _single_query(empty_logit=empty_logit)
        costs.append(
            float(_capture_cost(_matcher(True), pred_poly, pred_scores, _circle_gt(), torch.tensor([0]))[0, 0])
        )
    assert all(b > a for a, b in zip(costs, costs[1:])), costs


def test_empty_query_is_left_unused_instead_of_taking_a_gt():
    """One GT, two queries: the better-fitting one claims empty.

    q1 sits exactly on the centroid with a higher class logit, so the flag-off
    matcher spends it. With the flag on its empty penalty makes it lose to q0.
    """
    gt_poly = _circle_gt(0.1)
    gt_cls = torch.tensor([0])

    pred_poly = torch.zeros(2, 2 + N_RAYS)
    pred_poly[:, 2:] = 0.1
    pred_poly[0, :2] = torch.tensor([0.53, 0.5])
    pred_poly[1, :2] = torch.tensor([0.5, 0.5])

    pred_scores = torch.full((2, 6), -6.0)
    pred_scores[0, 0] = 6.0  # q0: good class, not empty
    pred_scores[1, 0] = 9.0  # q1: better class ...
    pred_scores[1, 5] = 9.0  # ... but confidently empty

    def assigned(matcher):
        idx = matcher(
            pred_poly.unsqueeze(0),
            pred_scores.unsqueeze(0),
            gt_polygons=gt_poly,
            gt_cls=gt_cls,
            gt_groups=[1],
        )[0]
        return [int(q) for q in idx[0]]

    assert assigned(_matcher(False)) == [1]
    assert assigned(_matcher(True, lam=4.0)) == [0]


def test_no_object_disabled_skips_the_term():
    """Without an explicit empty column the flag must be inert."""
    torch.manual_seed(3)
    pred_poly = torch.rand(3, 2 + N_RAYS) * 0.2 + 0.4
    pred_scores = torch.randn(3, 5)  # 5 real classes, no empty column
    gt_poly = _circle_gt()
    gt_cls = torch.tensor([0])

    off = _capture_cost(_matcher(False, has_no_object=False), pred_poly, pred_scores, gt_poly, gt_cls)
    on = _capture_cost(_matcher(True, has_no_object=False), pred_poly, pred_scores, gt_poly, gt_cls)
    assert torch.equal(off, on)
