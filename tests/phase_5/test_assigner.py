"""Phase 5 tests — RayCastAssigner.

Validates all key behaviours:
  - Constructor with custom params
  - Radius-based candidate containment (select_candidates_in_gts)
  - Zero-ray handling in containment
  - VRAM-safe Polar-IoU (get_box_metrics)
  - 34-dim targets (get_targets, inherited)
  - End-to-end forward pass

Run with: uv run python tests/phase_5/test_assigner.py
"""

import torch

from raycasted.model.tal import RayCastAssigner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 2
N_GT = 3
N_ANCHORS = 100  # small grid for speed
NC = 4
N_RAYS = 32
POLY_DIM = 34  # xy(2) + rays(32)


def _make_gt_polygons(batch=BATCH, n_gt=N_GT, cx_range=(0.3, 0.7), ray_val=0.05):
    """Create synthetic GT polygons [B, N_gt, 34] in normalised space."""
    gt = torch.zeros(batch, n_gt, POLY_DIM)
    for b in range(batch):
        for i in range(n_gt):
            gt[b, i, 0] = cx_range[0] + (cx_range[1] - cx_range[0]) * (i + 1) / (n_gt + 1)
            gt[b, i, 1] = cx_range[0] + (cx_range[1] - cx_range[0]) * (b + 1) / (batch + 1)
            gt[b, i, 2:] = ray_val  # uniform circular polygon
    return gt


def _make_anchor_grid(n_anchors=N_ANCHORS):
    """Create regular anchor grid [N_anchors, 2] in [0, 1].

    Pads with repeated last point if n_anchors is not a perfect square.
    """
    side = int(n_anchors**0.5)
    lin = torch.linspace(0.05, 0.95, side)
    gx, gy = torch.meshgrid(lin, lin, indexing='ij')
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    if grid.shape[0] < n_anchors:
        # Pad to exact size
        pad = grid[-1:].expand(n_anchors - grid.shape[0], -1)
        grid = torch.cat([grid, pad], dim=0)
    return grid[:n_anchors]


def _make_pd_polygons(batch=BATCH, n_anchors=N_ANCHORS, ray_val=0.05):
    """Create synthetic predicted polygons [B, N_anchors, 34]."""
    pd = torch.zeros(batch, n_anchors, POLY_DIM)
    grid = _make_anchor_grid(n_anchors)
    pd[:, :, :2] = grid[None, :, :]
    pd[:, :, 2:] = ray_val
    return pd


def _make_mask_gt(batch=BATCH, n_gt=N_GT):
    """All GTs valid: [B, N_gt, 1]."""
    return torch.ones(batch, n_gt, 1, dtype=torch.bool)


def _make_gt_labels(batch=BATCH, n_gt=N_GT):
    """GT labels: [B, N_gt, 1]."""
    return torch.zeros(batch, n_gt, 1, dtype=torch.long)


def _make_pd_scores(batch=BATCH, n_anchors=N_ANCHORS, nc=NC):
    """Predicted scores: [B, N_anchors, nc] — high confidence."""
    scores = torch.zeros(batch, n_anchors, nc)
    scores[:, :, 0] = 0.9  # all class 0
    return scores


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_constructor():
    """RayCastAssigner instantiates with correct params and inherits TaskAlignedAssigner."""
    from ultralytics.utils.tal import TaskAlignedAssigner

    assigner = RayCastAssigner(topk=10, num_classes=NC, radius_scale=1.5)
    assert isinstance(assigner, TaskAlignedAssigner), 'Must inherit from TaskAlignedAssigner'
    assert assigner.topk == 10
    assert assigner.radius_scale == 1.5
    assert assigner.alpha == 0.5
    assert assigner.beta == 6.0
    print('PASS: constructor — correct inheritance and params')


def test_candidate_containment():
    """Anchors near GT centroids are selected, far ones are rejected."""
    assigner = RayCastAssigner(topk=10, num_classes=NC, radius_scale=2.0)
    assigner.bs = BATCH
    assigner.n_max_boxes = N_GT

    # GT at (0.5, 0.5) with ray=0.05 → containment radius = 0.05 * 2.0 = 0.1
    gt = torch.zeros(1, 1, POLY_DIM)
    gt[0, 0, 0] = 0.5  # cx
    gt[0, 0, 1] = 0.5  # cy
    gt[0, 0, 2:] = 0.05  # rays

    # Anchors: one near (0.52, 0.50), one far (0.9, 0.9)
    anchors = torch.tensor([[0.52, 0.50], [0.9, 0.9]])
    mask_gt = torch.ones(1, 1, 1, dtype=torch.bool)

    mask = assigner.select_candidates_in_gts(anchors, gt, mask_gt)

    assert mask.shape == (1, 1, 2)
    # Anchor 0 is at distance 0.02 from GT → within 0.1 radius → selected
    assert mask[0, 0, 0], 'Near anchor should be selected'
    # Anchor 1 is at distance ~0.57 from GT → outside 0.1 radius → not selected
    assert not mask[0, 0, 1], 'Far anchor should not be selected'
    print('PASS: candidate containment — near selected, far rejected')


def test_candidate_zero_rays_fallback():
    """GT with < 8 non-zero rays falls back to max ray for radius."""
    assigner = RayCastAssigner(topk=10, num_classes=NC, radius_scale=2.0)
    assigner.bs = 1
    assigner.n_max_boxes = 1

    # GT at (0.5, 0.5) with only 4 non-zero rays (value 0.1)
    gt = torch.zeros(1, 1, POLY_DIM)
    gt[0, 0, 0] = 0.5
    gt[0, 0, 1] = 0.5
    gt[0, 0, 2:6] = 0.1  # 4 non-zero rays
    gt[0, 0, 6:] = 0.0  # 28 zero rays

    # containment radius should be max(0.1) * 2.0 = 0.2
    anchors = torch.tensor([[0.6, 0.5]])  # distance 0.1 < 0.2 → selected
    mask_gt = torch.ones(1, 1, 1, dtype=torch.bool)

    mask = assigner.select_candidates_in_gts(anchors, gt, mask_gt)
    assert mask[0, 0, 0], 'Anchor within fallback radius should be selected'
    print('PASS: zero-ray fallback — max ray used when < 8 non-zero rays')


def test_candidate_all_zero_rays():
    """GT with all-zero rays has no candidates (degenerate cell)."""
    assigner = RayCastAssigner(topk=10, num_classes=NC, radius_scale=2.0)
    assigner.bs = 1
    assigner.n_max_boxes = 1

    gt = torch.zeros(1, 1, POLY_DIM)
    gt[0, 0, 0] = 0.5
    gt[0, 0, 1] = 0.5
    # All rays = 0

    anchors = torch.tensor([[0.5, 0.5]])  # exactly at centroid
    mask_gt = torch.ones(1, 1, 1, dtype=torch.bool)

    mask = assigner.select_candidates_in_gts(anchors, gt, mask_gt)
    assert not mask[0, 0, 0], 'All-zero GT should have no candidates'
    print('PASS: all-zero rays — degenerate cell skipped')


def test_candidate_mask_gt_respected():
    """Invalid GTs (mask_gt=False) produce no candidates."""
    assigner = RayCastAssigner(topk=10, num_classes=NC, radius_scale=2.0)
    assigner.bs = 1
    assigner.n_max_boxes = 1

    gt = _make_gt_polygons(batch=1, n_gt=1, ray_val=0.1)
    anchors = _make_anchor_grid(100)
    mask_gt = torch.zeros(1, 1, 1, dtype=torch.bool)  # GT not valid

    mask = assigner.select_candidates_in_gts(anchors, gt, mask_gt)
    assert not mask.any(), 'Invalid GT should have no candidates'
    print('PASS: mask_gt respected — invalid GTs produce no candidates')


def test_box_metrics_iou_identity():
    """Identical rays produce IoU ≈ 1.0."""
    assigner = RayCastAssigner(topk=10, num_classes=NC)
    assigner.bs = 1
    assigner.n_max_boxes = 1

    ray_val = 0.05
    pd = _make_pd_polygons(batch=1, n_anchors=10, ray_val=ray_val)
    gt = _make_gt_polygons(batch=1, n_gt=1, ray_val=ray_val)
    # Place GT centroid near first anchor
    gt[0, 0, :2] = pd[0, 0, :2]

    gt_labels = _make_gt_labels(1, 1)
    pd_scores = _make_pd_scores(1, 10)

    # All anchors are candidates
    mask_gt = torch.ones(1, 1, 10, dtype=torch.bool)

    _, overlaps = assigner.get_box_metrics(pd_scores, pd, gt_labels, gt, mask_gt)

    # All IoU values should be ~1.0 for identical rays
    assert overlaps.shape == (1, 1, 10)
    iou_vals = overlaps[0, 0]
    assert (iou_vals > 0.99).all(), f'Identical rays should give IoU≈1.0, got {iou_vals}'
    print(f'PASS: IoU identity — identical rays → IoU min={iou_vals.min():.4f}')


def test_box_metrics_iou_different():
    """Different rays produce IoU < 1.0."""
    assigner = RayCastAssigner(topk=10, num_classes=NC)
    assigner.bs = 1
    assigner.n_max_boxes = 1

    pd = _make_pd_polygons(batch=1, n_anchors=10, ray_val=0.05)
    gt = _make_gt_polygons(batch=1, n_gt=1, ray_val=0.10)  # 2× prediction

    gt_labels = _make_gt_labels(1, 1)
    pd_scores = _make_pd_scores(1, 10)
    mask_gt = torch.ones(1, 1, 10, dtype=torch.bool)

    _, overlaps = assigner.get_box_metrics(pd_scores, pd, gt_labels, gt, mask_gt)

    # Analytical: Σ min² / Σ max² = 32×0.05² / 32×0.10² = 0.25
    iou_val = overlaps[0, 0, 0].item()
    assert 0.20 < iou_val < 0.30, f'Expected IoU≈0.25, got {iou_val}'
    print(f'PASS: IoU different — pred=0.05, gt=0.10 → IoU={iou_val:.4f} (expected ≈0.25)')


def test_box_metrics_shape():
    """get_box_metrics returns correct shapes."""
    assigner = RayCastAssigner(topk=10, num_classes=NC)
    assigner.bs = BATCH
    assigner.n_max_boxes = N_GT

    pd = _make_pd_polygons(batch=BATCH, n_anchors=N_ANCHORS)
    gt = _make_gt_polygons(batch=BATCH, n_gt=N_GT)
    gt_labels = _make_gt_labels()
    pd_scores = _make_pd_scores(batch=BATCH, n_anchors=N_ANCHORS)
    mask_gt = torch.ones(BATCH, N_GT, N_ANCHORS, dtype=torch.bool)

    align_metric, overlaps = assigner.get_box_metrics(pd_scores, pd, gt_labels, gt, mask_gt)

    assert align_metric.shape == (BATCH, N_GT, N_ANCHORS), f'align_metric shape: {align_metric.shape}'
    assert overlaps.shape == (BATCH, N_GT, N_ANCHORS), f'overlaps shape: {overlaps.shape}'
    print(f'PASS: box metrics shape — ({BATCH}, {N_GT}, {N_ANCHORS})')


def test_box_metrics_mask_applied():
    """IoU is only non-zero where mask_gt is True."""
    assigner = RayCastAssigner(topk=10, num_classes=NC)
    assigner.bs = 1
    assigner.n_max_boxes = 2

    pd = _make_pd_polygons(batch=1, n_anchors=20, ray_val=0.05)
    gt = _make_gt_polygons(batch=1, n_gt=2, ray_val=0.05)
    gt_labels = _make_gt_labels(1, 2)
    pd_scores = _make_pd_scores(1, 20)

    # Only first 5 anchors are candidates for GT 0; none for GT 1
    mask = torch.zeros(1, 2, 20, dtype=torch.bool)
    mask[0, 0, :5] = True

    _, overlaps = assigner.get_box_metrics(pd_scores, pd, gt_labels, gt, mask)

    # GT 0: only first 5 anchors should have non-zero IoU
    assert (overlaps[0, 0, :5] > 0).all(), 'Masked anchors should have IoU > 0'
    assert (overlaps[0, 0, 5:] == 0).all(), 'Non-masked anchors should have IoU = 0'
    # GT 1: no candidates at all
    assert (overlaps[0, 1, :] == 0).all(), 'GT with no candidates should have all-zero IoU'
    print('PASS: mask applied — IoU only non-zero where mask_gt is True')


def test_targets_inherited():
    """get_targets (inherited) returns 34-dim polygon targets."""
    assigner = RayCastAssigner(topk=10, num_classes=NC)
    assigner.bs = 1
    assigner.n_max_boxes = 1

    gt_bboxes = _make_gt_polygons(batch=1, n_gt=1, ray_val=0.05)
    gt_labels = _make_gt_labels(1, 1)
    # Every anchor assigned to GT 0
    target_gt_idx = torch.zeros(1, N_ANCHORS, dtype=torch.long)
    fg_mask = torch.ones(1, N_ANCHORS, dtype=torch.bool)

    target_labels, target_bboxes, target_scores = assigner.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

    assert target_bboxes.shape == (1, N_ANCHORS, POLY_DIM), (
        f'Expected (1, {N_ANCHORS}, {POLY_DIM}), got {target_bboxes.shape}'
    )
    # All targets should equal GT 0
    assert torch.allclose(target_bboxes[0, 0], gt_bboxes[0, 0]), 'Target should equal GT polygon'
    print(f'PASS: get_targets inherited — shape {target_bboxes.shape}, 34-dim targets correct')


def test_forward_end_to_end():
    """Full forward pass with synthetic polygon data produces valid assignments."""
    n_anchors = 400  # 20×20 grid
    assigner = RayCastAssigner(topk=10, num_classes=NC, radius_scale=3.0)

    anchors = _make_anchor_grid(n_anchors)
    pd = _make_pd_polygons(batch=1, n_anchors=n_anchors, ray_val=0.05)
    gt = _make_gt_polygons(batch=1, n_gt=2, ray_val=0.05)
    gt_labels = _make_gt_labels(1, 2)
    pd_scores = _make_pd_scores(1, n_anchors)
    mask_gt = _make_mask_gt(1, 2)

    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
        pd_scores, pd, anchors, gt_labels, gt, mask_gt
    )

    assert target_labels.shape == (1, n_anchors)
    assert target_bboxes.shape == (1, n_anchors, POLY_DIM), (
        f'Expected (1, {n_anchors}, {POLY_DIM}), got {target_bboxes.shape}'
    )
    assert target_scores.shape == (1, n_anchors, NC)
    assert fg_mask.shape == (1, n_anchors)
    assert target_gt_idx.shape == (1, n_anchors)

    # Some anchors should be positive
    n_pos = fg_mask.sum().item()
    assert n_pos > 0, 'Should have at least some positive assignments'
    assert n_pos <= n_anchors, f'Too many positives: {n_pos} > {n_anchors}'

    # Positive anchors should have non-zero target scores
    pos_scores = target_scores[fg_mask]
    assert pos_scores.sum() > 0, 'Positive anchors should have non-zero target scores'

    print(f'PASS: end-to-end — {n_pos}/{n_anchors} positive anchors, shapes correct')


def test_forward_multi_batch():
    """Forward pass works with batch size > 1."""
    n_anchors = 100
    assigner = RayCastAssigner(topk=5, num_classes=NC, radius_scale=3.0)

    anchors = _make_anchor_grid(n_anchors)
    pd = _make_pd_polygons(batch=BATCH, n_anchors=n_anchors, ray_val=0.05)
    gt = _make_gt_polygons(batch=BATCH, n_gt=N_GT, ray_val=0.05)
    gt_labels = _make_gt_labels()
    pd_scores = _make_pd_scores(batch=BATCH, n_anchors=n_anchors)
    mask_gt = _make_mask_gt()

    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
        pd_scores, pd, anchors, gt_labels, gt, mask_gt
    )

    assert target_bboxes.shape == (BATCH, n_anchors, POLY_DIM)
    assert target_labels.shape == (BATCH, n_anchors)
    assert fg_mask.shape == (BATCH, n_anchors)

    for b in range(BATCH):
        n_pos = fg_mask[b].sum().item()
        assert n_pos > 0, f'Batch {b}: should have positive assignments'

    print(f'PASS: multi-batch — shapes correct for batch={BATCH}')


def test_vram_chunking():
    """Large N_cand × N_gt triggers chunking without error.

    This test verifies the chunking code path is exercised without
    actually consuming large amounts of memory (we patch MAX_FLAT_PAIRS
    to a small value).
    """
    import raycasted.model.tal as tal_module

    original_max = tal_module.MAX_FLAT_PAIRS
    try:
        # Lower threshold to force chunking with small test data
        tal_module.MAX_FLAT_PAIRS = 10  # Very small to force chunking

        assigner = RayCastAssigner(topk=10, num_classes=NC)
        assigner.bs = 1
        assigner.n_max_boxes = 3

        n_anchors = 50
        pd = _make_pd_polygons(batch=1, n_anchors=n_anchors, ray_val=0.05)
        gt = _make_gt_polygons(batch=1, n_gt=3, ray_val=0.05)
        gt_labels = _make_gt_labels(1, 3)
        pd_scores = _make_pd_scores(1, n_anchors)
        mask_gt = torch.ones(1, 3, n_anchors, dtype=torch.bool)

        _, overlaps = assigner.get_box_metrics(pd_scores, pd, gt_labels, gt, mask_gt)

        assert overlaps.shape == (1, 3, n_anchors)
        assert not torch.isnan(overlaps).any(), 'Chunking produced NaN'
        assert (overlaps >= 0).all(), 'IoU should be non-negative'
        print('PASS: VRAM chunking — forced chunking completed, no NaN, shape correct')
    finally:
        tal_module.MAX_FLAT_PAIRS = original_max


def test_nan_safety_tiny_rays():
    """Very small rays (1e-6) produce no NaN or Inf in Polar-IoU.

    POLAR_IOU_EPS=1e-7 dominates the denominator for rays of 1e-6, so IoU
    won't be 1.0 — that's expected. We verify NaN/Inf safety only.
    For realistic tiny rays (0.001 ≈ 0.6px at 640 crop), IoU should be ≈1.0.
    """
    assigner = RayCastAssigner(topk=10, num_classes=NC)
    assigner.bs = 1
    assigner.n_max_boxes = 1

    # --- Sub-test A: extreme tiny rays → no NaN/Inf ---
    tiny = 1e-6
    pd = _make_pd_polygons(batch=1, n_anchors=10, ray_val=tiny)
    gt = _make_gt_polygons(batch=1, n_gt=1, ray_val=tiny)
    gt_labels = _make_gt_labels(1, 1)
    pd_scores = _make_pd_scores(1, 10)
    mask_gt = torch.ones(1, 1, 10, dtype=torch.bool)

    _, overlaps = assigner.get_box_metrics(pd_scores, pd, gt_labels, gt, mask_gt)

    assert not torch.isnan(overlaps).any(), 'NaN in overlaps with tiny rays'
    assert not torch.isinf(overlaps).any(), 'Inf in overlaps with tiny rays'

    # --- Sub-test B: realistic tiny rays → IoU ≈ 1.0 ---
    realistic_tiny = 0.001  # ~0.6px at crop_size=640
    pd2 = _make_pd_polygons(batch=1, n_anchors=10, ray_val=realistic_tiny)
    gt2 = _make_gt_polygons(batch=1, n_gt=1, ray_val=realistic_tiny)
    mask_gt2 = torch.ones(1, 1, 10, dtype=torch.bool)

    _, overlaps2 = assigner.get_box_metrics(pd_scores, pd2, gt_labels, gt2, mask_gt2)
    assert (overlaps2 > 0.99).all(), f'Realistic tiny rays should give IoU≈1.0, got min={overlaps2.min():.6f}'

    print('PASS: NaN safety — 1e-6 rays: no NaN/Inf; 0.001 rays: IoU>0.99')


def test_empty_gt_batch():
    """Forward pass with zero GT boxes returns valid zero tensors."""
    assigner = RayCastAssigner(topk=10, num_classes=NC)
    n_anchors = 50

    anchors = _make_anchor_grid(n_anchors)
    pd = _make_pd_polygons(batch=1, n_anchors=n_anchors)
    # Empty GT: shape (1, 0, 34)
    gt = torch.zeros(1, 0, POLY_DIM)
    gt_labels = torch.zeros(1, 0, 1, dtype=torch.long)
    mask_gt = torch.zeros(1, 0, 1, dtype=torch.bool)
    pd_scores = _make_pd_scores(1, n_anchors)

    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
        pd_scores, pd, anchors, gt_labels, gt, mask_gt
    )

    assert fg_mask.shape == (1, n_anchors)
    assert not fg_mask.any(), 'No positives when zero GTs'
    assert target_bboxes.shape[0] == 1 and target_bboxes.shape[2] == POLY_DIM
    print('PASS: empty GT — zero GT boxes handled without error')


def test_multi_gt_conflict_resolution():
    """Anchor assigned to multiple GTs resolves to the one with highest IoU."""
    assigner = RayCastAssigner(topk=10, num_classes=NC, radius_scale=10.0)

    n_anchors = 20
    anchors = _make_anchor_grid(n_anchors)

    # Two GTs very close together — both will claim the same anchors
    gt = torch.zeros(1, 2, POLY_DIM)
    gt[0, 0, :2] = torch.tensor([0.5, 0.5])
    gt[0, 0, 2:] = 0.05  # small rays → small IoU with predictions
    gt[0, 1, :2] = torch.tensor([0.5, 0.5])  # same centroid
    gt[0, 1, 2:] = 0.10  # larger rays → should win conflict

    # Predictions match GT 1's rays (larger)
    pd = _make_pd_polygons(batch=1, n_anchors=n_anchors, ray_val=0.10)
    gt_labels = torch.zeros(1, 2, 1, dtype=torch.long)
    pd_scores = _make_pd_scores(1, n_anchors)
    mask_gt = _make_mask_gt(1, 2)

    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
        pd_scores, pd, anchors, gt_labels, gt, mask_gt
    )

    assert fg_mask.any(), 'Should have positive assignments'
    # For anchors assigned to both GTs, should resolve to GT 1 (higher IoU
    # because pd rays=0.10 matches gt[1] rays=0.10, vs mismatch with gt[0] rays=0.05)
    pos_mask = fg_mask[0]
    assigned_gt = target_gt_idx[0, pos_mask]

    # All or most positive anchors should be assigned to GT 1 (index 1)
    n_assigned_to_gt1 = (assigned_gt == 1).sum().item()
    assert n_assigned_to_gt1 > 0, 'Some anchors should resolve to GT 1 (higher IoU)'
    print(f'PASS: multi-GT conflict — {n_assigned_to_gt1}/{pos_mask.sum().item()} anchors resolved to higher-IoU GT')


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_constructor()
    test_candidate_containment()
    test_candidate_zero_rays_fallback()
    test_candidate_all_zero_rays()
    test_candidate_mask_gt_respected()
    test_box_metrics_iou_identity()
    test_box_metrics_iou_different()
    test_box_metrics_shape()
    test_box_metrics_mask_applied()
    test_targets_inherited()
    test_forward_end_to_end()
    test_forward_multi_batch()
    test_vram_chunking()
    test_nan_safety_tiny_rays()
    test_empty_gt_batch()
    test_multi_gt_conflict_resolution()

    print('\nAll Phase 5 tests passed!')
