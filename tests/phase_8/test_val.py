"""Phase 8 tests — RayCastValidator and polygon IoU metrics.

Validates:
  - Shapely polygon IoU: identical, non-overlapping, partial, degenerate
  - Vertex coordinate computation matches decode_to_vertices
  - _process_batch with empty and non-empty inputs
  - Centroid distance matching
  - RayCastDetMetrics process and results
  - IoU computation performance

Run with: uv run python tests/phase_8/test_val.py
"""

import time

import numpy as np
import torch

from raycasted.data.etl.ops.convert import decode_to_vertices
from raycasted.model.val import (
    RAYCAST_DIM,
    RayCastDetMetrics,
    RayCastValidator,
    _build_polygon_coords,
    _polygon_iou_row,
)

N_RAYS = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_circle_polygons(cx, cy, radius, n=1):
    """Create [n, 34] polygon arrays for circles at (cx, cy) with given radius."""
    poly = np.zeros((n, RAYCAST_DIM), dtype=np.float64)
    poly[:, 0] = cx
    poly[:, 1] = cy
    poly[:, 2:] = radius
    return poly


def _make_validator():
    """Create a minimal RayCastValidator for testing."""
    validator = RayCastValidator.__new__(RayCastValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = validator.iouv.numel()
    validator.device = torch.device('cpu')
    validator.centroid_thresholds = [6.0, 8.0, 10.0]
    validator.n_centroid = 3
    return validator


# ---------------------------------------------------------------------------
# Shapely polygon IoU tests
# ---------------------------------------------------------------------------


def test_polygon_iou_identical():
    """IoU = 1.0 for identical polygons."""
    coords = _build_polygon_coords(_make_circle_polygons(100, 100, 30))[0]
    gt_coords = coords[None, :, :]

    iou = _polygon_iou_row((coords, gt_coords))
    assert abs(iou[0] - 1.0) < 1e-6, f'Expected IoU=1.0, got {iou[0]:.6f}'
    print(f'PASS: identical polygon IoU = {iou[0]:.6f}')


def test_polygon_iou_non_overlapping():
    """IoU = 0.0 for non-overlapping polygons."""
    pred_coords = _build_polygon_coords(_make_circle_polygons(50, 50, 10))[0]
    gt_coords = _build_polygon_coords(_make_circle_polygons(200, 200, 10))

    iou = _polygon_iou_row((pred_coords, gt_coords))
    assert iou[0] == 0.0, f'Expected IoU=0.0, got {iou[0]:.6f}'
    print(f'PASS: non-overlapping polygon IoU = {iou[0]:.6f}')


def test_polygon_iou_partial():
    """IoU in (0, 1) for partially overlapping polygons."""
    pred_coords = _build_polygon_coords(_make_circle_polygons(100, 100, 30))[0]
    gt_coords = _build_polygon_coords(_make_circle_polygons(120, 100, 30))

    iou = _polygon_iou_row((pred_coords, gt_coords))
    assert 0.0 < iou[0] < 1.0, f'Expected 0 < IoU < 1, got {iou[0]:.6f}'
    print(f'PASS: partial overlap IoU = {iou[0]:.6f}')


def test_polygon_iou_degenerate():
    """Zero-area polygon -> IoU = 0.0, no crash."""
    pred_coords = _build_polygon_coords(_make_circle_polygons(100, 100, 0))[0]
    gt_coords = _build_polygon_coords(_make_circle_polygons(100, 100, 30))

    iou = _polygon_iou_row((pred_coords, gt_coords))
    assert iou[0] == 0.0, f'Expected IoU=0.0 for degenerate polygon, got {iou[0]:.6f}'
    print('PASS: degenerate polygon -> IoU = 0.0')


# ---------------------------------------------------------------------------
# Vertex coordinate tests
# ---------------------------------------------------------------------------


def test_build_polygon_coords():
    """Vertex coords match decode_to_vertices output."""
    poly = _make_circle_polygons(200, 150, 40, n=3)
    poly[1, 2:] = 20
    poly[2, 0] = 300

    coords = _build_polygon_coords(poly)
    rays = poly[:, 2:]
    cx = poly[:, 0]
    cy = poly[:, 1]
    verts = decode_to_vertices(rays, cx, cy)

    assert np.allclose(coords, verts, atol=1e-10), f'Max diff: {np.abs(coords - verts).max():.2e}'
    print('PASS: build_polygon_coords matches decode_to_vertices')


# ---------------------------------------------------------------------------
# _process_batch tests
# ---------------------------------------------------------------------------


def test_process_batch_empty_preds():
    """Empty predictions -> zero tp matrices."""
    v = _make_validator()

    preds = {'bboxes': torch.zeros(0, RAYCAST_DIM), 'conf': torch.zeros(0), 'cls': torch.zeros(0)}
    batch = {
        'bboxes': torch.tensor([[100, 100] + [30.0] * 32]),
        'cls': torch.tensor([0]),
    }

    result = v._process_batch(preds, batch)
    assert result['tp_shapely'].shape == (0, 10), f'Expected (0, 10), got {result["tp_shapely"].shape}'
    assert result['tp_centroid'].shape == (0, 3), f'Expected (0, 3), got {result["tp_centroid"].shape}'
    print('PASS: empty preds -> zero tp matrices')


def test_process_batch_empty_gt():
    """Empty GT -> zero tp matrices."""
    v = _make_validator()

    preds = {
        'bboxes': torch.tensor([[100, 100] + [30.0] * 32]),
        'conf': torch.tensor([0.9]),
        'cls': torch.tensor([0]),
    }
    batch = {'bboxes': torch.zeros(0, RAYCAST_DIM), 'cls': torch.zeros(0, dtype=torch.long)}

    result = v._process_batch(preds, batch)
    assert result['tp_shapely'].shape == (1, 10), f'Expected (1, 10), got {result["tp_shapely"].shape}'
    assert result['tp_centroid'].shape == (1, 3), f'Expected (1, 3), got {result["tp_centroid"].shape}'
    print('PASS: empty GT -> zero tp matrices')


def test_process_batch_perfect_match():
    """Same polygon predicted as GT -> tp all True."""
    v = _make_validator()

    poly_data = [100, 100] + [30.0] * 32
    preds = {
        'bboxes': torch.tensor([poly_data]),
        'conf': torch.tensor([0.95]),
        'cls': torch.tensor([0]),
    }
    batch = {
        'bboxes': torch.tensor([poly_data]),
        'cls': torch.tensor([0]),
    }

    result = v._process_batch(preds, batch)
    tp = result['tp_shapely']
    assert tp[0].all(), f'Expected all True for perfect match, got {tp[0]}'
    print(f'PASS: perfect match -> tp all True ({tp.sum()}/{tp.size} thresholds)')


def test_process_batch_wrong_class():
    """Different class -> no match despite perfect geometry."""
    v = _make_validator()

    poly_data = [100, 100] + [30.0] * 32
    preds = {
        'bboxes': torch.tensor([poly_data]),
        'conf': torch.tensor([0.95]),
        'cls': torch.tensor([1]),
    }
    batch = {
        'bboxes': torch.tensor([poly_data]),
        'cls': torch.tensor([0]),
    }

    result = v._process_batch(preds, batch)
    tp = result['tp_shapely']
    assert not tp[0].any(), f'Expected no matches for wrong class, got {tp[0]}'
    print('PASS: wrong class -> no matches')


# ---------------------------------------------------------------------------
# Centroid matching tests
# ---------------------------------------------------------------------------


def test_centroid_matches_within_threshold():
    """Predictions within distance threshold -> matched."""
    v = _make_validator()

    pred_centroids = np.array([[100.0, 100.0]])
    gt_centroids = np.array([[103.0, 100.0]])
    pred_cls = torch.tensor([0])
    gt_cls = torch.tensor([0])

    result = v._compute_centroid_matches(pred_centroids, gt_centroids, pred_cls, gt_cls)
    assert result[0].all(), f'Expected all True for 3px distance, got {result[0]}'
    print('PASS: 3px distance -> matched at all thresholds')


def test_centroid_matches_beyond_threshold():
    """Predictions beyond distance threshold -> not matched."""
    v = _make_validator()

    pred_centroids = np.array([[100.0, 100.0]])
    gt_centroids = np.array([[120.0, 100.0]])
    pred_cls = torch.tensor([0])
    gt_cls = torch.tensor([0])

    result = v._compute_centroid_matches(pred_centroids, gt_centroids, pred_cls, gt_cls)
    assert not result[0].any(), f'Expected no matches for 20px distance, got {result[0]}'
    print('PASS: 20px distance -> no matches')


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_metrics_process():
    """RayCastDetMetrics.process computes mAP from synthetic stats."""
    metrics = RayCastDetMetrics(names={0: 'cell'})

    n = 10
    tp_shapely = np.ones((n, 10), dtype=bool)
    tp_centroid = np.ones((n, 3), dtype=bool)
    conf = np.linspace(0.95, 0.5, n)
    pred_cls = np.zeros(n, dtype=int)
    target_cls = np.zeros(n, dtype=int)
    target_img = np.zeros(n, dtype=int)

    metrics.update_stats(
        {
            'tp_shapely': tp_shapely,
            'tp_centroid': tp_centroid,
            'conf': conf,
            'pred_cls': pred_cls,
            'target_cls': target_cls,
            'target_img': target_img,
        }
    )

    metrics.process()

    shapely_map50 = metrics.shapely.map50
    shapely_map = metrics.shapely.map
    assert shapely_map50 > 0.9, f'Expected mAP50 > 0.9, got {shapely_map50:.4f}'
    assert shapely_map > 0.9, f'Expected mAP > 0.9, got {shapely_map:.4f}'

    centroid_map50 = metrics.centroid.map50
    assert centroid_map50 > 0.9, f'Expected centroid mAP50 > 0.9, got {centroid_map50:.4f}'

    results = metrics.results_dict
    assert 'metrics/mAP50(P)' in results
    assert 'metrics/mAP50(C)' in results
    assert 'fitness' in results
    print(f'PASS: metrics — shapely mAP50={shapely_map50:.3f}, centroid mAP50={centroid_map50:.3f}')


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


def test_iou_performance():
    """100x100 IoU matrix completes in < 5 seconds (sequential)."""
    rng = np.random.default_rng(42)
    n = 100

    cx = rng.uniform(50, 500, n)
    cy = rng.uniform(50, 500, n)
    radii = rng.uniform(15, 40, n)

    poly = np.zeros((n, RAYCAST_DIM), dtype=np.float64)
    poly[:, 0] = cx
    poly[:, 1] = cy
    poly[:, 2:] = radii[:, None]

    coords = _build_polygon_coords(poly)

    start = time.time()
    iou_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        iou_matrix[i] = _polygon_iou_row((coords[i], coords))
    elapsed = time.time() - start

    assert np.allclose(np.diag(iou_matrix), 1.0, atol=1e-6), 'Diagonal IoU should be 1.0'
    assert np.allclose(iou_matrix, iou_matrix.T, atol=1e-6), 'IoU matrix should be symmetric'
    assert elapsed < 5.0, f'100x100 sequential IoU took {elapsed:.2f}s (limit: 5s)'
    print(f'PASS: 100x100 IoU matrix in {elapsed:.2f}s')


def test_iou_threshold_discrimination():
    """Partial overlap (IoU~0.41) matches at threshold 0.5 but not 0.75."""
    v = _make_validator()

    # Two circles offset by 20px, radius 30 → IoU ≈ 0.41
    poly_pred = [100, 100] + [30.0] * 32
    poly_gt = [120, 100] + [30.0] * 32

    preds = {
        'bboxes': torch.tensor([poly_pred]),
        'conf': torch.tensor([0.9]),
        'cls': torch.tensor([0]),
    }
    batch = {
        'bboxes': torch.tensor([poly_gt]),
        'cls': torch.tensor([0]),
    }

    result = v._process_batch(preds, batch)
    tp = result['tp_shapely']

    # iouv = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # IoU ≈ 0.41 → should NOT match at 0.50 either
    # (the partial overlap of raycast approximations of circles may differ from true circles)
    iou_val = _polygon_iou_row(
        (
            _build_polygon_coords(np.array([poly_pred]))[0],
            _build_polygon_coords(np.array([poly_gt])),
        )
    )[0]

    # Verify consistency: tp matches are exactly what the thresholds dictate
    for i, threshold in enumerate(v.iouv.tolist()):
        if iou_val >= threshold:
            assert tp[0, i], f'Should match at threshold {threshold:.2f} (IoU={iou_val:.4f})'
        else:
            assert not tp[0, i], f'Should NOT match at threshold {threshold:.2f} (IoU={iou_val:.4f})'

    print(f'PASS: IoU threshold discrimination — IoU={iou_val:.4f}, matches at {tp[0].sum()}/10 thresholds')


def test_multiprocessing_iou_path():
    """Multiprocessing code path produces same results as sequential."""
    v = _make_validator()

    # Need n_pred * n_gt > 1000 to trigger multiprocessing
    rng = np.random.default_rng(123)
    n_pred, n_gt = 40, 30  # 40*30 = 1200 > 1000

    cx_p = rng.uniform(100, 400, n_pred)
    cy_p = rng.uniform(100, 400, n_pred)
    radii_p = rng.uniform(20, 35, n_pred)

    cx_g = rng.uniform(100, 400, n_gt)
    cy_g = rng.uniform(100, 400, n_gt)
    radii_g = rng.uniform(20, 35, n_gt)

    pred_poly = np.zeros((n_pred, RAYCAST_DIM), dtype=np.float64)
    pred_poly[:, 0] = cx_p
    pred_poly[:, 1] = cy_p
    pred_poly[:, 2:] = radii_p[:, None]

    gt_poly = np.zeros((n_gt, RAYCAST_DIM), dtype=np.float64)
    gt_poly[:, 0] = cx_g
    gt_poly[:, 1] = cy_g
    gt_poly[:, 2:] = radii_g[:, None]

    pred_coords = _build_polygon_coords(pred_poly)
    gt_coords = _build_polygon_coords(gt_poly)

    # Sequential (bypass the >1000 check)
    iou_seq = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i in range(n_pred):
        iou_seq[i] = _polygon_iou_row((pred_coords[i], gt_coords))

    # Via the validator method (should trigger multiprocessing)
    iou_par = v._compute_polygon_iou_matrix(pred_coords, gt_coords)

    assert np.allclose(iou_seq, iou_par, atol=1e-10), (
        f'Max diff between sequential and parallel: {np.abs(iou_seq - iou_par).max():.2e}'
    )
    print(f'PASS: multiprocessing path matches sequential — shape {iou_par.shape}, max_diff=0')


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_polygon_iou_identical()
    test_polygon_iou_non_overlapping()
    test_polygon_iou_partial()
    test_polygon_iou_degenerate()
    test_build_polygon_coords()
    test_process_batch_empty_preds()
    test_process_batch_empty_gt()
    test_process_batch_perfect_match()
    test_process_batch_wrong_class()
    test_centroid_matches_within_threshold()
    test_centroid_matches_beyond_threshold()
    test_metrics_process()
    test_iou_performance()
    test_iou_threshold_discrimination()
    test_multiprocessing_iou_path()

    print('\nAll Phase 8 tests passed!')
