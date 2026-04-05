"""Phase 9 tests — NumPy Post-Processing for Jetson Deployment.

Validates the post-processing pipeline that runs on the Jetson host.
These tests are hardware-independent and can run on any machine.

Run with: uv run python tests/phase_9/test_postprocess.py
"""

import numpy as np

from raycasted.export.postprocess import (
    RAYCAST_DIM,
    _build_anchor_grid,
    _decode_single,
    _dedup_by_distance,
    postprocess_raw_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(x))


# ---------------------------------------------------------------------------
# Activation tests
# ---------------------------------------------------------------------------


def test_sigmoid_values():
    """Raw logits → correct sigmoid values."""
    raw = np.array([[0.0, 0.0, 1.0, -1.0]], dtype=np.float32)
    sig = _sigmoid(raw)
    assert abs(sig[0, 0] - 0.5) < 1e-6
    assert abs(sig[0, 2] - 0.7311) < 1e-3
    assert abs(sig[0, 3] - 0.2689) < 1e-3
    print(f'PASS: sigmoid values — {sig[0]}')


def test_softplus_values():
    """Raw logits → correct softplus values."""
    raw = np.array([[0.0, 1.0, 5.0, -2.0]], dtype=np.float32)
    sp = _softplus(raw)
    assert abs(sp[0, 0] - 0.6931) < 1e-3  # ln(2)
    assert abs(sp[0, 1] - 1.3133) < 1e-3
    assert abs(sp[0, 2] - 5.0067) < 1e-3  # ≈ x for large x
    print(f'PASS: softplus values — {sp[0]}')


# ---------------------------------------------------------------------------
# Anchor grid tests
# ---------------------------------------------------------------------------


def test_anchor_grid_shape():
    """Anchor grid produces correct number of anchors for 640px."""
    strides = [8.0, 16.0, 32.0]
    ax, ay, s = _build_anchor_grid(strides, 640)

    # P3: 80x80=6400, P4: 40x40=1600, P5: 20x20=400 → total 8400
    expected = 6400 + 1600 + 400
    assert len(ax) == expected, f'Expected {expected} anchors, got {len(ax)}'
    print(f'PASS: anchor grid — {len(ax)} anchors for 640px')


def test_anchor_grid_stride_values():
    """Anchor grid strides are correctly assigned per scale."""
    strides = [8.0, 16.0, 32.0]
    _, _, s = _build_anchor_grid(strides, 640)

    assert np.all(s[:6400] == 8.0), 'P3 stride should be 8'
    assert np.all(s[6400:8000] == 16.0), 'P4 stride should be 16'
    assert np.all(s[8000:] == 32.0), 'P5 stride should be 32'
    print('PASS: anchor grid stride values correct')


# ---------------------------------------------------------------------------
# Decode tests
# ---------------------------------------------------------------------------


def test_decode_xy_centre():
    """Decoded xy centres are near anchor positions when logits = 0."""
    strides = [8.0, 16.0, 32.0]
    imgsz = 640
    nc = 4

    # Create raw logits for ALL anchors (8400 for 640px)
    n_anchors = 6400 + 1600 + 400
    raw = np.zeros((n_anchors, RAYCAST_DIM + nc), dtype=np.float32)
    # Set first anchor's class logit high
    raw[0, RAYCAST_DIM] = 5.0

    det = _decode_single(raw, strides, imgsz)

    # For P3 (stride=8), first anchor (0,0) = grid (0.5, 0.5)
    # sigmoid(0) = 0.5 → (0.5*2 - 0.5 + 0.5) * 8 = 8.0
    assert det.shape == (n_anchors, RAYCAST_DIM + 2), f'Expected ({n_anchors}, 36), got {det.shape}'
    assert abs(det[0, 0] - 8.0) < 0.1, f'Expected cx=8.0, got {det[0, 0]}'
    assert abs(det[0, 1] - 8.0) < 0.1, f'Expected cy=8.0, got {det[0, 1]}'
    print(f'PASS: decode xy centre — cx={det[0, 0]:.1f}, cy={det[0, 1]:.1f}')


def test_decode_ray_denorm():
    """Ray distances are denormalised to pixel space."""
    strides = [8.0]
    imgsz = 640

    n_anchors = 6400  # P3 only
    nc = 1
    raw = np.zeros((n_anchors, RAYCAST_DIM + nc), dtype=np.float32)
    raw[0, 2:34] = 1.0  # ray logits = 1.0 for first anchor

    det = _decode_single(raw, strides, imgsz)

    # softplus(1.0) ≈ 1.3133, * 640 ≈ 840.5
    expected_ray = float(_softplus(np.float32(1.0)) * 640)
    assert abs(det[0, 2] - expected_ray) < 0.1, f'Expected ray≈{expected_ray:.1f}, got {det[0, 2]:.1f}'
    print(f'PASS: ray denorm — softplus(1.0)*640 = {det[0, 2]:.1f}px')


# ---------------------------------------------------------------------------
# Full postprocess pipeline test
# ---------------------------------------------------------------------------


def test_full_postprocess_pipeline():
    """End-to-end: raw logits → filtered, deduped polygon detections."""
    strides = [8.0, 16.0, 32.0]
    imgsz = 640
    nc = 4

    # Create raw output: [1, 8400, nc + 34]
    n_anchors = 8400
    raw = np.random.randn(1, n_anchors, nc + RAYCAST_DIM).astype(np.float32) * 0.1

    # Make the first anchor have a high class score
    raw[0, 0, RAYCAST_DIM] = 5.0  # sigmoid(5) ≈ 0.993

    # Make the second anchor colocated with the first (same xy logits)
    raw[0, 1, :RAYCAST_DIM] = raw[0, 0, :RAYCAST_DIM]
    raw[0, 1, RAYCAST_DIM] = 3.0  # lower confidence

    results = postprocess_raw_output(
        raw,
        strides=strides,
        imgsz=imgsz,
        conf_threshold=0.25,
        dedup_radius_px=5.0,
    )

    assert len(results) == 1, f'Expected 1 image result, got {len(results)}'
    det = results[0]
    assert det.shape[1] == RAYCAST_DIM + 2, f'Expected 36 cols, got {det.shape[1]}'
    print(f'PASS: full pipeline — {det.shape[0]} detections from 8400 anchors')


def test_postprocess_empty_output():
    """All low-confidence predictions → empty result."""
    strides = [8.0, 16.0, 32.0]
    nc = 4
    n_anchors = 8400

    # All logits near zero → sigmoid ≈ 0.5, all class scores ≈ 0.5
    raw = np.zeros((1, n_anchors, nc + RAYCAST_DIM), dtype=np.float32)

    results = postprocess_raw_output(raw, strides=strides, conf_threshold=0.9)
    assert results[0].shape[0] == 0, f'Expected 0 detections, got {results[0].shape[0]}'
    print('PASS: empty output — 0 detections with high threshold')


def test_postprocess_batch():
    """Batch of 3 images produces 3 result arrays."""
    strides = [8.0, 16.0, 32.0]
    nc = 4
    n_anchors = 8400

    raw = np.random.randn(3, n_anchors, nc + RAYCAST_DIM).astype(np.float32) * 0.1
    # Give each image at least one high-confidence detection
    for b in range(3):
        raw[b, b * 100, RAYCAST_DIM] = 5.0

    results = postprocess_raw_output(raw, strides=strides, conf_threshold=0.25)
    assert len(results) == 3, f'Expected 3 results, got {len(results)}'
    print(f'PASS: batch of 3 — detections: {[r.shape[0] for r in results]}')


# ---------------------------------------------------------------------------
# Dedup test (standalone)
# ---------------------------------------------------------------------------


def test_dedup_numpy():
    """NumPy dedup matches the torch-based predictor dedup logic."""
    centroids = np.array([[100.0, 100.0], [101.0, 100.0], [200.0, 100.0]])
    scores = np.array([0.9, 0.8, 0.7])

    keep = _dedup_by_distance(centroids, scores, radius_px=5.0)
    # First two are 1px apart (within 5px) → keep only highest score
    # Third is 100px away → kept
    assert keep.sum() == 2, f'Expected 2 kept, got {keep.sum()}'
    assert keep[0], 'Should keep highest score at colocated position'
    assert keep[2], 'Should keep well-separated detection'
    print('PASS: numpy dedup — 2 kept from 3 (1 colocated pair)')


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_sigmoid_values()
    test_softplus_values()
    test_anchor_grid_shape()
    test_anchor_grid_stride_values()
    test_decode_xy_centre()
    test_decode_ray_denorm()
    test_full_postprocess_pipeline()
    test_postprocess_empty_output()
    test_postprocess_batch()
    test_dedup_numpy()

    print('\nAll Phase 9 postprocess tests passed!')
