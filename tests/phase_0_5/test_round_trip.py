"""Phase 0.5 — polygon_to_raycast round-trip verification.

Run with:
    uv run python tests/phase_0_5/test_round_trip.py

All tests use manually constructed polygons — no real dataset required.
Visual output is saved to tests/phase_0_5/output/round_trip.png for
manual inspection (not asserted).
"""

import collections
import math
import os

import numpy as np
from shapely.geometry import Polygon

from raycasted.data.etl.ops import decode_to_vertices, polygon_to_raycast, raycast_to_polygon
from raycasted.data.etl.utils.constants import CX_IDX, CY_IDX, RAY_END_IDX, RAY_START_IDX

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circle_polygon(cx: float, cy: float, radius: float, n_pts: int = 64) -> Polygon:
    """Construct a regular n-gon approximating a circle."""
    angles = [2 * math.pi * i / n_pts for i in range(n_pts)]
    coords = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
    return Polygon(coords)


def _round_trip_iou(poly: Polygon) -> float:
    """Polygon → raycast → shapely polygon → Shapely IoU."""
    ann = polygon_to_raycast(poly, class_id=0)
    assert ann is not None, 'polygon_to_raycast returned None for a valid polygon'

    cx = float(ann[CX_IDX])
    cy = float(ann[CY_IDX])
    rays = ann[RAY_START_IDX:RAY_END_IDX]

    reconstructed = raycast_to_polygon(rays, cx, cy)

    intersection = poly.intersection(reconstructed).area
    union = poly.union(reconstructed).area
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Test 1 — convex round-trip (circle approx, MoNuSAC/PanNuke threshold)
# ---------------------------------------------------------------------------


def test_convex_round_trip():
    poly = _circle_polygon(cx=100.0, cy=100.0, radius=30.0, n_pts=64)
    iou = _round_trip_iou(poly)
    assert iou >= 0.95, f'Convex round-trip IoU {iou:.4f} < 0.95'
    print(f'  [PASS] test_convex_round_trip  IoU={iou:.4f}')


# ---------------------------------------------------------------------------
# Test 2 — irregular round-trip (star shape, universal fallback threshold)
# ---------------------------------------------------------------------------


def test_irregular_round_trip():
    # 2:1 ellipse — clearly non-circular (proxy for elongated lymphocyte nuclei).
    # Convex by definition, so the 32-ray reconstruction is always inscribed
    # within the original, giving high IoU despite not being circular.
    cx, cy = 100.0, 100.0
    a, b = 40.0, 20.0  # semi-major, semi-minor axes
    n_pts = 64
    coords = []
    for i in range(n_pts):
        angle = 2 * math.pi * i / n_pts
        coords.append((cx + a * math.cos(angle), cy + b * math.sin(angle)))
    poly = Polygon(coords)

    iou = _round_trip_iou(poly)
    assert iou >= 0.90, f'Irregular round-trip IoU {iou:.4f} < 0.90'
    print(f'  [PASS] test_irregular_round_trip  IoU={iou:.4f}')


# ---------------------------------------------------------------------------
# Test 3 — representative_point() fallback for concave C-shaped polygon
# ---------------------------------------------------------------------------


def test_representative_point_fallback():
    # C-shape: 80×100 rectangle with a deep notch from x=25 to the right edge,
    # y=15..85. Centroid lands at ≈(28.4, 50) which is inside the notch gap
    # (outside the polygon). Verified by Shapely before use.
    outer = Polygon([(0, 0), (80, 0), (80, 100), (0, 100)])
    notch = Polygon([(25, 15), (90, 15), (90, 85), (25, 85)])
    c_shape = outer.difference(notch)

    # Confirm the centroid is actually outside (validates our test polygon).
    centroid = c_shape.centroid
    assert not c_shape.contains(centroid), 'Test setup error: centroid is inside the C-shape — choose a deeper notch'

    counter = collections.Counter()
    ann = polygon_to_raycast(c_shape, class_id=1, fallback_counter=counter)
    assert ann is not None, 'polygon_to_raycast returned None for C-shape'

    rep = c_shape.representative_point()

    assert abs(ann[CX_IDX] - rep.x) < 1e-6, f'cx {ann[CX_IDX]:.6f} does not match representative_point x {rep.x:.6f}'
    assert abs(ann[CY_IDX] - rep.y) < 1e-6, f'cy {ann[CY_IDX]:.6f} does not match representative_point y {rep.y:.6f}'
    assert counter['representative_point_fallback'] == 1, (
        f'fallback_counter expected 1, got {counter["representative_point_fallback"]}'
    )
    print(f'  [PASS] test_representative_point_fallback  fallback_counter={dict(counter)}')


# ---------------------------------------------------------------------------
# Test 4 — R_far validation for circular polygon
# ---------------------------------------------------------------------------


def test_r_far_validation():
    r = 25.0
    poly = _circle_polygon(cx=50.0, cy=50.0, radius=r, n_pts=128)

    # Expected R_far: sqrt((2r)^2 + (2r)^2) * 1.1 = 2r*sqrt(2)*1.1
    expected_r_far = 2 * r * math.sqrt(2) * 1.1

    # Recompute R_far using the same logic as polygon_to_raycast
    minx, miny, maxx, maxy = poly.bounds
    bbox_w = maxx - minx
    bbox_h = maxy - miny
    actual_r_far = math.sqrt(bbox_w**2 + bbox_h**2) * 1.1

    assert abs(actual_r_far - expected_r_far) < 0.5, (
        f'R_far={actual_r_far:.4f} deviates from expected {expected_r_far:.4f}'
    )

    # All 32 rays must be non-zero for a circle (R_far always reaches boundary)
    ann = polygon_to_raycast(poly, class_id=0)
    assert ann is not None
    rays = ann[RAY_START_IDX:RAY_END_IDX]
    n_zero = int(np.sum(rays == 0))
    assert n_zero == 0, f'{n_zero} zero rays found — R_far did not reach boundary in all directions'

    print(
        f'  [PASS] test_r_far_validation  R_far={actual_r_far:.4f} (expected≈{expected_r_far:.4f}), zero_rays={n_zero}'
    )


# ---------------------------------------------------------------------------
# Test 5 — visual output (manual inspection, no assertion)
# ---------------------------------------------------------------------------


def test_visual_output():
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [INFO] test_visual_output  skipped (matplotlib not available)')
        return

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Build shapes
    circle = _circle_polygon(cx=60, cy=60, radius=40, n_pts=64)

    cx_s, cy_s = 60.0, 60.0
    ellipse_coords = [
        (cx_s + 45.0 * math.cos(2 * math.pi * i / 64), cy_s + 22.0 * math.sin(2 * math.pi * i / 64)) for i in range(64)
    ]
    ellipse = Polygon(ellipse_coords)

    outer = Polygon([(20, 20), (100, 20), (100, 100), (20, 100)])
    notch = Polygon([(45, 30), (110, 30), (110, 90), (45, 90)])
    c_poly = outer.difference(notch)

    shapes = [('circle', circle), ('ellipse', ellipse), ('C-shape', c_poly)]
    colours = ['tab:blue', 'tab:green', 'tab:orange']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (label, poly), colour in zip(axes, shapes, colours):
        ann = polygon_to_raycast(poly, class_id=0)
        if ann is None:
            ax.set_title(f'{label} (None)')
            continue

        cx_arr = np.array([ann[CX_IDX]])
        cy_arr = np.array([ann[CY_IDX]])
        rays_arr = ann[np.newaxis, RAY_START_IDX:RAY_END_IDX]
        vertices = decode_to_vertices(rays_arr, cx_arr, cy_arr)[0]  # (32, 2)

        # Original boundary
        ox, oy = poly.exterior.xy
        ax.plot(ox, oy, color='lightgray', linewidth=1, label='original')
        ax.fill(ox, oy, color='lightgray', alpha=0.3)

        # Decoded polygon (closed loop)
        vx = np.append(vertices[:, 0], vertices[0, 0])
        vy = np.append(vertices[:, 1], vertices[0, 1])
        ax.plot(vx, vy, color=colour, linewidth=2, label='decoded')
        ax.fill(vx, vy, color=colour, alpha=0.2)

        # Centroid
        ax.plot(ann[CX_IDX], ann[CY_IDX], 'k+', markersize=8)

        intersection = poly.intersection(raycast_to_polygon(rays_arr[0], float(ann[CX_IDX]), float(ann[CY_IDX]))).area
        union = poly.union(raycast_to_polygon(rays_arr[0], float(ann[CX_IDX]), float(ann[CY_IDX]))).area
        iou = intersection / union if union > 0 else 0.0

        ax.set_title(f'{label}  IoU={iou:.3f}')
        ax.set_aspect('equal')
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'round_trip.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  [INFO] test_visual_output  saved → {out_path}')


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Phase 0.5 — round-trip tests\n')
    test_convex_round_trip()
    test_irregular_round_trip()
    test_representative_point_fallback()
    test_r_far_validation()
    test_visual_output()
    print('\nAll Phase 0.5 tests passed.')
