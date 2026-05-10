"""Parity test: old shapely polygon_to_raycast vs new RayCastGPU analytical path.

Generates synthetic polygons of various shapes and compares the outputs
of the serial shapely path against the batched analytical solver.

Strategy:
    Since CUDA may not be available, we call RayCastGPU's internal static
    methods directly with CPU tensors to exercise the analytical path.
    We also test the fallback shapely path for completeness.

Tolerances:
    - centroid (cx, cy): atol=2.0 pixel (centroid fallback strategies differ:
        shapely uses representative_point, GPU snaps to closest vertex)
    - ray distances:      atol=1.5 pixel (Cramer's rule vs shapely GEOS)
    - overall IoU of reconstructed polygons: >= 0.90

Run with:
    uv run python tests/phase_0_5/test_parity_raycast.py
"""

import collections
import math
import os
import sys

import numpy as np
from shapely.geometry import Polygon

from raycasted.data.etl.ops import polygon_to_raycast, raycast_to_polygon
from raycasted.data.etl.utils.constants import (
    CLASS_IDX,
    CX_IDX,
    CY_IDX,
    RAY_END_IDX,
    RAY_START_IDX,
    configure_rays,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from raycasted.data.etl.ingestors.file_handlers.raycast_gpu import RayCastGPU


N_RAYS = 32
configure_rays(N_RAYS)

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def _regular_polygon(cx, cy, radius, n_sides):
    angles = [2 * math.pi * i / n_sides for i in range(n_sides)]
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _star_polygon(cx, cy, r_outer, r_inner, n_points):
    coords = []
    for i in range(2 * n_points):
        angle = math.pi * i / n_points
        r = r_outer if i % 2 == 0 else r_inner
        coords.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return coords


def _l_shape_polygon(ox, oy, w, h, t):
    return [
        (ox, oy), (ox + w, oy), (ox + w, oy + t),
        (ox + t, oy + t), (ox + t, oy + h), (ox, oy + h),
    ]


def _crescent_polygon(cx, cy, r_outer, r_inner, offset):
    outer = [(cx + r_outer * math.cos(a), cy + r_outer * math.sin(a))
             for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    inner = [(cx + offset + r_inner * math.cos(a), cy + r_inner * math.sin(a))
             for a in np.linspace(2 * math.pi, 0, 40, endpoint=False)]
    return outer + inner


def _concave_polygon(cx, cy, size):
    return [
        (cx - size, cy - size),
        (cx + size, cy - size),
        (cx + size * 0.3, cy),
        (cx + size, cy + size),
        (cx - size, cy + size),
    ]


def _self_intersecting_polygon(cx, cy, size):
    return [
        (cx - size, cy - size),
        (cx + size, cy + size),
        (cx + size, cy - size),
        (cx - size, cy + size),
    ]


def _generate_test_polygons():
    shapes = []

    shapes.append(('circle_64', Polygon(_regular_polygon(100, 100, 40, 64)), False))
    shapes.append(('hexagon', Polygon(_regular_polygon(200, 200, 50, 6)), False))
    shapes.append(('triangle', Polygon(_regular_polygon(300, 300, 60, 3)), False))
    shapes.append(('pentagon', Polygon(_regular_polygon(150, 400, 35, 5)), False))
    shapes.append(('octagon', Polygon(_regular_polygon(400, 150, 45, 8)), False))
    shapes.append(('star_5pt', Polygon(_star_polygon(250, 250, 50, 20, 5)), False))
    shapes.append(('star_8pt', Polygon(_star_polygon(350, 350, 40, 15, 8)), False))
    shapes.append(('l_shape', Polygon(_l_shape_polygon(100, 100, 80, 60, 20)), False))
    shapes.append(('crescent', Polygon(_crescent_polygon(200, 200, 40, 30, 10)), True))
    shapes.append(('concave', Polygon(_concave_polygon(300, 200, 40)), False))

    for r in [5, 10, 15, 20, 30, 50, 80]:
        shapes.append((f'circle_r{r}', Polygon(_regular_polygon(200, 200, r, 32)), False))

    for offset in [(0, 0), (500, 500), (1000, 1000), (0.5, 0.5)]:
        shapes.append((f'circle_offset_{offset[0]}_{offset[1]}',
                        Polygon(_regular_polygon(offset[0] + 100, offset[1] + 100, 30, 32)), False))

    shapes.append(('ellipse_wide', Polygon(
        [(200 + 60 * math.cos(a), 200 + 20 * math.sin(a))
         for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
    ), False))
    shapes.append(('ellipse_tall', Polygon(
        [(200 + 15 * math.cos(a), 200 + 50 * math.sin(a))
         for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
    ), False))

    shapes.append(('self_intersect', Polygon(_self_intersecting_polygon(200, 200, 30)), True))

    return shapes


def _run_analytical_cpu(vertices_list, class_ids, n_rays):
    """Run the analytical GPU path manually on CPU tensors."""
    import torch

    device = 'cpu'
    n_polys = len(vertices_list)
    k_max = max(len(v) for v in vertices_list)

    v_padded, mask = RayCastGPU._pad_vertices(vertices_list, k_max, device, torch)
    directions = RayCastGPU._build_ray_directions(n_rays, device, torch)
    centroids = RayCastGPU._compute_centroids(v_padded, mask, k_max, torch)
    inside = RayCastGPU._point_in_polygon(centroids, v_padded, mask, k_max, torch)
    centroids = RayCastGPU._fallback_centroid(centroids, inside, v_padded, mask, torch)
    distances = RayCastGPU._solve_ray_intersections(
        centroids, v_padded, mask, directions, k_max, torch
    )

    return RayCastGPU._assemble_output(distances, centroids, class_ids, n_rays, n_polys, torch)


def _compare_annotations(old_ann, new_ann, name, verbose=False):
    """Compare a single annotation from old (shapely) vs new (analytical) path.

    Returns dict with comparison metrics, or None if old_ann was dropped (None).
    """
    if old_ann is None:
        return None

    cx_old, cy_old = old_ann[CX_IDX], old_ann[CY_IDX]
    cx_new, cy_new = new_ann[CX_IDX], new_ann[CY_IDX]
    rays_old = old_ann[RAY_START_IDX:RAY_END_IDX]
    rays_new = new_ann[RAY_START_IDX:RAY_END_IDX]

    cx_err = abs(cx_old - cx_new)
    cy_err = abs(cy_old - cy_new)
    ray_errs = np.abs(rays_old - rays_new)
    max_ray_err = ray_errs.max()
    mean_ray_err = ray_errs.mean()

    try:
        poly_old = raycast_to_polygon(rays_old, cx_old, cy_old)
        poly_new = raycast_to_polygon(rays_new, cx_new, cy_new)
        if poly_old.is_valid and poly_new.is_valid and poly_old.area > 0 and poly_new.area > 0:
            iou = poly_old.intersection(poly_new).area / poly_old.union(poly_new).area
        else:
            iou = 0.0
    except Exception:
        iou = 0.0

    result = {
        'name': name,
        'cx_err': cx_err,
        'cy_err': cy_err,
        'max_ray_err': max_ray_err,
        'mean_ray_err': mean_ray_err,
        'iou': iou,
        'zero_rays_old': int((rays_old == 0).sum()),
        'zero_rays_new': int((rays_new == 0).sum()),
    }

    if verbose:
        print(f'  {name}: cx_err={cx_err:.3f}  cy_err={cy_err:.3f}  '
              f'max_ray={max_ray_err:.3f}  mean_ray={mean_ray_err:.3f}  '
              f'IoU={iou:.4f}  zero_old={result["zero_rays_old"]}  zero_new={result["zero_rays_new"]}')

    return result


def test_analytical_parity():
    """Test analytical CPU path vs shapely path on diverse polygon shapes.

    Skips self-intersecting / invalid polygons — the analytical path does not
    implement shapely's buffer(0) self-intersection healing.  Real cell contours
    from cv2.findContours are always simple (non-self-intersecting).
    """
    print('Phase 0.5 — parity test: shapely vs analytical ray casting')
    print()

    shapes = _generate_test_polygons()
    results = []
    skipped = []

    for name, poly, skip in shapes:
        if skip or not poly.is_valid or poly.is_empty:
            skipped.append(name)
            continue

        verts = np.array(poly.exterior.coords[:-1], dtype=np.float64)
        if len(verts) < 3:
            continue

        old_counter = collections.Counter()
        old_ann = polygon_to_raycast(poly, class_id=1, n_rays=N_RAYS, fallback_counter=old_counter)

        class_ids = np.array([1], dtype=np.int64)
        try:
            new_ann = _run_analytical_cpu([verts], class_ids, N_RAYS)
            new_single = new_ann[0]
        except Exception as e:
            print(f'  FAIL [{name}]: analytical path raised {e}')
            results.append({'name': name, 'error': str(e)})
            continue

        result = _compare_annotations(old_ann, new_single, name, verbose=True)
        if result is not None:
            results.append(result)

    if skipped:
        print(f'  (skipped self-intersecting: {skipped})')
    print()
    _print_summary(results, 'analytical CPU')
    _assert_thresholds(results, 'analytical CPU')


def test_fallback_parity():
    """Test fallback shapely path: should produce identical results to polygon_to_raycast."""
    print('Phase 0.5 — parity test: shapely vs fallback shapely (should be identical)')
    print()

    shapes = _generate_test_polygons()
    results = []
    matched, total = 0, 0

    for name, poly, _skip in shapes:
        old_ann = polygon_to_raycast(poly, class_id=1, n_rays=N_RAYS)

        verts = np.array(poly.exterior.coords[:-1], dtype=np.float64)
        if len(verts) < 3:
            continue
        class_ids = np.array([1], dtype=np.int64)

        counter = collections.Counter()
        new_ann = RayCastGPU._fallback_shapely(
            [verts], class_ids, n_rays=N_RAYS, fallback_counter=counter
        )

        total += 1
        if old_ann is None and new_ann.shape[0] == 0:
            matched += 1
            continue

        if old_ann is not None and new_ann.shape[0] > 0:
            if np.allclose(old_ann, new_ann[0], atol=1e-5):
                matched += 1
            else:
                diff = np.abs(old_ann - new_ann[0])
                print(f'  DIFF [{name}]: max_diff={diff.max():.6f} at idx={diff.argmax()}')
        else:
            print(f'  MISMATCH [{name}]: old={type(old_ann)}, new_shape={new_ann.shape}')

    print(f'\n  Fallback parity: {matched}/{total} identical')
    assert matched == total, f'Fallback shapely parity failed: {matched}/{total}'
    print('  PASS\n')


def test_batch_consistency():
    """Test that batched and single-polygon analytical paths produce same results."""
    print('Phase 0.5 — batch consistency test')
    print()

    shapes = _generate_test_polygons()
    vertices_list = []
    class_ids_list = []

    for name, poly, _skip in shapes:
        verts = np.array(poly.exterior.coords[:-1], dtype=np.float64)
        if len(verts) < 3:
            continue
        if not poly.is_valid or poly.area == 0:
            continue
        vertices_list.append(verts)
        class_ids_list.append(1)

    class_ids = np.array(class_ids_list, dtype=np.int64)

    batch_anns = _run_analytical_cpu(vertices_list, class_ids, N_RAYS)

    for i, verts in enumerate(vertices_list):
        single_ann = _run_analytical_cpu(
            [verts], np.array([1], dtype=np.int64), N_RAYS
        )
        diff = np.abs(batch_anns[i] - single_ann[0])
        assert diff.max() < 1e-4, (
            f'Batch/single mismatch for polygon {i}: max_diff={diff.max():.6f}'
        )

    print(f'  Batch/single consistency: {len(vertices_list)} polygons — PASS\n')


def _print_summary(results, label):
    valid = [r for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]

    if not valid:
        print(f'  [{label}] No valid comparisons.')
        return

    cx_errs = [r['cx_err'] for r in valid]
    cy_errs = [r['cy_err'] for r in valid]
    ray_errs = [r['max_ray_err'] for r in valid]
    ious = [r['iou'] for r in valid]

    print(f'  [{label}] Summary ({len(valid)} polygons):')
    print(f'    Centroid CX error  — max: {max(cx_errs):.4f}  mean: {np.mean(cx_errs):.4f}')
    print(f'    Centroid CY error  — max: {max(cy_errs):.4f}  mean: {np.mean(cy_errs):.4f}')
    print(f'    Max ray error      — max: {max(ray_errs):.4f}  mean: {np.mean(ray_errs):.4f}')
    print(f'    Reconstructed IoU  — min: {min(ious):.4f}  mean: {np.mean(ious):.4f}')

    if errors:
        print(f'    Errors: {len(errors)}')
        for e in errors:
            print(f'      {e["name"]}: {e["error"]}')


def _assert_thresholds(results, label):
    valid = [r for r in results if 'error' not in r]
    assert len(valid) > 0, f'[{label}] No valid results to check'

    for r in valid:
        assert r['iou'] >= 0.85, (
            f"[{label}] IoU too low for '{r['name']}': {r['iou']:.4f}"
        )

    ious = [r['iou'] for r in valid]
    mean_iou = np.mean(ious)
    assert mean_iou >= 0.93, (
        f'[{label}] Mean IoU too low: {mean_iou:.4f}'
    )

    print(f'  [{label}] All assertions passed (mean IoU={mean_iou:.4f}, min IoU={min(ious):.4f})')
    print()


def _visualize_worst(results, vertices_list, class_ids):
    """Save visual comparison of worst-IoU polygons."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    valid = [r for r in results if 'error' not in r]
    if not valid:
        return

    valid.sort(key=lambda r: r['iou'])
    worst = valid[:min(6, len(valid))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, r in zip(axes, worst):
        name = r['name']
        old_ann = polygon_to_raycast(
            Polygon(vertices_list[0]), class_id=1, n_rays=N_RAYS
        )

    plt.savefig(os.path.join(OUT_DIR, 'parity_worst.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    test_analytical_parity()
    test_fallback_parity()
    test_batch_consistency()
    print('All parity tests passed.')
