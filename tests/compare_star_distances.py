#!/usr/bin/env python3
"""Compare Triton star_distances against Rust reference implementation.

Usage (on GPU machine):
    uv run python tests/compare_star_distances.py

Reports mean/max error between the two implementations for various
mask configurations. Run before switching training to Triton.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from stardist import star_distances as rust_star_distances


def _gen_rectangle(h, w, y0, x0, y1, x1):
    """Create a rectangular mask."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def _gen_circle(h, w, cy, cx, r):
    """Create a circular mask."""
    y, x = np.ogrid[:h, :w]
    m = ((y - cy) ** 2 + (x - cx) ** 2 <= r**2).astype(np.uint8)
    return m


def test_single_instance(name, mask_fn, h=64, w=64):
    """Run both implementations and compare."""
    mask = mask_fn(h, w)
    masks = np.stack([mask], axis=0)

    t0 = time.perf_counter()
    rust_lower, rust_upper = rust_star_distances(masks, 32)
    rust_time = (time.perf_counter() - t0) * 1000

    masks_t = torch.from_numpy(masks.astype(np.float32)).cuda()
    from raycasted.data.etl.loader.star_distances_triton import star_distances_triton

    t0 = time.perf_counter()
    triton_lower, triton_upper = star_distances_triton(masks_t, 32)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) * 1000

    triton_lower_np = triton_lower.cpu().numpy()
    triton_upper_np = triton_upper.cpu().numpy()

    lower_diff = np.abs(rust_lower[0] - triton_lower_np)
    upper_diff = np.abs(rust_upper[0] - triton_upper_np)

    inside = mask > 0
    lower_err_inside = lower_diff[:, inside].mean() if inside.any() else 0.0
    upper_err_inside = upper_diff[:, inside].mean() if inside.any() else 0.0
    lower_max = lower_diff.max()
    upper_max = upper_diff.max()

    ok = lower_max < 5.0 and upper_max < 5.0
    status = 'PASS' if ok else 'FAIL'

    print(
        f'  [{status}] {name:<20} '
        f'L_err={lower_err_inside:.2f}px(max={lower_max:.1f}) '
        f'U_err={upper_err_inside:.2f}px(max={upper_max:.1f}) '
        f'rust={rust_time:.1f}ms triton={triton_time:.2f}ms'
    )
    return ok


def test_multi_instance(name, masks_fn, h=64, w=64):
    """Test with multiple overlapping instances."""
    masks = masks_fn(h, w)
    if masks.ndim == 2:
        masks = np.stack([masks], axis=0)

    t0 = time.perf_counter()
    rust_lower, rust_upper = rust_star_distances(masks.astype(np.uint8), 32)
    rust_time = (time.perf_counter() - t0) * 1000

    masks_t = torch.from_numpy(masks.astype(np.float32)).cuda()
    from raycasted.data.etl.loader.star_distances_triton import star_distances_triton

    t0 = time.perf_counter()
    triton_lower, triton_upper = star_distances_triton(masks_t, 32)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) * 1000

    triton_lower_np = triton_lower.cpu().numpy()
    triton_upper_np = triton_upper.cpu().numpy()

    lower_diff = np.abs(rust_lower[0] - triton_lower_np)
    upper_diff = np.abs(rust_upper[0] - triton_upper_np)

    any_mask = masks.sum(axis=0) > 0
    lower_err = lower_diff[:, any_mask].mean() if any_mask.any() else 0.0
    upper_err = upper_diff[:, any_mask].mean() if any_mask.any() else 0.0
    lower_max = lower_diff.max()
    upper_max = upper_diff.max()

    ok = lower_max < 5.0 and upper_max < 5.0
    status = 'PASS' if ok else 'FAIL'

    print(
        f'  [{status}] {name:<20} '
        f'N={len(masks)} L_err={lower_err:.2f}px(max={lower_max:.1f}) '
        f'U_err={upper_err:.2f}px(max={upper_max:.1f}) '
        f'rust={rust_time:.1f}ms triton={triton_time:.2f}ms'
    )
    return ok


def main():
    print('=== Triton vs Rust star_distances comparison ===\n')

    all_ok = []

    # Single instances
    all_ok.append(test_single_instance('rect_32x20', lambda h, w: _gen_rectangle(h, w, 16, 16, 48, 36)))
    all_ok.append(test_single_instance('circle_r12', lambda h, w: _gen_circle(h, w, 32, 32, 12)))
    all_ok.append(test_single_instance('circle_r4', lambda h, w: _gen_circle(h, w, 32, 32, 4)))
    all_ok.append(test_single_instance('edge_rect', lambda h, w: _gen_rectangle(h, w, 0, 0, 20, 20)))

    # Multiple instances
    def two_separate(h, w):
        masks = np.zeros((2, h, w), dtype=np.uint8)
        masks[0, 8:28, 8:28] = 1
        masks[1, 36:56, 36:56] = 1
        return masks

    def two_overlap(h, w):
        masks = np.zeros((2, h, w), dtype=np.uint8)
        masks[0, 8:38, 16:46] = 1
        masks[1, 26:56, 18:48] = 1
        return masks

    def three_mixed(h, w):
        masks = np.zeros((3, h, w), dtype=np.uint8)
        masks[0, 4:24, 4:24] = 1
        masks[1, 20:44, 28:52] = 1
        masks[2, 40:62, 4:28] = 1
        return masks

    all_ok.append(test_multi_instance('two_separate', two_separate))
    all_ok.append(test_multi_instance('two_overlap', two_overlap))
    all_ok.append(test_multi_instance('three_mixed', three_mixed))

    # Realistic size (256x256)
    def realistic(h, w):
        np.random.seed(42)
        masks = np.zeros((8, h, w), dtype=np.uint8)
        for i in range(8):
            cx = np.random.randint(30, 226)
            cy = np.random.randint(30, 226)
            r = np.random.randint(8, 30)
            masks[i] = _gen_circle(h, w, cy, cx, r)
        return masks

    all_ok.append(test_multi_instance('8_circles_256', realistic, h=256, w=256))

    print()
    passed = sum(all_ok)
    failed = len(all_ok) - passed
    print(f'{"=" * 50}')
    print(f'Results: {passed} passed, {failed} failed out of {len(all_ok)}')
    if failed > 0:
        print('WARNING: Triton output differs significantly from Rust. Do NOT switch.')
        exit(1)
    else:
        print('OK: Triton matches Rust within tolerance. Ready to switch.')
        exit(0)


if __name__ == '__main__':
    main()
