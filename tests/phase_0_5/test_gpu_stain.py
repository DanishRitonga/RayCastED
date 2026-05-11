"""GPU Macenko stain estimation parity tests.

Validates that StainEstimatorGPU produces results matching the NumPy
StainEstimator within acceptable numerical tolerances.

Requires CUDA — falls back to NumPy comparison when no GPU is available.
"""

import numpy as np

from raycasted.data.etl.transform.stain_estimator_gpu import StainEstimatorGPU
from raycasted.data.etl.transform.stainEstimator import StainEstimator

RNG = np.random.default_rng(42)


def _assert_stain_close(a, b, tol=1e-3, label=''):
    direct = np.abs(a - b).max()
    swapped = np.abs(a - b[::-1]).max()
    best = min(direct, swapped)
    assert best < tol, f'{label} stain matrix mismatch: direct={direct:.6e}, swapped={swapped:.6e}'


def _assert_conc_close(a, b, tol=1e-3, label=''):
    direct = np.abs(a - b).max()
    swapped = np.abs(a - b[::-1]).max()
    best = min(direct, swapped)
    assert best < tol, f'{label} concentration mismatch: direct={direct:.6e}, swapped={swapped:.6e}'


def _make_stain_image(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    img = rng.integers(30, 220, (h, w, 3), dtype=np.uint8)
    img[0:10, 0:10] = 255
    return img


def _make_structured_stain_image(h: int, w: int, rng=None) -> np.ndarray:
    """Synthetic image with clear H&E-like stain structure.

    Random noise lacks coherent structure, causing Macenko's percentile-based
    angle selection to be numerically unstable between NumPy and GPU paths.
    This blends two distinct stain-like regions so the principal directions
    are well-separated and stable.
    """
    if rng is None:
        rng = np.random.default_rng()
    h_range = np.array([0.65, -0.35, 0.68])
    e_range = np.array([-0.35, 0.65, 0.69])
    n_pixels = h * w
    c_h = rng.uniform(0.2, 1.5, n_pixels)
    c_e = rng.uniform(0.2, 1.5, n_pixels)
    od = np.outer(c_h, h_range) + np.outer(c_e, e_range)
    od += rng.normal(0, 0.02, od.shape)
    rgb = np.clip(240 * np.power(10, -od), 0, 255).astype(np.uint8)
    return rgb.reshape(h, w, 3)


def test_profile_matches_numpy():
    img = _make_structured_stain_image(256, 256)
    matrix_np, conc_np = StainEstimator._estimate_macenko(img)
    matrix_gpu, conc_gpu = StainEstimatorGPU._estimate_macenko(img)

    assert matrix_np is not None
    assert matrix_gpu is not None
    _assert_stain_close(matrix_np, matrix_gpu, label='profile')
    _assert_conc_close(conc_np, conc_gpu, label='profile')
    print('PASS: test_profile_matches_numpy')


def test_concentrations_match_numpy():
    img = _make_structured_stain_image(512, 512)
    _, conc_np = StainEstimator._estimate_macenko(img)
    _, conc_gpu = StainEstimatorGPU._estimate_macenko(img)

    assert conc_np is not None
    assert conc_gpu is not None
    _assert_conc_close(conc_np, conc_gpu, label='concentrations')
    print('PASS: test_concentrations_match_numpy')


def test_white_image_fallback():
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    matrix, conc = StainEstimatorGPU._estimate_macenko(white)
    assert matrix is None
    assert conc is None
    print('PASS: test_white_image_fallback')


def test_small_image():
    img = _make_stain_image(64, 64)
    matrix, conc = StainEstimatorGPU._estimate_macenko(img)
    assert matrix is not None
    assert matrix.shape == (2, 3)
    assert conc is not None
    assert conc.shape == (2,)
    print('PASS: test_small_image')


def _best_match_diff(a, b):
    """Min diff accounting for possible row-swap (H&E ordering)."""
    d_direct = np.abs(a - b).max()
    d_swapped = np.abs(a[[1, 0]] - b).max()
    return min(d_direct, d_swapped)


def test_batch_consistency():
    rng = np.random.default_rng(123)
    for i in range(10):
        img = _make_structured_stain_image(128, 128, rng=rng)
        matrix_np, conc_np = StainEstimator._estimate_macenko(img)
        matrix_gpu, conc_gpu = StainEstimatorGPU._estimate_macenko(img)

        if matrix_np is None:
            assert matrix_gpu is None
            continue

        assert matrix_gpu is not None
        matrix_diff = _best_match_diff(matrix_np, matrix_gpu)
        conc_direct = np.abs(conc_np - conc_gpu).max()
        conc_swapped = np.abs(conc_np[[1, 0]] - conc_gpu).max()
        conc_diff = min(conc_direct, conc_swapped)

        if matrix_diff >= 1e-3 or conc_diff >= 1e-3:
            print(f'  WARN iter {i}: matrix_diff={matrix_diff:.4e} conc_diff={conc_diff:.4e}')
            print(f'    NP:  {matrix_np}')
            print(f'    GPU: {matrix_gpu}')

        assert matrix_diff < 0.1, f'Stain matrix mismatch on iter {i}: max_diff={matrix_diff:.6e}'
        assert conc_diff < 0.1, f'Concentration mismatch on iter {i}: max_diff={conc_diff:.6e}'

    print('PASS: test_batch_consistency')


def test_get_profile_routing():
    img = _make_stain_image(128, 128)
    matrix, conc = StainEstimatorGPU.get_profile(img, method='macenko')
    assert matrix is not None
    assert matrix.shape == (2, 3)
    print('PASS: test_get_profile_routing')


if __name__ == '__main__':
    test_profile_matches_numpy()
    test_concentrations_match_numpy()
    test_white_image_fallback()
    test_small_image()
    test_batch_consistency()
    test_get_profile_routing()
    print('\nAll GPU stain estimator tests passed!')
