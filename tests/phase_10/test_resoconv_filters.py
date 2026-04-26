"""Test ResoConv wavelet filter values.

This test verifies that the hardcoded fallback values in resoconv.py
match the reference implementation from pywt.Wavelet('bior2.2').

Run with: uv run python tests/phase_10/test_resoconv_filters.py
"""

import torch


def test_bior22_filters_match_pywt():
    """Verify hardcoded bior2.2 filters match pywt reference."""
    try:
        import pywt

        w = pywt.Wavelet('bior2.2')
        ref_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        ref_hi = torch.tensor(w.dec_hi, dtype=torch.float32)

        from raycasted.model.blocks.resoconv import _BIOR_LO, _BIOR_HI

        assert torch.allclose(_BIOR_LO, ref_lo, rtol=1e-6), 'LO filters mismatch'
        assert torch.allclose(_BIOR_HI, ref_hi, rtol=1e-6), 'HI filters mismatch'

        print('✓ bior2.2 filters match pywt.Wavelet("bior2.2") reference')
        print(f'  dec_lo: {_BIOR_LO.tolist()}')
        print(f'  dec_hi: {_BIOR_HI.tolist()}')
        return True

    except ImportError:
        print('⚠ pywt not installed, skipping verification')
        print('  (hardcoded fallback values will be used)')
        return False


def test_dwt2d_forward_shape():
    """Test DWT2D produces expected output shape."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import DWT2D

    B, C, H, W = 2, 16, 32, 32
    x = torch.randn(B, C, H, W)

    dwt = DWT2D(C)
    x_dwt = dwt(x)

    expected_shape = (B, 4 * C, H // 2, W // 2)
    assert x_dwt.shape == expected_shape, f'Expected {expected_shape}, got {x_dwt.shape}'

    print(f'✓ DWT2D forward pass shape correct: {x_dwt.shape}')
    return True


def test_dwt2d_drop_hh_shape():
    """Test DWT2D with drop_hh produces 3 sub-bands."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import DWT2D

    B, C, H, W = 2, 16, 32, 32
    x = torch.randn(B, C, H, W)

    dwt = DWT2D(C, drop_hh=True)
    x_dwt = dwt(x)

    expected_shape = (B, 3 * C, H // 2, W // 2)
    assert x_dwt.shape == expected_shape, f'Expected {expected_shape}, got {x_dwt.shape}'

    print(f'✓ DWT2D (drop_hh) forward pass shape correct: {x_dwt.shape}')
    return True


def test_resoconv_forward_shape():
    """Test ResoConv halves spatial resolution (DWT downsampling)."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import ResoConv

    B, C_in, H, W, C_out = 2, 64, 32, 32, 128
    x = torch.randn(B, C_in, H, W)

    reso = ResoConv(C_in, C_out, shortcut=True)
    x_out = reso(x)

    expected_shape = (B, C_out, H // 2, W // 2)
    assert x_out.shape == expected_shape, f'Expected {expected_shape}, got {x_out.shape}'

    print(f'✓ ResoConv forward pass shape correct: {x_out.shape}')
    print(f'  Input:  {(B, C_in, H, W)}')
    print(f'  Output: {(B, C_out, H // 2, W // 2)} (spatial size halved)')
    return True


def test_resoconv_no_shortcut_shape():
    """Test ResoConv without shortcut."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import ResoConv

    B, C_in, H, W, C_out = 2, 64, 32, 32, 128
    x = torch.randn(B, C_in, H, W)

    reso = ResoConv(C_in, C_out, shortcut=False)
    x_out = reso(x)

    expected_shape = (B, C_out, H // 2, W // 2)
    assert x_out.shape == expected_shape, f'Expected {expected_shape}, got {x_out.shape}'

    print(f'✓ ResoConv (no shortcut) forward pass shape correct: {x_out.shape}')
    return True


def test_filters_are_frozen():
    """Test that DWT2D filters are a buffer (no gradient)."""
    from raycasted.model.blocks.resoconv import DWT2D

    dwt = DWT2D(16)

    assert not dwt.dwt_weight.requires_grad, 'DWT weight buffer should not require grad'
    assert isinstance(dwt.dwt_weight, torch.Tensor), 'DWT weight should be a buffer, not a Parameter'
    assert 'dwt_weight' in dict(dwt.named_buffers()), 'DWT weight should be registered buffer'

    print('✓ DWT2D filters are registered buffer (no gradient, unfreeze-safe)')
    return True


def test_bior22_filter_shape():
    """Test that bior2.2 filter kernel is 6x6 (not 4x4 like db2)."""
    from raycasted.model.blocks.resoconv import DWT2D

    dwt = DWT2D(16)
    assert dwt.dwt_weight.shape[2:] == (6, 6), f'Expected 6x6 kernel, got {dwt.dwt_weight.shape[2:]}'

    print(f'✓ bior2.2 filter kernel shape: {dwt.dwt_weight.shape[2:]} (6x6, was 4x4 for db2)')
    return True


if __name__ == '__main__':
    print('=' * 60)
    print('ResoConv Wavelet Filter Tests (bior2.2)')
    print('=' * 60)

    test_bior22_filters_match_pywt()
    test_dwt2d_forward_shape()
    test_dwt2d_drop_hh_shape()
    test_resoconv_forward_shape()
    test_resoconv_no_shortcut_shape()
    test_filters_are_frozen()
    test_bior22_filter_shape()

    print('=' * 60)
    print('All tests passed!')
    print('=' * 60)