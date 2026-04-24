"""Test ResoConv wavelet filter values.

This test verifies that the hardcoded fallback values in resoconv.py
match the reference implementation from pywt.Wavelet('db2').

Run with: uv run python tests/phase_10/test_resoconv_filters.py
"""

import torch


def test_db2_filters_match_pywt():
    """Verify hardcoded DB2 filters match pywt reference."""
    try:
        import pywt

        # Get reference values from pywt
        w = pywt.Wavelet('db2')
        ref_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        ref_hi = torch.tensor(w.dec_hi, dtype=torch.float32)

        # Get our values
        from raycasted.model.blocks.resoconv import _DB2_LO, _DB2_HI

        # Verify they match (within floating-point precision)
        assert torch.allclose(_DB2_LO, ref_lo, rtol=1e-6), f'LO filters mismatch'
        assert torch.allclose(_DB2_HI, ref_hi, rtol=1e-6), f'HI filters mismatch'

        print('✓ DB2 filters match pywt.Wavelet("db2") reference')
        print(f'  dec_lo: {_DB2_LO.tolist()}')
        print(f'  dec_hi: {_DB2_HI.tolist()}')
        return True

    except ImportError:
        print('⚠ pywt not installed, skipping verification')
        print('  (hardcoded fallback values will be used)')
        return False


def test_dwt2d_forward_shape():
    """Test DWT2D produces expected output shape."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import DWT2D

    # Create test input
    B, C, H, W = 2, 16, 32, 32
    x = torch.randn(B, C, H, W)

    # Apply DWT
    dwt = DWT2D(C)
    x_dwt = dwt(x)

    # Expected: [B, 4*C, H//2, W//2] (DWT halves spatial resolution)
    expected_shape = (B, 4 * C, H // 2, W // 2)
    assert x_dwt.shape == expected_shape, f'Expected {expected_shape}, got {x_dwt.shape}'

    print(f'✓ DWT2D forward pass shape correct: {x_dwt.shape}')
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


def test_filters_are_frozen():
    """Test that DWT2D filters are a buffer (no gradient)."""
    from raycasted.model.blocks.resoconv import DWT2D

    dwt = DWT2D(16)

    assert not dwt.dwt_weight.requires_grad, 'DWT weight buffer should not require grad'
    assert isinstance(dwt.dwt_weight, torch.Tensor), 'DWT weight should be a buffer, not a Parameter'
    assert 'dwt_weight' in dict(dwt.named_buffers()), 'DWT weight should be registered buffer'

    print('✓ DWT2D filters are registered buffer (no gradient, unfreeze-safe)')
    return True


if __name__ == '__main__':
    print('=' * 60)
    print('ResoConv Wavelet Filter Tests')
    print('=' * 60)

    test_db2_filters_match_pywt()
    test_dwt2d_forward_shape()
    test_resoconv_forward_shape()
    test_filters_are_frozen()

    print('=' * 60)
    print('All tests passed!')
    print('=' * 60)
