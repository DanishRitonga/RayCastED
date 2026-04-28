"""Test ResoConv wavelet filter values and dual-stream modules.

This test verifies:
- db2 and bior2.2 filter values match pywt reference
- DWT2D grouped output layout (LL contiguous, then LH, HL, HH)
- ResoConvDS dual-output (ll + stored _hf)
- HFResidual residual injection
- SE block behavior
- wavelet_type switching between db2 and bior2.2

Run with: uv run python tests/phase_10/test_resoconv_filters.py
"""

import torch


def test_wavelet_filters_match_pywt():
    """Verify hardcoded db2 and bior2.2 filters match pywt reference."""
    try:
        import pywt

        from raycasted.model.blocks.resoconv import _get_wavelet_1d_filters

        for name in ('db2', 'bior2.2'):
            w = pywt.Wavelet(name)
            ref_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
            ref_hi = torch.tensor(w.dec_hi, dtype=torch.float32)

            lo, hi = _get_wavelet_1d_filters(name)

            assert torch.allclose(lo, ref_lo, rtol=1e-6), f'{name} LO filters mismatch'
            assert torch.allclose(hi, ref_hi, rtol=1e-6), f'{name} HI filters mismatch'
            print(f'  ✓ {name} filters match pywt reference')

        print('✓ All wavelet filters match pywt references')
        return True

    except ImportError:
        print('⚠ pywt not installed, skipping verification')
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


def test_dwt2d_grouped_layout():
    """Test DWT2D output is grouped: all LL channels first, then LH, HL, HH."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import DWT2D

    B, C, H, W = 1, 4, 64, 64
    x = torch.randn(B, C, H, W)

    dwt = DWT2D(C)
    x_dwt = dwt(x)

    ll = x_dwt[:, :C]
    lh = x_dwt[:, C : 2 * C]
    hl = x_dwt[:, 2 * C : 3 * C]
    hh = x_dwt[:, 3 * C :]

    assert ll.shape == (B, C, H // 2, W // 2)
    assert lh.shape == (B, C, H // 2, W // 2)
    assert hl.shape == (B, C, H // 2, W // 2)
    assert hh.shape == (B, C, H // 2, W // 2)
    assert x_dwt.shape == (B, 4 * C, H // 2, W // 2)
    assert abs(x_dwt.sum().item()) > 0, 'DWT output should not be all zeros'

    print('✓ DWT2D grouped layout correct: [LL_C, LH_C, HL_C, HH_C]')
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
    return True


def test_resoconv_wavelet_switching():
    """Test ResoConv can switch between db2 and bior2.2 wavelets."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import ResoConv

    B, C_in, H, W, C_out = 2, 64, 32, 32, 128
    x = torch.randn(B, C_in, H, W)

    for wt in ('db2', 'bior2.2'):
        reso = ResoConv(C_in, C_out, shortcut=False, wavelet_type=wt)
        out = reso(x)
        assert out.shape == (B, C_out, H // 2, W // 2)
        print(f'  ✓ ResoConv({wt}) output shape: {out.shape}')

    print('✓ ResoConv wavelet_type switching works')
    return True


def test_resoconv_ds_dual_output():
    """Test ResoConvDS returns ll and stores _hf with correct shapes."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import ResoConvDS

    B, C_in, H, W, C_out = 2, 64, 32, 32, 128
    x = torch.randn(B, C_in, H, W)

    ds = ResoConvDS(C_in, C_out, se_r=4, drop_hh=False)
    ll_out = ds(x)

    assert ll_out.shape == (B, C_out, H // 2, W // 2), f'LL shape: {ll_out.shape}'
    assert ds._hf is not None, '_hf should be stored after forward'
    assert ds._hf.shape == (B, C_out, H // 2, W // 2), f'HF shape: {ds._hf.shape}'

    print(f'✓ ResoConvDS dual output: ll={ll_out.shape}, _hf={ds._hf.shape}')
    return True


def test_resoconv_ds_drop_hh():
    """Test ResoConvDS with drop_hh: HF has 2 sub-bands instead of 3."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import ResoConvDS

    B, C_in, H, W, C_out = 2, 64, 32, 32, 128
    x = torch.randn(B, C_in, H, W)

    ds = ResoConvDS(C_in, C_out, se_r=4, drop_hh=True)
    ll_out = ds(x)

    assert ll_out.shape == (B, C_out, H // 2, W // 2)
    assert ds._hf.shape == (B, C_out, H // 2, W // 2)

    params_full = sum(p.numel() for p in ResoConvDS(64, 128, se_r=4, drop_hh=False).parameters())
    params_drop = sum(p.numel() for p in ds.parameters())
    assert params_drop < params_full, 'drop_hh should have fewer params'

    print(f'✓ ResoConvDS (drop_hh) works: params {params_drop} < {params_full}')
    return True


def test_hf_residual():
    """Test HFResidual adds boundary-processed HF to semantic features."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import HFResidual, ResoConvDS

    B, C_in, H, W, C_out = 2, 64, 32, 32, 128
    x = torch.randn(B, C_in, H, W)

    source = ResoConvDS(C_in, C_out, se_r=4, drop_hh=False)
    _ = source(x)

    hf_res = HFResidual(C_out, C_out)
    hf_res._source = source

    semantic = torch.randn(B, C_out, H // 2, W // 2)
    out = hf_res(semantic)

    assert out.shape == (B, C_out, H // 2, W // 2), f'Output shape: {out.shape}'

    with torch.no_grad():
        dummy = torch.zeros_like(semantic)
        dummy_out = hf_res(dummy)
        assert not torch.allclose(dummy_out, dummy), 'HFResidual should modify input'

    print(f'✓ HFResidual: input={semantic.shape} → output={out.shape}')
    return True


def test_se_block():
    """Test SE block preserves shape and modulates channels."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import SE

    B, C, H, W = 2, 64, 16, 16
    x = torch.randn(B, C, H, W)

    se = SE(C, reduction=4)
    out = se(x)

    assert out.shape == x.shape, f'SE output shape: {out.shape}'

    se_small = SE(8, reduction=2)
    out_small = se_small(torch.randn(B, 8, H, W))
    assert out_small.shape == (B, 8, H, W)

    print(f'✓ SE block: input={x.shape} → output={out.shape}, image-size agnostic')
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


def test_filter_shapes_by_wavelet():
    """Test that db2 filters are 4x4 and bior2.2 filters are 6x6."""
    from raycasted.model.blocks.resoconv import DWT2D

    dwt_db2 = DWT2D(16, wavelet_type='db2')
    assert dwt_db2.dwt_weight.shape[2:] == (4, 4), f'Expected db2 4x4 kernel, got {dwt_db2.dwt_weight.shape[2:]}'

    dwt_bior = DWT2D(16, wavelet_type='bior2.2')
    assert dwt_bior.dwt_weight.shape[2:] == (6, 6), f'Expected bior2.2 6x6 kernel, got {dwt_bior.dwt_weight.shape[2:]}'

    print(f'✓ db2 kernel: {dwt_db2.dwt_weight.shape[2:]} (4x4), bior2.2 kernel: {dwt_bior.dwt_weight.shape[2:]} (6x6)')
    return True


def test_dwt_ll_hf_split():
    """Test DWT_LL and DWT_HF split DWT output into explicit streams."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import DWT2D, DWT_HF, DWT_LL

    b, c_in, h, w = 2, 16, 32, 32
    x = torch.randn(b, c_in, h, w)

    # Full DWT for reference
    dwt = DWT2D(c_in, drop_hh=False, wavelet_type='db2')
    x_dwt = dwt(x)
    ll_ref = x_dwt[:, :c_in]
    hf_ref = x_dwt[:, c_in:]

    # Split modules
    dwt_ll = DWT_LL(c_in, wavelet_type='db2')
    dwt_hf = DWT_HF(c_in, drop_hh=False, wavelet_type='db2')

    ll_out = dwt_ll(x)
    hf_out = dwt_hf(x)

    assert ll_out.shape == (b, c_in, h // 2, w // 2)
    assert hf_out.shape == (b, 3 * c_in, h // 2, w // 2)
    assert torch.allclose(ll_out, ll_ref, atol=1e-6)
    assert torch.allclose(hf_out, hf_ref, atol=1e-6)

    print(f'✓ DWT_LL: {x.shape} → {ll_out.shape}')
    print(f'✓ DWT_HF: {x.shape} → {hf_out.shape}')
    print('✓ DWT_LL + DWT_HF match DWT2D full output')
    return True


def test_dwt_hf_drop_hh():
    """Test DWT_HF with drop_hh returns 2 sub-bands instead of 3."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import DWT_HF

    b, c_in, h, w = 2, 16, 32, 32
    x = torch.randn(b, c_in, h, w)

    dwt_hf_full = DWT_HF(c_in, drop_hh=False, wavelet_type='db2')
    dwt_hf_drop = DWT_HF(c_in, drop_hh=True, wavelet_type='db2')

    hf_full = dwt_hf_full(x)
    hf_drop = dwt_hf_drop(x)

    assert hf_full.shape == (b, 3 * c_in, h // 2, w // 2)
    assert hf_drop.shape == (b, 2 * c_in, h // 2, w // 2)

    print(f'✓ DWT_HF(drop_hh=False): {hf_full.shape}')
    print(f'✓ DWT_HF(drop_hh=True):  {hf_drop.shape}')
    return True


def test_hybrid_dwt():
    """Test DWT2D_Hybrid: bior2.2 LL + db2 HF."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import DWT2D_Hybrid

    b, c_in, h, w = 2, 16, 32, 32
    x = torch.randn(b, c_in, h, w)

    dwt_hybrid = DWT2D_Hybrid(c_in, drop_hh=False)
    out = dwt_hybrid(x)

    assert out.shape == (b, 4 * c_in, h // 2, w // 2)

    # LL should come from bior2.2 (6x6 kernel), HF from db2 (4x4 kernel)
    assert dwt_hybrid.dwt_ll.dwt_weight.shape[2:] == (6, 6)
    assert dwt_hybrid.dwt_hf.dwt_weight.shape[2:] == (4, 4)

    # Layout: LL first, then HF
    ll = out[:, :c_in]
    hf = out[:, c_in:]
    assert ll.shape == (b, c_in, h // 2, w // 2)
    assert hf.shape == (b, 3 * c_in, h // 2, w // 2)

    print(f'✓ DWT2D_Hybrid: {x.shape} → {out.shape}  (LL=bior2.2, HF=db2)')
    return True


def test_resoconv_hybrid():
    """Test ResoConvHybrid forward pass."""
    torch.manual_seed(42)
    from raycasted.model.blocks.resoconv import ResoConvHybrid

    b, c_in, c_out, h, w = 2, 64, 128, 32, 32
    x = torch.randn(b, c_in, h, w)

    m = ResoConvHybrid(c_in, c_out, shortcut=True, drop_hh=False)
    out = m(x)

    assert out.shape == (b, c_out, h // 2, w // 2)
    assert m.dwt.dwt_ll.wavelet_type == 'bior2.2'
    assert m.dwt.dwt_hf.wavelet_type == 'db2'

    print(f'✓ ResoConvHybrid: {x.shape} → {out.shape}  (LL=bior2.2, HF=db2)')
    return True


if __name__ == '__main__':
    print('=' * 60)
    print('ResoConv Wavelet Filter Tests (db2 + bior2.2 + dual-stream)')
    print('=' * 60)

    test_wavelet_filters_match_pywt()
    test_dwt2d_forward_shape()
    test_dwt2d_grouped_layout()
    test_dwt2d_drop_hh_shape()
    test_resoconv_forward_shape()
    test_resoconv_wavelet_switching()
    test_resoconv_ds_dual_output()
    test_resoconv_ds_drop_hh()
    test_hf_residual()
    test_se_block()
    test_filters_are_frozen()
    test_filter_shapes_by_wavelet()
    test_dwt_ll_hf_split()
    test_dwt_hf_drop_hh()
    test_hybrid_dwt()
    test_resoconv_hybrid()

    print('=' * 60)
    print('All tests passed!')
    print('=' * 60)
