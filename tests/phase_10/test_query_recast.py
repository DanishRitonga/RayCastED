"""Checks for fixed-budget, inter-layer segmentation-guided recasting."""

import pytest
import torch

from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder, SegmentationQueryRecast


def test_identity_and_gradient():
    """Identity initialization learns a correction without changing routing evidence."""
    module = SegmentationQueryRecast(32)
    q = torch.randn(2, 16, 32, requires_grad=True)
    feat = torch.randn(2, 32, 8, 8, requires_grad=True)
    pos = torch.rand(2, 16, 2, requires_grad=True)
    scores = torch.randn(2, 16, 6, requires_grad=True)
    seed = torch.randn(2, 1, 32, 32, requires_grad=True)
    out = module(q, pos, scores, feat, seed)
    torch.testing.assert_close(out, q, rtol=0, atol=0)
    out.square().mean().backward()
    assert module.projection.weight.grad.abs().sum() > 0
    assert scores.grad is None and seed.grad is None and pos.grad is None
    with torch.no_grad():
        module.projection.weight.fill_(0.01)
    module.zero_grad()
    module(q, pos, scores, feat, seed).square().mean().backward()
    assert feat.grad.abs().sum() > 0
    assert module.gate[1].weight.grad.abs().sum() > 0


def make_head(**kwargs):
    """Build a small CPU decoder without downloading weights."""
    return NuLiteDETRDecoder(
        nc=5,
        ch=(16, 32, 64),
        hd=32,
        nh=4,
        ndl=4,
        d_ffn=64,
        n_rays=8,
        grid_size=0.25,
        seed_feature=False,
        mds=False,
        **kwargs,
    )


def test_decoder_recast_preserves_budget_and_backward():
    """Exercise the full decoder in train/eval modes with the same query budget."""
    torch.set_num_threads(2)
    head = make_head(recast=True, recast_after_layer=2)
    feats = [torch.randn(2, c, s, s) for c, s in ((16, 8), (32, 4), (64, 2))]
    seed = torch.randn(2, 1, 32, 32)
    calls = []
    handles = [layer.register_forward_hook(lambda *args: calls.append(1)) for layer in head.decoder.layers]
    out = head(feats, seed_map=seed)
    assert out[0].shape == (4, 2, 16, 10)
    assert len(calls) == 4
    (out[0][-1].square().mean() + out[1][-1].square().mean()).backward()
    assert head.recast_module.projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(head.recast_module.projection.weight.grad).all()
    head.eval()
    calls.clear()
    with torch.no_grad():
        enabled = head(feats, seed_map=seed)[0]
        assert len(calls) == 4
        module = head.recast_module
        head.recast_module = None
        disabled = head(feats, seed_map=seed)[0]
        torch.testing.assert_close(enabled, disabled, rtol=0, atol=0)
        head.recast_module = module
        module.projection.weight.normal_(std=0.1)
        corrected = head(feats, seed_map=seed)[0]
        assert corrected.shape == disabled.shape
        assert not torch.equal(corrected, disabled)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize('kwargs', [dict(respawn=True), dict(no_object=False), dict(recast_after_layer=4)])
def test_invalid_combinations(kwargs):
    """Reject incompatible or ineffective experiment settings."""
    with pytest.raises(ValueError):
        make_head(recast=True, **kwargs)
